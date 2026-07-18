"""Engine do Concept Fingerprint — Documento 112.7D, seções 12–16, 27 e 30.

Pipeline determinístico por evento:

    extract_terms → resolução direta (TermIndex) → expansões explícitas
    (ancestrais com decaimento, related de 1 salto) → filtro de
    minimum_contribution → agregação noisy-or → ordenação canônica → hash.

Regras centrais:

- contribuição direta = peso do campo (correspondências concept_id,
  canonical_name e synonym valem 1.00 na V1 — seção 12);
- ancestral a distância d recebe contribuição × parent_decay^d (seção 13),
  no máximo `max_parent_depth` saltos, nunca inferindo descendentes;
- related recebe contribuição × related_decay, exatamente 1 salto, sem
  mistura recursiva com parent (seções 14–15); em expansões, o
  `canonical_concept_id` da evidência registra o conceito que a originou;
- `minimum_contribution` descarta contribuições ANTES da agregação:
  não geram MatchEvidence nem entram em parcela; conceito que perde todas
  as contribuições é omitido (seção 27);
- evidências idênticas (mesmo campo, termo, conceito, tipo e contribuição)
  são registradas uma única vez — duplicatas literais não carregam sinal
  novo e a saída permanece estável sob permutação de keywords/entities;
- termos não resolvidos vão para `unmatched_terms` e nunca criam conceitos.

O engine não consulta Research Profile, não calcula relevância ou
significância, não acessa banco, não chama LLM e não gera embeddings
(invariantes da seção 46).
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from sentinela.core.models import Event
from sentinela.fingerprint.aggregation import (
    build_signal,
    signal_sort_key,
)
from sentinela.fingerprint.extractor import FieldTerm, extract_terms
from sentinela.fingerprint.hashing import (
    canonical_payload,
    compute_fingerprint_hash,
)
from sentinela.fingerprint.models import (
    ConceptFingerprint,
    ConceptMatchType,
    FingerprintConfig,
    FingerprintField,
    MatchEvidence,
    UnmatchedTerm,
)
from sentinela.fingerprint.resolver import (
    TermEntry,
    TermIndex,
    TermResolution,
)
from sentinela.taxonomy.models import normalize_term
from sentinela.taxonomy.taxonomy import TaxonomyIndex

#: versão do algoritmo de construção do fingerprint (seção 24).
ALGORITHM_VERSION = "1"

REASON_NOT_IN_TAXONOMY = "not_in_taxonomy"
REASON_AMBIGUOUS = "ambiguous_term"


class FingerprintConfigLoadError(ValueError):
    """Erro de carregamento da configuração, com o caminho do arquivo."""


def load_fingerprint_config(path: str | Path) -> FingerprintConfig:
    """Carrega o YAML de configuração (seção 27) e valida o contrato."""
    p = Path(path)
    if not p.exists():
        raise FingerprintConfigLoadError(
            f"arquivo de configuração não encontrado: {p}"
        )
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise FingerprintConfigLoadError(f"YAML inválido em {p.name}: {e}") from e
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise FingerprintConfigLoadError(
            f"{p.name}: esperado um mapeamento de configuração no topo"
        )
    try:
        return FingerprintConfig.model_validate(data)
    except ValidationError as e:
        raise FingerprintConfigLoadError(
            f"{p.name}: configuração inválida:\n{e}"
        ) from e


class ConceptFingerprintEngine:
    """Constrói ConceptFingerprint determinísticos para eventos validados.

    O índice de termos e o matcher textual são construídos uma única vez,
    no `__init__`, e reutilizados por todos os eventos (seção 37).
    """

    def __init__(self, taxonomy: TaxonomyIndex, config: FingerprintConfig) -> None:
        self._taxonomy = taxonomy
        self._config = config
        self._index = TermIndex(taxonomy.taxonomy.concepts)

    # ------------------------------------------------------------ público
    def build(self, event: Event) -> ConceptFingerprint:
        cfg = self._config

        direct: list[MatchEvidence] = []
        unmatched: list[UnmatchedTerm] = []

        for term in extract_terms(event):
            if term.is_text:
                self._resolve_text(term, direct, unmatched)
            else:
                self._resolve_discrete(term, direct, unmatched)

        direct = list(dict.fromkeys(direct))

        # (conceito do signal, evidência): em expansões o conceito do signal
        # é o ancestral/relacionado; a evidência registra a origem.
        pairs: list[tuple[str, MatchEvidence]] = [
            (ev.canonical_concept_id, ev) for ev in direct
        ]
        pairs.extend(self._expand_parents(direct))
        pairs.extend(self._expand_related(direct))
        pairs = list(dict.fromkeys(pairs))

        by_concept: dict[str, list[MatchEvidence]] = {}
        for concept_id, evidence in pairs:
            by_concept.setdefault(concept_id, []).append(evidence)

        signals = [
            build_signal(concept_id, self._taxonomy.get(concept_id).domain, evs)
            for concept_id, evs in by_concept.items()
        ]
        signals.sort(key=signal_sort_key)

        unmatched = sorted(
            dict.fromkeys(unmatched),
            key=lambda u: (u.field.value, u.normalized_term),
        )

        source_fields = tuple(
            sorted({ev.field for _, ev in pairs}, key=lambda f: f.value)
        )

        fingerprint = ConceptFingerprint(
            event_id=event.id,
            taxonomy_version=self._taxonomy.taxonomy.version,
            algorithm_version=ALGORITHM_VERSION,
            weights_version=cfg.version,
            concepts=tuple(signals),
            unmatched_terms=tuple(unmatched),
            source_fields=source_fields,
            fingerprint_hash="",
        )
        return fingerprint.model_copy(
            update={
                "fingerprint_hash": compute_fingerprint_hash(
                    canonical_payload(fingerprint)
                )
            }
        )

    # -------------------------------------------------------- resolução
    def _resolve_discrete(
        self,
        term: FieldTerm,
        direct: list[MatchEvidence],
        unmatched: list[UnmatchedTerm],
    ) -> None:
        resolution = self._index.resolve_discrete(term.input_term)
        if resolution.status is TermResolution.MATCH:
            evidence = self._direct_evidence(term, resolution.entry)
            if evidence is not None:
                direct.append(evidence)
            return
        reason = (
            REASON_AMBIGUOUS
            if resolution.status is TermResolution.AMBIGUOUS
            else REASON_NOT_IN_TAXONOMY
        )
        unmatched.append(self._unmatched(term, reason))

    def _resolve_text(
        self,
        term: FieldTerm,
        direct: list[MatchEvidence],
        unmatched: list[UnmatchedTerm],
    ) -> None:
        for match in self._index.find_in_text(term.input_term):
            if match.resolution.status is TermResolution.AMBIGUOUS:
                unmatched.append(self._unmatched(term, REASON_AMBIGUOUS))
                continue
            evidence = self._direct_evidence(term, match.resolution.entry)
            if evidence is not None:
                direct.append(evidence)

    def _direct_evidence(
        self, term: FieldTerm, entry: TermEntry
    ) -> MatchEvidence | None:
        contribution = self._field_weight(term.field)
        if contribution < self._config.minimum_contribution:
            return None
        return MatchEvidence(
            field=term.field,
            input_term=term.input_term,
            normalized_term=normalize_term(term.input_term),
            matched_term=entry.matched_term,
            canonical_concept_id=entry.concept_id,
            match_type=entry.match_type,
            contribution=contribution,
        )

    def _field_weight(self, field: FingerprintField) -> float:
        # os valores do enum coincidem com os nomes dos pesos (seção 11).
        return getattr(self._config.weights, field.value)

    @staticmethod
    def _unmatched(term: FieldTerm, reason: str) -> UnmatchedTerm:
        return UnmatchedTerm(
            field=term.field,
            input_term=term.input_term,
            normalized_term=normalize_term(term.input_term),
            reason=reason,
        )

    # -------------------------------------------------------- expansões
    def _expand_parents(
        self, direct: list[MatchEvidence]
    ) -> list[tuple[str, MatchEvidence]]:
        cfg = self._config
        if not cfg.include_parents or cfg.max_parent_depth == 0:
            return []
        out: list[tuple[str, MatchEvidence]] = []
        for origin in direct:
            ancestors = self._taxonomy.ancestors(origin.canonical_concept_id)
            for distance, ancestor in enumerate(
                ancestors[: cfg.max_parent_depth], start=1
            ):
                contribution = origin.contribution * (
                    cfg.weights.parent_decay**distance
                )
                if contribution < cfg.minimum_contribution:
                    continue
                out.append(
                    (
                        ancestor.id,
                        self._expansion_evidence(
                            origin, ConceptMatchType.PARENT, contribution
                        ),
                    )
                )
        return out

    def _expand_related(
        self, direct: list[MatchEvidence]
    ) -> list[tuple[str, MatchEvidence]]:
        cfg = self._config
        if not cfg.include_related or cfg.max_related_depth == 0:
            return []
        out: list[tuple[str, MatchEvidence]] = []
        for origin in direct:
            contribution = origin.contribution * cfg.weights.related_decay
            if contribution < cfg.minimum_contribution:
                continue
            for related in self._taxonomy.related(origin.canonical_concept_id):
                out.append(
                    (
                        related.id,
                        self._expansion_evidence(
                            origin, ConceptMatchType.RELATED, contribution
                        ),
                    )
                )
        return out

    @staticmethod
    def _expansion_evidence(
        origin: MatchEvidence,
        match_type: ConceptMatchType,
        contribution: float,
    ) -> MatchEvidence:
        # canonical_concept_id registra o conceito que ORIGINOU a expansão.
        return MatchEvidence(
            field=origin.field,
            input_term=origin.input_term,
            normalized_term=origin.normalized_term,
            matched_term=origin.matched_term,
            canonical_concept_id=origin.canonical_concept_id,
            match_type=match_type,
            contribution=contribution,
        )


__all__ = [
    "ALGORITHM_VERSION",
    "ConceptFingerprintEngine",
    "FingerprintConfigLoadError",
    "REASON_AMBIGUOUS",
    "REASON_NOT_IN_TAXONOMY",
    "load_fingerprint_config",
]
