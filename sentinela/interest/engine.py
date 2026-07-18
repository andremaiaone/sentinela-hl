"""Interest Engine — Documento 112.7E, seções 5, 6, 8, 11 e 18.

Mede o grau de aderência de um evento científico à agenda de UM Research
Profile, combinando o ConceptFingerprint já construído (112.7D) com as
entradas do perfil (112.7B), pela fórmula declarada na seção 9.

Regra fundamental (seção 8): todo sinal conceitual vem exclusivamente do
fingerprint. O engine não lê texto do evento — do `Event` consome exatamente
`event.id` (seção 6.1); não acessa a Taxonomia, não refaz matching, não
consulta banco/rede, não chama LLM, não usa embeddings e não persiste nada.

O perfil é vinculado no `__init__` (seção 18.1): os mapas de consulta são
construídos uma única vez e uma instância conhece exatamente um perfil —
N pesquisadores = N instâncias baratas sobre o mesmo fingerprint.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from sentinela.core.models import Event
from sentinela.fingerprint.hashing import canonical_json
from sentinela.fingerprint.models import ConceptFingerprint

from .models import ResearchProfile
from .results import (
    InterestContribution,
    InterestLayer,
    InterestResult,
    MatchedResearchLine,
)
from .scoring import (
    CONCEPT_LAYERS,
    aggregate,
    atomic_contribution,
    compose,
    contribution_sort_key,
    domain_contribution,
)
from .validator import normalize_free_term

#: versão da fórmula deste documento (seções 13 e 16).
ALGORITHM_VERSION = "1"

#: campos fora do payload canônico do hash (seção 14).
_HASH_EXCLUDED = {"generated_at", "interest_hash"}


@dataclass(frozen=True)
class _LinePlan:
    """Uma ResearchLine pré-compilada: entradas (layer, concept_id, weight)."""
    line_id: str
    priority: float
    entries: tuple[tuple[InterestLayer, str, float], ...]


def _line_plan(line) -> _LinePlan:
    entries = [
        (InterestLayer.CONCEPT, c.concept_id, c.weight) for c in line.concepts
    ]
    entries += [
        (InterestLayer.REGION, r.concept_id, r.weight) for r in line.regions
    ]
    entries += [
        (InterestLayer.INSTRUMENT, i.concept_id, i.weight)
        for i in line.instruments
    ]
    entries.sort(key=lambda e: (e[0].value, e[1]))
    return _LinePlan(
        line_id=line.id,
        priority=line.priority,
        entries=tuple(entries),
    )


class InterestEngine:
    """Avalia a aderência de eventos à agenda de um único perfil."""

    ALGORITHM_VERSION = ALGORITHM_VERSION

    def __init__(self, profile: ResearchProfile) -> None:
        self._profile = profile
        self._layers: dict[InterestLayer, dict[str, float]] = {
            InterestLayer.CONCEPT: {
                c.concept_id: c.weight for c in profile.concepts
            },
            InterestLayer.REGION: {
                r.concept_id: r.weight for r in profile.regions
            },
            InterestLayer.INSTRUMENT: {
                i.concept_id: i.weight for i in profile.instruments
            },
        }
        self._domains: dict[str, float] = {
            d.domain_id: d.weight for d in profile.domains
        }
        self._lines: tuple[_LinePlan, ...] = tuple(
            sorted(
                (_line_plan(line) for line in profile.research_lines),
                key=lambda plan: plan.line_id,
            )
        )
        self._excluded: frozenset[str] = frozenset(
            term
            for term in (normalize_free_term(t) for t in profile.excluded_topics)
            if term
        )

    # ------------------------------------------------------------ público
    def evaluate(
        self,
        event: Event,
        fingerprint: ConceptFingerprint,
    ) -> InterestResult:
        signals = fingerprint.concepts
        by_concept = {s.concept_id: s for s in signals}

        contributions: list[InterestContribution] = []

        # camadas globais por conceito (seção 9.1)
        layer_scores: dict[InterestLayer, float] = {}
        for layer in CONCEPT_LAYERS:
            entries = self._layers[layer]
            layer_contributions = [
                atomic_contribution(layer, s.concept_id, entries[s.concept_id], s.weight)
                for s in signals
                if s.concept_id in entries
            ]
            contributions.extend(layer_contributions)
            layer_scores[layer] = aggregate(layer_contributions)

        # camada de domínio — uma contribuição por domínio (seção 9.1)
        domain_contributions = [
            contribution
            for domain_id in sorted(self._domains)
            if (
                contribution := domain_contribution(
                    domain_id,
                    self._domains[domain_id],
                    [s for s in signals if s.domain_id == domain_id],
                )
            )
            is not None
        ]
        contributions.extend(domain_contributions)
        domain_score = aggregate(domain_contributions)

        # linhas de pesquisa (seções 9.3–9.4)
        matched: list[MatchedResearchLine] = []
        line_terms: list[tuple[str, float]] = []
        for plan in self._lines:
            line_contributions = [
                atomic_contribution(
                    layer, concept_id, weight, by_concept[concept_id].weight,
                    plan.line_id,
                )
                for layer, concept_id, weight in plan.entries
                if concept_id in by_concept
            ]
            contributions.extend(line_contributions)
            line_score = aggregate(line_contributions)
            if line_score > 0.0:
                matched.append(
                    MatchedResearchLine(
                        line_id=plan.line_id,
                        priority=plan.priority,
                        line_score=line_score,
                    )
                )
                line_terms.append((plan.line_id, plan.priority * line_score))

        # composição final em ordem fixa (seções 9.4 e 9.5.3)
        score = compose(
            [
                domain_score,
                layer_scores[InterestLayer.CONCEPT],
                layer_scores[InterestLayer.REGION],
                layer_scores[InterestLayer.INSTRUMENT],
                *(term for _, term in sorted(line_terms)),
            ]
        )

        excluded, exclusion_reason = self._check_exclusions(fingerprint)
        if excluded:
            score = 0.0                       # veto declarado (seção 11.1)

        result = InterestResult(
            event_id=event.id,
            profile_id=self._profile.researcher.id,
            profile_version=self._profile.researcher.version,
            fingerprint_hash=fingerprint.fingerprint_hash,
            taxonomy_version=fingerprint.taxonomy_version,
            algorithm_version=ALGORITHM_VERSION,
            interest_score=score,
            contributions=tuple(
                sorted(contributions, key=contribution_sort_key)
            ),
            matched_lines=tuple(
                sorted(matched, key=lambda m: m.line_id)
            ),
            excluded=excluded,
            exclusion_reason=exclusion_reason,
            interest_hash="",
        )
        return result.model_copy(
            update={"interest_hash": self._compute_hash(result)}
        )

    # ---------------------------------------------------------- exclusões
    def _check_exclusions(
        self, fingerprint: ConceptFingerprint
    ) -> tuple[bool, str | None]:
        """Veto do perfil por igualdade exata sobre o conteúdo do fingerprint
        (seção 11.1): {concept_id} ∪ {domain_id} ∪ {matched_term}."""
        if not self._excluded:
            return False, None
        targets: set[str] = set()
        for signal in fingerprint.concepts:
            targets.add(signal.concept_id)
            targets.add(signal.domain_id)
            for evidence in signal.evidence:
                targets.add(evidence.matched_term)
        matches = sorted(
            (term, raw)
            for term in self._excluded
            for raw in targets
            if normalize_free_term(raw) == term
        )
        if not matches:
            return False, None
        term, raw = matches[0]
        return True, f"excluded topic {term!r} matched {raw!r}"

    # --------------------------------------------------------------- hash
    @staticmethod
    def _compute_hash(result: InterestResult) -> str:
        """SHA-256 do JSON canônico (seção 14) — canonização única do
        projeto, reutilizada de `sentinela.fingerprint.hashing`."""
        payload = result.model_dump(mode="json", exclude=_HASH_EXCLUDED)
        return hashlib.sha256(canonical_json(payload)).hexdigest()


__all__ = ["ALGORITHM_VERSION", "InterestEngine"]
