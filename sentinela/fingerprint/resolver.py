"""Resolução taxonômica de termos — Documento 112.7D, seções 12, 18 e 19.

Índice determinístico de termos da Taxonomia, construído UMA vez no
`__init__` do engine a partir de `taxonomy_index.taxonomy.concepts` (seção 37)
— o `TaxonomyIndex` não é modificado.

Chaves do índice:

- `concept_id` é indexado pela sua forma canônica (`normalize_term`, sem dobra
  de pontuação): o trecho precisa ser literalmente o id ("cme", "mineral_dust");
- nome canônico e sinônimos são indexados pela forma de comparação (seção 10).

Resolução de uma entrada:

- entradas de conceitos distintos sob a mesma chave → AMBIGUOUS (seção 19):
  nunca escolher arbitrariamente;
- várias entradas do MESMO conceito → uma só, pela prioridade da seção 12:
  `concept_id > canonical_name > synonym`.

Busca textual (seções 18.2–18.3): correspondência por fronteira de palavra
(tokens da forma de comparação), preferência por termos mais longos e nenhuma
duplicação por sobreposição — os termos são processados na ordenação da seção
18.3 (mais caracteres, mais tokens, ordem lexical) e cada um reivindica seus
trechos com exclusividade, inclusive entre conceitos diferentes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional

from sentinela.fingerprint.models import ConceptMatchType
from sentinela.fingerprint.normalization import comparison_form, comparison_tokens
from sentinela.taxonomy.models import Concept, normalize_term


class TermResolution(str, Enum):
    NO_MATCH = "no_match"
    MATCH = "match"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class TermEntry:
    """Um termo taxonômico indexado."""
    concept_id: str
    match_type: ConceptMatchType           # CONCEPT_ID | CANONICAL_NAME | SYNONYM
    matched_term: str                      # forma canônica do termo declarado


@dataclass(frozen=True)
class Resolution:
    """Resultado da resolução de um termo de entrada."""
    status: TermResolution
    entry: Optional[TermEntry] = None      # presente quando status == MATCH


@dataclass(frozen=True)
class TextMatch:
    """Um trecho de texto reivindicado por uma chave do índice."""
    start: int                             # token inicial (na forma de comparação)
    length: int                            # número de tokens do trecho
    resolution: Resolution                 # MATCH ou AMBIGUOUS (nunca NO_MATCH)


#: prioridade da seção 12 (menor = mais prioritário).
_TYPE_PRIORITY = {
    ConceptMatchType.CONCEPT_ID: 0,
    ConceptMatchType.CANONICAL_NAME: 1,
    ConceptMatchType.SYNONYM: 2,
}


class TermIndex:
    """Índice imutável de termos taxonômicos. Somente leitura após construído."""

    def __init__(self, concepts: Iterable[Concept]):
        by_key: dict[str, list[TermEntry]] = {}

        def add(key: str, entry: TermEntry) -> None:
            if not key:
                return
            entries = by_key.setdefault(key, [])
            if entry not in entries:
                entries.append(entry)

        for concept in concepts:
            add(
                normalize_term(concept.id),
                TermEntry(concept.id, ConceptMatchType.CONCEPT_ID, concept.id),
            )
            add(
                comparison_form(concept.name),
                TermEntry(
                    concept.id,
                    ConceptMatchType.CANONICAL_NAME,
                    normalize_term(concept.name),
                ),
            )
            for synonym in concept.synonyms:
                add(
                    comparison_form(synonym),
                    TermEntry(
                        concept.id,
                        ConceptMatchType.SYNONYM,
                        normalize_term(synonym),
                    ),
                )

        self._by_key: dict[str, tuple[TermEntry, ...]] = {
            key: tuple(entries) for key, entries in by_key.items()
        }
        # ordenação da seção 18.3: mais caracteres, mais tokens, ordem lexical.
        self._ordered_keys: tuple[str, ...] = tuple(
            sorted(
                self._by_key,
                key=lambda k: (-len(k), -(k.count(" ") + 1), k),
            )
        )
        self._key_tokens: dict[str, tuple[str, ...]] = {
            key: tuple(key.split(" ")) for key in self._by_key
        }

    # --------------------------------------------------------- resolução
    @staticmethod
    def _decide(entries: Iterable[TermEntry]) -> Resolution:
        pool = list(entries)
        if not pool:
            return Resolution(TermResolution.NO_MATCH)
        if len({e.concept_id for e in pool}) > 1:
            return Resolution(TermResolution.AMBIGUOUS)
        best = min(
            pool,
            key=lambda e: (
                _TYPE_PRIORITY[e.match_type],
                e.matched_term,
                e.concept_id,
            ),
        )
        return Resolution(TermResolution.MATCH, best)

    def resolve_discrete(self, term: str) -> Resolution:
        """Resolve um valor discreto (keyword, entity, category...).

        Consulta a forma canônica (captura concept_id e termos sem pontuação)
        e a forma de comparação (captura termos com pontuação dobrada).
        """
        seen: set[TermEntry] = set()
        entries: list[TermEntry] = []
        for key in (normalize_term(term), comparison_form(term)):
            for entry in self._by_key.get(key, ()):
                if entry not in seen:
                    seen.add(entry)
                    entries.append(entry)
        return self._decide(entries)

    def _resolve_key(self, key: str) -> Resolution:
        return self._decide(self._by_key[key])

    # -------------------------------------------------------- busca texto
    def find_in_text(self, text: str) -> list[TextMatch]:
        """Localiza termos taxonômicos no texto por fronteira de palavra.

        Cada chave, na ordenação da seção 18.3, reivindica com exclusividade
        os trechos livres onde ocorre; um trecho reivindicado não gera
        correspondência para termos mais curtos/sobrepostos.
        """
        tokens = comparison_tokens(text)
        n = len(tokens)
        if n == 0:
            return []

        positions: dict[str, list[int]] = {}
        for i, token in enumerate(tokens):
            positions.setdefault(token, []).append(i)

        owner: list[Optional[str]] = [None] * n
        matches: list[TextMatch] = []

        for key in self._ordered_keys:
            key_tokens = self._key_tokens[key]
            size = len(key_tokens)
            if size > n:
                continue
            for start in positions.get(key_tokens[0], ()):
                end = start + size
                if end > n or owner[start] is not None:
                    continue
                if tuple(tokens[start:end]) != key_tokens:
                    continue
                if any(owner[i] is not None for i in range(start, end)):
                    continue
                for i in range(start, end):
                    owner[i] = key
                matches.append(TextMatch(start, size, self._resolve_key(key)))

        matches.sort(key=lambda m: (m.start, m.length))
        return matches


__all__ = [
    "Resolution",
    "TermEntry",
    "TermIndex",
    "TermResolution",
    "TextMatch",
]
