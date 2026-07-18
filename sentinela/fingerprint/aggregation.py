"""Agregação de contribuições — Documento 112.7D, seções 16, 17 e 23.

A agregação padrão é o "noisy-or":

    peso = 1 - produto(1 - contribuição_i)

cresce com múltiplas evidências, permanece em [0, 1] e não permite que
repetições gerem score ilimitado. Ocorre em dois níveis: primeiro dentro de
cada parcela (direct/inherited/related), depois entre as parcelas.

Estabilidade bit a bit (seção 16): a multiplicação de ponto flutuante não é
associativa, então o produto é SEMPRE aplicado sobre as contribuições na
ordem canônica das evidências (seção 23) e as parcelas são combinadas em
ordem fixa (direct → inherited → related). Nenhum valor sofre arredondamento.
"""

from __future__ import annotations

from typing import Iterable

from sentinela.fingerprint.models import (
    DIRECT_MATCH_TYPES,
    ConceptMatchType,
    ConceptSignal,
    MatchEvidence,
)


def noisy_or(contributions: Iterable[float]) -> float:
    """1 - produto(1 - c_i), aplicado na ordem recebida."""
    product = 1.0
    for contribution in contributions:
        product *= 1.0 - contribution
    return 1.0 - product


def evidence_sort_key(evidence: MatchEvidence) -> tuple:
    """Ordem canônica das evidências (seção 23):
    campo, contribuição decrescente, termo normalizado, concept_id."""
    return (
        evidence.field.value,
        -evidence.contribution,
        evidence.normalized_term,
        evidence.canonical_concept_id,
    )


def signal_sort_key(signal: ConceptSignal) -> tuple:
    """Ordem canônica dos signals (seção 23):
    weight decrescente, direct_weight decrescente, concept_id crescente."""
    return (-signal.weight, -signal.direct_weight, signal.concept_id)


def build_signal(
    concept_id: str,
    domain_id: str,
    evidences: Iterable[MatchEvidence],
) -> ConceptSignal:
    """Monta um ConceptSignal a partir de todas as suas evidências.

    As evidências são repartidas por origem (direta, herdada, relacionada),
    agregadas por parcela na ordem canônica e combinadas em ordem fixa.
    """
    ordered = sorted(evidences, key=evidence_sort_key)
    direct = [e for e in ordered if e.match_type in DIRECT_MATCH_TYPES]
    inherited = [e for e in ordered if e.match_type is ConceptMatchType.PARENT]
    related = [e for e in ordered if e.match_type is ConceptMatchType.RELATED]

    direct_weight = noisy_or(e.contribution for e in direct)
    inherited_weight = noisy_or(e.contribution for e in inherited)
    related_weight = noisy_or(e.contribution for e in related)
    weight = noisy_or((direct_weight, inherited_weight, related_weight))

    return ConceptSignal(
        concept_id=concept_id,
        domain_id=domain_id,
        weight=weight,
        direct_weight=direct_weight,
        inherited_weight=inherited_weight,
        related_weight=related_weight,
        evidence=tuple(ordered),
    )


__all__ = [
    "build_signal",
    "evidence_sort_key",
    "noisy_or",
    "signal_sort_key",
]
