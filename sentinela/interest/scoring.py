"""Modelo de aderência do Interest Engine — Documento 112.7E, seção 9.

Contribuição atômica:

    contribution = profile_weight × fingerprint_weight

Agregação em dois níveis, com a fórmula já aprovada no 112.7D (noisy-or,
reutilizada de `sentinela.fingerprint.aggregation` — implementação única):

1. dentro de cada camada (global ou linha de pesquisa), sobre as
   contribuições atômicas em ordem canônica (seção 9.5.1);
2. na composição final, em ordem fixa: domain, concept, region, instrument,
   depois as linhas ordenadas por line_id (seção 9.5.3).

Propriedades: limitado em [0, 1], monotônico em cada contribuição, cresce
com evidências independentes sem soma ilimitada, bit-estável sob permutação
das coleções de entrada. Nenhum valor sofre arredondamento.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from sentinela.fingerprint.aggregation import noisy_or
from sentinela.fingerprint.models import ConceptSignal

from .results import InterestContribution, InterestLayer

#: camadas globais por conceito, na ordem fixa da composição (seção 9.5.3).
CONCEPT_LAYERS: tuple[InterestLayer, ...] = (
    InterestLayer.CONCEPT,
    InterestLayer.REGION,
    InterestLayer.INSTRUMENT,
)


def atomic_contribution(
    layer: InterestLayer,
    interest_id: str,
    profile_weight: float,
    fingerprint_weight: float,
    research_line_id: str | None = None,
) -> InterestContribution:
    """profile_weight × fingerprint_weight (seção 9.1)."""
    return InterestContribution(
        layer=layer,
        interest_id=interest_id,
        profile_weight=profile_weight,
        fingerprint_weight=fingerprint_weight,
        contribution=profile_weight * fingerprint_weight,
        research_line_id=research_line_id,
    )


def _aggregation_sort_key(contribution: InterestContribution) -> tuple:
    """Ordem canônica para o noisy-or dentro de uma camada (seção 9.5.1):
    interest_id, depois research_line_id com None primeiro."""
    return (
        contribution.interest_id,
        contribution.research_line_id is not None,
        contribution.research_line_id or "",
    )


def aggregate(contributions: Iterable[InterestContribution]) -> float:
    """noisy-or sobre as contribuições, em ordem canônica (seção 9.5.1).

    Vazio → 0.0. Usado tanto nas camadas globais quanto no line_score.
    """
    ordered = sorted(contributions, key=_aggregation_sort_key)
    return noisy_or(c.contribution for c in ordered)


def compose(scores: Sequence[float]) -> float:
    """noisy-or sobre os scores de camada NA ORDEM RECEBIDA — o chamador
    garante a ordem fixa da seção 9.5.3."""
    return noisy_or(scores)


def contribution_sort_key(contribution: InterestContribution) -> tuple:
    """Ordem canônica da lista `contributions` da saída (seção 12):
    layer, interest_id, research_line_id com None primeiro."""
    return (
        contribution.layer.value,
        contribution.interest_id,
        contribution.research_line_id is not None,
        contribution.research_line_id or "",
    )


def domain_contribution(
    domain_id: str,
    profile_weight: float,
    signals: Iterable[ConceptSignal],
) -> InterestContribution | None:
    """A única contribuição de um domínio (seção 9.1):

        fingerprint_weight = noisy_or(weight dos sinais do domínio)
        contribution       = profile_weight × fingerprint_weight

    Os sinais são agregados em ordem de concept_id (seção 9.5.2). Retorna
    None quando o domínio não tem sinal no fingerprint.
    """
    ordered = sorted(signals, key=lambda s: s.concept_id)
    if not ordered:
        return None
    fingerprint_weight = noisy_or(s.weight for s in ordered)
    return atomic_contribution(
        InterestLayer.DOMAIN,
        domain_id,
        profile_weight,
        fingerprint_weight,
    )


__all__ = [
    "CONCEPT_LAYERS",
    "aggregate",
    "atomic_contribution",
    "compose",
    "contribution_sort_key",
    "domain_contribution",
    "noisy_or",
]
