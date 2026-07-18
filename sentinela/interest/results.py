"""Contratos de saída do Interest Engine — Documento 112.7E, seções 7 e 17.

O `InterestResult` mede exclusivamente o grau de aderência de um evento
científico à agenda declarada no Research Profile (112.7B), computada sobre o
Concept Fingerprint (112.7D). Não mede significância (Event Radar, 112.7C),
não decide prioridade nem apresentação (camada de composição posterior).

Todos os modelos são Pydantic v2, imutáveis (frozen), rejeitam campos extras
e validam pesos/scores em [0, 1] — sem campos ocultos, sem pesos novos.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

Weight = Annotated[float, Field(ge=0.0, le=1.0)]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class InterestLayer(str, Enum):
    """Camada do perfil que produziu a aderência (seção 7.3)."""
    DOMAIN = "domain"
    CONCEPT = "concept"
    REGION = "region"
    INSTRUMENT = "instrument"


class InterestContribution(_StrictModel):
    """Um par auditável de aderência (seção 7.1).

    `research_line_id` é None quando a entrada pertence às camadas globais
    do perfil; preenchido quando a entrada vem de uma ResearchLine.
    """
    layer: InterestLayer
    interest_id: str                    # concept_id ou domain_id da entrada
    profile_weight: Weight              # peso declarado pelo pesquisador
    fingerprint_weight: Weight          # força do conceito no evento
    contribution: Weight                # profile_weight × fingerprint_weight
    research_line_id: str | None = None


class MatchedResearchLine(_StrictModel):
    """Uma linha de pesquisa atingida (seção 7.2). Só entram linhas com
    line_score > 0."""
    line_id: str
    priority: Weight
    line_score: Weight


class InterestResult(_StrictModel):
    """Saída do InterestEngine (seção 7). Imutável; `generated_at` não
    participa do hash nem da comparação lógica (seções 14–15).

    Quando `excluded` é True, `interest_score` é 0.0 por regra declarada e
    `contributions`/`matched_lines` permanecem preenchidos para auditoria
    (seção 11.1).
    """
    event_id: UUID | None
    profile_id: str
    profile_version: int = Field(ge=1)
    fingerprint_hash: str
    taxonomy_version: str
    algorithm_version: str
    generated_at: datetime = Field(default_factory=_utcnow)
    interest_score: Weight
    contributions: tuple[InterestContribution, ...]
    matched_lines: tuple[MatchedResearchLine, ...]
    excluded: bool = False
    exclusion_reason: str | None = None
    interest_hash: str


__all__ = [
    "InterestContribution",
    "InterestLayer",
    "InterestResult",
    "MatchedResearchLine",
    "Weight",
]
