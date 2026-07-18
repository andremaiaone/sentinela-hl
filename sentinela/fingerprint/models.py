"""Contratos do Concept Fingerprint — Documento 112.7D.

Representação conceitual determinística de um evento validado: quais conceitos
da Scientific Taxonomy estão explicitamente representados, por quais sinais e
com qual força. Não calcula interesse, relevância ou significância (isso é do
Event Radar / Interest Engine); não conhece pesquisador; não usa LLM nem
embeddings (ADR-007, invariantes da seção 46).

Todos os modelos são Pydantic v2, imutáveis (frozen) e rejeitam campos extras:
o fingerprint é um contrato estável, serializado de forma determinística e
protegido por hash canônico (seções 25–26).
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


class FingerprintField(str, Enum):
    """Campos do Event autorizados a gerar sinal conceitual (seção 9.1)."""
    TITLE = "title"
    SUMMARY = "summary"
    CATEGORY = "category"
    SCIENTIFIC_AREA = "scientific_area"
    KEYWORD = "keyword"
    ENTITY = "entity"
    COUNTRY = "country"
    EVIDENCE = "evidence"


class ConceptMatchType(str, Enum):
    """Tipo da correspondência taxonômica (seção 9.2). Sem fuzzy na V1."""
    CONCEPT_ID = "concept_id"
    CANONICAL_NAME = "canonical_name"
    SYNONYM = "synonym"
    PARENT = "parent"
    RELATED = "related"


#: correspondências que compõem a parcela `direct_weight` (seção 17).
DIRECT_MATCH_TYPES = (
    ConceptMatchType.CONCEPT_ID,
    ConceptMatchType.CANONICAL_NAME,
    ConceptMatchType.SYNONYM,
)


class MatchEvidence(_StrictModel):
    """Proveniência de uma contribuição (seção 22).

    `canonical_concept_id` é o conceito canônico do termo que casou. Em
    evidências diretas coincide com o conceito do signal; em expansões
    (parent/related) registra o conceito que ORIGINOU a expansão (seção 14).
    """
    field: FingerprintField
    input_term: str                      # texto original, nunca normalizado
    normalized_term: str                 # normalize_term() da taxonomia
    matched_term: str                    # termo taxonômico (forma canônica)
    canonical_concept_id: str
    match_type: ConceptMatchType
    contribution: Weight


class ConceptSignal(_StrictModel):
    """Um conceito canônico encontrado no evento, com pesos por origem."""
    concept_id: str
    domain_id: str
    weight: Weight
    direct_weight: Weight
    inherited_weight: Weight
    related_weight: Weight
    evidence: tuple[MatchEvidence, ...]


class UnmatchedTerm(_StrictModel):
    """Termo não resolvido — lacuna auditável, nunca cria conceito (seção 20)."""
    field: FingerprintField
    input_term: str
    normalized_term: str
    reason: str


class ConceptFingerprint(_StrictModel):
    """Saída do engine (seção 8). Imutável; `generated_at` não participa do
    hash nem da comparação lógica (seção 25)."""
    event_id: UUID | None
    taxonomy_version: str
    algorithm_version: str
    weights_version: str
    generated_at: datetime = Field(default_factory=_utcnow)
    concepts: tuple[ConceptSignal, ...]
    unmatched_terms: tuple[UnmatchedTerm, ...]
    source_fields: tuple[FingerprintField, ...]
    fingerprint_hash: str


class FingerprintWeights(_StrictModel):
    """Pesos-base por campo e decaimentos de expansão (seção 11)."""
    keyword: Weight = 1.00
    scientific_area: Weight = 1.00
    category: Weight = 0.90
    entity: Weight = 0.90
    country: Weight = 0.85
    title: Weight = 0.85
    summary: Weight = 0.70
    evidence: Weight = 0.65
    parent_decay: Weight = 0.70
    related_decay: Weight = 0.40


class FingerprintConfig(_StrictModel):
    """Configuração versionada do engine (seção 27).

    Os limites de profundidade carregam os tetos da V1: hierarquia até 3
    saltos (seção 13) e related com exatamente 1 salto (seções 14–15).
    """
    version: str = "1"
    weights: FingerprintWeights = Field(default_factory=FingerprintWeights)
    include_parents: bool = True
    max_parent_depth: int = Field(default=3, ge=0, le=3)
    include_related: bool = True
    max_related_depth: int = Field(default=1, ge=0, le=1)
    minimum_contribution: Weight = 0.05


__all__ = [
    "ConceptFingerprint",
    "ConceptMatchType",
    "ConceptSignal",
    "DIRECT_MATCH_TYPES",
    "FingerprintConfig",
    "FingerprintField",
    "FingerprintWeights",
    "MatchEvidence",
    "UnmatchedTerm",
    "Weight",
]
