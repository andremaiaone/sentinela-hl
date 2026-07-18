"""Testes dos contratos Pydantic — Documento 112.7D, seção 40.1."""
from __future__ import annotations

import json
from uuid import uuid4

import pytest
from pydantic import ValidationError

from sentinela.fingerprint.models import (
    ConceptFingerprint,
    ConceptMatchType,
    ConceptSignal,
    FingerprintConfig,
    FingerprintField,
    FingerprintWeights,
    MatchEvidence,
    UnmatchedTerm,
)


def _evidence(**over) -> MatchEvidence:
    base = dict(
        field=FingerprintField.TITLE,
        input_term="Mineral dust",
        normalized_term="mineral dust",
        matched_term="mineral dust",
        canonical_concept_id="mineral_dust",
        match_type=ConceptMatchType.CANONICAL_NAME,
        contribution=0.85,
    )
    base.update(over)
    return MatchEvidence(**base)


def _signal(**over) -> ConceptSignal:
    base = dict(
        concept_id="mineral_dust",
        domain_id="atmospheric_science",
        weight=1.0,
        direct_weight=1.0,
        inherited_weight=0.0,
        related_weight=0.0,
        evidence=(_evidence(),),
    )
    base.update(over)
    return ConceptSignal(**base)


# ------------------------- limites de peso (0..1) -------------------------
@pytest.mark.parametrize("field,value", [
    ("keyword", 1.5), ("title", -0.1), ("parent_decay", 2.0), ("related_decay", -1.0),
])
def test_weights_fora_de_0_a_1_rejeitados(field, value):
    with pytest.raises(ValidationError):
        FingerprintWeights(**{field: value})


@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_contribution_fora_de_0_a_1_rejeitada(value):
    with pytest.raises(ValidationError):
        _evidence(contribution=value)


def test_pesos_nos_limitos_aceitos():
    w = FingerprintWeights(keyword=0.0, summary=1.0)
    assert w.keyword == 0.0 and w.summary == 1.0


# --------------------------- campos extras -------------------------------
def test_campos_extras_rejeitados():
    with pytest.raises(ValidationError):
        _evidence(campo_inventado="x")
    with pytest.raises(ValidationError):
        _signal(outro=1)
    with pytest.raises(ValidationError):
        FingerprintConfig(desconhecido=True)
    with pytest.raises(ValidationError):
        UnmatchedTerm(
            field=FingerprintField.KEYWORD,
            input_term="x",
            normalized_term="x",
            reason="not_in_taxonomy",
            extra="nope",
        )


# -------------------------------- enums ----------------------------------
def test_enums_valores_validos():
    assert FingerprintField("title") is FingerprintField.TITLE
    assert FingerprintField("scientific_area") is FingerprintField.SCIENTIFIC_AREA
    assert ConceptMatchType("concept_id") is ConceptMatchType.CONCEPT_ID
    assert ConceptMatchType("related") is ConceptMatchType.RELATED


def test_enums_valores_invalidos():
    with pytest.raises(ValueError):
        FingerprintField("subtitle")
    with pytest.raises(ValueError):
        ConceptMatchType("fuzzy")           # fuzzy não existe na V1 (seção 9.2)


# ----------------------------- imutabilidade -----------------------------
def test_modelos_imutaveis():
    ev = _evidence()
    with pytest.raises(ValidationError):
        ev.contribution = 0.1
    sig = _signal()
    with pytest.raises(ValidationError):
        sig.weight = 0.0
    cfg = FingerprintConfig()
    with pytest.raises(ValidationError):
        cfg.minimum_contribution = 0.9


# ----------------------------- config padrão -----------------------------
def test_config_defaults_do_contrato():
    cfg = FingerprintConfig()
    assert cfg.version == "1"
    assert cfg.include_parents is True
    assert cfg.max_parent_depth == 3
    assert cfg.include_related is True
    assert cfg.max_related_depth == 1
    assert cfg.minimum_contribution == 0.05
    w = cfg.weights
    assert (w.keyword, w.scientific_area) == (1.0, 1.0)
    assert (w.category, w.entity) == (0.90, 0.90)
    assert (w.country, w.title) == (0.85, 0.85)
    assert (w.summary, w.evidence) == (0.70, 0.65)
    assert (w.parent_decay, w.related_decay) == (0.70, 0.40)


def test_limites_da_v1_na_config():
    with pytest.raises(ValidationError):
        FingerprintConfig(max_related_depth=2)     # V1: apenas 1 salto (seção 14)
    with pytest.raises(ValidationError):
        FingerprintConfig(max_parent_depth=4)      # V1: máximo de 3 (seção 13)
    with pytest.raises(ValidationError):
        FingerprintConfig(minimum_contribution=1.5)


# ---------------------- serialização determinística ----------------------
def test_serializacao_deterministica():
    a = ConceptFingerprint(
        event_id=uuid4(),
        taxonomy_version="1",
        algorithm_version="1",
        weights_version="1",
        concepts=(_signal(),),
        unmatched_terms=(),
        source_fields=(FingerprintField.TITLE,),
        fingerprint_hash="abc",
    )
    b = a.model_copy()
    assert json.dumps(a.model_dump(mode="json"), sort_keys=True) == json.dumps(
        b.model_dump(mode="json"), sort_keys=True
    )
    # enums serializam como seus valores string
    dumped = a.model_dump(mode="json")
    assert dumped["source_fields"] == ["title"]
    assert dumped["concepts"][0]["evidence"][0]["field"] == "title"
