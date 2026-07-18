"""Testes dos contratos do Interest Engine — Documento 112.7E, seções 7, 17 e 24.1."""
from __future__ import annotations

import json
from uuid import uuid4

import pytest
from pydantic import ValidationError

from sentinela.interest.results import (
    InterestContribution,
    InterestLayer,
    InterestResult,
    MatchedResearchLine,
)


def _contribution(**over) -> InterestContribution:
    base = dict(
        layer=InterestLayer.CONCEPT,
        interest_id="mineral_dust",
        profile_weight=1.0,
        fingerprint_weight=1.0,
        contribution=1.0,
        research_line_id=None,
    )
    base.update(over)
    return InterestContribution(**base)


def _result(**over) -> InterestResult:
    base = dict(
        event_id=uuid4(),
        profile_id="henrique_lobo",
        profile_version=1,
        fingerprint_hash="a" * 64,
        taxonomy_version="1",
        algorithm_version="1",
        interest_score=1.0,
        contributions=(_contribution(),),
        matched_lines=(
            MatchedResearchLine(
                line_id="transatlantic_mineral_dust",
                priority=1.0,
                line_score=1.0,
            ),
        ),
        excluded=False,
        exclusion_reason=None,
        interest_hash="b" * 64,
    )
    base.update(over)
    return InterestResult(**base)


# ------------------------- limites (0..1) -------------------------
@pytest.mark.parametrize("field", ["profile_weight", "fingerprint_weight", "contribution"])
@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_contribution_pesos_fora_de_0_a_1(field, value):
    with pytest.raises(ValidationError):
        _contribution(**{field: value})


@pytest.mark.parametrize("value", [-0.1, 1.5])
def test_score_fora_de_0_a_1(value):
    with pytest.raises(ValidationError):
        _result(interest_score=value)


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_line_score_e_priority_fora_de_0_a_1(value):
    with pytest.raises(ValidationError):
        MatchedResearchLine(line_id="x", priority=value, line_score=0.5)
    with pytest.raises(ValidationError):
        MatchedResearchLine(line_id="x", priority=0.5, line_score=value)


def test_profile_version_minima():
    with pytest.raises(ValidationError):
        _result(profile_version=0)


# --------------------------- extras ------------------------------
def test_campos_extras_rejeitados():
    with pytest.raises(ValidationError):
        _contribution(extra="x")
    with pytest.raises(ValidationError):
        _result(outro=1)
    with pytest.raises(ValidationError):
        MatchedResearchLine(
            line_id="x", priority=1.0, line_score=1.0, campo="y"
        )


# ---------------------------- enums -------------------------------
def test_interest_layer_valores():
    assert InterestLayer("domain") is InterestLayer.DOMAIN
    assert InterestLayer("concept") is InterestLayer.CONCEPT
    assert InterestLayer("region") is InterestLayer.REGION
    assert InterestLayer("instrument") is InterestLayer.INSTRUMENT
    with pytest.raises(ValueError):
        InterestLayer("research_line")
    with pytest.raises(ValueError):
        InterestLayer("source")


# ------------------------- imutabilidade -------------------------
def test_modelos_imutaveis():
    c = _contribution()
    with pytest.raises(ValidationError):
        c.contribution = 0.0
    r = _result()
    with pytest.raises(ValidationError):
        r.interest_score = 0.0
    with pytest.raises(ValidationError):
        r.excluded = True


# -------------------- serialização determinística -----------------
def test_serializacao_deterministica():
    a = _result()
    b = a.model_copy()
    assert json.dumps(a.model_dump(mode="json"), sort_keys=True) == json.dumps(
        b.model_dump(mode="json"), sort_keys=True
    )
    dumped = a.model_dump(mode="json")
    assert dumped["contributions"][0]["layer"] == "concept"
    assert dumped["excluded"] is False
    assert dumped["exclusion_reason"] is None


def test_estado_excluido_preserva_contribuicoes():
    r = _result(
        excluded=True,
        exclusion_reason="excluded topic 'mineral dust' matched 'mineral_dust'",
        interest_score=0.0,
    )
    assert r.excluded is True
    assert r.interest_score == 0.0
    assert len(r.contributions) == 1          # auditoria preservada (seção 11.1)
    assert len(r.matched_lines) == 1
