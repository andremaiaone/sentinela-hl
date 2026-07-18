"""Testes do modelo de aderência — Documento 112.7E, seções 9 e 24.1."""
from __future__ import annotations

import random

import pytest

from sentinela.fingerprint.models import (
    ConceptMatchType,
    ConceptSignal,
    FingerprintField,
    MatchEvidence,
)
from sentinela.interest.results import InterestLayer
from sentinela.interest.scoring import (
    aggregate,
    atomic_contribution,
    compose,
    contribution_sort_key,
    domain_contribution,
)


def _cont(layer=InterestLayer.CONCEPT, iid="mineral_dust", pw=1.0, fw=1.0, rid=None):
    return atomic_contribution(layer, iid, pw, fw, rid)


def _signal(concept_id: str, domain_id: str, weight: float) -> ConceptSignal:
    ev = MatchEvidence(
        field=FingerprintField.KEYWORD,
        input_term=concept_id,
        normalized_term=concept_id,
        matched_term=concept_id,
        canonical_concept_id=concept_id,
        match_type=ConceptMatchType.CONCEPT_ID,
        contribution=weight,
    )
    return ConceptSignal(
        concept_id=concept_id,
        domain_id=domain_id,
        weight=weight,
        direct_weight=weight,
        inherited_weight=0.0,
        related_weight=0.0,
        evidence=(ev,),
    )


# ----------------------- contribuição atômica ----------------------
def test_contribuicao_atomica_e_produto():
    c = atomic_contribution(InterestLayer.REGION, "south_atlantic", 0.8, 0.5)
    assert c.contribution == pytest.approx(0.4)
    assert c.research_line_id is None


# ------------------------- agregação -------------------------------
def test_aggregate_vazio_e_unico():
    assert aggregate([]) == 0.0
    assert aggregate([_cont(pw=1.0, fw=0.7)]) == pytest.approx(0.7)


def test_aggregate_noisy_or():
    contribs = [_cont(iid="a", pw=1.0, fw=0.7), _cont(iid="b", pw=1.0, fw=0.65)]
    assert aggregate(contribs) == pytest.approx(0.895)


def test_aggregate_limites():
    assert 0.0 <= aggregate([_cont(iid=str(i), fw=0.9) for i in range(50)]) <= 1.0


def test_aggregate_independente_da_ordem():
    contribs = [
        _cont(iid="a", fw=0.7),
        _cont(iid="b", fw=0.9, rid="linha"),
        _cont(iid="b", fw=0.5),
        _cont(iid="c", fw=0.4, rid="linha"),
    ]
    base = aggregate(contribs)
    rng = random.Random(7)
    for _ in range(20):
        shuffled = list(contribs)
        rng.shuffle(shuffled)
        assert aggregate(shuffled) == base          # bit a bit


def test_compose_ordem_recebida():
    assert compose([0.0, 0.0]) == 0.0
    assert compose([1.0, 0.5]) == pytest.approx(1.0)
    assert compose([0.5, 0.5]) == pytest.approx(0.75)


# ---------------------- camada de domínio --------------------------
def test_dominio_uma_unica_contribuicao():
    signals = [
        _signal("mineral_dust", "atmospheric_science", 1.0),
        _signal("aerosols", "atmospheric_science", 0.7),
        _signal("south_atlantic", "regions", 0.9),
    ]
    c = domain_contribution(
        "atmospheric_science", 0.8,
        [s for s in signals if s.domain_id == "atmospheric_science"],
    )
    assert c is not None
    assert c.layer is InterestLayer.DOMAIN
    assert c.interest_id == "atmospheric_science"
    assert c.fingerprint_weight == pytest.approx(1.0)     # noisy_or(1.0, 0.7)
    assert c.contribution == pytest.approx(0.8)


def test_dominio_sem_sinal_retorna_none():
    assert domain_contribution("climate_science", 1.0, []) is None


def test_dominio_agrega_em_ordem_de_concept_id():
    s1 = _signal("a", "dom", 0.3)
    s2 = _signal("b", "dom", 0.5)
    s3 = _signal("c", "dom", 0.7)
    a = domain_contribution("dom", 1.0, [s1, s2, s3])
    b = domain_contribution("dom", 1.0, [s3, s2, s1])
    assert a.fingerprint_weight == b.fingerprint_weight    # bit a bit


# ---------------------- ordenação da saída -------------------------
def test_contribution_sort_key_canonica():
    a = _cont(layer=InterestLayer.REGION, iid="z")
    b = _cont(layer=InterestLayer.CONCEPT, iid="b")
    c = _cont(layer=InterestLayer.CONCEPT, iid="a", rid="linha")
    d = _cont(layer=InterestLayer.CONCEPT, iid="a")
    ordered = sorted([a, b, c, d], key=contribution_sort_key)
    # layer, interest_id, research_line_id (None primeiro)
    assert ordered == [d, c, b, a]
