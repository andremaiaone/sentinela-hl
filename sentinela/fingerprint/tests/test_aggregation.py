"""Testes de agregação — Documento 112.7D, seções 16, 17, 23 e 40.6."""
from __future__ import annotations

import random

import pytest

from sentinela.fingerprint.aggregation import (
    build_signal,
    evidence_sort_key,
    noisy_or,
    signal_sort_key,
)
from sentinela.fingerprint.models import (
    ConceptMatchType,
    FingerprintField,
    MatchEvidence,
)


def _ev(
    field=FingerprintField.TITLE,
    match_type=ConceptMatchType.CANONICAL_NAME,
    contribution=0.85,
    concept="mineral_dust",
    term="mineral dust",
) -> MatchEvidence:
    return MatchEvidence(
        field=field,
        input_term=term,
        normalized_term=term,
        matched_term=term,
        canonical_concept_id=concept,
        match_type=match_type,
        contribution=contribution,
    )


def test_noisy_or_uma_evidencia():
    assert noisy_or([0.85]) == pytest.approx(0.85)


def test_noisy_or_multiplas_evidencias():
    # exemplo da seção 16: summary 0.70 + evidence 0.65 → 0.895
    assert noisy_or([0.70, 0.65]) == pytest.approx(0.895)
    # title 0.85 + keyword 1.00 → 1.00
    assert noisy_or([0.85, 1.00]) == pytest.approx(1.0)


def test_noisy_or_limites():
    assert noisy_or([]) == 0.0
    assert 0.0 <= noisy_or([0.9] * 50) <= 1.0


def test_parcelas_preservadas():
    # exemplo da seção 17: direct 0.70 + inherited 0.49 → weight 0.847
    signal = build_signal(
        "aerosols",
        "atmospheric_science",
        [
            _ev(match_type=ConceptMatchType.SYNONYM, contribution=0.70),
            _ev(match_type=ConceptMatchType.PARENT, contribution=0.49,
                concept="mineral_dust"),
        ],
    )
    assert signal.direct_weight == pytest.approx(0.70)
    assert signal.inherited_weight == pytest.approx(0.49)
    assert signal.related_weight == 0.0
    assert signal.weight == pytest.approx(0.847)


def test_independencia_da_ordem_de_entrada():
    evidences = [
        _ev(field=FingerprintField.TITLE, contribution=0.85),
        _ev(field=FingerprintField.KEYWORD, contribution=1.0),
        _ev(field=FingerprintField.SUMMARY, contribution=0.70, term="dust"),
        _ev(field=FingerprintField.ENTITY, match_type=ConceptMatchType.RELATED,
            contribution=0.36, concept="south_atlantic", term="south atlantic"),
        _ev(field=FingerprintField.TITLE, match_type=ConceptMatchType.PARENT,
            contribution=0.595, concept="saharan_dust", term="saharan dust"),
    ]
    base = build_signal("mineral_dust", "atmospheric_science", evidences)
    rng = random.Random(42)
    for _ in range(20):
        shuffled = list(evidences)
        rng.shuffle(shuffled)
        other = build_signal("mineral_dust", "atmospheric_science", shuffled)
        assert other.model_dump(mode="json") == base.model_dump(mode="json")


def test_ordenacao_canonica_das_evidencias():
    a = _ev(field=FingerprintField.TITLE, contribution=0.85)
    b = _ev(field=FingerprintField.KEYWORD, contribution=0.40)
    c = _ev(field=FingerprintField.KEYWORD, contribution=1.0)
    d = _ev(field=FingerprintField.KEYWORD, contribution=1.0, term="aerosol")
    ordered = sorted([a, b, c, d], key=evidence_sort_key)
    # 1) campo; 2) contribuição decrescente; 3) termo normalizado
    assert ordered == [d, c, b, a]


def test_ordenacao_canonica_dos_signals():
    s1 = build_signal("b", "d", [_ev(contribution=0.5)])
    s2 = build_signal("a", "d", [_ev(contribution=0.5)])
    s3 = build_signal("c", "d", [_ev(contribution=0.9)])
    ordered = sorted([s1, s2, s3], key=signal_sort_key)
    # weight desc; empate → direct desc → concept_id crescente
    assert [s.concept_id for s in ordered] == ["c", "a", "b"]
