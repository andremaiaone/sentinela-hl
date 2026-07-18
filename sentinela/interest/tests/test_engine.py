"""Testes do InterestEngine — Documento 112.7E, seções 6, 8, 9, 11, 14, 15 e 24."""
from __future__ import annotations

from uuid import uuid4

import pytest

from sentinela.core.models import EpistemicStatus, Event, PipelineStatus
from sentinela.fingerprint.models import (
    ConceptFingerprint,
    ConceptMatchType,
    ConceptSignal,
    FingerprintField,
    MatchEvidence,
)
from sentinela.interest.engine import ALGORITHM_VERSION, InterestEngine
from sentinela.interest.models import (
    ResearchConcept,
    ResearchDomain,
    ResearchIdentity,
    ResearchInstrument,
    ResearchLine,
    ResearchProfile,
    ResearchRegion,
)


# ----------------------------------------------------------------- helpers
def _profile(**over) -> ResearchProfile:
    base = dict(
        researcher=ResearchIdentity(id="tester", name="Test", version=3),
    )
    base.update(over)
    return ResearchProfile(**base)


def _signal(concept_id: str, domain_id: str, weight: float) -> ConceptSignal:
    ev = MatchEvidence(
        field=FingerprintField.KEYWORD,
        input_term=concept_id.replace("_", " "),
        normalized_term=concept_id.replace("_", " "),
        matched_term=concept_id.replace("_", " "),
        canonical_concept_id=concept_id,
        match_type=ConceptMatchType.CANONICAL_NAME,
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


def _fingerprint(*signals: ConceptSignal, hash_: str = "f" * 64,
                 taxonomy_version: str = "1") -> ConceptFingerprint:
    return ConceptFingerprint(
        event_id=None,
        taxonomy_version=taxonomy_version,
        algorithm_version="1",
        weights_version="1",
        concepts=tuple(signals),
        unmatched_terms=(),
        source_fields=(),
        fingerprint_hash=hash_,
    )


def _event(**over) -> Event:
    base = dict(title="t", epistemic_status=EpistemicStatus.CONFIRMED_FACT)
    base.update(over)
    return Event(**base)


def _engine(profile: ResearchProfile) -> InterestEngine:
    return InterestEngine(profile)


# ------------------------------ matemática -------------------------
def test_duas_camadas_compoem_em_dois_niveis():
    profile = _profile(
        domains=(ResearchDomain(domain_id="atmospheric_science", weight=0.5),),
        concepts=(ResearchConcept(concept_id="mineral_dust", weight=0.5),),
    )
    fp = _fingerprint(_signal("mineral_dust", "atmospheric_science", 1.0))
    result = _engine(profile).evaluate(_event(), fp)
    # noisy_or(domain 0.5, concept 0.5) = 0.75
    assert result.interest_score == pytest.approx(0.75)
    layers = {c.layer.value for c in result.contributions}
    assert layers == {"domain", "concept"}


def test_linha_escalonada_por_priority():
    profile = _profile(
        research_lines=(
            ResearchLine(
                id="linha_a",
                title="A",
                priority=0.5,
                concepts=(ResearchConcept(concept_id="mineral_dust", weight=1.0),),
            ),
        ),
    )
    fp = _fingerprint(_signal("mineral_dust", "atmospheric_science", 0.8))
    result = _engine(profile).evaluate(_event(), fp)
    (line,) = result.matched_lines
    assert line.line_score == pytest.approx(0.8)
    # contribuição da linha = 0.5 × 0.8 = 0.4
    assert result.interest_score == pytest.approx(0.4)
    assert all(
        c.research_line_id == "linha_a" for c in result.contributions
    )


def test_linha_sem_match_nao_entra_na_lista():
    profile = _profile(
        research_lines=(
            ResearchLine(
                id="linha_b",
                title="B",
                priority=1.0,
                concepts=(ResearchConcept(concept_id="earthquake", weight=1.0),),
            ),
        ),
    )
    fp = _fingerprint(_signal("mineral_dust", "atmospheric_science", 1.0))
    result = _engine(profile).evaluate(_event(), fp)
    assert result.matched_lines == ()
    assert result.interest_score == 0.0


def test_matched_lines_ordenadas_por_line_id():
    profile = _profile(
        research_lines=(
            ResearchLine(
                id="z_linha", title="Z", priority=1.0,
                concepts=(ResearchConcept(concept_id="mineral_dust", weight=1.0),),
            ),
            ResearchLine(
                id="a_linha", title="A", priority=1.0,
                concepts=(ResearchConcept(concept_id="mineral_dust", weight=1.0),),
            ),
        ),
    )
    fp = _fingerprint(_signal("mineral_dust", "atmospheric_science", 0.5))
    result = _engine(profile).evaluate(_event(), fp)
    assert [m.line_id for m in result.matched_lines] == ["a_linha", "z_linha"]


def test_perfil_vazio_e_fingerprint_vazio():
    engine = _engine(_profile())
    assert engine.evaluate(_event(), _fingerprint()).interest_score == 0.0
    profile = _profile(
        concepts=(ResearchConcept(concept_id="mineral_dust", weight=1.0),)
    )
    result = _engine(profile).evaluate(_event(), _fingerprint())
    assert result.interest_score == 0.0
    assert result.contributions == ()


# ------------------------------ exclusões --------------------------
def test_exclusao_por_igualdade_exata_zera_score_preservando_auditoria():
    profile = _profile(
        concepts=(ResearchConcept(concept_id="mineral_dust", weight=1.0),),
        excluded_topics=("mineral dust",),
    )
    fp = _fingerprint(_signal("mineral_dust", "atmospheric_science", 1.0))
    result = _engine(profile).evaluate(_event(), fp)
    assert result.excluded is True
    assert result.interest_score == 0.0
    assert "mineral dust" in result.exclusion_reason
    assert len(result.contributions) == 1          # aderência preservada


def test_exclusao_casa_concept_id_com_dobra_de_underscore():
    profile = _profile(excluded_topics=("mineral dust",))
    fp = _fingerprint(_signal("mineral_dust", "atmospheric_science", 1.0))
    result = _engine(profile).evaluate(_event(), fp)
    assert result.excluded is True


def test_exclusao_nao_casa_substring():
    profile = _profile(
        concepts=(ResearchConcept(concept_id="mineral_dust", weight=1.0),),
        excluded_topics=("dust",),                  # ≠ "mineral dust"
    )
    fp = _fingerprint(_signal("mineral_dust", "atmospheric_science", 1.0))
    result = _engine(profile).evaluate(_event(), fp)
    assert result.excluded is False
    assert result.interest_score == pytest.approx(1.0)


def test_exclusao_editorial_nao_casa_conceito_cientifico():
    profile = _profile(excluded_topics=("sports",))
    fp = _fingerprint(_signal("cme", "space_weather", 1.0))
    result = _engine(profile).evaluate(_event(), fp)
    assert result.excluded is False                # "sports" ≠ "space weather"


# --------------------- prova de não-reextração (24.2) ---------------
def test_engine_nao_enxerga_texto_do_evento():
    profile = _profile(
        concepts=(ResearchConcept(concept_id="mineral_dust", weight=1.0),),
        regions=(ResearchRegion(concept_id="south_atlantic", weight=1.0),),
    )
    event = _event(
        title="Mineral dust plume crosses the South Atlantic",
        summary="Trade winds carry aerosols.",
        keywords=["mineral dust", "south atlantic"],
        entities=[{"name": "CALIPSO"}],
    )
    # fingerprint SEM os conceitos mencionados no texto do evento
    fp = _fingerprint(_signal("earthquake", "seismology", 1.0))
    result = _engine(profile).evaluate(event, fp)
    assert result.interest_score == 0.0
    assert result.contributions == ()
    assert result.matched_lines == ()


# ------------------------- determinismo e hash ----------------------
def _cenario():
    profile = _profile(
        domains=(ResearchDomain(domain_id="atmospheric_science", weight=0.9),),
        concepts=(ResearchConcept(concept_id="mineral_dust", weight=1.0),),
        research_lines=(
            ResearchLine(
                id="dust", title="Dust", priority=1.0,
                concepts=(ResearchConcept(concept_id="mineral_dust", weight=1.0),),
            ),
        ),
    )
    signals = (
        _signal("mineral_dust", "atmospheric_science", 1.0),
        _signal("aerosols", "atmospheric_science", 0.7),
        _signal("south_atlantic", "regions", 0.9),
    )
    return profile, signals


def test_reexecucao_idempotente():
    profile, signals = _cenario()
    engine = _engine(profile)
    ev = _event(id=uuid4())
    fp = _fingerprint(*signals)
    a, b = engine.evaluate(ev, fp), engine.evaluate(ev, fp)
    assert a.interest_hash == b.interest_hash
    assert a.model_dump_json(exclude={"generated_at"}) == b.model_dump_json(
        exclude={"generated_at"}
    )


def test_bit_estavel_sob_permutacao_dos_sinais():
    profile, signals = _cenario()
    engine = _engine(profile)
    ev = _event(id=uuid4())
    a = engine.evaluate(ev, _fingerprint(*signals))
    b = engine.evaluate(ev, _fingerprint(*reversed(signals)))
    assert a.interest_hash == b.interest_hash
    assert a.model_dump_json(exclude={"generated_at"}) == b.model_dump_json(
        exclude={"generated_at"}
    )


def test_versoes_e_vinculos_propagados():
    profile, signals = _cenario()
    ev = _event(id=uuid4())
    fp = _fingerprint(*signals, hash_="abc123", taxonomy_version="7")
    result = _engine(profile).evaluate(ev, fp)
    assert result.event_id == ev.id
    assert result.profile_id == "tester"
    assert result.profile_version == 3
    assert result.fingerprint_hash == "abc123"
    assert result.taxonomy_version == "7"
    assert result.algorithm_version == ALGORITHM_VERSION == "1"


def test_generated_at_fora_do_hash():
    profile, signals = _cenario()
    engine = _engine(profile)
    ev = _event(id=uuid4())
    fp = _fingerprint(*signals)
    a = engine.evaluate(ev, fp)
    b = engine.evaluate(ev, fp)
    assert a.generated_at != b.generated_at or a.interest_hash == b.interest_hash
    payload = a.model_dump(mode="json")
    assert "generated_at" in payload          # existe no contrato…
    assert a.interest_hash == b.interest_hash  # …mas não no hash


def test_evento_sem_id_e_requires_human_review_nao_copiado():
    profile, signals = _cenario()
    ev = _event(
        pipeline_status=PipelineStatus.ESCALATED,
        requires_human_review=True,
    )
    result = _engine(profile).evaluate(ev, _fingerprint(*signals))
    assert result.event_id is None
    assert "requires_human_review" not in result.model_dump()


def test_fingerprint_e_perfil_nao_modificados():
    profile, signals = _cenario()
    engine = _engine(profile)
    fp = _fingerprint(*signals)
    before = fp.model_dump_json()
    engine.evaluate(_event(), fp)
    assert fp.model_dump_json() == before
