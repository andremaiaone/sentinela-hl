"""Testes de integração do Interest Engine — Documento 112.7E, seção 24.3.

Taxonomia real de `taxonomy/`, perfil real `profiles/henrique_lobo.yaml`,
fingerprints reais produzidos pelo engine do 112.7D. Sem rede, sem LLM, sem
banco, sem embeddings.
"""
from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from sentinela.core.models import EpistemicStatus, Event, PipelineStatus
from sentinela.fingerprint import (
    ConceptFingerprintEngine,
    load_fingerprint_config,
)
from sentinela.interest.engine import InterestEngine
from sentinela.interest.loader import load_research_profile
from sentinela.interest.models import (
    ResearchConcept,
    ResearchIdentity,
    ResearchProfile,
)
from sentinela.taxonomy.loader import load_taxonomy
from sentinela.taxonomy.taxonomy import TaxonomyIndex

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def taxonomy_index() -> TaxonomyIndex:
    return TaxonomyIndex(load_taxonomy(ROOT / "taxonomy"))


@pytest.fixture(scope="module")
def fingerprint_engine(taxonomy_index: TaxonomyIndex) -> ConceptFingerprintEngine:
    config = load_fingerprint_config(ROOT / "config" / "concept_fingerprint.yaml")
    return ConceptFingerprintEngine(taxonomy=taxonomy_index, config=config)


@pytest.fixture(scope="module")
def engine(taxonomy_index: TaxonomyIndex) -> InterestEngine:
    profile = load_research_profile(ROOT / "profiles" / "henrique_lobo.yaml", taxonomy_index)
    return InterestEngine(profile)


def _event(**over) -> Event:
    base = dict(title="t", epistemic_status=EpistemicStatus.CONFIRMED_FACT)
    base.update(over)
    return Event(**base)


def _dust_event() -> Event:
    return _event(
        id=uuid4(),
        title="Mineral dust plume crosses the South Atlantic",
        summary=(
            "Satellite observations show an aerosol plume transported "
            "westward by trade winds."
        ),
        keywords=["mineral dust", "aerosols", "trade winds", "CALIPSO"],
        entities=[
            {"name": "CALIPSO", "type": "instrument"},
            {"name": "South Atlantic", "type": "region"},
        ],
    )


# 1 — evento de poeira no Atlântico Sul: aderência máxima (seção 10).
def test_integracao_poeira_aderencia_maxima(engine, fingerprint_engine):
    event = _dust_event()
    result = engine.evaluate(event, fingerprint_engine.build(event))

    assert result.interest_score == pytest.approx(1.0)
    layers = {c.layer.value for c in result.contributions}
    assert {"concept", "region", "instrument", "domain"} <= layers

    by_layer_id = {(c.layer.value, c.interest_id): c for c in result.contributions}
    assert by_layer_id[("concept", "mineral_dust")].contribution == pytest.approx(1.0)
    # weight total do sinal = noisy_or(direto 0.985, relacionado 0.568)
    assert by_layer_id[("region", "south_atlantic")].contribution == pytest.approx(0.99352)
    assert by_layer_id[("instrument", "satellite_instruments")].contribution == pytest.approx(0.9)
    assert by_layer_id[("domain", "atmospheric_science")].contribution == pytest.approx(1.0)
    assert by_layer_id[("domain", "remote_sensing")].contribution == pytest.approx(0.9)

    (line,) = [m for m in result.matched_lines if m.line_id == "transatlantic_mineral_dust"]
    assert line.priority == pytest.approx(1.0)
    assert line.line_score == pytest.approx(1.0)

    assert result.excluded is False
    assert result.profile_id == "henrique_lobo"
    assert result.profile_version == 1
    assert len(result.interest_hash) == 64


# 2 — grande terremoto: fora da agenda, score zero (Radar mede significância).
def test_integracao_terremoto_score_zero(engine, fingerprint_engine):
    event = _event(
        id=uuid4(),
        title="Magnitude 7.8 earthquake strikes coastal Chile",
        keywords=["earthquake", "tsunami", "seismic magnitude"],
    )
    result = engine.evaluate(event, fingerprint_engine.build(event))
    assert result.interest_score == 0.0
    assert result.contributions == ()
    assert result.matched_lines == ()


# 3 — evento espacial: aderência explicável via domínio remote_sensing.
def test_integracao_evento_espacial_dominio_remoto(engine, fingerprint_engine):
    event = _event(
        id=uuid4(),
        title="X-class flare observed by space weather monitors",
        summary="A coronal mass ejection is expected to trigger a geomagnetic storm.",
        keywords=["CME", "solar flare"],
    )
    result = engine.evaluate(event, fingerprint_engine.build(event))

    # satellite_failure chega por related de geomagnetic_storm (0.70 × 0.40);
    # o domínio remote_sensing do perfil (0.9) converte em 0.9 × 0.28.
    assert result.interest_score == pytest.approx(0.252)
    (contrib,) = result.contributions
    assert contrib.layer.value == "domain"
    assert contrib.interest_id == "remote_sensing"
    assert contrib.profile_weight == pytest.approx(0.9)
    assert contrib.fingerprint_weight == pytest.approx(0.28)
    assert result.matched_lines == ()


# 4 — evento escalado com requires_human_review: avaliado, metadado não copiado.
def test_integracao_requires_human_review(engine, fingerprint_engine):
    event = _event(
        id=uuid4(),
        title="Preliminary report of mineral dust over the South Atlantic",
        keywords=["mineral dust"],
        pipeline_status=PipelineStatus.ESCALATED,
        requires_human_review=True,
    )
    result = engine.evaluate(event, fingerprint_engine.build(event))
    assert result.interest_score > 0.0
    assert "requires_human_review" not in result.model_dump()


# 5 — reexecução determinística.
def test_integracao_reexecucao_deterministica(engine, fingerprint_engine):
    event = _dust_event()
    a = engine.evaluate(event, fingerprint_engine.build(event))
    b = engine.evaluate(event, fingerprint_engine.build(event))
    assert a.interest_hash == b.interest_hash
    assert a.model_dump_json(exclude={"generated_at"}) == b.model_dump_json(
        exclude={"generated_at"}
    )


# 6 — multi-perfil: o mesmo fingerprint, agendas diferentes, resultados
# independentes.
def test_integracao_multi_perfil(engine, fingerprint_engine):
    event = _dust_event()
    fingerprint = fingerprint_engine.build(event)

    sismologo = InterestEngine(
        ResearchProfile(
            researcher=ResearchIdentity(id="sismologo", name="Sismólogo", version=1),
            concepts=(ResearchConcept(concept_id="earthquake", weight=1.0),),
        )
    )
    henrique = engine.evaluate(event, fingerprint)
    outro = sismologo.evaluate(event, fingerprint)

    assert henrique.interest_score == pytest.approx(1.0)
    assert outro.interest_score == 0.0
    assert henrique.fingerprint_hash == outro.fingerprint_hash   # mesmo fingerprint
    assert henrique.interest_hash != outro.interest_hash


# 7 — exclusão colidindo com conceito presente: veto com razão registrada.
def test_integracao_exclusao_com_veto(fingerprint_engine, taxonomy_index):
    from sentinela.interest.models import ResearchProfile, ResearchIdentity

    event = _dust_event()
    fingerprint = fingerprint_engine.build(event)
    veto_profile = ResearchProfile(
        researcher=ResearchIdentity(id="vetado", name="Vetado", version=1),
        concepts=(ResearchConcept(concept_id="mineral_dust", weight=1.0),),
        excluded_topics=("mineral dust",),
    )
    result = InterestEngine(veto_profile).evaluate(event, fingerprint)

    assert result.excluded is True
    assert result.interest_score == 0.0
    assert result.exclusion_reason is not None
    assert "mineral dust" in result.exclusion_reason
    assert len(result.contributions) >= 1          # auditoria preservada
