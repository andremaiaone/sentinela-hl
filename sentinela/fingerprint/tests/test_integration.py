"""Testes de integração com a Taxonomia real — Documento 112.7D, seção 41.

Usa a Taxonomia de `taxonomy/`, o `TaxonomyIndex` real, o contrato `Event`
real e a configuração de `config/concept_fingerprint.yaml`. Sem rede, sem
LLM, sem banco.
"""
from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from sentinela.core.models import EpistemicStatus, Event, PipelineStatus
from sentinela.fingerprint.engine import (
    ConceptFingerprintEngine,
    load_fingerprint_config,
)
from sentinela.fingerprint.models import ConceptMatchType, FingerprintField
from sentinela.taxonomy.loader import load_taxonomy
from sentinela.taxonomy.taxonomy import TaxonomyIndex

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def taxonomy_index() -> TaxonomyIndex:
    return TaxonomyIndex(load_taxonomy(ROOT / "taxonomy"))


@pytest.fixture(scope="module")
def engine(taxonomy_index: TaxonomyIndex) -> ConceptFingerprintEngine:
    config = load_fingerprint_config(ROOT / "config" / "concept_fingerprint.yaml")
    return ConceptFingerprintEngine(taxonomy=taxonomy_index, config=config)


def _event(**over) -> Event:
    base = dict(title="t", epistemic_status=EpistemicStatus.CONFIRMED_FACT)
    base.update(over)
    return Event(**base)


def _signal(fp, concept_id):
    for s in fp.concepts:
        if s.concept_id == concept_id:
            return s
    return None


def _unmatched(fp):
    return {(u.field, u.input_term): u.reason for u in fp.unmatched_terms}


# 1 — transporte de poeira mineral no Atlântico Sul (exemplo da seção 31).
@pytest.fixture()
def dust_event() -> Event:
    return _event(
        id=uuid4(),
        title="Mineral dust plume crosses the South Atlantic",
        summary=(
            "Satellite observations show an aerosol plume transported "
            "westward by trade winds."
        ),
        category="atmospheric science",
        scientific_area="atmospheric_science",
        keywords=["mineral dust", "aerosols", "trade winds", "CALIPSO"],
        entities=[
            {"name": "CALIPSO", "type": "instrument"},
            {"name": "South Atlantic", "type": "region"},
        ],
        confidence_score=0.85,
    )


def test_integracao_poeira_atlantico_sul(engine, taxonomy_index, dust_event):
    fp = engine.build(dust_event)

    mineral = _signal(fp, "mineral_dust")
    assert mineral.direct_weight == pytest.approx(1.0)       # title + keyword
    assert mineral.weight == pytest.approx(1.0)

    aerosols = _signal(fp, "aerosols")
    assert aerosols.direct_weight == pytest.approx(1.0)      # summary + keyword
    assert aerosols.inherited_weight == pytest.approx(0.8785)  # herda de mineral_dust

    atlantic = _signal(fp, "south_atlantic")
    assert atlantic.direct_weight == pytest.approx(0.985)    # title 0.85 + entity 0.90

    instruments = _signal(fp, "satellite_instruments")
    assert instruments.direct_weight == pytest.approx(1.0)   # keyword + entity

    trade = _signal(fp, "trade_winds")
    assert trade.direct_weight == pytest.approx(1.0)         # summary + keyword

    # regra de sobreposição: "atlantic" não casa dentro de "south atlantic";
    # atlantic_ocean só entra via related de south_atlantic.
    ocean = _signal(fp, "atlantic_ocean")
    assert ocean.direct_weight == 0.0
    assert ocean.related_weight == pytest.approx(1 - (1 - 0.34) * (1 - 0.36))

    # categoria/área não constam no vocabulário de conceitos → auditadas.
    unmatched = _unmatched(fp)
    assert unmatched[(FingerprintField.CATEGORY, "atmospheric science")] == (
        "not_in_taxonomy"
    )
    assert unmatched[
        (FingerprintField.SCIENTIFIC_AREA, "atmospheric_science")
    ] == "not_in_taxonomy"

    # somente campos que produziram correspondência, em ordem lexical.
    assert [f.value for f in fp.source_fields] == [
        "entity",
        "keyword",
        "summary",
        "title",
    ]

    # todo conceito do fingerprint existe na Taxonomia (invariante 2).
    for signal in fp.concepts:
        assert taxonomy_index.get(signal.concept_id) is not None


# 2 — grande terremoto.
def test_integracao_grande_terremoto(engine):
    ev = _event(
        id=uuid4(),
        title="Magnitude 7.8 earthquake strikes coastal Chile",
        keywords=["earthquake", "tsunami", "seismic magnitude"],
    )
    fp = engine.build(ev)

    quake = _signal(fp, "earthquake")
    assert quake.direct_weight == pytest.approx(1.0)         # title + keyword
    assert quake.domain_id == "seismology"

    magnitude = _signal(fp, "seismic_magnitude")
    assert magnitude.direct_weight == pytest.approx(1.0)     # title + keyword

    tsunami = _signal(fp, "tsunami")
    assert tsunami.direct_weight == pytest.approx(1.0)

    rupture = _signal(fp, "fault_rupture")
    assert rupture.direct_weight == 0.0
    assert rupture.related_weight > 0.0                      # via earthquake

    # aftershock é FILHO de earthquake: a hierarquia nunca o infere
    # (inherited_weight == 0). Só pode aparecer via relação curada
    # (related sísmico de seismic_magnitude), nunca como descendente.
    aftershock = _signal(fp, "aftershock")
    assert aftershock is not None
    assert aftershock.direct_weight == 0.0
    assert aftershock.inherited_weight == 0.0
    assert aftershock.related_weight > 0.0


# 3 — erupção vulcânica com emissão de SO₂.
def test_integracao_erupcao_so2(engine):
    ev = _event(
        id=uuid4(),
        title="Volcanic eruption at Mount Etna",
        keywords=["volcanic eruption", "SO2 emission"],
    )
    fp = engine.build(ev)

    eruption = _signal(fp, "volcanic_eruption")
    assert eruption.direct_weight == pytest.approx(1.0)
    # "volcanic eruption" vence o trecho; "eruption" não gera 2ª evidência.
    title_hits = [
        e for e in eruption.evidence if e.field is FingerprintField.TITLE
    ]
    assert len(title_hits) == 1
    assert title_hits[0].matched_term == "volcanic eruption"

    gas = _signal(fp, "volcanic_gas")
    assert gas.direct_weight == pytest.approx(1.0)           # sinônimo so2 emission

    ash = _signal(fp, "volcanic_ash")
    assert ash.direct_weight == 0.0
    assert ash.related_weight > 0.0                          # via volcanic_eruption

    # sem mistura parent+related: aerosols só chega via related de
    # volcanic_gas — volcanic_ash (apenas related) NÃO expande hierarquia.
    aerosols = _signal(fp, "aerosols")
    assert aerosols.inherited_weight == 0.0
    assert aerosols.related_weight == pytest.approx(0.40)


# 4 — tempestade solar/CME.
def test_integracao_tempestade_solar(engine):
    ev = _event(
        id=uuid4(),
        title="X-class flare observed by space weather monitors",
        summary="A coronal mass ejection is expected to trigger a geomagnetic storm.",
        keywords=["CME", "solar flare"],
    )
    fp = engine.build(ev)

    flare = _signal(fp, "solar_flare")
    assert flare.direct_weight == pytest.approx(1.0)         # title + keyword

    cme = _signal(fp, "cme")
    assert cme.direct_weight == pytest.approx(1.0)           # summary + keyword
    keyword_hits = [e for e in cme.evidence if e.field is FingerprintField.KEYWORD]
    assert keyword_hits[0].match_type is ConceptMatchType.CONCEPT_ID

    storm = _signal(fp, "geomagnetic_storm")
    assert storm.direct_weight == pytest.approx(0.70)        # summary
    assert storm.related_weight > 0.0                        # via cme/solar_flare


# 5 — conflito geopolítico com fechamento de espaço aéreo.
def test_integracao_conflito_espaco_aereo(engine):
    ev = _event(
        id=uuid4(),
        title="Escalation forces airspace closure over the region",
        keywords=["armed conflict", "airspace closure"],
    )
    fp = engine.build(ev)

    conflict = _signal(fp, "armed_conflict")
    closure = _signal(fp, "airspace_closure")
    assert conflict.direct_weight == pytest.approx(1.0)
    assert closure.direct_weight == pytest.approx(1.0)
    # relação simétrica: cada um aparece como related no sinal do outro.
    assert conflict.related_weight > 0.0
    assert closure.related_weight > 0.0
    assert _signal(fp, "maritime_disruption") is not None    # via armed_conflict


# 6 — evento sem nenhum conceito conhecido.
def test_integracao_evento_sem_conceitos(engine):
    ev = _event(
        id=uuid4(),
        title="Local food festival draws crowds",
        summary="A celebration of regional cuisine and music.",
        category="lifestyle",
        keywords=["cuisine", "folk music"],
        entities=[{"name": "Springfield", "type": "city"}],
    )
    fp = engine.build(ev)
    assert fp.concepts == ()
    assert fp.source_fields == ()
    unmatched = _unmatched(fp)
    assert unmatched[(FingerprintField.KEYWORD, "cuisine")] == "not_in_taxonomy"
    assert unmatched[(FingerprintField.ENTITY, "Springfield")] == "not_in_taxonomy"
    assert unmatched[(FingerprintField.CATEGORY, "lifestyle")] == "not_in_taxonomy"
    assert len(fp.fingerprint_hash) == 64                    # hash ainda válido


# 7 — evento multidisciplinar.
def test_integracao_multidisciplinar(engine):
    ev = _event(
        id=uuid4(),
        title="Dust deposition links atmosphere, ocean and climate",
        keywords=["mineral dust", "ocean fertilization", "climate anomaly"],
    )
    fp = engine.build(ev)
    domains = {s.domain_id for s in fp.concepts}
    assert {
        "atmospheric_science",
        "oceanography",
        "climate_science",
    } <= domains
    # ponte curada: ocean_fertilization declara related com mineral_dust.
    dust = _signal(fp, "mineral_dust")
    assert dust.related_weight > 0.0


# 8 — evento com termo sinônimo de instrumento.
def test_integracao_sinonimo_de_instrumento(engine):
    ev = _event(
        id=uuid4(),
        title="New data from MERRA-2 released",
        keywords=["CALIPSO"],
        entities=[{"name": "TROPOMI", "type": "instrument"}],
    )
    fp = engine.build(ev)

    instruments = _signal(fp, "satellite_instruments")
    assert instruments.direct_weight == pytest.approx(1.0)   # keyword + entity
    for e in instruments.evidence:
        assert e.match_type is ConceptMatchType.SYNONYM

    reanalysis = _signal(fp, "reanalysis")
    assert reanalysis.direct_weight == pytest.approx(0.85)   # título
    assert reanalysis.evidence[0].match_type is ConceptMatchType.SYNONYM
    assert reanalysis.evidence[0].matched_term == "merra-2"


# 9 — evento com requires_human_review=True.
def test_integracao_requires_human_review(engine):
    ev = _event(
        id=uuid4(),
        title="Preliminary report of mineral dust over the South Atlantic",
        keywords=["mineral dust"],
        pipeline_status=PipelineStatus.ESCALATED,
        requires_human_review=True,
    )
    fp = engine.build(ev)
    # o engine não verifica estado do pipeline e não copia o metadado (7.4).
    assert _signal(fp, "mineral_dust") is not None
    assert "requires_human_review" not in fp.model_dump()


# 10 — reexecução determinística.
def test_integracao_reexecucao_deterministica(engine, dust_event):
    a = engine.build(dust_event)
    b = engine.build(dust_event)
    assert a.fingerprint_hash == b.fingerprint_hash
    assert a.model_dump_json(exclude={"generated_at"}) == b.model_dump_json(
        exclude={"generated_at"}
    )


# seção 43 — bit-estável sob permutação de keywords e entities.
def test_integracao_bit_estavel_sob_permutacao(engine, dust_event):
    a = engine.build(dust_event)
    shuffled = dust_event.model_copy(
        update={
            "keywords": list(reversed(dust_event.keywords)),
            "entities": list(reversed(dust_event.entities)),
        }
    )
    b = engine.build(shuffled)
    assert a.fingerprint_hash == b.fingerprint_hash
    assert a.model_dump_json(exclude={"generated_at"}) == b.model_dump_json(
        exclude={"generated_at"}
    )
