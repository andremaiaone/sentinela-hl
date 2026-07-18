"""Testes do engine com taxonomia sintética — Documento 112.7D, seções 12–16, 27, 40.4 e 40.5."""
from __future__ import annotations

from uuid import uuid4

import pytest

from sentinela.core.models import EpistemicStatus, Event, PipelineStatus
from sentinela.fingerprint.engine import (
    ALGORITHM_VERSION,
    ConceptFingerprintEngine,
)
from sentinela.fingerprint.models import (
    ConceptMatchType,
    FingerprintConfig,
    FingerprintField,
    FingerprintWeights,
)
from sentinela.taxonomy.models import Concept, Domain, Taxonomy
from sentinela.taxonomy.taxonomy import TaxonomyIndex


def _c(cid, name=None, *, synonyms=(), related=(), parent=None, domain="dom"):
    return Concept(
        id=cid,
        name=name or cid.replace("_", " ").title(),
        domain=domain,
        synonyms=list(synonyms),
        related=list(related),
        parent=parent,
    )


def _taxonomy(*concepts: Concept, version="1") -> TaxonomyIndex:
    return TaxonomyIndex(
        Taxonomy(
            version=version,
            domains=[Domain(id="dom", name="Dom", concepts=list(concepts))],
        )
    )


# hierarquia de teste espelha os exemplos das seções 13 e 14.
CHAIN = (
    _c("aerosols"),
    _c("mineral_dust", parent="aerosols",
       synonyms=["desert dust"], related=["atmospheric_transport"]),
    _c("saharan_dust", parent="mineral_dust", synonyms=["sahara dust"]),
    _c("atmospheric_transport"),
    _c("trade_winds", related=["atmospheric_circulation"]),
    _c("atmospheric_circulation"),
)


def _engine(*concepts, config=None, version="1") -> ConceptFingerprintEngine:
    return ConceptFingerprintEngine(
        taxonomy=_taxonomy(*(concepts or CHAIN), version=version),
        config=config or FingerprintConfig(),
    )


def _event(**over) -> Event:
    base = dict(title="t", epistemic_status=EpistemicStatus.CONFIRMED_FACT)
    base.update(over)
    return Event(**base)


def _signal(fp, concept_id):
    for s in fp.concepts:
        if s.concept_id == concept_id:
            return s
    return None


# ---------------------------- hierarquia (40.4) --------------------------
def test_pai_imediato_e_decaimento():
    fp = _engine().build(_event(title="x", keywords=["sahara dust"]))
    mineral = _signal(fp, "mineral_dust")
    assert mineral is not None
    assert mineral.direct_weight == 0.0
    assert mineral.inherited_weight == pytest.approx(0.70)   # 1.0 × 0.7^1
    ev = mineral.evidence[0]
    assert ev.match_type is ConceptMatchType.PARENT
    assert ev.canonical_concept_id == "saharan_dust"         # origem registrada
    assert ev.field is FingerprintField.KEYWORD


def test_multiplos_ancestrais_decaimento_por_salto():
    fp = _engine().build(_event(title="x", keywords=["sahara dust"]))
    # exemplo da seção 13: 1.00 → 0.70 → 0.49
    assert _signal(fp, "saharan_dust").direct_weight == pytest.approx(1.0)
    assert _signal(fp, "mineral_dust").inherited_weight == pytest.approx(0.70)
    assert _signal(fp, "aerosols").inherited_weight == pytest.approx(0.49)


def test_limite_de_profundidade():
    cfg = FingerprintConfig(max_parent_depth=1)
    fp = _engine(config=cfg).build(_event(title="x", keywords=["sahara dust"]))
    assert _signal(fp, "mineral_dust") is not None
    assert _signal(fp, "aerosols") is None                   # distância 2 cortada


def test_include_parents_false():
    cfg = FingerprintConfig(include_parents=False)
    fp = _engine(config=cfg).build(_event(title="x", keywords=["sahara dust"]))
    assert _signal(fp, "mineral_dust") is None
    assert _signal(fp, "aerosols") is None


def test_sem_inferencia_descendente():
    fp = _engine().build(_event(title="x", keywords=["aerosols"]))
    assert _signal(fp, "aerosols") is not None
    assert _signal(fp, "mineral_dust") is None               # filho nunca inferido
    assert _signal(fp, "saharan_dust") is None


# ---------------------------- relações (40.5) ----------------------------
def test_related_um_salto_com_decaimento():
    fp = _engine().build(_event(title="x", keywords=["desert dust"]))
    transport = _signal(fp, "atmospheric_transport")
    assert transport is not None
    assert transport.related_weight == pytest.approx(0.40)   # 1.0 × 0.40
    ev = transport.evidence[0]
    assert ev.match_type is ConceptMatchType.RELATED
    assert ev.canonical_concept_id == "mineral_dust"         # origem registrada


def test_related_simetrico_inverso():
    # trade_winds declara related=[atmospheric_circulation]; casar
    # atmospheric_circulation deve alcançar trade_winds pelo sentido inverso.
    fp = _engine().build(_event(title="x", keywords=["atmospheric circulation"]))
    winds = _signal(fp, "trade_winds")
    assert winds is not None
    assert winds.related_weight == pytest.approx(0.40)
    assert winds.direct_weight == 0.0


def test_sem_transitividade_related():
    # mineral_dust → atmospheric_transport; casar mineral_dust NÃO pode
    # alcançar os vizinhos de atmospheric_transport (2º salto).
    fp = _engine().build(_event(title="x", keywords=["desert dust"]))
    assert _signal(fp, "atmospheric_transport") is not None
    assert _signal(fp, "trade_winds") is None
    assert _signal(fp, "atmospheric_circulation") is None


def test_sem_mistura_parent_related():
    # saharan_dust: herda mineral_dust; o related de mineral_dust NÃO pode
    # partir do conceito herdado — só do conceito direto.
    fp = _engine().build(_event(title="x", keywords=["sahara dust"]))
    assert _signal(fp, "atmospheric_transport") is None


def test_include_related_false():
    cfg = FingerprintConfig(include_related=False)
    fp = _engine(config=cfg).build(_event(title="x", keywords=["desert dust"]))
    assert _signal(fp, "atmospheric_transport") is None


def test_related_nunca_supera_origem():
    fp = _engine().build(_event(title="x", keywords=["desert dust"]))
    origin = _signal(fp, "mineral_dust")
    related = _signal(fp, "atmospheric_transport")
    assert related.weight <= origin.weight


# ---------------------- agregação e separação (17) -----------------------
def test_conceito_direto_e_herdado_agrega_parcelas():
    fp = _engine().build(
        _event(title="x", keywords=["mineral dust", "sahara dust"])
    )
    mineral = _signal(fp, "mineral_dust")
    assert mineral.direct_weight == pytest.approx(1.0)       # keyword direta
    assert mineral.inherited_weight == pytest.approx(0.70)   # via saharan_dust
    assert mineral.weight == pytest.approx(1.0)
    aerosols = _signal(fp, "aerosols")
    assert aerosols.inherited_weight == pytest.approx(
        1 - (1 - 0.70) * (1 - 0.49)
    )                                                        # duas evidências herdadas


# --------------------- minimum_contribution (27) -------------------------
def test_minimum_contribution_descarta_expansao():
    cfg = FingerprintConfig(minimum_contribution=0.45)
    fp = _engine(config=cfg).build(_event(title="x", keywords=["desert dust"]))
    assert _signal(fp, "mineral_dust") is not None           # direto 1.0 ≥ 0.45
    assert _signal(fp, "atmospheric_transport") is None      # related 0.40 < 0.45


def test_minimum_contribution_omite_conceito_sem_parcelas():
    cfg = FingerprintConfig(
        weights=FingerprintWeights(summary=0.04),
        minimum_contribution=0.05,
    )
    fp = _engine(config=cfg).build(_event(title="x", summary="desert dust"))
    assert fp.concepts == ()                                 # 0.04 < 0.05 → sem signal
    assert fp.source_fields == ()


# ------------------------- regras de saída (23) --------------------------
def test_conceitos_ordenados_canonicamente():
    fp = _engine().build(
        _event(title="sahara dust", keywords=["mineral dust", "aerosols"])
    )
    keys = [(-s.weight, -s.direct_weight, s.concept_id) for s in fp.concepts]
    assert keys == sorted(keys)


def test_source_fields_dedup_lexico():
    fp = _engine().build(_event(title="desert dust", keywords=["aerosols"]))
    assert fp.source_fields == (FingerprintField.KEYWORD, FingerprintField.TITLE)


def test_versoes_propagadas():
    fp = _engine(version="7").build(_event(title="x", keywords=["aerosols"]))
    assert fp.taxonomy_version == "7"
    assert fp.algorithm_version == ALGORITHM_VERSION
    assert fp.weights_version == "1"


def test_evento_sem_id_e_requires_human_review():
    ev = _event(
        title="desert dust",
        pipeline_status=PipelineStatus.ESCALATED,
        requires_human_review=True,
    )
    fp = _engine().build(ev)
    assert fp.event_id is None
    assert _signal(fp, "mineral_dust") is not None
    assert "requires_human_review" not in fp.model_dump()    # seção 7.4


def test_termos_nao_resolvidos_auditaveis():
    fp = _engine().build(
        _event(title="x", keywords=["aerosols", "new atmospheric river index"])
    )
    assert [u.input_term for u in fp.unmatched_terms] == [
        "new atmospheric river index"
    ]
    assert fp.unmatched_terms[0].reason == "not_in_taxonomy"
    assert fp.unmatched_terms[0].field is FingerprintField.KEYWORD


def test_bit_estavel_sob_permutacao_de_keywords_e_entities():
    base = _c("south_atlantic", "South Atlantic", domain="dom")
    kw = ["mineral dust", "aerosols", "sahara dust"]
    ent = [{"name": "South Atlantic"}, {"name": "Atmospheric Transport"}]
    engine = _engine(*CHAIN, base)
    a = engine.build(_event(title="x", keywords=kw, entities=ent, id=uuid4()))
    b = engine.build(
        _event(
            title="x",
            keywords=list(reversed(kw)),
            entities=list(reversed(ent)),
            id=a.event_id,
        )
    )
    assert a.model_dump_json(exclude={"generated_at"}) == b.model_dump_json(
        exclude={"generated_at"}
    )
    assert a.fingerprint_hash == b.fingerprint_hash


def test_reexecucao_idempotente():
    engine = _engine()
    ev = _event(title="desert dust", keywords=["aerosols"], id=uuid4())
    a, b = engine.build(ev), engine.build(ev)
    assert a.fingerprint_hash == b.fingerprint_hash
    assert a.model_dump_json(exclude={"generated_at"}) == b.model_dump_json(
        exclude={"generated_at"}
    )


def test_summary_none_e_entities_desconhecidas_sem_erro():
    ev = _event(
        title="desert dust",
        summary=None,
        entities=[{"tipo": 1}, {"name": "Aerosols"}],
    )
    fp = _engine().build(ev)
    assert _signal(fp, "mineral_dust") is not None
    assert _signal(fp, "aerosols") is not None
