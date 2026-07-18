"""Testes do índice de termos e da correspondência — Documento 112.7D, seções 12, 18, 19 e 40.3."""
from __future__ import annotations

from sentinela.fingerprint.models import ConceptMatchType
from sentinela.fingerprint.resolver import TermIndex, TermResolution
from sentinela.taxonomy.models import Concept


def _concept(
    cid: str,
    name: str,
    *,
    synonyms: list[str] | None = None,
    domain: str = "dom",
) -> Concept:
    return Concept(id=cid, name=name, domain=domain, synonyms=synonyms or [])


def _index(*concepts: Concept) -> TermIndex:
    return TermIndex(list(concepts))


# -------------------------- resolução discreta ---------------------------
def test_resolve_por_concept_id():
    idx = _index(_concept("mineral_dust", "Mineral Dust"))
    r = idx.resolve_discrete("mineral_dust")
    assert r.status is TermResolution.MATCH
    assert r.entry.match_type is ConceptMatchType.CONCEPT_ID
    assert r.entry.matched_term == "mineral_dust"


def test_resolve_por_nome_canonico():
    idx = _index(_concept("mineral_dust", "Mineral Dust"))
    r = idx.resolve_discrete("mineral dust")
    assert r.status is TermResolution.MATCH
    assert r.entry.match_type is ConceptMatchType.CANONICAL_NAME
    assert r.entry.matched_term == "mineral dust"


def test_resolve_por_sinonimo():
    idx = _index(_concept("mineral_dust", "Mineral Dust", synonyms=["desert dust"]))
    r = idx.resolve_discrete("Desert Dust")
    assert r.status is TermResolution.MATCH
    assert r.entry.match_type is ConceptMatchType.SYNONYM
    assert r.entry.concept_id == "mineral_dust"


def test_resolve_dobra_de_pontuacao():
    idx = _index(_concept("reanalysis", "Reanalysis", synonyms=["merra-2"]))
    for term in ("MERRA-2", "merra 2", "Merra 2"):
        r = idx.resolve_discrete(term)
        assert r.status is TermResolution.MATCH
        assert r.entry.match_type is ConceptMatchType.SYNONYM
        assert r.entry.matched_term == "merra-2"


def test_termo_inexistente():
    idx = _index(_concept("mineral_dust", "Mineral Dust"))
    assert idx.resolve_discrete("new atmospheric river index").status is (
        TermResolution.NO_MATCH
    )


def test_termo_ambiguo_apos_dobra():
    # "merra-2" e "merra 2" são distintos para o validator da Taxonomia,
    # mas colidem na forma de comparação (seção 19).
    idx = _index(
        _concept("aaa", "AAA", synonyms=["merra-2"]),
        _concept("bbb", "BBB", synonyms=["merra 2"]),
    )
    assert idx.resolve_discrete("merra-2").status is TermResolution.AMBIGUOUS
    assert idx.resolve_discrete("merra 2").status is TermResolution.AMBIGUOUS


def test_prioridade_concept_id_sobre_nome():
    idx = _index(_concept("aerosols", "Aerosols"))
    r = idx.resolve_discrete("aerosols")
    assert r.entry.match_type is ConceptMatchType.CONCEPT_ID


def test_prioridade_nome_sobre_sinonimo():
    # o mesmo termo é nome canônico e consta como sinônimo do próprio conceito.
    idx = _index(_concept("dust_bowl", "Dust Bowl", synonyms=["dust bowl"]))
    r = idx.resolve_discrete("dust bowl")
    assert r.entry.match_type is ConceptMatchType.CANONICAL_NAME


# ---------------------------- busca em texto -----------------------------
def _text_concept_ids(idx: TermIndex, text: str):
    return [
        (m.resolution.entry.concept_id, m.resolution.entry.matched_term)
        for m in idx.find_in_text(text)
        if m.resolution.status is TermResolution.MATCH
    ]


def test_fronteira_de_palavra():
    idx = _index(_concept("mineral_dust", "Mineral Dust", synonyms=["dust"]))
    assert _text_concept_ids(idx, "dust crosses the ocean") == [
        ("mineral_dust", "dust")
    ]
    # "dust" não pode casar dentro de "stardust" (fronteira de token).
    assert _text_concept_ids(idx, "stardust crosses the ocean") == []


def test_termo_composto_em_texto():
    idx = _index(_concept("cme", "Coronal Mass Ejection"))
    assert _text_concept_ids(idx, "A coronal mass ejection was observed") == [
        ("cme", "coronal mass ejection")
    ]


def test_sobreposicao_mesmo_conceito_registra_melhor():
    idx = _index(
        _concept("mineral_dust", "Mineral Dust", synonyms=["dust", "mineral dust"])
    )
    matches = idx.find_in_text("mineral dust transport")
    matched = [m for m in matches if m.resolution.status is TermResolution.MATCH]
    assert len(matched) == 1                       # apenas a melhor do trecho
    assert matched[0].resolution.entry.matched_term == "mineral dust"


def test_sobreposicao_entre_conceitos_distintos():
    # o termo mais longo vence o trecho com exclusividade (seção 18.2).
    idx = _index(
        _concept("south_atlantic", "South Atlantic"),
        _concept("atlantic_ocean", "Atlantic Ocean", synonyms=["atlantic"]),
    )
    assert _text_concept_ids(idx, "plume over the south atlantic") == [
        ("south_atlantic", "south atlantic")
    ]
    # sem sobreposição, o termo curto casa normalmente.
    assert _text_concept_ids(idx, "plume over the atlantic") == [
        ("atlantic_ocean", "atlantic")
    ]


def test_desempate_lexical_na_ordenacao():
    # mesmos caracteres e tokens: a ordem lexical decide (seção 18.3).
    idx = _index(
        _concept("segundo", "b c"),
        _concept("primeiro", "a b"),
    )
    assert _text_concept_ids(idx, "a b c") == [("primeiro", "a b")]


def test_ambiguidade_em_texto():
    idx = _index(
        _concept("aaa", "AAA", synonyms=["merra-2"]),
        _concept("bbb", "BBB", synonyms=["merra 2"]),
    )
    matches = idx.find_in_text("data from merra-2 reanalysis")
    assert len(matches) == 1
    assert matches[0].resolution.status is TermResolution.AMBIGUOUS


def test_texto_sem_correspondencia():
    idx = _index(_concept("mineral_dust", "Mineral Dust"))
    assert idx.find_in_text("nothing known here") == []


def test_texto_vazio_e_pontuacao():
    idx = _index(_concept("mineral_dust", "Mineral Dust"))
    assert idx.find_in_text("") == []
    assert idx.find_in_text("---") == []


def test_concept_id_em_texto_quando_literal():
    idx = _index(_concept("cme", "Coronal Mass Ejection"))
    matches = idx.find_in_text("A CME was observed")
    assert matches[0].resolution.entry.match_type is ConceptMatchType.CONCEPT_ID
