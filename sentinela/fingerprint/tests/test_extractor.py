"""Testes da extração de termos do Event — Documento 112.7D, seções 7, 18, 21 e 40.

As formas flexíveis de `entities`/`evidence` (strings soltas, itens não-dict)
são testadas diretamente nos helpers: o contrato `Event` atual só admite
listas de dicts — a tolerância existe para a evolução do contrato (seção 21).
"""
from __future__ import annotations

from sentinela.core.models import EpistemicStatus, Event
from sentinela.fingerprint.extractor import (
    _entity_terms,
    _evidence_terms,
    extract_terms,
)
from sentinela.fingerprint.models import FingerprintField


def _event(**over) -> Event:
    base = dict(
        title="Um título qualquer",
        epistemic_status=EpistemicStatus.CONFIRMED_FACT,
    )
    base.update(over)
    return Event(**base)


def _by_field(terms, field):
    return [t for t in terms if t.field is field]


def test_campos_discretos_independentes():
    ev = _event(
        category="atmospheric science",
        scientific_area="atmospheric_science",
        country="Brazil",
        keywords=["mineral dust", "aerosols"],
    )
    terms = extract_terms(ev)
    assert [t.input_term for t in _by_field(terms, FingerprintField.CATEGORY)] == [
        "atmospheric science"
    ]
    assert [t.input_term for t in _by_field(terms, FingerprintField.SCIENTIFIC_AREA)] == [
        "atmospheric_science"
    ]
    assert [t.input_term for t in _by_field(terms, FingerprintField.COUNTRY)] == ["Brazil"]
    assert [t.input_term for t in _by_field(terms, FingerprintField.KEYWORD)] == [
        "mineral dust",
        "aerosols",
    ]
    assert all(not t.is_text for t in terms if t.field is not FingerprintField.TITLE)


def test_titulo_e_resumo_sao_texto():
    ev = _event(summary="um resumo")
    terms = extract_terms(ev)
    (title,) = _by_field(terms, FingerprintField.TITLE)
    (summary,) = _by_field(terms, FingerprintField.SUMMARY)
    assert title.is_text and summary.is_text


def test_summary_none_ignorado_sem_erro():
    terms = extract_terms(_event(summary=None))
    assert _by_field(terms, FingerprintField.SUMMARY) == []


def test_campos_none_ou_vazios_ignorados():
    ev = _event(category=None, country="   ", keywords=["", "  "])
    terms = extract_terms(ev)
    assert _by_field(terms, FingerprintField.CATEGORY) == []
    assert _by_field(terms, FingerprintField.COUNTRY) == []
    assert _by_field(terms, FingerprintField.KEYWORD) == []


def test_entities_lista_de_strings():
    terms = _entity_terms(["NASA", "CALIPSO", "South Atlantic"])
    assert [t.input_term for t in terms] == ["NASA", "CALIPSO", "South Atlantic"]
    assert all(t.field is FingerprintField.ENTITY and not t.is_text for t in terms)


def test_entities_lista_de_objetos_com_name():
    ev = _event(
        entities=[
            {"name": "CALIPSO", "type": "instrument"},
            {"name": "South Atlantic", "type": "region"},
        ]
    )
    terms = _by_field(extract_terms(ev), FingerprintField.ENTITY)
    assert [t.input_term for t in terms] == ["CALIPSO", "South Atlantic"]


def test_entities_formas_desconhecidas_ignoradas_sem_erro():
    terms = _entity_terms(
        [
            {"tipo": "sem nome"},
            42,
            ["aninhado"],
            {"name": 123},
            {"name": "Válido"},
        ]
    )
    assert [t.input_term for t in terms] == ["Válido"]


def test_evidence_formas_suportadas():
    terms = _evidence_terms(
        [
            "texto livre de evidência",
            {"text": "trecho identificado"},
            {"excerpt": "excerto de fonte"},
        ]
    )
    assert [t.input_term for t in terms] == [
        "texto livre de evidência",
        "trecho identificado",
        "excerto de fonte",
    ]
    assert all(t.field is FingerprintField.EVIDENCE and t.is_text for t in terms)


def test_evidence_estruturas_arbitrarias_ignoradas():
    terms = _evidence_terms([{"url": "https://x"}, {"nested": {"a": 1}}, 7, None])
    assert terms == []


def test_evidence_dicts_validos_no_contrato_event():
    ev = _event(evidence=[{"text": "mineral dust observed"}, {"url": "https://x"}])
    terms = _by_field(extract_terms(ev), FingerprintField.EVIDENCE)
    assert [t.input_term for t in terms] == ["mineral dust observed"]


def test_campos_contextuais_nunca_extraidos():
    ev = _event(
        summary=None,
        confidence_score=0.9,
        requires_human_review=True,
    )
    inputs = [t.input_term for t in extract_terms(ev)]
    assert "0.9" not in inputs and "confirmed_fact" not in inputs
