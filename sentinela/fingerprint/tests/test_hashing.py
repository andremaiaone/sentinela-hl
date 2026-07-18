"""Testes do hash canônico — Documento 112.7D, seções 25, 26 e 40.7."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

from sentinela.fingerprint.hashing import (
    canonical_payload,
    compute_fingerprint_hash,
)
from sentinela.fingerprint.models import (
    ConceptFingerprint,
    ConceptMatchType,
    ConceptSignal,
    FingerprintField,
    MatchEvidence,
    UnmatchedTerm,
)


def _fingerprint(**over) -> ConceptFingerprint:
    evidence = MatchEvidence(
        field=FingerprintField.TITLE,
        input_term="Mineral dust crosses Atlantic",
        normalized_term="mineral dust crosses atlantic",
        matched_term="mineral dust",
        canonical_concept_id="mineral_dust",
        match_type=ConceptMatchType.CANONICAL_NAME,
        contribution=0.85,
    )
    base = dict(
        event_id=uuid4(),
        taxonomy_version="1",
        algorithm_version="1",
        weights_version="1",
        generated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        concepts=(
            ConceptSignal(
                concept_id="mineral_dust",
                domain_id="atmospheric_science",
                weight=0.85,
                direct_weight=0.85,
                inherited_weight=0.0,
                related_weight=0.0,
                evidence=(evidence,),
            ),
        ),
        unmatched_terms=(
            UnmatchedTerm(
                field=FingerprintField.KEYWORD,
                input_term="mystery",
                normalized_term="mystery",
                reason="not_in_taxonomy",
            ),
        ),
        source_fields=(FingerprintField.TITLE,),
        fingerprint_hash="",
    )
    base.update(over)
    fp = ConceptFingerprint(**base)
    return fp.model_copy(
        update={"fingerprint_hash": compute_fingerprint_hash(canonical_payload(fp))}
    )


def test_mesma_entrada_mesmo_hash():
    event_id = uuid4()
    assert _fingerprint(event_id=event_id).fingerprint_hash == _fingerprint(
        event_id=event_id
    ).fingerprint_hash


def test_hash_e_sha256_hex():
    h = _fingerprint().fingerprint_hash
    assert len(h) == 64
    int(h, 16)


def test_alteracao_de_conceito_muda_hash():
    a = _fingerprint(event_id=uuid4())
    concepts = a.concepts[0].model_copy(update={"weight": 0.90})
    b = _fingerprint(event_id=a.event_id, concepts=(concepts,))
    assert a.fingerprint_hash != b.fingerprint_hash


def test_alteracao_de_generated_at_nao_muda_hash():
    event_id = uuid4()
    a = _fingerprint(event_id=event_id)
    b = _fingerprint(
        event_id=event_id,
        generated_at=a.generated_at + timedelta(days=30),
    )
    assert a.fingerprint_hash == b.fingerprint_hash


def test_alteracao_de_versao_muda_hash():
    event_id = uuid4()
    base = _fingerprint(event_id=event_id)
    for field in ("taxonomy_version", "algorithm_version", "weights_version"):
        other = _fingerprint(event_id=event_id, **{field: "2"})
        assert base.fingerprint_hash != other.fingerprint_hash


def test_alteracao_de_unmatched_muda_hash():
    event_id = uuid4()
    a = _fingerprint(event_id=event_id)
    b = _fingerprint(event_id=event_id, unmatched_terms=())
    assert a.fingerprint_hash != b.fingerprint_hash


def test_payload_exclui_campos_operacionais():
    payload = canonical_payload(_fingerprint())
    assert "generated_at" not in payload
    assert "fingerprint_hash" not in payload
    assert "source_fields" not in payload          # derivável das evidências
    assert set(payload) == {
        "event_id",
        "taxonomy_version",
        "algorithm_version",
        "weights_version",
        "concepts",
        "unmatched_terms",
    }
