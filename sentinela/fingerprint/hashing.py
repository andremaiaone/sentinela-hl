"""Hash canônico do fingerprint — Documento 112.7D, seções 25 e 26.

SHA-256 sobre a serialização canônica:

    model_dump(mode="json")
    → json.dumps(sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    → UTF-8

O payload exclui `generated_at`, `fingerprint_hash` e `source_fields`
(derivável das evidências). Floats são serializados em precisão plena (menor
ida e volta do runtime); a reprodutibilidade garantida é para o mesmo runtime.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from sentinela.fingerprint.models import ConceptFingerprint

#: campos fora do payload canônico (seção 26).
_EXCLUDED = {"generated_at", "fingerprint_hash", "source_fields"}


def canonical_payload(fingerprint: ConceptFingerprint) -> dict[str, Any]:
    """O payload que entra no hash: versões, evento, conceitos e não
    resolvidos — sem timestamp, sem o próprio hash, sem campos derivados."""
    return fingerprint.model_dump(mode="json", exclude=_EXCLUDED)


def canonical_json(payload: dict[str, Any]) -> bytes:
    """JSON canônico: chaves ordenadas, sem espaços, UTF-8."""
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def compute_fingerprint_hash(payload: dict[str, Any]) -> str:
    """SHA-256 (hexdigest) do JSON canônico do payload."""
    return hashlib.sha256(canonical_json(payload)).hexdigest()


__all__ = ["canonical_json", "canonical_payload", "compute_fingerprint_hash"]
