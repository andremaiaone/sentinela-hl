"""Concept Fingerprint — Documento 112.7D.

Representação conceitual canônica, determinística, explicável e reutilizável
de um evento científico validado. Sem LLM, sem embeddings, sem banco.
"""

from sentinela.fingerprint.engine import (
    ALGORITHM_VERSION,
    ConceptFingerprintEngine,
    FingerprintConfigLoadError,
    load_fingerprint_config,
)
from sentinela.fingerprint.models import (
    ConceptFingerprint,
    ConceptMatchType,
    ConceptSignal,
    FingerprintConfig,
    FingerprintField,
    FingerprintWeights,
    MatchEvidence,
    UnmatchedTerm,
)

__all__ = [
    "ALGORITHM_VERSION",
    "ConceptFingerprint",
    "ConceptFingerprintEngine",
    "ConceptMatchType",
    "ConceptSignal",
    "FingerprintConfig",
    "FingerprintConfigLoadError",
    "FingerprintField",
    "FingerprintWeights",
    "MatchEvidence",
    "UnmatchedTerm",
    "load_fingerprint_config",
]
