"""Benchmark offline do Concept Fingerprint — Documento 112.7D, seção 42.

Processa um conjunto local de eventos sintéticos (sem rede, sem LLM, sem
banco) sobre a Taxonomia real e registra:

    eventos por segundo, tempo médio por evento, p95,
    conceitos médios por evento, termos não resolvidos

Uso:

    uv run python benchmarks/benchmark_concept_fingerprint.py [n_eventos]
"""
from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uuid import uuid4

from sentinela.core.models import EpistemicStatus, Event
from sentinela.fingerprint.engine import (
    ConceptFingerprintEngine,
    load_fingerprint_config,
)
from sentinela.taxonomy.loader import load_taxonomy
from sentinela.taxonomy.taxonomy import TaxonomyIndex

#: massa de eventos representativa dos domínios da Taxonomia (seção 41).
_TEMPLATES: list[dict] = [
    dict(
        title="Mineral dust plume crosses the South Atlantic",
        summary="Satellite observations show an aerosol plume transported "
        "westward by trade winds.",
        keywords=["mineral dust", "aerosols", "trade winds", "CALIPSO"],
        entities=[{"name": "South Atlantic", "type": "region"}],
    ),
    dict(
        title="Magnitude 7.8 earthquake strikes coastal Chile",
        summary="A major seismic event with aftershock sequence.",
        keywords=["earthquake", "tsunami", "seismic magnitude"],
        entities=[],
    ),
    dict(
        title="Volcanic eruption at Mount Etna",
        summary="Ash plume and sulfur dioxide plume detected by satellite.",
        keywords=["volcanic eruption", "SO2 emission", "volcanic ash"],
        entities=[{"name": "TROPOMI", "type": "instrument"}],
    ),
    dict(
        title="X-class flare observed by space weather monitors",
        summary="A coronal mass ejection may trigger a geomagnetic storm.",
        keywords=["CME", "solar flare", "geomagnetic storm"],
        entities=[],
    ),
    dict(
        title="Escalation forces airspace closure over the region",
        summary="Military conflict disrupts routes and observation campaigns.",
        keywords=["armed conflict", "airspace closure"],
        entities=[],
    ),
    dict(
        title="Warmest January on record amid strong El Nino",
        summary="Sea surface temperature anomaly drives extreme weather.",
        keywords=["climate record", "enso", "sea surface temperature"],
        entities=[],
    ),
    dict(
        title="Dust deposition may fertilize the South Atlantic",
        summary="Iron oxides from saharan dust reach the ocean.",
        keywords=["ocean fertilization", "saharan dust", "iron oxides"],
        entities=[{"name": "South Atlantic", "type": "region"}],
    ),
    dict(
        title="Local food festival draws crowds",
        summary="A celebration of regional cuisine and music.",
        keywords=["cuisine", "folk music"],
        entities=[{"name": "Springfield", "type": "city"}],
    ),
]


def _build_events(n: int) -> list[Event]:
    events: list[Event] = []
    for i in range(n):
        template = _TEMPLATES[i % len(_TEMPLATES)]
        events.append(
            Event(
                id=uuid4(),
                title=template["title"],
                summary=template["summary"],
                keywords=list(template["keywords"]),
                entities=[dict(e) for e in template["entities"]],
                epistemic_status=EpistemicStatus.CONFIRMED_FACT,
            )
        )
    return events


def run(n_events: int = 300) -> int:
    taxonomy = TaxonomyIndex(load_taxonomy(ROOT / "taxonomy"))
    config = load_fingerprint_config(ROOT / "config" / "concept_fingerprint.yaml")
    engine = ConceptFingerprintEngine(taxonomy=taxonomy, config=config)

    events = _build_events(n_events)

    samples_ms: list[float] = []
    concepts_found = 0
    unmatched = 0

    start = time.perf_counter()
    for event in events:
        t0 = time.perf_counter()
        fingerprint = engine.build(event)
        samples_ms.append((time.perf_counter() - t0) * 1000)
        concepts_found += len(fingerprint.concepts)
        unmatched += len(fingerprint.unmatched_terms)
    elapsed = time.perf_counter() - start

    sorted_ms = sorted(samples_ms)
    p95 = sorted_ms[min(len(sorted_ms) - 1, int(len(sorted_ms) * 0.95))]

    print("Benchmark Concept Fingerprint (offline, taxonomia real)")
    print("-" * 56)
    print(f"eventos processados:        {n_events}")
    print(f"tempo total:                {elapsed:.3f} s")
    print(f"eventos por segundo:        {n_events / elapsed:.1f}")
    print(f"tempo médio por evento:     {statistics.fmean(samples_ms):.3f} ms")
    print(f"p95 por evento:             {p95:.3f} ms")
    print(f"conceitos médios por evento:{concepts_found / n_events:6.1f}")
    print(f"termos não resolvidos:      {unmatched}")
    print("-" * 56)
    print("critério: muito mais rápido que qualquer chamada LLM;")
    print("adequado ao processamento em lote de centenas de eventos.")
    return 0


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    raise SystemExit(run(n))
