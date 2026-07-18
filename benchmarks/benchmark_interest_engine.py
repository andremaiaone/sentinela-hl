"""Benchmark offline do Interest Engine — Documento 112.7E, seção 25.

Sem rede, sem LLM, sem banco. Usa a Taxonomia real, o perfil real
`profiles/henrique_lobo.yaml` e fingerprints pré-computados pelo engine do
112.7D sobre um lote local de eventos sintéticos.

Registra: eventos por segundo, tempo médio por evento, p95, score médio e
taxa de zero-score — e compara com o custo do Concept Fingerprint por evento
(o Interest Engine deve ser estritamente mais barato).

Uso:

    uv run python benchmarks/benchmark_interest_engine.py [n_eventos]
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
from sentinela.fingerprint import (
    ConceptFingerprintEngine,
    load_fingerprint_config,
)
from sentinela.interest.engine import InterestEngine
from sentinela.interest.loader import load_research_profile
from sentinela.taxonomy.loader import load_taxonomy
from sentinela.taxonomy.taxonomy import TaxonomyIndex

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
        title="X-class flare observed by space weather monitors",
        summary="A coronal mass ejection may trigger a geomagnetic storm.",
        keywords=["CME", "solar flare", "geomagnetic storm"],
        entities=[],
    ),
    dict(
        title="New MERRA-2 reanalysis covers the South Atlantic",
        summary="CALIPSO and MODIS observations over four decades.",
        keywords=["reanalysis", "satellite observation"],
        entities=[{"name": "TROPOMI", "type": "instrument"}],
    ),
    dict(
        title="Warmest January on record amid strong El Nino",
        summary="Sea surface temperature anomaly drives extreme weather.",
        keywords=["climate record", "enso", "sea surface temperature"],
        entities=[],
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
    fp_engine = ConceptFingerprintEngine(taxonomy=taxonomy, config=config)
    profile = load_research_profile(ROOT / "profiles" / "henrique_lobo.yaml", taxonomy)
    engine = InterestEngine(profile)

    events = _build_events(n_events)

    # fingerprints pré-computados (custo medido à parte, para comparação)
    fingerprints = []
    fp_ms: list[float] = []
    for event in events:
        t0 = time.perf_counter()
        fingerprints.append(fp_engine.build(event))
        fp_ms.append((time.perf_counter() - t0) * 1000)

    samples_ms: list[float] = []
    scores: list[float] = []
    zero_score = 0

    start = time.perf_counter()
    for event, fingerprint in zip(events, fingerprints):
        t0 = time.perf_counter()
        result = engine.evaluate(event, fingerprint)
        samples_ms.append((time.perf_counter() - t0) * 1000)
        scores.append(result.interest_score)
        zero_score += result.interest_score == 0.0
    elapsed = time.perf_counter() - start

    sorted_ms = sorted(samples_ms)
    p95 = sorted_ms[min(len(sorted_ms) - 1, int(len(sorted_ms) * 0.95))]
    mean_ms = statistics.fmean(samples_ms)
    fp_mean_ms = statistics.fmean(fp_ms)

    print("Benchmark Interest Engine (offline, perfil e taxonomia reais)")
    print("-" * 60)
    print(f"eventos avaliados:          {n_events}")
    print(f"tempo total:                {elapsed:.3f} s")
    print(f"eventos por segundo:        {n_events / elapsed:.1f}")
    print(f"tempo médio por evento:     {mean_ms:.4f} ms")
    print(f"p95 por evento:             {p95:.4f} ms")
    print(f"score médio:                {statistics.fmean(scores):.3f}")
    print(f"taxa de zero-score:         {zero_score / n_events:.1%}")
    print("-" * 60)
    print(f"referência — fingerprint:   {fp_mean_ms:.4f} ms/evento "
          f"({fp_mean_ms / mean_ms:.1f}× mais lento)")
    print("critério: estritamente mais barato que o Concept Fingerprint;")
    print("adequado a lotes de centenas de eventos por pesquisador.")
    return 0


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    raise SystemExit(run(n))
