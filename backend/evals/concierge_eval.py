"""Concierge evaluation runner.

Scores the Property & Lease Concierge on a small labeled dataset with three
deterministic metrics — route accuracy, retrieval hit-rate and groundedness.

By default (``use_llm=False``) synthesis is forced down the templated path so
the run is reproducible and fast (no Claude calls). The metrics themselves are
deterministic heuristics, so they never need an LLM judge.

Usage:  python backend/evals/concierge_eval.py
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import concierge  # noqa: E402

from evals.concierge_dataset import CONCIERGE_DATASET, item_kind  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def _grounded(answer: str, sources: list, must_include: list) -> bool:
    """Every required fact appears in the answer or an assembled source snippet
    (case-insensitive)."""
    haystack = (answer or "").lower()
    for s in sources or []:
        haystack += " " + str(s.get("snippet", "")).lower()
        haystack += " " + str(s.get("label", "")).lower()
    return all(str(term).lower() in haystack for term in (must_include or []))


def _retrieval_ok(item: dict, result: dict) -> bool:
    kind = item_kind(item)
    sources = result.get("sources") or []
    if kind == "lease":
        return item["expected_section"] in {s.get("section") for s in sources}
    if kind == "compare":
        got = {c.get("id") for c in (result.get("comparison") or [])}
        return set(item["expected_property_ids"]).issubset(got)
    # property
    return any(s.get("type") == "property" for s in sources)


def run(use_llm: bool = False) -> dict:
    """Evaluate the concierge over the labeled dataset and return a metrics
    dict. ``use_llm=False`` (default) forces the deterministic templated path."""
    orig_llm = concierge.get_langchain_llm
    if not use_llm:
        concierge.get_langchain_llm = lambda: None
    try:
        items = []
        route_hits = retrieval_hits = grounded_hits = 0
        for row in CONCIERGE_DATASET:
            result = concierge.answer(
                question=row["question"],
                property_id=row.get("property_id"),
                history=None,
            )
            predicted = result.get("route")
            route_ok = predicted == row["expected_route"]
            retrieval_ok = _retrieval_ok(row, result)
            grounded_ok = _grounded(
                result.get("answer", ""),
                result.get("sources", []),
                row.get("must_include", []),
            )
            route_hits += int(route_ok)
            retrieval_hits += int(retrieval_ok)
            grounded_hits += int(grounded_ok)
            items.append(
                {
                    "id": row["id"],
                    "question": row["question"],
                    "expected_route": row["expected_route"],
                    "predicted_route": predicted,
                    "route_ok": route_ok,
                    "retrieval_ok": retrieval_ok,
                    "grounded_ok": grounded_ok,
                }
            )
    finally:
        concierge.get_langchain_llm = orig_llm

    n = len(CONCIERGE_DATASET)

    def frac(x):
        return round(x / n, 4) if n else 0.0

    results = {
        "route_accuracy": frac(route_hits),
        "retrieval_hit_rate": frac(retrieval_hits),
        "groundedness": frac(grounded_hits),
        "n": n,
        "use_llm": use_llm,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "items": items,
    }
    _persist(results)
    return results


def _persist(results: dict) -> None:
    try:
        RESULTS.mkdir(exist_ok=True)
        (RESULTS / "concierge_latest.json").write_text(json.dumps(results, indent=2))
    except OSError:
        pass


def load_latest() -> dict:
    path = RESULTS / "concierge_latest.json"
    try:
        return json.loads(path.read_text()) if path.exists() else {}
    except (json.JSONDecodeError, OSError):
        return {}


if __name__ == "__main__":
    import pprint

    pprint.pp(run())
