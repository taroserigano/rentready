"""Source-quality audit for the Ask/Concierge page chat.

The existing ``concierge_eval.py`` / ``chat_golden_eval.py`` only check that the
ONE expected lease section is *somewhere* among the returned sources (a recall
check). They never ask two questions a user actually cares about:

  1. Is every source shown in the "Show N sources" panel actually relevant to
     the question, or is some of it padding that just fills out ``k``?
  2. Does the source COUNT adapt to the question at all, or is it always the
     same fixed number regardless of how narrow/specific the question is?

This script measures both, directly against ``knowledge.search()`` /
``concierge._assemble()`` (the real retrieval path, no mocking), using the
existing labeled dataset in ``concierge_dataset.py`` (each lease item already
carries the single correct ``expected_section`` for its property).

Usage:  python backend/evals/concierge_source_audit.py [--live N]
"""

import sys
from collections import Counter
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import concierge  # noqa: E402
import graph  # noqa: E402
import knowledge  # noqa: E402

from evals.concierge_dataset import CONCIERGE_DATASET  # noqa: E402

_LEASE_ITEMS = [it for it in CONCIERGE_DATASET if it.get("expected_section")]


def audit_retrieval_precision() -> dict:
    """For every lease-route golden item, call the REAL knowledge.search() and
    check how many of the k=4 returned passages are actually the on-topic
    section vs a different (off-topic, padding) section of the same lease."""
    graph.seed_graph()
    knowledge.ingest_all()

    counts = Counter()
    off_topic_total = 0
    n_sources_total = 0
    expected_rank_misses = 0
    per_item = []

    for item in _LEASE_ITEMS:
        passages = knowledge.search(
            item["question"], property_id=item["property_id"], k=4
        )
        sections = [p.get("section", "") for p in passages]
        counts[len(passages)] += 1
        n_sources_total += len(passages)
        off_topic = sum(1 for s in sections if s != item["expected_section"])
        off_topic_total += off_topic
        rank = (sections.index(item["expected_section"]) + 1
                if item["expected_section"] in sections else -1)
        if rank != 1:
            expected_rank_misses += 1
        per_item.append({
            "id": item["id"], "question": item["question"],
            "expected_section": item["expected_section"],
            "returned_sections": sections, "off_topic_count": off_topic,
            "expected_rank": rank,
        })

    n = len(_LEASE_ITEMS)
    return {
        "n_items": n,
        "source_count_distribution": dict(counts),
        "always_returns_k": len(counts) == 1 and 4 in counts,
        "avg_sources_per_answer": round(n_sources_total / n, 2),
        "avg_off_topic_sources_per_answer": round(off_topic_total / n, 2),
        "pct_answers_with_off_topic_source": round(
            100 * sum(1 for r in per_item if r["off_topic_count"] > 0) / n, 1
        ),
        "expected_section_not_rank1_count": expected_rank_misses,
        "worst_examples": sorted(
            per_item, key=lambda r: -r["off_topic_count"]
        )[:8],
    }


def _scoped_offtopic_probe() -> dict:
    """The concrete failure mode: a question with ZERO lease-relevant content,
    scoped to a real property. Checks what concierge._Plan (the real assembled
    sources, post citation-filter via concierge.answer) actually surfaces."""
    props = graph.load_properties()
    pid = props[0]["id"] if props else "PROP-001"
    off_topic_questions = [
        "Hi there, how's your day going?",
        "What do you think of this place?",
        "Give me a quick summary.",
        "What are my next steps?",
    ]
    out = []
    for q in off_topic_questions:
        route = concierge.route(q, pid)
        plan = concierge._Plan(q, pid)
        lease_sources_retrieved = [s for s in plan.sources if s["type"] == "lease"]
        # Full pipeline (post citation-filter), deterministic path forced off
        # so we see what today's default (LLM-off / offline) users get.
        det_text = plan.deterministic_answer(q)
        det_sources = concierge._filter_cited_sources(det_text, plan.sources)
        det_lease = [s for s in det_sources if s["type"] == "lease"]
        out.append({
            "question": q, "property_id": pid, "route": route,
            "n_lease_sources_retrieved": len(lease_sources_retrieved),
            "sections_retrieved": [s["section"] for s in lease_sources_retrieved],
            "n_lease_sources_shown_deterministic": len(det_lease),
            "deterministic_answer": det_text,
        })
    return {"probe": out}


def audit_citation_faithfulness(live_n: int = 0) -> dict:
    """Live-LLM check: of the sources shown in the UI, how many does the model
    actually cite with [n]? Requires a real LLM; skipped if live_n<=0 or the
    key isn't configured (falls back gracefully, never raises)."""
    if live_n <= 0:
        return {"skipped": True, "reason": "pass --live N to run this (hits the real LLM)"}

    sample = _LEASE_ITEMS[:: max(1, len(_LEASE_ITEMS) // live_n)][:live_n]
    rows = []
    for item in sample:
        result = concierge.answer(
            question=item["question"], property_id=item["property_id"], history=None
        )
        if result.get("source") != "anthropic":
            continue
        n_shown = len(result.get("sources", []))
        rows.append({
            "id": item["id"], "n_sources_shown_after_filter": n_shown,
            "cite_numbers": [s.get("cite") for s in result.get("sources", [])],
        })

    if not rows:
        return {"skipped": True, "reason": "no live LLM answers produced (no Anthropic key?)"}

    avg_shown = sum(r["n_sources_shown_after_filter"] for r in rows) / len(rows)
    return {
        "n": len(rows), "avg_sources_shown_after_filter": round(avg_shown, 2),
        "rows": rows,
    }


def run(live_n: int = 0) -> dict:
    return {
        "retrieval_precision": audit_retrieval_precision(),
        "scoped_offtopic_probe": _scoped_offtopic_probe(),
        "citation_faithfulness": audit_citation_faithfulness(live_n),
    }


if __name__ == "__main__":
    live = 0
    if "--live" in sys.argv:
        idx = sys.argv.index("--live")
        live = int(sys.argv[idx + 1]) if idx + 1 < len(sys.argv) else 12

    res = run(live_n=live)

    rp = res["retrieval_precision"]
    print("=" * 78)
    print("RETRIEVAL PRECISION (raw knowledge.search k=4, before citation-filter)")
    print("=" * 78)
    print(f"source_count_distribution: {rp['source_count_distribution']}")
    print(f"avg off-topic sources per raw retrieval: {rp['avg_off_topic_sources_per_answer']} / 4")
    print(f"% of raw retrievals with >=1 off-topic source: {rp['pct_answers_with_off_topic_source']}%")

    print("\n" + "=" * 78)
    print("SCOPED OFF-TOPIC PROBE (small talk WITH a property selected)")
    print("=" * 78)
    for p in res["scoped_offtopic_probe"]["probe"]:
        print(f"  Q: {p['question']!r}  route={p['route']}")
        print(f"      lease sources RETRIEVED (raw): {p['n_lease_sources_retrieved']} -> {p['sections_retrieved']}")
        print(f"      lease sources SHOWN (deterministic path, post-fix): {p['n_lease_sources_shown_deterministic']}")
        print(f"      deterministic answer: {p['deterministic_answer'][:100]!r}")

    print("\n" + "=" * 78)
    print("CITATION FAITHFULNESS (live LLM, post citation-filter)")
    print("=" * 78)
    cf = res["citation_faithfulness"]
    if cf.get("skipped"):
        print(f"  SKIPPED: {cf['reason']}")
    else:
        print(f"  n={cf['n']}  avg_sources_shown_after_filter={cf['avg_sources_shown_after_filter']}")
        for r in cf["rows"]:
            print(f"    {r['id']:<22} shown={r['n_sources_shown_after_filter']} cite_numbers={r['cite_numbers']}")
    print("=" * 78)
