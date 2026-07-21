"""Run the evaluation suites and persist the results.

Two tiers:
  * deterministic (always): eligibility, profile extraction, recommendation
    ranking -- no LLM, no Neo4j, reproducible, gate CI.
  * LLM tier (auto): LLM-as-judge groundedness + RAGAS RAG quality. These need
    ANTHROPIC_API_KEY, so they run only when a key is present and are skipped
    cleanly otherwise.

Usage:  python backend/evals/run_evals.py
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from evals import evaluators
from evals import judges
from evals import rag_eval
from llm import get_langchain_llm

HERE = Path(__file__).resolve().parent
DATASETS = HERE / "datasets"
RESULTS = HERE / "results"


def _load(name: str) -> list:
    with open(DATASETS / name) as f:
        return [json.loads(line) for line in f if line.strip()]


def run(include_llm: Optional[bool] = None) -> dict:
    """Run every suite and return a combined summary dict.

    include_llm: None = auto (run the LLM tier iff a key is configured),
    True/False to force it on/off.
    """
    if include_llm is None:
        include_llm = get_langchain_llm() is not None

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "eligibility": evaluators.eval_eligibility(_load("eligibility.jsonl")),
        "extraction": evaluators.eval_extraction(_load("profile_extraction.jsonl")),
        "recommendations": evaluators.eval_ranking(_load("recommendations.jsonl")),
        "llm_tier": include_llm,
    }
    if include_llm:
        results["judge"] = judges.run_judge_suite()
        results["ragas"] = rag_eval.run_isolated()
    else:
        results["judge"] = {"skipped": True, "reason": "LLM tier disabled."}
        results["ragas"] = {"skipped": True, "reason": "LLM tier disabled."}

    _persist(results)
    return results


def _persist(results: dict) -> None:
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "latest.json").write_text(json.dumps(results, indent=2))
    stamp = results["generated_at"].replace(":", "-")
    (RESULTS / f"{stamp}.json").write_text(json.dumps(results, indent=2))


def load_latest() -> dict:
    path = RESULTS / "latest.json"
    return json.loads(path.read_text()) if path.exists() else {}


def _summarize(results: dict) -> dict:
    """A compact one-row snapshot of a run, for trend charts."""
    judge = results.get("judge") or {}
    rec_judge = judge.get("recommendation") or {}
    ragas = results.get("ragas") or {}
    ragas_metrics = ragas.get("metrics") or {}
    return {
        "generated_at": results.get("generated_at"),
        "eligibility_accuracy": (results.get("eligibility") or {}).get("accuracy"),
        "extraction_field_accuracy": (results.get("extraction") or {}).get(
            "field_accuracy"
        ),
        "ndcg_at_5": (results.get("recommendations") or {}).get("ndcg_at_5"),
        "judge_groundedness_pct": rec_judge.get("mean_groundedness_pct"),
        "ragas_faithfulness": ragas_metrics.get("faithfulness"),
        "ragas_answer_correctness": ragas_metrics.get("answer_correctness"),
    }


def load_history(limit: int = 30) -> list:
    """Compact metric snapshots across all persisted runs, oldest -> newest."""
    if not RESULTS.exists():
        return []
    snaps = []
    for path in RESULTS.glob("*.json"):
        if path.name == "latest.json":
            continue
        try:
            snaps.append(_summarize(json.loads(path.read_text())))
        except (json.JSONDecodeError, OSError):
            continue
    snaps.sort(key=lambda s: s.get("generated_at") or "")
    return snaps[-limit:]


if __name__ == "__main__":
    import pprint

    pprint.pp(run())
