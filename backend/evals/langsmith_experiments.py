"""Run the eligibility suite as a tracked LangSmith experiment.

This pushes our golden eligibility dataset to LangSmith, runs the deterministic
rule engine as the "system under test", and scores each prediction with a
custom evaluator. The result is a versioned, shareable experiment in the
LangSmith UI (great for showing eval-over-time on a portfolio).

Skipped cleanly when LANGSMITH_API_KEY is absent, so it never breaks CI.
Run directly:  python backend/evals/langsmith_experiments.py
"""

import json
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

import eligibility  # noqa: E402
from models import ApplicantProfile  # noqa: E402
from settings import settings  # noqa: E402

HERE = Path(__file__).resolve().parent
DATASET_NAME = "rentready-eligibility"


def _load(name: str) -> list:
    with open(HERE / "datasets" / name) as f:
        return [json.loads(line) for line in f if line.strip()]


def _predict(inputs: dict) -> dict:
    """System under test: deterministic eligibility verdict (no LLM)."""
    profile = ApplicantProfile(**inputs["profile"])
    result = eligibility.evaluate(profile, explain=False)
    return {"verdict": result.verdict}


def _verdict_match(outputs: dict, reference_outputs: dict) -> dict:
    """Custom evaluator: exact-match on the verdict."""
    ok = outputs.get("verdict") == reference_outputs.get("verdict")
    return {"key": "verdict_exact_match", "score": int(ok)}


def _ensure_dataset(client):
    """Create (or reuse) the LangSmith dataset from our local golden file."""
    rows = _load("eligibility.jsonl")
    if client.has_dataset(dataset_name=DATASET_NAME):
        return DATASET_NAME
    dataset = client.create_dataset(dataset_name=DATASET_NAME)
    client.create_examples(
        inputs=[{"profile": r["profile"]} for r in rows],
        outputs=[{"verdict": r["expected_verdict"]} for r in rows],
        dataset_id=dataset.id,
    )
    return DATASET_NAME


def run() -> dict:
    if not settings.has_langsmith:
        return {
            "skipped": True,
            "reason": "No LANGSMITH_API_KEY; experiments need LangSmith.",
        }
    try:
        from langsmith import Client
        from langsmith.evaluation import evaluate

        client = Client(api_key=settings.langsmith_api_key)
        dataset_name = _ensure_dataset(client)
        results = evaluate(
            _predict,
            data=dataset_name,
            evaluators=[_verdict_match],
            client=client,
            experiment_prefix="eligibility-rules",
            metadata={"suite": "eligibility", "model": "deterministic"},
        )
        name = getattr(results, "experiment_name", None)
        return {
            "skipped": False,
            "dataset": dataset_name,
            "experiment_name": name,
            "project": settings.langsmith_project,
        }
    except Exception as exc:  # noqa: BLE001 - SDK/network is environment-specific
        return {"skipped": True, "reason": f"{type(exc).__name__}: {exc}"}


if __name__ == "__main__":
    import pprint

    pprint.pp(run())
