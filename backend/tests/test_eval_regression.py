"""Regression gate: fail if evaluation metrics drop below thresholds.

This is the CI quality bar. It runs the deterministic suites (no LLM, no
Neo4j) and asserts each headline metric meets its threshold.
"""

import json
from pathlib import Path

from evals import evaluators

HERE = Path(__file__).resolve().parent.parent / "evals"
THRESHOLDS = json.loads((HERE / "thresholds.json").read_text())


def _load(name: str) -> list:
    with open(HERE / "datasets" / name) as f:
        return [json.loads(line) for line in f if line.strip()]


def test_eligibility_accuracy_meets_threshold():
    res = evaluators.eval_eligibility(_load("eligibility.jsonl"))
    assert res["accuracy"] >= THRESHOLDS["eligibility_accuracy"], res


def test_extraction_field_accuracy_meets_threshold():
    res = evaluators.eval_extraction(_load("profile_extraction.jsonl"))
    assert res["field_accuracy"] >= THRESHOLDS["extraction_field_accuracy"], res


def test_recommendation_ndcg_meets_threshold():
    res = evaluators.eval_ranking(_load("recommendations.jsonl"))
    assert res["ndcg_at_5"] >= THRESHOLDS["ndcg_at_5"], res
