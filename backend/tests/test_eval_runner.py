"""Offline tests for the eval runner: deterministic tier + history aggregation.

These never touch the LLM tier (we force include_llm=False) so they stay
hermetic and fast in CI.
"""

import json

from evals import run_evals


def test_run_deterministic_tier_offline(tmp_path, monkeypatch):
    monkeypatch.setattr(run_evals, "RESULTS", tmp_path / "results")
    results = run_evals.run(include_llm=False)
    assert results["llm_tier"] is False
    assert results["judge"]["skipped"] is True
    assert results["ragas"]["skipped"] is True
    # Deterministic suites are present and sane.
    assert 0.0 <= results["eligibility"]["accuracy"] <= 1.0
    assert "field_accuracy" in results["extraction"]
    assert "ndcg_at_5" in results["recommendations"]


def test_summarize_pulls_headline_metrics():
    sample = {
        "generated_at": "2026-01-01T00:00:00+00:00",
        "eligibility": {"accuracy": 1.0},
        "extraction": {"field_accuracy": 0.95},
        "recommendations": {"ndcg_at_5": 0.85},
        "judge": {"recommendation": {"mean_groundedness_pct": 0.9}},
        "ragas": {"metrics": {"faithfulness": 0.8, "answer_correctness": 0.7}},
    }
    snap = run_evals._summarize(sample)
    assert snap["eligibility_accuracy"] == 1.0
    assert snap["judge_groundedness_pct"] == 0.9
    assert snap["ragas_faithfulness"] == 0.8


def test_load_history_sorted_and_limited(tmp_path, monkeypatch):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    monkeypatch.setattr(run_evals, "RESULTS", results_dir)
    stamps = [
        "2026-01-03T00:00:00+00:00",
        "2026-01-01T00:00:00+00:00",
        "2026-01-02T00:00:00+00:00",
    ]
    for s in stamps:
        payload = {
            "generated_at": s,
            "eligibility": {"accuracy": 1.0},
            "extraction": {"field_accuracy": 0.9},
            "recommendations": {"ndcg_at_5": 0.8},
        }
        (results_dir / f"{s.replace(':', '-')}.json").write_text(
            json.dumps(payload)
        )
    # latest.json must be ignored by history.
    (results_dir / "latest.json").write_text(json.dumps({"generated_at": "x"}))

    history = run_evals.load_history(limit=2)
    assert [h["generated_at"] for h in history] == [
        "2026-01-02T00:00:00+00:00",
        "2026-01-03T00:00:00+00:00",
    ]
