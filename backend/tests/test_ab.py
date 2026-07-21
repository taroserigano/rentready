"""Offline tests for the A/B comparison harness.

We test the deterministic decision/aggregation logic and the graceful skip.
The live generation/judging path needs a key and is exercised manually.
"""

from evals import ab


def test_list_variants_shape():
    variants = ab.list_variants()
    assert len(variants) >= 2
    for v in variants:
        assert {"id", "label", "model"} <= set(v)


def test_cost_uses_model_price_table():
    # Haiku is cheaper than Sonnet for the same token counts.
    sonnet = ab._cost("claude-sonnet-4-6", 1000, 1000)
    haiku = ab._cost("claude-haiku-4-5-20251001", 1000, 1000)
    assert haiku < sonnet
    # Unknown model falls back to the default price (no crash).
    assert ab._cost("mystery-model", 1000, 0) > 0


def test_decide_prefers_higher_groundedness():
    a = {"id": "a", "label": "A", "mean_groundedness_pct": 0.9, "est_cost_usd": 0.01}
    b = {"id": "b", "label": "B", "mean_groundedness_pct": 0.6, "est_cost_usd": 0.001}
    winner, rationale = ab._decide(a, b)
    assert winner == "a"
    assert "groundedness" in rationale.lower()


def test_decide_breaks_tie_on_cost():
    a = {"id": "a", "label": "A", "mean_groundedness_pct": 0.90, "est_cost_usd": 0.05}
    b = {"id": "b", "label": "B", "mean_groundedness_pct": 0.91, "est_cost_usd": 0.01}
    winner, rationale = ab._decide(a, b)  # within 0.02 -> tie -> cheaper wins
    assert winner == "b"
    assert "cheaper" in rationale.lower()


def test_delta_handles_missing():
    a = {"x": 0.5}
    b = {"x": 0.8}
    assert ab._delta(a, b, "x") == 0.3
    assert ab._delta({"x": None}, b, "x") is None


def test_compare_skips_without_llm(monkeypatch):
    monkeypatch.setattr(ab.judges, "get_langchain_llm", lambda: None)
    out = ab.compare("concise", "detailed")
    assert out["skipped"] is True


def test_compare_rejects_unknown_variant(monkeypatch):
    monkeypatch.setattr(ab.judges, "get_langchain_llm", lambda: object())
    out = ab.compare("concise", "does-not-exist")
    assert out["skipped"] is True
    assert "Unknown variant" in out["reason"]
