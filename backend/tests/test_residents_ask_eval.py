"""Tests for the Residents-chat ("Ask about residents") evaluation runner.

Runs fully offline: the Anthropic LLM is disabled so every item takes the
deterministic templated path. This is the full-dataset regression gate —
CI-safe, free, and 100% reproducible.
"""

import pytest

import residents_chat as rc
from evals import residents_ask_eval
from evals.residents_ask_dataset import RESIDENTS_ASK_DATASET


@pytest.fixture(autouse=True)
def _no_residents_llm(monkeypatch):
    monkeypatch.setattr(rc, "get_langchain_llm", lambda: None)


def test_run_returns_all_metrics_in_range():
    res = residents_ask_eval.run(use_llm=False)
    for key in ("intent_accuracy", "groundedness", "safety"):
        assert key in res
        assert 0.0 <= res[key] <= 1.0
    assert res["n"] == len(RESIDENTS_ASK_DATASET) > 0
    assert res["use_llm"] is False
    assert len(res["items"]) == res["n"]
    for item in res["items"]:
        assert {
            "id", "question", "expected_intent", "predicted_intent",
            "intent_ok", "grounded_ok", "safety_ok", "source", "answer",
        }.issubset(set(item))


def test_run_is_deterministic():
    a = residents_ask_eval.run(use_llm=False)
    b = residents_ask_eval.run(use_llm=False)
    assert a["intent_accuracy"] == b["intent_accuracy"]
    assert a["groundedness"] == b["groundedness"]
    assert a["safety"] == b["safety"]


def test_all_items_pass_all_three_metrics():
    """Every item in the golden set should pass intent/grounding/safety on the
    deterministic path — the full-dataset regression gate."""
    res = residents_ask_eval.run(use_llm=False)
    failing = [
        it["id"] for it in res["items"]
        if not (it["intent_ok"] and it["grounded_ok"] and it["safety_ok"])
    ]
    assert not failing, f"{len(failing)} item(s) failed: {failing}"


def test_dataset_covers_all_intents():
    """The golden set exercises every intent the router can produce."""
    intents = {it["expected_intent"] for it in RESIDENTS_ASK_DATASET}
    assert intents == {
        "horizon", "frequency", "severity", "arrears", "cure", "retention",
        "explain", "compare", "property_health", "at_risk_residents",
        "general", "governance",
    }


def test_dataset_has_no_duplicate_ids():
    ids = [it["id"] for it in RESIDENTS_ASK_DATASET]
    assert len(ids) == len(set(ids))


def test_adversarial_safety_probes_stay_safe():
    """The safety-probe items (eviction/denial/protected-attribute baiting)
    must never trip the unsafe-lexicon scan, even on the templated path."""
    res = residents_ask_eval.run(use_llm=False)
    by_id = {it["id"]: it for it in res["items"]}
    adversarial = [it for it in RESIDENTS_ASK_DATASET if it["id"].startswith("adversarial-")]
    assert len(adversarial) >= 5
    for it in adversarial:
        assert by_id[it["id"]]["safety_ok"], f"{it['id']} tripped the safety scan"
