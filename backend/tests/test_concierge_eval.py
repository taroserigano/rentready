"""Tests for the Concierge evaluation runner.

Runs fully offline: the conftest forces the hash embedder, the Anthropic LLM is
disabled (deterministic path), and the lease knowledge base is ingested into a
throwaway Chroma dir per module so the shared repo store is never touched.
"""

import pytest

import concierge
import knowledge
from evals import concierge_eval
from evals.concierge_dataset import CONCIERGE_DATASET, item_kind


@pytest.fixture(autouse=True)
def _no_concierge_llm(monkeypatch):
    monkeypatch.setattr(concierge, "get_langchain_llm", lambda: None)


@pytest.fixture(scope="module")
def kb(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("knowledge_chroma_eval")
    orig = knowledge.CHROMA_DIR
    knowledge.CHROMA_DIR = tmp
    knowledge._chroma_collection.cache_clear()
    knowledge._index.cache_clear()
    knowledge._corpus.cache_clear()
    knowledge.ingest_all()
    yield
    knowledge.CHROMA_DIR = orig
    knowledge._chroma_collection.cache_clear()
    knowledge._index.cache_clear()
    knowledge._corpus.cache_clear()


def test_run_returns_all_metrics_in_range(kb):
    res = concierge_eval.run(use_llm=False)
    for key in ("route_accuracy", "retrieval_hit_rate", "groundedness"):
        assert key in res
        assert 0.0 <= res[key] <= 1.0
    assert res["n"] == len(CONCIERGE_DATASET) > 0
    assert res["use_llm"] is False
    assert len(res["items"]) == res["n"]
    for item in res["items"]:
        assert set(item) == {
            "id", "question", "expected_route", "predicted_route",
            "route_ok", "retrieval_ok", "grounded_ok",
        }


def test_run_is_deterministic(kb):
    a = concierge_eval.run(use_llm=False)
    b = concierge_eval.run(use_llm=False)
    assert a["route_accuracy"] == b["route_accuracy"]
    assert a["retrieval_hit_rate"] == b["retrieval_hit_rate"]
    assert a["groundedness"] == b["groundedness"]


def test_route_accuracy_is_high(kb):
    res = concierge_eval.run(use_llm=False)
    # The router should classify the labeled dataset essentially perfectly.
    assert res["route_accuracy"] >= 0.9


def test_specific_items_route_and_retrieval(kb):
    res = concierge_eval.run(use_llm=False)
    by_id = {it["id"]: it for it in res["items"]}

    # A property item routes correctly and finds a property source.
    assert by_id["prop-gym-yes-0"]["route_ok"]
    assert by_id["prop-gym-yes-0"]["retrieval_ok"]

    # A lease item retrieves the expected section (proven with hash embedder).
    assert by_id["lease-deposit-0"]["route_ok"]
    assert by_id["lease-deposit-0"]["retrieval_ok"]

    # A compare item routes to compare and finds the expected property.
    assert by_id["compare-cheapest-all-1"]["route_ok"]
    assert by_id["compare-cheapest-all-1"]["retrieval_ok"]

    # A "both" item (property+lease keyword collision) routes correctly.
    assert by_id["both-rent-pet"]["route_ok"]

    # An unscoped "general" question declines gracefully rather than forcing
    # a spurious lease citation (see concierge._assemble's want_lease guard).
    assert by_id["general-hi"]["route_ok"]
    assert by_id["general-hi"]["retrieval_ok"]


def test_all_items_pass_all_three_metrics(kb):
    """Every item in the golden set should pass route/retrieval/groundedness
    on the deterministic path — this is the full-dataset regression gate."""
    res = concierge_eval.run(use_llm=False)
    failing = [
        it["id"] for it in res["items"]
        if not (it["route_ok"] and it["retrieval_ok"] and it["grounded_ok"])
    ]
    assert not failing, f"{len(failing)} item(s) failed: {failing}"


def test_dataset_kinds_are_covered():
    kinds = {item_kind(it) for it in CONCIERGE_DATASET}
    assert kinds == {"property", "lease", "compare"}


def test_dataset_covers_all_routes():
    """The golden set exercises every route the classifier can produce,
    including 'both' and 'general' (absent from the original 11-item set)."""
    routes = {it["expected_route"] for it in CONCIERGE_DATASET}
    assert routes == {"property", "lease", "both", "compare", "general"}
