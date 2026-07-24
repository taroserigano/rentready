"""Tests for rag_llamaindex.query()'s error-degradation contract.

Regression: an LLM/query-engine failure (network error, rate limit, timeout)
used to propagate uncaught out of query(), surfacing as a bare 500 from
POST /ask — inconsistent with every other chat agent's "never raises"
contract. query() must now degrade to the same retrieve-only answer used
when no Anthropic key is configured.
"""

import rag_llamaindex as rli


class _FakeNode:
    def __init__(self, text):
        self._text = text

    def get_content(self):
        return self._text


class _FakeNodeWithScore:
    def __init__(self, text):
        self.node = _FakeNode(text)


class _FakeRetriever:
    def retrieve(self, question):
        return [_FakeNodeWithScore("some retrieved application context")]


class _FakeEngine:
    def query(self, question):
        raise RuntimeError("simulated Anthropic outage")


class _FakeIndex:
    def as_query_engine(self, **kwargs):
        return _FakeEngine()

    def as_retriever(self, **kwargs):
        return _FakeRetriever()


def test_query_degrades_to_retrieve_only_on_llm_failure(monkeypatch):
    monkeypatch.setattr(rli, "_index", lambda: _FakeIndex())
    # Truthy stand-in so query() takes the LLM branch (and hits the engine
    # failure above) rather than skipping straight to retrieve-only.
    monkeypatch.setattr(rli, "get_llamaindex_llm", lambda: object())

    result = rli.query("APP-TEST", "What is the income?")
    assert result["source"] == "mock"
    assert "some retrieved application context" in result["answer"]
    assert result["sources"]


def test_query_degrades_fully_if_even_retrieval_fails(monkeypatch):
    """Belt-and-suspenders: if the retrieve-only fallback ALSO fails (e.g. the
    index itself is broken), query() still returns a safe dict, never raises."""
    def _boom():
        raise RuntimeError("chroma is down")

    monkeypatch.setattr(rli, "_index", _boom)
    monkeypatch.setattr(rli, "get_llamaindex_llm", lambda: None)

    result = rli.query("APP-TEST", "What is the income?")
    assert result["source"] == "mock"
    assert result["answer"]
    assert result["sources"] == []
