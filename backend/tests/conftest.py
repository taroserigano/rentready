"""Shared test fixtures.

Tests run fully offline: we force the offline embedder and disable the LLM
so no network calls happen and results are deterministic.
"""

import os
import sys
from pathlib import Path

import pytest

# Make the backend package importable and force offline embeddings.
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))
os.environ["EMBEDDING_BACKEND"] = "hash"


@pytest.fixture(autouse=True)
def no_llm(monkeypatch):
    """Disable both LLM clients so logic is deterministic and offline.

    The consuming modules do `from llm import get_*`, binding the name in
    their own namespace, so we patch the reference in each module that uses
    it (resolved at call time from that module's globals).
    """
    import eligibility
    import graph
    import graphrag

    monkeypatch.setattr(eligibility, "get_langchain_llm", lambda: None)
    monkeypatch.setattr(graphrag, "get_langchain_llm", lambda: None)
    # Force the in-memory graph fallback so tests are deterministic and never
    # depend on a running Neo4j instance or its current data.
    monkeypatch.setattr(graph, "is_available", lambda: False)
    yield
