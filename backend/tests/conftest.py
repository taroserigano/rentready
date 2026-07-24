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
# Force the local SQLite/Chroma path even though the repo-root .env may carry
# a real DATABASE_URL/PINECONE_API_KEY for the deployed instance -- tests must
# never touch that shared, real infrastructure. Real env vars beat .env values
# in pydantic-settings' precedence, so these two lines are what actually keep
# `settings.database_url`/`settings.vector_provider` empty/"chroma" here.
os.environ["DATABASE_URL"] = ""
os.environ["VECTOR_PROVIDER"] = "chroma"


@pytest.fixture(scope="session", autouse=True)
def _isolate_db(tmp_path_factory):
    """Point the SQLite store at a throwaway temp DB for the whole test session.

    Without this, any test that saves an applicant/booking/decision would write
    to the real ``rentready.db``. Tests that need their own isolated DB still
    monkeypatch ``store.DB_PATH`` per-test (that just overrides this default and
    restores back to the temp path, never the real file).
    """
    import store

    original = store.DB_PATH
    store.DB_PATH = tmp_path_factory.mktemp("store") / "test.db"
    store.init_db()
    try:
        yield
    finally:
        store.DB_PATH = original


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
    import residents_chat
    import risk_chat

    monkeypatch.setattr(eligibility, "get_langchain_llm", lambda: None)
    monkeypatch.setattr(graphrag, "get_langchain_llm", lambda: None)
    # The grounded chat agents synthesize prose via the LLM; disabling it forces
    # the deterministic "rules" fallback so chat tests are offline + repeatable
    # (grounding/artifacts are built regardless of the LLM).
    monkeypatch.setattr(risk_chat, "get_langchain_llm", lambda: None)
    monkeypatch.setattr(residents_chat, "get_langchain_llm", lambda: None)
    # Force the in-memory graph fallback so tests are deterministic and never
    # depend on a running Neo4j instance or its current data.
    monkeypatch.setattr(graph, "is_available", lambda: False)
    yield
