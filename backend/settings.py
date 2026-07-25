"""Central configuration + the embedding model (with an offline fallback).

All environment-driven config lives here in one typed place
(pydantic-settings), so the rest of the code just imports `settings`.
"""

import hashlib
import math
import re
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CHROMA_DIR = ROOT / "chroma_db"
UPLOAD_DIR = ROOT / "uploads"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT / ".env"), extra="ignore"
    )

    # Anthropic
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-5"

    # Postgres (e.g. a shared Neon project). Empty -> store.py uses its local
    # SQLite file instead (the default for local dev and every test).
    database_url: str = ""

    # Pinecone (applicant RAG vector store). vector_provider="chroma" (default)
    # keeps rag_llamaindex.py on the local Chroma dir; "pinecone" switches it to
    # a hosted Pinecone index using pinecone_api_key/pinecone_index.
    vector_provider: str = "chroma"
    pinecone_api_key: str = ""
    pinecone_index: str = "rentready-applications"
    # Free/Starter Pinecone projects cap out at 5 serverless indexes, so a
    # shared project (used by other apps too) often has none to spare --
    # RentReady's vectors then live in their own namespace within an
    # existing index (see PINECONE_INDEX/PINECONE_NAMESPACE in .env) rather
    # than a dedicated one.
    pinecone_namespace: str = "rentready"

    # LangSmith
    langsmith_tracing: bool = True
    langsmith_api_key: str = ""
    langsmith_project: str = "rentready"
    langsmith_endpoint: str = "https://api.smith.langchain.com"

    # Phoenix
    phoenix_enabled: bool = True
    phoenix_collector_endpoint: str = "http://localhost:6006"

    # Datadog (optional metric sink for monitoring; skipped if no key)
    datadog_api_key: str = ""
    datadog_site: str = "datadoghq.com"

    # Neo4j (local dev only — this default matches the standalone instance
    # this app expects on localhost. If Neo4j is ever exposed beyond
    # localhost, override NEO4J_PASSWORD via env instead of relying on this.)
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = "rentready123"
    neo4j_database: str = "neo4j"

    # Embeddings
    embedding_backend: str = "auto"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # RAG / eligibility knobs
    retriever_k: int = 4
    income_rent_multiple: float = 3.0
    min_credit_score: int = 620

    # Upload cap for /upload's PDF (real applications run a few pages; this is
    # generous headroom while still bounding memory/disk on one request).
    max_upload_mb: int = 20

    # CORS: comma-separated origins, or "*" (default -- fine for local dev,
    # where the frontend's dev-server origin varies by port). Set explicitly
    # to the real frontend origin(s) for any non-localhost deployment, e.g.
    #   CORS_ALLOWED_ORIGINS=https://rentready.example.com
    cors_allowed_origins: str = "*"

    @property
    def has_anthropic(self) -> bool:
        return bool(self.anthropic_api_key)

    @property
    def has_langsmith(self) -> bool:
        return self.langsmith_tracing and bool(self.langsmith_api_key)

    @property
    def has_datadog(self) -> bool:
        return bool(self.datadog_api_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()


# --------------------------------------------------------------------------
# Embeddings: local HuggingFace model by default, with an offline fallback
# (a tiny hashing vectorizer) so the app runs even when huggingface.co is
# unreachable. The same instance must be shared by ingest and query code.
# --------------------------------------------------------------------------
_HASH_DIM = 512
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _hash_embed(text: str, dim: int = _HASH_DIM) -> list:
    vec = [0.0] * dim
    for token in _TOKEN_RE.findall(text.lower()):
        h = int(hashlib.md5(token.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm else vec


@lru_cache(maxsize=1)
def get_embeddings():
    """Return a LlamaIndex-compatible embedding model.

    Tries the local HuggingFace model first; on any failure (e.g. blocked
    network) it falls back to an offline hashing embedder so the app always
    runs. Force with EMBEDDING_BACKEND = huggingface | hash.
    """
    from llama_index.core.embeddings import BaseEmbedding

    class HashEmbedding(BaseEmbedding):
        """Offline, dependency-free embedding (word-overlap similarity)."""

        def _get_query_embedding(self, query: str) -> list:
            return _hash_embed(query)

        async def _aget_query_embedding(self, query: str) -> list:
            return _hash_embed(query)

        def _get_text_embedding(self, text: str) -> list:
            return _hash_embed(text)

    backend = settings.embedding_backend.lower()
    if backend == "hash":
        print("Embeddings: offline HashEmbedding (forced).")
        return HashEmbedding()

    try:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        emb = HuggingFaceEmbedding(model_name=settings.embedding_model)
        emb.get_text_embedding("warm up")
        print(f"Embeddings: HuggingFace '{settings.embedding_model}'.")
        return emb
    except Exception as exc:  # noqa: BLE001 - any failure -> safe fallback
        if backend == "huggingface":
            raise
        print(
            f"Embeddings: HuggingFace unavailable ({type(exc).__name__}); "
            "using offline HashEmbedding. Set EMBEDDING_BACKEND=huggingface "
            "once huggingface.co is reachable."
        )
        return HashEmbedding()
