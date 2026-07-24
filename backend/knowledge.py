"""Lease knowledge base: Chroma ingestion + HYBRID retrieval.

We generate a standard lease per property (``leases.py``) and ingest every
lease SECTION as its own chunk into a dedicated Chroma collection
(``"knowledge"``) — kept separate from the applicant RAG collection
(``"applications"`` in ``rag_llamaindex.py``).

Retrieval is HYBRID because this environment runs the offline HASH embedder
(``EMBEDDING_BACKEND=hash``), where pure vector similarity is weak:

    vector (Chroma / LlamaIndex)  ∪  in-process BM25 keyword scorer
        → reciprocal-rank fusion → FlashRank rerank → top-k

The BM25 scorer is dependency-free (plain Python over the chunk texts), so the
keyword path carries retrieval quality even when embeddings are just hashes.

Public API:
    ingest_all() -> dict                       # idempotent; skips if populated
    search(query, property_id=None, k=4) -> list[dict]
    count() -> int
"""

from __future__ import annotations

import math
import re
from functools import lru_cache

from settings import CHROMA_DIR, get_embeddings
import graph
import leases
import rerank

COLLECTION = "knowledge"

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# BM25 parameters (standard defaults).
_BM25_K1 = 1.5
_BM25_B = 0.75
# Reciprocal-rank-fusion constant.
_RRF_K = 60


# ---------------------------------------------------------------------------
# Chroma / LlamaIndex plumbing
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _chroma_client():
    import chromadb

    return chromadb.PersistentClient(path=str(CHROMA_DIR))


@lru_cache(maxsize=1)
def _chroma_collection():
    return _chroma_client().get_or_create_collection(COLLECTION)


@lru_cache(maxsize=1)
def _index():
    from llama_index.core import StorageContext, VectorStoreIndex, Settings
    from llama_index.vector_stores.chroma import ChromaVectorStore

    Settings.embed_model = get_embeddings()
    Settings.llm = None  # retrieval only; synthesis lives in concierge.py

    vector_store = ChromaVectorStore(chroma_collection=_chroma_collection())
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_vector_store(
        vector_store, storage_context=storage_context
    )


def count() -> int:
    """Number of ingested lease chunks."""
    try:
        return _chroma_collection().count()
    except Exception:  # noqa: BLE001
        return 0


# ---------------------------------------------------------------------------
# Ingestion (idempotent)
# ---------------------------------------------------------------------------
def ingest_all() -> dict:
    """Ingest every property's lease sections. Idempotent: skips re-ingesting
    when the collection already holds the CURRENT expected chunk count. If a
    lease section was added/removed since the collection was built, the count
    no longer matches -- the stale collection is wiped and rebuilt so the new
    content is never silently missing from retrieval."""
    props = graph.load_properties()
    expected = len(props) * leases.SECTION_COUNT
    existing = count()
    if existing == expected and existing > 0:
        return {"indexed": existing, "skipped": True, "collection": COLLECTION}

    from llama_index.core.schema import TextNode

    if existing:
        _chroma_client().delete_collection(COLLECTION)
        _chroma_collection.cache_clear()
        _index.cache_clear()

    nodes = []
    for prop in props:
        pid = prop.get("id", "")
        pname = prop.get("name", "")
        for section, text in leases.lease_sections(prop):
            nodes.append(
                TextNode(
                    text=f"{section} — {pname}\n{text}",
                    metadata={
                        "property_id": pid,
                        "property_name": pname,
                        "doc_type": "lease",
                        "section": section,
                    },
                )
            )

    if nodes:
        _index().insert_nodes(nodes)

    _corpus.cache_clear()
    return {
        "indexed": count(),
        "skipped": False,
        "collection": COLLECTION,
        "properties": len(props),
    }


# ---------------------------------------------------------------------------
# In-process BM25 keyword corpus (built from the Chroma contents)
# ---------------------------------------------------------------------------
def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


@lru_cache(maxsize=8)
def _corpus(count_key: int):
    """Load all chunks from Chroma and precompute BM25 statistics.

    Cached on the collection count so a fresh ingest transparently rebuilds it.
    Returns a dict with per-doc token frequencies plus corpus-wide idf/avgdl.
    """
    try:
        raw = _chroma_collection().get(include=["documents", "metadatas"])
    except Exception:  # noqa: BLE001
        raw = {"documents": [], "metadatas": []}

    docs = raw.get("documents") or []
    metas = raw.get("metadatas") or []

    entries = []
    df: dict[str, int] = {}
    total_len = 0
    for text, meta in zip(docs, metas):
        tokens = _tokenize(text)
        tf: dict[str, int] = {}
        for tok in tokens:
            tf[tok] = tf.get(tok, 0) + 1
        entries.append(
            {"text": text, "meta": meta or {}, "tf": tf, "len": len(tokens)}
        )
        total_len += len(tokens)
        for tok in tf:
            df[tok] = df.get(tok, 0) + 1

    n = len(entries)
    avgdl = (total_len / n) if n else 0.0
    idf = {
        tok: math.log(1 + (n - d + 0.5) / (d + 0.5)) for tok, d in df.items()
    }
    return {"entries": entries, "idf": idf, "avgdl": avgdl, "n": n}


def _bm25_scores(query: str, property_id: str | None):
    """BM25 score for every (optionally property-filtered) chunk. Returns a
    list of (score, entry) sorted high→low, positive scores only."""
    corpus = _corpus(count())
    idf = corpus["idf"]
    avgdl = corpus["avgdl"] or 1.0
    q_terms = set(_tokenize(query))

    scored = []
    for entry in corpus["entries"]:
        if property_id and entry["meta"].get("property_id") != property_id:
            continue
        tf = entry["tf"]
        dl = entry["len"] or 1
        score = 0.0
        for term in q_terms:
            f = tf.get(term)
            if not f:
                continue
            denom = f + _BM25_K1 * (1 - _BM25_B + _BM25_B * dl / avgdl)
            score += idf.get(term, 0.0) * (f * (_BM25_K1 + 1)) / denom
        if score > 0:
            scored.append((score, entry))
    scored.sort(key=lambda t: t[0], reverse=True)
    return scored


# ---------------------------------------------------------------------------
# Vector retrieval
# ---------------------------------------------------------------------------
def _vector_hits(query: str, property_id: str | None, k: int) -> list[dict]:
    """Chroma/LlamaIndex vector retrieval. Best-effort: [] on any failure."""
    try:
        kwargs = {"similarity_top_k": k}
        if property_id:
            from llama_index.core.vector_stores import (
                FilterOperator,
                MetadataFilter,
                MetadataFilters,
            )

            kwargs["filters"] = MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="property_id",
                        value=property_id,
                        operator=FilterOperator.EQ,
                    )
                ]
            )
        retriever = _index().as_retriever(**kwargs)
        out = []
        for node in retriever.retrieve(query):
            out.append(
                {"text": node.node.get_content(), "meta": dict(node.node.metadata or {})}
            )
        return out
    except Exception as exc:  # noqa: BLE001
        print(f"knowledge: vector retrieval failed ({type(exc).__name__}); keyword only.")
        return []


# ---------------------------------------------------------------------------
# Hybrid search
# ---------------------------------------------------------------------------
def _passage_from(entry_or_hit: dict) -> dict:
    meta = entry_or_hit.get("meta", {})
    text = entry_or_hit["text"]
    # Strip the "Section — Property\n" header we prepended at ingest for display.
    body = text.split("\n", 1)[1] if "\n" in text else text
    return {
        "text": body.strip(),
        "raw": text,
        "section": meta.get("section", ""),
        "property_id": meta.get("property_id", ""),
        "property_name": meta.get("property_name", ""),
        "doc_type": meta.get("doc_type", "lease"),
    }


def search(query: str, property_id: str | None = None, k: int = 4) -> list[dict]:
    """Hybrid retrieval over lease chunks.

    Fuses vector hits (Chroma) with BM25 keyword hits via reciprocal-rank
    fusion, reranks the fused set with FlashRank, and returns the top-k as
    passage dicts: ``{text, section, property_id, property_name, doc_type}``.
    """
    if not query or not query.strip():
        return []

    pool = max(k * 3, 8)
    vector = _vector_hits(query, property_id, pool)
    keyword = [entry for _, entry in _bm25_scores(query, property_id)[:pool]]

    # Reciprocal-rank fusion. Key by raw chunk text (unique per section/property).
    fused: dict[str, dict] = {}
    for rank, hit in enumerate(vector):
        key = hit["text"]
        fused.setdefault(key, {"hit": hit, "score": 0.0})
        fused[key]["score"] += 1.0 / (_RRF_K + rank)
    for rank, entry in enumerate(keyword):
        key = entry["text"]
        fused.setdefault(key, {"hit": entry, "score": 0.0})
        fused[key]["score"] += 1.0 / (_RRF_K + rank)

    if not fused:
        return []

    ordered = sorted(fused.values(), key=lambda v: v["score"], reverse=True)
    candidates = [v["hit"] for v in ordered[:pool]]

    # Rerank on the display body (what actually answers the question).
    passages = [_passage_from(h) for h in candidates]
    by_body = {p["text"]: p for p in passages}
    reranked_bodies = rerank.rerank(query, [p["text"] for p in passages], top_k=k)

    seen = set()
    result = []
    for body in reranked_bodies:
        p = by_body.get(body)
        if p and body not in seen:
            seen.add(body)
            result.append(p)
    # Safety net if rerank returned nothing recognizable.
    if not result:
        result = passages[:k]
    return result
