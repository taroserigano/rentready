"""Optional reranking with FlashRank.

A reranker reorders retrieved chunks so the most relevant are first, which
usually improves RAG answers. FlashRank downloads a small model on first use;
if that's blocked, this degrades to a no-op (returns the input order).
"""

from functools import lru_cache


@lru_cache(maxsize=1)
def _ranker():
    try:
        from flashrank import Ranker

        return Ranker(max_length=256)
    except Exception as exc:  # noqa: BLE001
        print(f"FlashRank unavailable ({type(exc).__name__}); skipping rerank.")
        return None


def rerank(query: str, passages: list, top_k: int = 4) -> list:
    """Return passages reordered by relevance. No-op if FlashRank is absent."""
    ranker = _ranker()
    if ranker is None or not passages:
        return passages[:top_k]
    try:
        from flashrank import RerankRequest

        req = RerankRequest(
            query=query,
            passages=[{"id": i, "text": p} for i, p in enumerate(passages)],
        )
        results = ranker.rerank(req)
        return [passages[r["id"]] for r in results][:top_k]
    except Exception as exc:  # noqa: BLE001
        print(f"FlashRank rerank failed ({type(exc).__name__}); original order.")
        return passages[:top_k]
