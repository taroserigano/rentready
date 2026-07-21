"""Evaluate the PDF RAG quality with RAGAS.

RAGAS scores answers on things like faithfulness (did the answer stick to the
retrieved context?) and answer relevancy. This needs an LLM + embeddings, so
it really shines once ANTHROPIC_API_KEY is set. Run it directly:

    python backend/eval.py
"""

from settings import settings, get_embeddings
from llm import get_langchain_llm
import rag_llamaindex

# A tiny gold set of questions about the sample application.
QUESTIONS = [
    "What is the applicant's monthly income?",
    "What rent does the applicant want to pay?",
    "Does the applicant have pets?",
]


def run(applicant_id: str) -> dict:
    """Run a small RAGAS evaluation for one applicant's documents."""
    if get_langchain_llm() is None:
        return {
            "skipped": True,
            "reason": "No ANTHROPIC_API_KEY; RAGAS needs an LLM to judge.",
        }

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, faithfulness
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LlamaIndexEmbeddingsWrapper

        rows = {"question": [], "answer": [], "contexts": []}
        for q in QUESTIONS:
            res = rag_llamaindex.query(applicant_id, q)
            rows["question"].append(q)
            rows["answer"].append(res["answer"])
            rows["contexts"].append(res["sources"] or ["(no context)"])

        dataset = Dataset.from_dict(rows)
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy],
            llm=LangchainLLMWrapper(get_langchain_llm()),
            embeddings=LlamaIndexEmbeddingsWrapper(get_embeddings()),
        )
        return {"skipped": False, "scores": result.to_pandas().to_dict()}
    except Exception as exc:  # noqa: BLE001
        return {"skipped": True, "reason": f"{type(exc).__name__}: {exc}"}


if __name__ == "__main__":
    import sys

    aid = sys.argv[1] if len(sys.argv) > 1 else ""
    print(run(aid))
