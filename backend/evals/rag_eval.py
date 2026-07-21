"""RAGAS evaluation of the PDF question-answering pipeline.

RAGAS grades the RAG answers on:
  * faithfulness        -- did the answer stick to the retrieved context?
  * answer_relevancy    -- is the answer on-topic for the question?
  * answer_correctness  -- does it match the gold answer? (needs a reference)

This needs an LLM to judge, so it's skipped cleanly when ANTHROPIC_API_KEY is
absent (e.g. in CI). Run directly:  python backend/evals/rag_eval.py

NOTE on async: RAGAS uses nest_asyncio, which cannot patch uvicorn's uvloop.
So when called from the API we run it in an isolated subprocess
(``run_isolated``) that gets a clean standard event loop.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

_JSON_SENTINEL = "RAGAS_RESULT_JSON:"

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

import pdf_ingest  # noqa: E402
import rag_llamaindex  # noqa: E402
from llm import get_langchain_llm  # noqa: E402
from settings import get_embeddings  # noqa: E402

HERE = Path(__file__).resolve().parent
DATASETS = HERE / "datasets"
APPLICATIONS_DIR = BACKEND.parent / "data" / "applications"


def _load(name: str) -> list:
    with open(DATASETS / name) as f:
        return [json.loads(line) for line in f if line.strip()]


def _ensure_ingested(slug: str) -> str:
    """Index a sample applicant's PDF under a stable eval id (idempotent-ish)."""
    applicant_id = f"eval-{slug}"
    pdf = APPLICATIONS_DIR / f"{slug}.pdf"
    text = pdf_ingest.extract_text(str(pdf))
    rag_llamaindex.ingest(applicant_id, text)
    return applicant_id


def run() -> dict:
    """Run RAGAS over the gold Q&A set. Returns a metrics dict (or skipped)."""
    if get_langchain_llm() is None:
        return {
            "skipped": True,
            "reason": "No ANTHROPIC_API_KEY; RAGAS needs an LLM to judge.",
        }

    try:
        from ragas import EvaluationDataset, evaluate
        from ragas.embeddings import LlamaIndexEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (
            answer_correctness,
            answer_relevancy,
            faithfulness,
        )

        rows = _load("pdf_qa.jsonl")
        ingested: dict = {}
        samples, per_case = [], []
        for row in rows:
            slug = row["slug"]
            if slug not in ingested:
                ingested[slug] = _ensure_ingested(slug)
            res = rag_llamaindex.query(ingested[slug], row["question"])
            contexts = rag_llamaindex.retrieve_contexts(
                ingested[slug], row["question"]
            )
            samples.append(
                {
                    "user_input": row["question"],
                    "retrieved_contexts": contexts or ["(no context)"],
                    "response": res["answer"],
                    "reference": row["ground_truth"],
                }
            )
            per_case.append({"id": row["id"], "question": row["question"]})

        dataset = EvaluationDataset.from_list(samples)
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, answer_correctness],
            llm=LangchainLLMWrapper(get_langchain_llm()),
            embeddings=LlamaIndexEmbeddingsWrapper(get_embeddings()),
            show_progress=False,
        )
        df = result.to_pandas()

        def col_mean(name):
            return round(float(df[name].mean()), 4) if name in df else None

        return {
            "skipped": False,
            "n": len(samples),
            "metrics": {
                "faithfulness": col_mean("faithfulness"),
                "answer_relevancy": col_mean("answer_relevancy"),
                "answer_correctness": col_mean("answer_correctness"),
            },
            "per_case": per_case,
        }
    except Exception as exc:  # noqa: BLE001 - RAGAS API/version is fragile
        return {"skipped": True, "reason": f"{type(exc).__name__}: {exc}"}


def run_isolated(timeout: int = 300) -> dict:
    """Run RAGAS in a subprocess to avoid uvloop/nest_asyncio conflicts.

    Safe to call from inside the uvicorn (uvloop) server. Falls back to a
    skip dict on any failure so the caller never crashes.
    """
    if get_langchain_llm() is None:
        return {
            "skipped": True,
            "reason": "No ANTHROPIC_API_KEY; RAGAS needs an LLM to judge.",
        }
    try:
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve())],
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ},
        )
        for line in reversed(proc.stdout.splitlines()):
            if line.startswith(_JSON_SENTINEL):
                return json.loads(line[len(_JSON_SENTINEL):])
        return {
            "skipped": True,
            "reason": f"RAGAS subprocess produced no result (exit "
            f"{proc.returncode}).",
        }
    except Exception as exc:  # noqa: BLE001
        return {"skipped": True, "reason": f"{type(exc).__name__}: {exc}"}


if __name__ == "__main__":
    print(_JSON_SENTINEL + json.dumps(run()))
