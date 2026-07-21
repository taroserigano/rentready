"""API endpoints for the evaluation dashboard.

Tiers:
  * deterministic suites (eligibility / extraction / ranking) -- always
    available, no LLM or DB needed, gate CI.
  * LLM tier (judge / RAGAS) and LangSmith experiments -- on-demand, since
    they call Claude. They skip cleanly when no key is configured.
"""

from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import store
from evals import (
    ab,
    concierge_eval,
    judges,
    langsmith_experiments,
    rag_eval,
    report,
    risk_eval,
    run_evals,
)

router = APIRouter(prefix="/evals", tags=["evaluations"])

SUITES = [
    {
        "id": "eligibility",
        "name": "Eligibility rules",
        "description": "Verdict exact-match accuracy vs ground truth.",
        "metric": "accuracy",
        "tier": "deterministic",
    },
    {
        "id": "extraction",
        "name": "Profile extraction",
        "description": "Per-field accuracy of the offline extractor.",
        "metric": "field_accuracy",
        "tier": "deterministic",
    },
    {
        "id": "recommendations",
        "name": "Recommendation ranking",
        "description": "NDCG@5 of the deterministic scorer vs graded relevance.",
        "metric": "ndcg_at_5",
        "tier": "deterministic",
    },
    {
        "id": "judge",
        "name": "LLM-as-judge",
        "description": "Claude grades how grounded recommendation/eligibility "
        "explanations are. Guardrails: temp 0, JSON output, reference-anchored.",
        "metric": "mean_groundedness",
        "tier": "llm",
    },
    {
        "id": "ragas",
        "name": "RAGAS (RAG quality)",
        "description": "Faithfulness, answer relevancy and correctness of the "
        "PDF Q&A pipeline against a gold answer set.",
        "metric": "faithfulness",
        "tier": "llm",
    },
    {
        "id": "risk",
        "name": "Late-payment risk",
        "description": "Discrimination (AUC/PR-AUC), calibration (Brier/ECE), "
        "confusion @0.40 and non-protected slices on a held-out synthetic set.",
        "metric": "auc",
        "tier": "deterministic",
    },
]


@router.get("/suites")
def suites() -> list:
    return SUITES


@router.post("/run")
def run() -> dict:
    """Run all suites (LLM tier auto-included when a key is configured)."""
    return run_evals.run()


@router.post("/run-deterministic")
def run_deterministic() -> dict:
    """Run only the fast, hermetic suites (no LLM calls)."""
    return run_evals.run(include_llm=False)


@router.get("/latest")
def latest() -> dict:
    return run_evals.load_latest()


@router.get("/history")
def history(limit: int = 30) -> list:
    return run_evals.load_history(limit)


@router.get("/report", response_class=HTMLResponse)
def report_html() -> HTMLResponse:
    """A standalone, printable HTML evaluation report from the latest run."""
    return HTMLResponse(report.build_html(run_evals.load_latest()))


@router.post("/judge")
def judge() -> dict:
    """Run the LLM-as-judge suite on demand."""
    return judges.run_judge_suite()


@router.post("/ragas")
def ragas() -> dict:
    """Run the RAGAS RAG-quality suite on demand (isolated subprocess)."""
    return rag_eval.run_isolated()


@router.post("/langsmith")
def langsmith() -> dict:
    """Push the eligibility suite to LangSmith as a tracked experiment."""
    return langsmith_experiments.run()


@router.post("/concierge/run")
def concierge_run(use_llm: bool = False) -> dict:
    """Run the Concierge suite: route accuracy, retrieval hit-rate,
    groundedness. Deterministic by default (``?use_llm=false``)."""
    result = concierge_eval.run(use_llm=use_llm)
    store.log_event(
        endpoint="evals_concierge_run",
        meta={
            "n": result.get("n"),
            "route_accuracy": result.get("route_accuracy"),
            "retrieval_hit_rate": result.get("retrieval_hit_rate"),
            "groundedness": result.get("groundedness"),
            "use_llm": use_llm,
        },
    )
    return result


@router.get("/concierge/latest")
def concierge_latest() -> dict:
    return concierge_eval.load_latest()


@router.post("/risk/run")
def risk_run() -> dict:
    """Run the late-payment risk suite on a held-out synthetic set."""
    result = risk_eval.run()
    store.log_event(
        endpoint="evals_risk_run",
        source=result.get("source"),
        meta={
            "n": result.get("n"),
            "auc": result.get("auc"),
            "brier": result.get("brier"),
            "ece": result.get("ece"),
        },
    )
    return result


@router.get("/risk/latest")
def risk_latest() -> dict:
    return risk_eval.load_latest()


@router.get("/ab/variants")
def ab_variants() -> list:
    """The prompt/model variants available to A/B test."""
    return ab.list_variants()


class ABRequest(BaseModel):
    a: str
    b: str


@router.post("/ab/run")
def ab_run(req: ABRequest) -> dict:
    """Compare two variants on the same data + judge and pick a winner."""
    return ab.compare(req.a, req.b)
