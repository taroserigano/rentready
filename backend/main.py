"""RentReady FastAPI backend.

Ties together: PDF RAG (LlamaIndex), the Neo4j property graph, GraphRAG
recommendations (LangChain + Claude), eligibility rules, and tracing
(LangSmith + Phoenix).
"""

import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from settings import DATA_DIR, UPLOAD_DIR, settings

SAMPLES_DIR = DATA_DIR / "applications"
from observability import init_observability
from models import (
    AskRequest,
    AskResponse,
    EligibilityResult,
    RecommendResponse,
    UploadResponse,
)
import pdf_ingest
import rag_llamaindex
import eligibility
import graphrag
import graph
import store
import strength
import eval_api
import apply_api
import properties_api
import dashboard_api
import tours
import tours_api
import knowledge
import concierge_api
import risk
import risk_api
import residents_risk
import resident_api
import monitoring
from ratelimit import rate_limit_middleware
from evals import judges

app = FastAPI(title="RentReady API")

_cors_origins = [o.strip() for o in settings.cors_allowed_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins or ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.middleware("http")(rate_limit_middleware)

TRACING_STATUS: dict = {}

app.include_router(eval_api.router)
app.include_router(apply_api.router)
app.include_router(properties_api.router)
app.include_router(dashboard_api.router)
app.include_router(tours_api.router)
app.include_router(concierge_api.router)
app.include_router(risk_api.router)
app.include_router(resident_api.router)


@app.on_event("startup")
def _startup() -> None:
    global TRACING_STATUS
    TRACING_STATUS = init_observability()
    UPLOAD_DIR.mkdir(exist_ok=True)
    store.init_db()
    tours_result = tours.seed_tours()
    print(f"Tours seeded: {tours_result}")
    result = graph.seed_graph()
    print(f"Graph seeded: {result}")
    # Ingest lease docs into the "knowledge" collection. Best-effort so a
    # Chroma/embedding hiccup never blocks startup.
    try:
        k_result = knowledge.ingest_all()
        print(f"Knowledge ingested: {k_result}")
    except Exception as exc:  # noqa: BLE001
        print(f"Knowledge ingest failed ({type(exc).__name__}: {exc}); continuing.")
    # Warm the retrieval pipeline (LlamaIndex retriever + FlashRank ranker):
    # both lazily load on first use, which otherwise costs the FIRST real
    # concierge question several extra seconds. Pay that cost here instead.
    try:
        t0 = time.perf_counter()
        knowledge.search("warm up the retrieval pipeline", k=1)
        print(f"Knowledge search warmed up ({(time.perf_counter() - t0) * 1000:.0f}ms).")
    except Exception as exc:  # noqa: BLE001
        print(f"Knowledge warm-up failed ({type(exc).__name__}: {exc}); continuing.")
    # Best-effort: train the risk model if its artifact is missing. Wrapped so a
    # training hiccup never blocks startup (scoring degrades to the heuristic).
    try:
        risk_result = risk.ensure_model()
        print(f"Risk model: {risk_result}")
    except Exception as exc:  # noqa: BLE001
        print(f"Risk model ensure failed ({type(exc).__name__}: {exc}); continuing.")
    # Best-effort: train the resident-risk bundle if its artifact is missing.
    # Wrapped so a training hiccup never blocks startup (scoring degrades to the
    # transparent heuristic).
    try:
        residents_result = residents_risk.ensure_model()
        print(f"Residents model: {residents_result}")
    except Exception as exc:  # noqa: BLE001
        print(f"Residents model ensure failed ({type(exc).__name__}: {exc}); continuing.")


def _process_pdf(pdf_bytes: bytes) -> UploadResponse:
    """Shared pipeline: save the PDF under the applicant id, extract text,
    index it, extract the profile. Saving here (rather than in the caller)
    guarantees the stored PDF is keyed by the SAME applicant id we return, so
    it can be served back at /applicants/{id}/pdf."""
    if not pdf_bytes.startswith(b"%PDF-"):
        raise HTTPException(400, "That file doesn't look like a valid PDF.")

    applicant_id = uuid.uuid4().hex[:12]
    UPLOAD_DIR.mkdir(exist_ok=True)
    dest = UPLOAD_DIR / f"{applicant_id}.pdf"
    dest.write_bytes(pdf_bytes)

    try:
        text = pdf_ingest.extract_text(str(dest))
    except Exception as exc:  # noqa: BLE001 - both PDF parsers exhausted
        dest.unlink(missing_ok=True)
        print(f"PDF extraction failed ({type(exc).__name__}: {exc}).")
        raise HTTPException(422, "Could not read that PDF — it may be corrupted.")
    if not text:
        dest.unlink(missing_ok=True)
        raise HTTPException(422, "Could not read any text from that PDF.")

    chunks = rag_llamaindex.ingest(applicant_id, text)
    profile = rag_llamaindex.extract_profile(text)
    store.save_applicant(applicant_id, profile, chunks)
    return UploadResponse(
        applicant_id=applicant_id, profile=profile, chunks_indexed=chunks, has_pdf=True
    )


@app.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)) -> UploadResponse:
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Please upload a PDF file.")
    # Bounded read (not `await file.read()`): reading one extra byte past the
    # cap tells us the real file is too large without ever buffering it fully
    # into memory.
    max_bytes = settings.max_upload_mb * 1024 * 1024
    pdf_bytes = await file.read(max_bytes + 1)
    if len(pdf_bytes) > max_bytes:
        raise HTTPException(
            413, f"PDF exceeds the {settings.max_upload_mb}MB upload limit."
        )
    return _process_pdf(pdf_bytes)


@app.get("/samples")
def list_samples() -> list[dict]:
    """List the bundled sample application PDFs."""
    if not SAMPLES_DIR.exists():
        return []
    samples = []
    for pdf in sorted(SAMPLES_DIR.glob("*.pdf")):
        name = pdf.stem.replace("-", " ").title()
        samples.append({"slug": pdf.stem, "name": name})
    return samples


@app.post("/samples/{slug}", response_model=UploadResponse)
def load_sample(slug: str) -> UploadResponse:
    """Process a bundled sample PDF by name (no file upload needed)."""
    # Guard against path traversal: only allow known files in SAMPLES_DIR.
    pdf_path = (SAMPLES_DIR / f"{slug}.pdf").resolve()
    if SAMPLES_DIR.resolve() not in pdf_path.parents or not pdf_path.exists():
        raise HTTPException(404, "Unknown sample.")
    return _process_pdf(pdf_path.read_bytes())


def _get_profile(applicant_id: str):
    profile = store.get_profile(applicant_id)
    if profile is None:
        raise HTTPException(404, "Unknown applicant id. Upload a PDF first.")
    return profile


@app.get("/applicants")
def list_applicants() -> list:
    rows = store.list_applicants()
    statuses = store.latest_statuses()
    for r in rows:
        r["status"] = statuses.get(r["id"], "new")
    return rows


@app.get("/applicants/{applicant_id}/strength")
def get_strength(applicant_id: str) -> dict:
    """Advisory 0-100 tenant-quality score (F4). Does not affect the verdict."""
    return strength.compute(_get_profile(applicant_id))


class DecisionRequest(BaseModel):
    action: str  # approve | decline | waitlist | request_info
    note: str = ""
    reviewer: str = ""


@app.post("/applicants/{applicant_id}/decision")
def post_decision(applicant_id: str, req: DecisionRequest) -> dict:
    """Record a reviewer decision (F6). Additive audit trail."""
    _get_profile(applicant_id)
    store.save_decision(applicant_id, req.action, req.note or None, req.reviewer or None)
    store.log_event(endpoint="decision", applicant_id=applicant_id,
                    meta={"action": req.action})
    return {"ok": True, "status": req.action}


@app.get("/applicants/{applicant_id}/decisions")
def get_decisions(applicant_id: str) -> dict:
    _get_profile(applicant_id)
    return {"applicant_id": applicant_id, "decisions": store.decisions_for(applicant_id)}


@app.get("/applicants/{applicant_id}")
def get_applicant(applicant_id: str) -> dict:
    profile = _get_profile(applicant_id)
    return {"applicant_id": applicant_id, "profile": profile.model_dump()}


@app.get("/applicants/{applicant_id}/pdf")
def get_applicant_pdf(applicant_id: str):
    """Serve the application PDF (uploaded, sample, or generated on apply),
    inline so it can be embedded in the Workspace viewer."""
    _get_profile(applicant_id)  # 404s for unknown ids (also guards the path)
    path = UPLOAD_DIR / f"{applicant_id}.pdf"
    if not path.exists():
        raise HTTPException(404, "No PDF on file for this applicant.")
    return FileResponse(
        str(path),
        media_type="application/pdf",
        content_disposition_type="inline",
        filename=f"application-{applicant_id}.pdf",
    )


@app.delete("/applicants/{applicant_id}")
def delete_applicant(applicant_id: str) -> dict:
    if not store.delete_applicant(applicant_id):
        raise HTTPException(404, "Unknown applicant id.")
    # Best-effort cleanup of what the DB row doesn't own: the uploaded PDF on
    # disk and this applicant's chunks in the RAG index. Neither failing
    # should undo the delete that already succeeded above.
    try:
        path = UPLOAD_DIR / f"{applicant_id}.pdf"
        if path.exists():
            path.unlink()
    except OSError as exc:
        print(f"delete_applicant: couldn't remove PDF for {applicant_id} ({exc}).")
    rag_llamaindex.delete_applicant(applicant_id)
    return {"deleted": applicant_id}


@app.get("/eligibility/{applicant_id}", response_model=EligibilityResult)
def get_eligibility(applicant_id: str) -> EligibilityResult:
    profile = _get_profile(applicant_id)
    t0 = time.perf_counter()
    result = eligibility.evaluate(profile)
    store.log_event(
        endpoint="eligibility",
        applicant_id=applicant_id,
        latency_ms=(time.perf_counter() - t0) * 1000,
        meta={"verdict": result.verdict},
    )
    return result


@app.get("/recommend/{applicant_id}", response_model=RecommendResponse)
def get_recommend(applicant_id: str) -> RecommendResponse:
    t0 = time.perf_counter()
    profile = _get_profile(applicant_id)
    try:
        result = graphrag.recommend(profile)
        # Cheap online quality signal: deterministic hallucination tripwire
        # over the explanations we just served.
        violations = sum(
            len(judges.faithfulness_violations(r.match_reason, r.amenities))
            for r in result["recommendations"]
        )
    except Exception as exc:  # noqa: BLE001
        # Log the failure too — a bare 500 that skips log_event is invisible
        # to the monitoring dashboard, which otherwise reads an outage as
        # silence instead of a spike.
        store.log_event(
            endpoint="recommend",
            applicant_id=applicant_id,
            latency_ms=(time.perf_counter() - t0) * 1000,
            source="error",
            meta={"error": f"{type(exc).__name__}: {exc}"},
        )
        raise
    store.log_event(
        endpoint="recommend",
        applicant_id=applicant_id,
        latency_ms=(time.perf_counter() - t0) * 1000,
        source=result.get("source"),
        faithfulness_violations=violations,
        meta={"count": len(result["recommendations"])},
    )
    return RecommendResponse(**result)


class SimulateRequest(BaseModel):
    applicant_id: str
    monthly_income: float | None = None
    desired_rent: float | None = None
    credit_score: int | None = None


@app.post("/simulate")
def simulate(req: SimulateRequest) -> dict:
    """What-if: re-run eligibility + ranking on a modified copy of the profile.

    A scratch computation — nothing is persisted or logged. The LLM
    explanation pass is skipped (``explain=False``) so it stays fast and
    fully deterministic, which is exactly what the slider UI needs.
    """
    profile = _get_profile(req.applicant_id)
    overrides = {
        k: v
        for k, v in {
            "monthly_income": req.monthly_income,
            "desired_rent": req.desired_rent,
            "credit_score": req.credit_score,
        }.items()
        if v is not None
    }
    modified = profile.model_copy(update=overrides)
    eligibility_result = eligibility.evaluate(modified, explain=False)
    rec_result = graphrag.recommend(modified, explain=False)
    return {
        "eligibility": eligibility_result,
        "recommendations": RecommendResponse(**rec_result),
    }


class GoalSeekRequest(BaseModel):
    applicant_id: str
    solve_for: str = "monthly_income"  # "monthly_income" | "desired_rent"


@app.post("/simulate/goal-seek")
def goal_seek(req: GoalSeekRequest) -> dict:
    """Solve for the exact income/rent threshold that reaches 'qualified' (F10).
    Bisection over the deterministic rule (monotonic in income & rent)."""
    profile = _get_profile(req.applicant_id)

    def qualifies(income: float, rent: float) -> bool:
        p = profile.model_copy(update={"monthly_income": income, "desired_rent": rent})
        return eligibility.evaluate(p, explain=False).verdict == "qualified"

    field = req.solve_for
    cur_income = profile.monthly_income
    cur_rent = profile.desired_rent

    if field == "monthly_income":
        cap = max(cur_rent * 10, 50000)
        current = cur_income
        if not qualifies(cap, cur_rent):
            reason = eligibility.evaluate(
                profile.model_copy(update={"monthly_income": cap}), explain=False
            )
            return {"solve_for": field, "current": current, "achievable": False,
                    "reason": "Income alone can't reach 'qualified' — " + reason.reasons[-1]}
        lo, hi = 0.0, cap
        for _ in range(40):
            mid = (lo + hi) / 2
            if qualifies(mid, cur_rent):
                hi = mid
            else:
                lo = mid
        threshold = round(hi / 50) * 50
        return {"solve_for": field, "current": current, "threshold": threshold,
                "achievable": True,
                "delta": round(threshold - current)}
    else:  # desired_rent — find the MAX rent that still qualifies
        current = cur_rent
        if not qualifies(cur_income, 100):
            reason = eligibility.evaluate(
                profile.model_copy(update={"desired_rent": 100}), explain=False
            )
            return {"solve_for": field, "current": current, "achievable": False,
                    "reason": "Lowering rent can't reach 'qualified' — " + reason.reasons[-1]}
        lo, hi = 100.0, max(cur_income, cur_rent * 2, 10000)
        for _ in range(40):
            mid = (lo + hi) / 2
            if qualifies(cur_income, mid):
                lo = mid
            else:
                hi = mid
        threshold = round(lo / 10) * 10
        return {"solve_for": field, "current": current, "threshold": threshold,
                "achievable": True, "delta": round(threshold - current)}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    _get_profile(req.applicant_id)
    t0 = time.perf_counter()
    try:
        result = rag_llamaindex.query(req.applicant_id, req.question)
    except Exception as exc:  # noqa: BLE001 — rag_llamaindex.query() itself
        # never raises now, but log any failure that somehow still reaches
        # here rather than let it skip log_event invisibly.
        store.log_event(
            endpoint="ask",
            applicant_id=req.applicant_id,
            latency_ms=(time.perf_counter() - t0) * 1000,
            source="error",
            meta={"error": f"{type(exc).__name__}: {exc}"},
        )
        raise
    store.log_event(
        endpoint="ask",
        applicant_id=req.applicant_id,
        latency_ms=(time.perf_counter() - t0) * 1000,
        source=result.get("source"),
    )
    return AskResponse(**result)


class FeedbackRequest(BaseModel):
    applicant_id: str = ""
    target: str  # "recommendation" | "eligibility" | "answer"
    rating: str  # "up" | "down"
    item_id: str = ""
    comment: str = ""


@app.post("/feedback")
def submit_feedback(req: FeedbackRequest) -> dict:
    if req.rating not in ("up", "down"):
        raise HTTPException(400, "rating must be 'up' or 'down'.")
    store.save_feedback(
        applicant_id=req.applicant_id,
        target=req.target,
        rating=req.rating,
        item_id=req.item_id or None,
        comment=req.comment or None,
    )
    return {"ok": True}


@app.get("/monitoring/overview")
def monitoring_overview() -> dict:
    return monitoring.snapshot()


@app.post("/monitoring/push-datadog")
def monitoring_push_datadog() -> dict:
    return monitoring.push_to_datadog()


class GraphAskRequest(BaseModel):
    question: str


@app.post("/graph-ask")
def graph_ask(req: GraphAskRequest) -> dict:
    """Natural-language question answered over the Neo4j property graph."""
    return graphrag.graph_ask(req.question)


@app.post("/seed-graph")
def seed() -> dict:
    return graph.seed_graph()


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "anthropic_key_set": settings.has_anthropic,
        "langsmith": TRACING_STATUS.get("langsmith", False),
        "phoenix": TRACING_STATUS.get("phoenix", False),
        "neo4j_available": graph.is_available(),
        "datadog": settings.has_datadog,
        "applicants_loaded": len(store.list_applicants()),
        "tours_loaded": len(store.list_agents()),
        "knowledge_indexed": knowledge.count(),
        "risk_model": risk.status(),
        "residents_model": residents_risk.status(),
    }
