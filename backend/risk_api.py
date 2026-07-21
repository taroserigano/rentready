"""HTTP API for Resident Late-Payment Risk (decision-support).

Mirrors the concierge/tours style: an ``APIRouter`` with ``log_event`` on each
call. Scoring NEVER 500s — ``risk.predict`` degrades to a transparent heuristic
server-side. The only typed error is 404 for an unknown applicant id.
"""

import json
import time

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

import risk
import risk_chat
import store
from models import RiskChatRequest, RiskChatResponse, RiskScoreRequest

router = APIRouter(tags=["risk"])


def _profile_or_404(applicant_id: str):
    profile = store.get_profile(applicant_id)
    if profile is None:
        raise HTTPException(404, "Unknown applicant id. Upload a PDF first.")
    return profile


@router.get("/risk")
def list_risk() -> dict:
    """Score every applicant (RiskListResponse). Per-row try/except so one bad
    row never fails the batch. Rows sorted by probability, highest first."""
    t0 = time.perf_counter()
    rows = []
    applicants = store.list_applicants()
    for a in applicants:
        try:
            profile = store.get_profile(a["id"])
            if profile is None:
                continue
            result = risk.predict(profile, applicant_id=a["id"], name=a.get("name", ""))
            rows.append(
                {
                    "applicant_id": a["id"],
                    "name": result.get("name") or a.get("name", ""),
                    "probability": result["probability"],
                    "band": result["band"],
                    "top_driver": risk.top_driver(result),
                }
            )
        except Exception as exc:  # noqa: BLE001 — skip a bad row, never fail the batch
            print(f"risk_api: scoring {a.get('id')} failed ({type(exc).__name__}: {exc}).")

    rows.sort(key=lambda r: r["probability"], reverse=True)
    scored = len(rows)
    avg = round(sum(r["probability"] for r in rows) / scored, 4) if scored else 0.0
    high_pct = (
        round(100.0 * sum(1 for r in rows if r["band"] == "high") / scored, 1) if scored else 0.0
    )
    store.log_event(
        endpoint="risk_list",
        latency_ms=(time.perf_counter() - t0) * 1000,
        meta={"total": len(applicants), "scored": scored, "high_risk_pct": high_pct},
    )
    return {
        "rows": rows,
        "total": len(applicants),
        "scored": scored,
        "avg_probability": avg,
        "high_risk_pct": high_pct,
    }


@router.get("/risk/model-card")
def model_card() -> dict:
    """The model card: intended use, features used/excluded, metrics, limits."""
    card = risk.model_card()
    store.log_event(endpoint="risk_model_card", source=card.get("source"))
    return card


@router.post("/risk/score")
def score(req: RiskScoreRequest) -> dict:
    """Score an ad-hoc profile (not persisted). Always 200 — degrades server-side."""
    t0 = time.perf_counter()
    result = risk.predict(req.profile, name=req.profile.name)
    store.log_event(
        endpoint="risk_score",
        latency_ms=(time.perf_counter() - t0) * 1000,
        source=result.get("source"),
        meta={"band": result.get("band"), "confidence": result.get("confidence")},
    )
    return result


@router.post("/risk/chat")
def chat(req: RiskChatRequest) -> RiskChatResponse:
    """Risk decision-support chat. ALWAYS 200 — ``risk_chat.answer`` degrades
    server-side and never raises. An unknown ``applicant_id`` is NOT a 404 here:
    the agent deflects with a "select an applicant" reply rather than erroring
    (chat should degrade, not break). Declared before ``/risk/{applicant_id}``
    so the literal path is not swallowed by that catch-all."""
    t0 = time.perf_counter()
    history = [m.model_dump() for m in (req.history or [])]
    result = risk_chat.answer(
        question=req.question,
        applicant_id=req.applicant_id,
        history=history,
    )
    store.log_event(
        endpoint="risk_chat",
        applicant_id=req.applicant_id or None,
        latency_ms=(time.perf_counter() - t0) * 1000,
        source=result.get("source"),
        meta={
            "intent": result.get("intent"),
            "scope": result.get("scope"),
            "applicant_id": req.applicant_id or "",
            "artifact": (result.get("artifact") or {}).get("kind"),
            "sources": len(result.get("sources") or []),
        },
    )
    return result


@router.post("/risk/chat/stream")
def chat_stream(req: RiskChatRequest) -> StreamingResponse:
    """Stream a risk decision-support answer as Server-Sent Events. Emits a
    ``meta`` frame (the deterministic ``_RiskPlan`` pass — scope, intent,
    follow-ups and the artifact) first, then ``token`` frames as the LLM prose
    streams, then a final ``done``. Degrades to a single deterministic token on
    any error — the generator never crashes the response. An unknown
    ``applicant_id`` is NOT a 404: the agent deflects gracefully. Declared before
    ``/risk/{applicant_id}`` so the literal path is not swallowed by the
    catch-all."""
    t0 = time.perf_counter()
    history = [m.model_dump() for m in (req.history or [])]

    def _event_stream():
        intent = ""
        source = "rules"
        artifact_kind = ""
        try:
            for event in risk_chat.answer_stream(
                question=req.question,
                applicant_id=req.applicant_id,
                history=history,
            ):
                etype = event.get("type")
                if etype == "meta":
                    intent = event.get("intent", "")
                    artifact_kind = (event.get("artifact") or {}).get("kind", "")
                elif etype == "done":
                    source = event.get("source", "rules")
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as exc:  # noqa: BLE001 — last-ditch guard
            print(f"risk_api: stream failed ({type(exc).__name__}: {exc}).")
            fallback = {
                "type": "token",
                "text": "Sorry, I hit a snag. Please try again.",
            }
            yield f"data: {json.dumps(fallback)}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'source': 'rules'})}\n\n"
        finally:
            store.log_event(
                endpoint="risk_chat_stream",
                applicant_id=req.applicant_id or None,
                latency_ms=(time.perf_counter() - t0) * 1000,
                source=source,
                meta={
                    "intent": intent,
                    "scope": "applicant" if req.applicant_id else "portfolio",
                    "applicant_id": req.applicant_id or "",
                    "artifact": artifact_kind,
                },
            )

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/risk/{applicant_id}")
def get_risk(applicant_id: str) -> dict:
    """Score one stored applicant. 404 if the id is unknown."""
    t0 = time.perf_counter()
    profile = _profile_or_404(applicant_id)
    result = risk.predict(profile, applicant_id=applicant_id, name=profile.name)
    store.log_event(
        endpoint="risk_get",
        applicant_id=applicant_id,
        latency_ms=(time.perf_counter() - t0) * 1000,
        source=result.get("source"),
        meta={"band": result.get("band"), "confidence": result.get("confidence")},
    )
    return result
