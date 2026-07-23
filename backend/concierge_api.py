"""HTTP API for the Property & Lease Concierge.

Mirrors the tours_api style: an ``APIRouter`` with ``log_event`` on each call.
``POST /concierge/ask`` ALWAYS returns 200 — the agent degrades server-side
(mirrors ``graphrag.graph_ask`` / ``tours_chat.handle``). The only typed error
is 404 when a caller passes an unknown ``property_id``.
"""

import json
import time

from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import concierge
import graph
import knowledge
import leases
import lease_pdf
import store

router = APIRouter(tags=["concierge"])


def _property_or_404(property_id: str) -> dict:
    for p in graph.load_properties():
        if p.get("id") == property_id:
            return p
    raise HTTPException(404, "Unknown property id.")


def _key_terms(p: dict) -> dict:
    """At-a-glance lease terms, straight from the structured data."""
    return {
        "rent": p.get("monthly_rent"),
        "deposit": p.get("security_deposit") or p.get("monthly_rent"),
        "term_months": p.get("lease_term_months"),
        "pets": bool(p.get("pets_allowed")),
        "parking": p.get("parking_type") or "none",
        "furnished": bool(p.get("furnished")),
    }


class ConciergeChatMessage(BaseModel):
    role: str
    content: str


class ConciergeAskRequest(BaseModel):
    question: str
    property_id: str | None = None
    history: list[ConciergeChatMessage] | None = None


def _property_exists(property_id: str) -> bool:
    return any(p.get("id") == property_id for p in graph.load_properties())


@router.post("/concierge/ask")
def ask(req: ConciergeAskRequest) -> dict:
    """Answer a property/lease question. ALWAYS 200 — degrades server-side.
    404 only when a given property_id is unknown."""
    t0 = time.perf_counter()

    if req.property_id and not _property_exists(req.property_id):
        raise HTTPException(404, "Unknown property id.")

    history = [m.model_dump() for m in (req.history or [])]
    result = concierge.answer(
        question=req.question,
        property_id=req.property_id,
        history=history,
    )

    store.log_event(
        endpoint="concierge_ask",
        latency_ms=(time.perf_counter() - t0) * 1000,
        source=result.get("source"),
        meta={
            "route": result.get("route"),
            "property_id": req.property_id or "",
            "sources": len(result.get("sources") or []),
        },
    )
    return result


@router.post("/concierge/ask/stream")
def ask_stream(req: ConciergeAskRequest) -> StreamingResponse:
    """Stream an answer as Server-Sent Events. Emits a ``meta`` frame (the
    deterministic retrieval pass) first, then ``token`` frames as the LLM prose
    streams, then a final ``done``. Degrades to a single deterministic token on
    any error — the generator never crashes the response. 404 only when a given
    property_id is unknown."""
    t0 = time.perf_counter()

    if req.property_id and not _property_exists(req.property_id):
        raise HTTPException(404, "Unknown property id.")

    history = [m.model_dump() for m in (req.history or [])]

    def _event_stream():
        route_ = ""
        source = "rules"
        n_sources = 0
        try:
            for event in concierge.answer_stream(
                question=req.question,
                property_id=req.property_id,
                history=history,
            ):
                etype = event.get("type")
                if etype == "meta":
                    route_ = event.get("route", "")
                    n_sources = len(event.get("sources") or [])
                elif etype == "done":
                    source = event.get("source", "rules")
                yield f"data: {json.dumps(event)}\n\n"
        except GeneratorExit:
            # Client disconnected mid-stream (navigated away, closed the tab).
            # Don't log latency here — time-to-teardown is not backend latency
            # and can be arbitrarily (and misleadingly) large.
            raise
        except Exception as exc:  # noqa: BLE001 — last-ditch guard
            print(f"concierge_api: stream failed ({type(exc).__name__}: {exc}).")
            fallback = {
                "type": "token",
                "text": "Sorry, I hit a snag. Please try again.",
            }
            yield f"data: {json.dumps(fallback)}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'source': 'rules'})}\n\n"
        store.log_event(
            endpoint="concierge_ask_stream",
            latency_ms=(time.perf_counter() - t0) * 1000,
            source=source,
            meta={
                "route": route_,
                "property_id": req.property_id or "",
                "sources": n_sources,
            },
        )

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/concierge/status")
def status() -> dict:
    """Ingestion status for the lease knowledge base."""
    indexed = knowledge.count()
    store.log_event(endpoint="concierge_status", meta={"indexed": indexed})
    return {"indexed": indexed, "collection": knowledge.COLLECTION}


@router.get("/concierge/lease/{property_id}")
def lease(property_id: str) -> dict:
    """The full generated lease for a property — the source the concierge cites.
    Powers the in-app lease viewer + clickable citations. 404 if unknown."""
    t0 = time.perf_counter()
    p = _property_or_404(property_id)
    sections = [
        {"section": title, "text": text} for title, text in leases.lease_sections(p)
    ]
    store.log_event(
        endpoint="concierge_lease",
        latency_ms=(time.perf_counter() - t0) * 1000,
        meta={"property_id": property_id, "sections": len(sections)},
    )
    return {
        "property_id": property_id,
        "property_name": p.get("name", ""),
        "sections": sections,
        "key_terms": _key_terms(p),
    }


@router.get("/concierge/lease/{property_id}/pdf")
def lease_pdf_route(property_id: str) -> Response:
    """The generated lease as a real PDF — same 18 sections as ``lease()``
    above, rendered for download/inline viewing. 404 if unknown."""
    t0 = time.perf_counter()
    p = _property_or_404(property_id)
    data = lease_pdf.render_lease_pdf(p, _key_terms(p))
    store.log_event(
        endpoint="concierge_lease_pdf",
        latency_ms=(time.perf_counter() - t0) * 1000,
        meta={"property_id": property_id, "bytes": len(data)},
    )
    return Response(
        content=data,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'inline; filename="{property_id}-lease.pdf"'
        },
    )
