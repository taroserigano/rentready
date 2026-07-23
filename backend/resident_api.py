"""HTTP API for Resident Risk (decision-support; synthetic-data model).

Mirrors the applicant-risk API style: an ``APIRouter`` with ``store.log_event``
on every call, per-row ``try/except`` so one bad resident never fails a batch,
and graceful degradation. Scoring NEVER 500s — ``residents_risk.predict_resident``
degrades to a transparent heuristic server-side and never raises. The only typed
error is 404 for a genuinely unknown resident id on the detail/score routes.

Literal paths (``/residents/model-card``, ``/residents/portfolio/summary``) are
declared BEFORE the ``/residents/{resident_id}`` catch-all so they are not
swallowed by it.

DECISION-SUPPORT ONLY: proactive outreach and retention. Never eviction,
denial, pricing, lease conditioning, or automated action.
"""

import json
import time
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

import graph
import residents_chat
import residents_risk
import store
from models import (
    ArrearsPrediction,
    BandDistribution,
    ChurnPrediction,
    LatePrediction,
    LedgerStats,
    PortfolioHealthResponse,
    PortfolioSummary,
    PropertyHealth,
    PropertyResidentRollup,
    PropertyResidentsResponse,
    Resident,
    ResidentChatRequest,
    ResidentChatResponse,
    ResidentDetail,
    ResidentListResponse,
    ResidentModelCard,
    ResidentPredictions,
    ResidentPropertiesResponse,
    ResidentPropertyOption,
    ResidentRow,
    ResidentRollup,
    SeriousPrediction,
)

router = APIRouter(tags=["residents"])


# ---------------------------------------------------------------------------
# Internal helpers — score one resident into a table row + an aggregate record.
# ---------------------------------------------------------------------------
def _resident_or_404(resident_id: str) -> dict:
    r = residents_risk.get_resident(resident_id)
    if r is None:
        raise HTTPException(404, "Unknown resident id.")
    return r


def _score_row(resident: dict, pred: dict | None = None) -> tuple[ResidentRow, dict]:
    """Score one resident and build (table row, aggregate record). The aggregate
    record carries just the numbers the rollups need. ``pred`` may be a
    precomputed (fast, no-reasons) prediction from the batched bulk scorer."""
    if pred is None:
        # Fast bulk path: only the heads the row/rollup need, no TreeSHAP reasons.
        pred = residents_risk.predict_resident(
            resident, with_reasons=False, heads=residents_risk.BULK_HEADS
        )
    feats = residents_risk.extract_resident_features(resident)

    late = pred.get("late") or {}
    arrears = pred.get("arrears") or {}
    churn = pred.get("churn") or {}
    serious = pred.get("serious") or {}

    churn_prob = churn.get("probability")
    churn_band = churn.get("band", "not_applicable")

    row = ResidentRow(
        resident_id=resident.get("resident_id", ""),
        property_id=resident.get("property_id", ""),
        unit_id=resident.get("unit_id", ""),
        name=resident.get("name", ""),
        base_rent=float(resident.get("base_rent") or 0.0),
        tenure_months=int(round(float(feats.get("tenure_months", 0.0)))),
        late_probability=float(late.get("probability") or 0.0),
        late_band=late.get("band", "low"),
        expected_arrears=float(arrears.get("expected_balance") or 0.0),
        churn_probability=churn_prob,
        churn_status=churn_band,
        serious_probability=float(serious.get("probability") or 0.0),
        serious_band=serious.get("band", "low"),
        current_balance=round(float(feats.get("current_balance", 0.0)), 2),
        top_driver=residents_risk.heuristic_top_driver(feats),
    )

    agg = {
        "property_id": row.property_id,
        "late_prob": row.late_probability,
        "late_band": row.late_band,
        "expected_arrears": row.expected_arrears,
        "churn_band": churn_band,
        "churn_eligible": churn_band != "not_applicable",
        "serious_prob": row.serious_probability,
        "serious_band": row.serious_band,
    }
    return row, agg


def _score_many(residents: list) -> tuple[list, list, str]:
    """Score a list of residents; return (rows, aggregates, source). Per-resident
    guarded so one bad record never sinks the batch."""
    rows: list = []
    aggs: list = []
    source = "heuristic"
    # One VECTORIZED model pass over the whole batch (one call per head, not one
    # per resident) — the fast path. Reason codes are skipped for bulk.
    try:
        preds = residents_risk.predict_bulk(
            residents or [], heads=residents_risk.BULK_HEADS
        )
    except Exception:  # noqa: BLE001 — fall back to per-resident inside the loop
        preds = [None] * len(residents or [])
    for r, pred in zip(residents or [], preds):
        try:
            row, agg = _score_row(r, pred)
            rows.append(row)
            aggs.append(agg)
        except Exception as exc:  # noqa: BLE001 — skip a bad row, never fail the batch
            print(f"resident_api: scoring {r.get('resident_id')} failed ({type(exc).__name__}: {exc}).")
    # One status read tells us whether the trained bundle is in play.
    try:
        source = "model" if residents_risk.status().get("trained") else "heuristic"
    except Exception:  # noqa: BLE001
        source = "heuristic"
    return rows, aggs, source


def _rollup(aggs: list) -> dict:
    """Aggregate a list of per-resident records into a rollup dict."""
    n = len(aggs)
    late_bands = BandDistribution()
    serious_bands = BandDistribution()
    churn_bands = BandDistribution()
    late_sum = 0.0
    serious_sum = 0.0
    arrears_sum = 0.0
    churn_eligible = 0
    churn_risk = 0
    serious_flag = 0

    for a in aggs:
        late_sum += a["late_prob"]
        serious_sum += a["serious_prob"]
        arrears_sum += a["expected_arrears"]
        setattr(late_bands, a["late_band"], getattr(late_bands, a["late_band"], 0) + 1)
        setattr(serious_bands, a["serious_band"], getattr(serious_bands, a["serious_band"], 0) + 1)
        cb = a["churn_band"]
        setattr(churn_bands, cb, getattr(churn_bands, cb, 0) + 1)
        if a["churn_eligible"]:
            churn_eligible += 1
            if cb == "high":
                churn_risk += 1
        if a["serious_band"] == "high":
            serious_flag += 1

    return {
        "resident_count": n,
        "predicted_late_rate": round(late_sum / n, 4) if n else 0.0,
        "total_expected_arrears": round(arrears_sum, 2),
        "avg_serious_probability": round(serious_sum / n, 4) if n else 0.0,
        "churn_eligible_count": churn_eligible,
        "churn_risk_count": churn_risk,
        "serious_flag_count": serious_flag,
        "late_bands": late_bands,
        "serious_bands": serious_bands,
        "churn_bands": churn_bands,
    }


def _ledger_stats(resident: dict) -> LedgerStats:
    """Derive the read-time ledger statistics via the same feature extraction the
    models use (no train/serve skew), plus a couple of lifetime totals."""
    feats = residents_risk.extract_resident_features(resident)
    ledger = list(resident.get("ledger") or [])
    lifetime_paid = round(sum(float(e.get("amount_paid", 0.0)) for e in ledger), 2)
    lifetime_late_fees = round(sum(float(e.get("late_fee", 0.0)) for e in ledger), 2)
    return LedgerStats(
        ledger_months=len(ledger),
        tenure_months=int(round(float(feats.get("tenure_months", 0.0)))),
        current_balance=round(float(feats.get("current_balance", 0.0)), 2),
        current_balance_ratio=float(feats.get("current_balance_ratio", 0.0)),
        balance_trend_6mo=float(feats.get("balance_trend_6mo", 0.0)),
        times_late_3mo=int(feats.get("late_count_3mo", 0.0)),
        times_late_6mo=int(feats.get("late_count_6mo", 0.0)),
        times_late_12mo=int(feats.get("late_count_12mo", 0.0)),
        times_late_24mo=int(feats.get("late_count_24mo", 0.0)),
        missed_count_12mo=int(feats.get("missed_count_12mo", 0.0)),
        partial_count_12mo=int(feats.get("partial_count_12mo", 0.0)),
        on_time_streak_months=int(feats.get("on_time_streak_months", 0.0)),
        months_since_last_late=int(feats.get("months_since_last_late", 0.0)),
        max_days_late_12mo=int(feats.get("max_days_late_12mo", 0.0)),
        avg_days_late_12mo=round(float(feats.get("avg_days_late_12mo", 0.0)), 2),
        late_fees_12mo=round(float(feats.get("late_fees_12mo", 0.0)), 2),
        notice_response_rate=float(feats.get("notice_response_rate", 0.0)),
        rent_to_income=float(feats.get("rent_to_income", 0.0)),
        autopay_enrolled=bool(resident.get("autopay_enrolled")),
        income_verified=bool(resident.get("income_verified")),
        lifetime_paid=lifetime_paid,
        lifetime_late_fees=lifetime_late_fees,
    )


# ---------------------------------------------------------------------------
# Routes — literal paths first, then the /{resident_id} catch-all.
# ---------------------------------------------------------------------------
@router.get("/residents")
def list_residents(property_id: str | None = None) -> ResidentListResponse:
    """Portfolio residents table. Optional ``?property_id`` filter. Per-row
    guarded; always 200 (unknown property just yields an empty list)."""
    t0 = time.perf_counter()
    residents = residents_risk.load_residents()
    if property_id:
        residents = [r for r in residents if r.get("property_id") == property_id]
    rows, _aggs, source = _score_many(residents)
    rows.sort(key=lambda r: r.late_probability, reverse=True)
    store.log_event(
        endpoint="residents_list",
        latency_ms=(time.perf_counter() - t0) * 1000,
        source=source,
        meta={"count": len(rows), "property_id": property_id or ""},
    )
    return ResidentListResponse(
        residents=rows, count=len(rows), property_id=property_id, source=source
    )


@router.get("/residents/properties")
def resident_properties() -> ResidentPropertiesResponse:
    """Cheap property picker for the Residents page: the properties that actually
    have residents, with a display name and headcount — NO scoring. The UI loads
    residents only when a property is chosen, never the whole portfolio."""
    t0 = time.perf_counter()
    counts: dict[str, int] = {}
    for r in residents_risk.load_residents():
        pid = r.get("property_id") or ""
        if pid:
            counts[pid] = counts.get(pid, 0) + 1

    names: dict[str, str] = {}
    try:
        for p in graph.load_properties():
            pid = p.get("id")
            if pid:
                names[pid] = p.get("name") or pid
    except Exception:  # noqa: BLE001 — names are cosmetic; fall back to the id
        pass

    # Preserve the canonical property order, then any stragglers.
    ordered = [p for p in residents_risk.RESIDENT_PROPERTY_IDS if p in counts]
    ordered += [p for p in counts if p not in ordered]
    options = [
        ResidentPropertyOption(
            property_id=p, name=names.get(p, p), resident_count=counts[p]
        )
        for p in ordered
    ]
    store.log_event(
        endpoint="residents_properties",
        latency_ms=(time.perf_counter() - t0) * 1000,
        meta={"properties": len(options)},
    )
    return ResidentPropertiesResponse(properties=options, count=len(options))


@router.get("/residents/model-card")
def resident_model_card() -> ResidentModelCard:
    """The resident-risk model card: intended use, features per target,
    excluded protected-class proxies, metrics, limitations."""
    card = residents_risk.model_card()
    store.log_event(endpoint="residents_model_card", source=card.get("source"))
    return ResidentModelCard(**card)


@router.get("/residents/portfolio/summary")
def portfolio_summary() -> PortfolioSummary:
    """Per-property + overall rollups: predicted late-rate next quarter, total
    expected arrears, churn-risk and serious-flag counts, and band distributions."""
    t0 = time.perf_counter()
    residents = residents_risk.load_residents()
    rows, aggs, source = _score_many(residents)

    # Bucket aggregates by property (preserve the canonical property order).
    by_prop: dict = {}
    for a in aggs:
        by_prop.setdefault(a["property_id"], []).append(a)

    ordered = [p for p in residents_risk.RESIDENT_PROPERTY_IDS if p in by_prop]
    ordered += [p for p in by_prop if p not in ordered]

    properties = [
        PropertyResidentRollup(property_id=p, **_rollup(by_prop[p])) for p in ordered
    ]
    overall = ResidentRollup(**_rollup(aggs))

    store.log_event(
        endpoint="residents_portfolio_summary",
        latency_ms=(time.perf_counter() - t0) * 1000,
        source=source,
        meta={
            "properties": len(properties),
            "residents": overall.resident_count,
            "predicted_late_rate": overall.predicted_late_rate,
            "total_expected_arrears": overall.total_expected_arrears,
            "churn_risk": overall.churn_risk_count,
            "serious_flags": overall.serious_flag_count,
        },
    )
    return PortfolioSummary(
        properties=properties,
        overall=overall,
        property_count=len(properties),
        resident_count=overall.resident_count,
        snapshot_date=residents_risk.RESIDENT_SNAPSHOT.isoformat(),
        source=source,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


@router.get("/properties/{property_id}/residents")
def property_residents(property_id: str) -> PropertyResidentsResponse:
    """Residents for one property + that property's rollup. Always 200 — an
    unknown property just yields an empty list and a zeroed rollup."""
    t0 = time.perf_counter()
    residents = [
        r for r in residents_risk.load_residents() if r.get("property_id") == property_id
    ]
    rows, aggs, source = _score_many(residents)
    rows.sort(key=lambda r: r.late_probability, reverse=True)
    rollup = PropertyResidentRollup(property_id=property_id, **_rollup(aggs))
    store.log_event(
        endpoint="residents_by_property",
        latency_ms=(time.perf_counter() - t0) * 1000,
        source=source,
        meta={"property_id": property_id, "count": len(rows)},
    )
    return PropertyResidentsResponse(
        property_id=property_id,
        residents=rows,
        count=len(rows),
        rollup=rollup,
        source=source,
    )


@router.get("/residents/health")
def residents_health() -> PortfolioHealthResponse:
    """Regional-director property-health ranking (healthiest first). Composite
    0-100 score + letter grade per property, from its residents' predictions,
    with the property display name attached and an explicit worst-property
    callout. Always 200 — health degrades server-side and never raises."""
    t0 = time.perf_counter()
    ranked = residents_risk.portfolio_health()

    # Attach the property display name (map property_id -> name), like
    # /residents/properties does. Names are cosmetic — fall back to the id.
    names: dict[str, str] = {}
    try:
        for p in graph.load_properties():
            pid = p.get("id")
            if pid:
                names[pid] = p.get("name") or pid
    except Exception:  # noqa: BLE001
        pass

    items = [
        PropertyHealth(**{**h, "name": names.get(h.get("property_id"), h.get("property_id") or "")})
        for h in ranked
    ]
    # portfolio_health already sorts best->worst; be explicit and defensive.
    items.sort(key=lambda h: h.score, reverse=True)

    try:
        source = "model" if residents_risk.status().get("trained") else "heuristic"
    except Exception:  # noqa: BLE001
        source = "heuristic"

    store.log_event(
        endpoint="residents_health",
        latency_ms=(time.perf_counter() - t0) * 1000,
        source=source,
        meta={
            "properties": len(items),
            "healthiest": items[0].property_id if items else "",
            "needs_attention": items[-1].property_id if items else "",
        },
    )
    return PortfolioHealthResponse(
        properties=items,
        count=len(items),
        healthiest=items[0] if items else None,
        needs_attention=items[-1] if items else None,
        snapshot_date=residents_risk.RESIDENT_SNAPSHOT.isoformat(),
        source=source,
    )


@router.post("/residents/chat")
def residents_chat_ask(req: ResidentChatRequest) -> ResidentChatResponse:
    """Residents decision-support chat. ALWAYS 200 — the agent degrades
    server-side and never raises. ``resident_id`` scopes to one resident;
    ``property_id`` (or no id) scopes to property / portfolio health. Unknown ids
    deflect gracefully (never a 404)."""
    t0 = time.perf_counter()
    history = [m.model_dump() for m in (req.history or [])]
    result = residents_chat.answer(
        question=req.question,
        resident_id=req.resident_id,
        property_id=req.property_id,
        history=history,
    )
    store.log_event(
        endpoint="residents_chat",
        latency_ms=(time.perf_counter() - t0) * 1000,
        source=result.get("source"),
        meta={
            "intent": result.get("intent"),
            "scope": result.get("scope"),
            "resident_id": req.resident_id or "",
            "property_id": req.property_id or "",
            "sources": len(result.get("sources") or []),
        },
    )
    return ResidentChatResponse(**result)


@router.post("/residents/chat/stream")
def residents_chat_stream(req: ResidentChatRequest) -> StreamingResponse:
    """Stream a residents-chat answer as Server-Sent Events: a ``meta`` frame
    (the deterministic pass — scope, intent, artifact, follow_ups) first, then
    ``token`` frames as the LLM prose streams, then a final ``done`` (source).
    Degrades to a single deterministic token on any error — the generator never
    crashes the response, and this endpoint never 404s."""
    t0 = time.perf_counter()
    history = [m.model_dump() for m in (req.history or [])]

    def _event_stream():
        intent = ""
        scope = ""
        source = "rules"
        try:
            for event in residents_chat.answer_stream(
                question=req.question,
                resident_id=req.resident_id,
                property_id=req.property_id,
                history=history,
            ):
                etype = event.get("type")
                if etype == "meta":
                    intent = event.get("intent", "")
                    scope = event.get("scope", "")
                elif etype == "done":
                    source = event.get("source", "rules")
                yield f"data: {json.dumps(event)}\n\n"
        except GeneratorExit:
            # Client disconnected mid-stream (navigated away, closed the tab).
            # Don't log latency here — time-to-teardown is not backend latency
            # and can be arbitrarily (and misleadingly) large.
            raise
        except Exception as exc:  # noqa: BLE001 — last-ditch guard
            print(f"resident_api: chat stream failed ({type(exc).__name__}: {exc}).")
            fallback = {"type": "token", "text": "Sorry, I hit a snag. Please try again."}
            yield f"data: {json.dumps(fallback)}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'source': 'rules'})}\n\n"
        store.log_event(
            endpoint="residents_chat_stream",
            latency_ms=(time.perf_counter() - t0) * 1000,
            source=source,
            meta={
                "intent": intent,
                "scope": scope,
                "resident_id": req.resident_id or "",
                "property_id": req.property_id or "",
            },
        )

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/residents/{resident_id}")
def get_resident(resident_id: str) -> ResidentDetail:
    """Full drill-down for one resident: committed record + four predictions +
    derived ledger stats. 404 only for a genuinely unknown resident id."""
    t0 = time.perf_counter()
    resident = _resident_or_404(resident_id)
    pred = residents_risk.predict_resident(resident)
    detail = ResidentDetail(
        resident=Resident(**resident),
        predictions=ResidentPredictions(**pred),
        ledger_stats=_ledger_stats(resident),
        source=(pred.get("late") or {}).get("source", "heuristic"),
    )
    store.log_event(
        endpoint="residents_get",
        latency_ms=(time.perf_counter() - t0) * 1000,
        source=detail.source,
        meta={
            "resident_id": resident_id,
            "late_band": detail.predictions.late.band,
            "serious_band": detail.predictions.serious.band,
            "churn_status": detail.predictions.churn.band,
        },
    )
    return detail


@router.post("/residents/{resident_id}/score")
def score_resident(resident_id: str) -> ResidentPredictions:
    """Re-score one resident on all four targets. 404 for an unknown id; always
    200 otherwise (scoring degrades server-side and never raises)."""
    t0 = time.perf_counter()
    resident = _resident_or_404(resident_id)
    pred = residents_risk.predict_resident(resident)
    result = ResidentPredictions(**pred)
    store.log_event(
        endpoint="residents_score",
        latency_ms=(time.perf_counter() - t0) * 1000,
        source=result.late.source,
        meta={
            "resident_id": resident_id,
            "late_band": result.late.band,
            "serious_band": result.serious.band,
        },
    )
    return result
