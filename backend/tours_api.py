"""HTTP API for the Tour Scheduler (F: book a property tour).

Wires the pure slot engine (``tours``), persistence (``store``), the property
inventory (``graph.load_properties``), and the conversational brain
(``tours_chat``). Nothing here may 500: the chat endpoint degrades server-side
(mirrors ``graphrag.graph_ask``); the CRUD endpoints raise typed HTTP errors
(404/409/422) only.
"""

import time
from datetime import date, datetime, timedelta
from urllib.parse import quote, urlencode

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

import graph
import store
import tours
import tours_chat
from models import (
    BookTourRequest,
    Slot,
    TourAgent,
    TourBooking,
    TourChatRequest,
    TourChatResponse,
)

router = APIRouter(tags=["tours"])

# Austin is US Central; tours are stored as naive local wall-clock. We pin the
# timezone on calendar exports (gcal `ctz` + ics `TZID`) so a booked 2:00 PM
# lands at 2:00 PM Central on the prospect's real calendar, wherever they are.
_TZID = "America/Chicago"


def _cal_stamp(iso_local: str) -> str:
    """"2026-07-21T14:00:00" -> "20260721T140000" (calendar basic format)."""
    return iso_local.replace("-", "").replace(":", "")


def _gcal_url(b: TourBooking) -> str:
    """A self-contained 'Add to Google Calendar' template link. Opening it in a
    browser creates a real event on the user's actual calendar — no server-side
    Google credentials needed."""
    params = {
        "action": "TEMPLATE",
        "text": f"Tour: {b.property_name}",
        "dates": f"{_cal_stamp(b.start)}/{_cal_stamp(b.end)}",
        "ctz": _TZID,
        "details": (
            f"In-person tour of {b.property_name} with {b.agent_name}.\n"
            f"Booked via RentReady."
        ),
        "location": b.property_name,
    }
    return "https://calendar.google.com/calendar/render?" + urlencode(params, quote_via=quote)


def _augment(b: TourBooking) -> TourBooking:
    """Attach the calendar link to a booking before returning it."""
    b.gcal_url = _gcal_url(b)
    return b


def _ics(b: TourBooking) -> str:
    """A minimal RFC-5545 VEVENT for the booking (importable by any calendar)."""
    dt = _cal_stamp(b.start)
    de = _cal_stamp(b.end)
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//RentReady//Tour Scheduler//EN",
        "CALSCALE:GREGORIAN",
        "METHOD:PUBLISH",
        "BEGIN:VEVENT",
        f"UID:{b.id}@rentready",
        f"DTSTAMP:{stamp}",
        f"DTSTART;TZID={_TZID}:{dt}",
        f"DTEND;TZID={_TZID}:{de}",
        f"SUMMARY:Tour: {b.property_name}",
        f"DESCRIPTION:In-person tour of {b.property_name} with {b.agent_name}. "
        f"Booked via RentReady.",
        f"LOCATION:{b.property_name}",
        "STATUS:CONFIRMED",
        "END:VEVENT",
        "END:VCALENDAR",
    ]
    return "\r\n".join(lines) + "\r\n"


def _property_or_404(property_id: str) -> dict:
    for p in graph.load_properties():
        if p.get("id") == property_id:
            return p
    raise HTTPException(404, "Unknown property id.")


def _area_of(prop: dict) -> str:
    # Historically the property's neighborhood name; now the property's own id
    # -- every property has exactly one dedicated agent (see tours._SEED_AGENTS)
    # so eligibility is keyed 1:1 to the property, never shared across a
    # neighborhood of several properties.
    return prop.get("id", "")


def _open_to_slot(o: tours.OpenSlot) -> Slot:
    return Slot(
        slot_id=o.slot_id,
        property_id=o.property_id,
        start=o.start.isoformat(timespec="seconds"),
        end=o.end.isoformat(timespec="seconds"),
        agent_id=o.agent_id,
        agent_name=o.agent_name,
        label=o.label,
    )


@router.get("/tours/staff")
def get_staff(property_id: str = Query(None)) -> dict:
    """All tour agents; if ``property_id`` is given, only the staff member
    dedicated to that property."""
    t0 = time.perf_counter()
    agents = store.list_agents()
    if property_id:
        prop = _property_or_404(property_id)
        area = _area_of(prop)
        agents = [a for a in agents if tours.agent_covers(a, area)]
    staff = [TourAgent(**a) for a in agents]
    store.log_event(
        endpoint="tours_staff",
        latency_ms=(time.perf_counter() - t0) * 1000,
        meta={"property_id": property_id, "total": len(staff)},
    )
    return {"staff": staff, "total": len(staff)}


@router.get("/tours/slots")
def get_slots(
    property_id: str = Query(...),
    date_from: str = Query(None),
    date_to: str = Query(None),
    time_of_day: str = Query("any"),
) -> dict:
    """Open (unbooked, non-past) tour slots for a property. Defaults:
    date_from=today, date_to=today+7."""
    t0 = time.perf_counter()
    prop = _property_or_404(property_id)
    area = _area_of(prop)
    today = date.today()
    try:
        df = date.fromisoformat(date_from) if date_from else today
        dt = date.fromisoformat(date_to) if date_to else today + timedelta(days=7)
    except ValueError:
        raise HTTPException(422, "Dates must look like 2026-07-20 (year-month-day).")

    opens = tours.open_slots(
        property_id=property_id,
        area=area,
        agents=store.list_agents(),
        windows=store.list_windows(),
        bookings=store.list_bookings(status="booked"),
        date_from=df,
        date_to=dt,
        time_of_day=time_of_day or "any",
    )
    slots = [_open_to_slot(o) for o in opens]
    store.log_event(
        endpoint="tours_slots",
        latency_ms=(time.perf_counter() - t0) * 1000,
        meta={"property_id": property_id, "total": len(slots),
              "time_of_day": time_of_day},
    )
    return {"property_id": property_id, "slots": slots, "total": len(slots)}


@router.post("/tours/book")
def book(req: BookTourRequest) -> dict:
    """Book a tour from a slot_id. Atomic: the store re-checks the conflict in
    a transaction. 404 unknown property/agent, 422 invalid/out-of-window slot,
    409 slot no longer free."""
    t0 = time.perf_counter()
    prop = _property_or_404(req.property_id)
    property_name = prop.get("name", req.property_id)

    if not (req.prospect_name or "").strip():
        raise HTTPException(422, "A prospect name is required to book a tour.")

    try:
        agent_id, start = tours.parse_slot_id(req.slot_id)
    except ValueError:
        raise HTTPException(422, "Malformed slot_id.")

    agents = {a["id"]: a for a in store.list_agents()}
    agent = agents.get(agent_id)
    if agent is None:
        raise HTTPException(404, "Unknown agent for that slot.")

    windows = store.list_windows()
    if not tours.slot_is_structurally_valid(agent_id, start, list(agents.values()), windows):
        raise HTTPException(422, "That slot isn't a valid, future tour time.")

    end = start + timedelta(minutes=tours.DEFAULT_DURATION_MIN)
    try:
        row = store.book_tour(
            property_id=req.property_id,
            property_name=property_name,
            agent_id=agent_id,
            agent_name=agent["name"],
            start=start.isoformat(timespec="seconds"),
            end=end.isoformat(timespec="seconds"),
            prospect_name=req.prospect_name.strip(),
            duration_minutes=tours.DEFAULT_DURATION_MIN,
            prospect_email=req.prospect_email or "",
            prospect_phone=req.prospect_phone or "",
            applicant_id=req.applicant_id or "",
            notes=req.notes or "",
        )
    except store.SlotConflict:
        store.log_event(endpoint="tours_book", meta={"result": "conflict",
                        "slot_id": req.slot_id})
        raise HTTPException(409, "That slot was just booked. Please pick another.")

    booking = _augment(TourBooking(**row))
    store.log_event(
        endpoint="tours_book",
        applicant_id=req.applicant_id or None,
        latency_ms=(time.perf_counter() - t0) * 1000,
        meta={"result": "booked", "booking_id": booking.id,
              "property_id": req.property_id, "agent_id": agent_id},
    )
    return {"ok": True, "booking": booking}


def _redact_other_prospects(rows: list, email: str | None, applicant_id: str | None) -> list:
    """Blank out contact info (email/phone/applicant_id) on any row that isn't
    the caller's own booking. A property_id-scoped listing (the staff console)
    needs name/time/agent to manage the schedule, but has no business handing
    back every OTHER prospect's contact info in bulk — property ids are public,
    so "requires property_id" alone doesn't stop a stranger from harvesting
    contact info for a property they merely know the id of. Only a row that
    matches the caller's own email/applicant_id filter is trusted to be theirs,
    and keeps its real contact fields."""
    q_email = (email or "").strip().lower()
    q_applicant = (applicant_id or "").strip()
    out = []
    for r in rows:
        is_own = (q_email and (r.get("prospect_email") or "").strip().lower() == q_email) or (
            q_applicant and (r.get("applicant_id") or "").strip() == q_applicant
        )
        if not is_own:
            r = {**r, "prospect_email": "", "prospect_phone": "", "applicant_id": ""}
        out.append(r)
    return out


@router.get("/tours")
def list_tours(
    email: str = Query(None),
    applicant_id: str = Query(None),
    property_id: str = Query(None),
    status: str = Query(None),
) -> dict:
    """Bookings for one property/prospect, soonest first. Requires at least one
    of property_id/email/applicant_id — an unscoped call would dump every
    prospect's name/email/phone across the whole portfolio. Contact info
    (email/phone/applicant_id) is further redacted on any row that isn't the
    caller's own booking — see _redact_other_prospects."""
    if not any([email, applicant_id, property_id]):
        raise HTTPException(
            422, "Provide at least one of property_id, email, or applicant_id."
        )
    effective_status = status
    rows = store.list_bookings(
        status=effective_status,
        property_id=property_id,
        email=email,
        applicant_id=applicant_id,
    )
    rows = _redact_other_prospects(rows, email, applicant_id)
    tours_out = [_augment(TourBooking(**r)) for r in rows]
    store.log_event(endpoint="tours_list", meta={"total": len(tours_out)})
    return {"tours": tours_out, "total": len(tours_out)}


@router.get("/tours/{booking_id}/calendar.ics")
def calendar_ics(booking_id: str) -> Response:
    """Download the booking as an .ics file (import into any calendar app)."""
    row = store.get_booking(booking_id)
    if row is None:
        raise HTTPException(404, "Unknown booking id.")
    body = _ics(TourBooking(**row))
    return Response(
        content=body,
        media_type="text/calendar; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="tour-{booking_id}.ics"'},
    )


@router.delete("/tours/{booking_id}")
def cancel(booking_id: str) -> dict:
    """Cancel a booking (frees the slot). 404 if unknown.

    Security model, given this app has no login/session system anywhere: a
    booking_id ("TOUR-" + 12 hex chars, 48 bits) is a high-entropy capability
    token, not a guessable sequence. The only ways to learn one are (a) being
    the person who booked it (it's returned directly in POST /tours/book's
    response) or (b) the property-scoped GET /tours listing used by the
    internal staff console -- which no longer echoes back OTHER prospects'
    contact info (see _redact_other_prospects), so knowing a property_id alone
    no longer lets a stranger harvest a specific person's booking to target.
    An earlier version of this endpoint additionally required the caller to
    supply the booking's own email/applicant_id as "proof" -- that check was
    circular (GET returned exactly the value DELETE demanded) and, once the
    redaction above closed the real leak, would have broken the legitimate
    staff-console cancel flow (which has no session to prove identity with
    either). Removed rather than kept as security theater.
    """
    row = store.get_booking(booking_id)
    if row is None:
        raise HTTPException(404, "Unknown booking id.")
    store.cancel_booking(booking_id)
    store.log_event(endpoint="tours_cancel", meta={"booking_id": booking_id})
    return {"ok": True, "status": "cancelled"}


@router.post("/tours/chat", response_model=TourChatResponse)
def chat(req: TourChatRequest) -> TourChatResponse:
    """Conversational booking. ALWAYS returns 200 — degrades server-side."""
    t0 = time.perf_counter()
    resp = tours_chat.handle(req)
    store.log_event(
        endpoint="tours_chat",
        latency_ms=(time.perf_counter() - t0) * 1000,
        source=resp.source,
        meta={"phase": resp.state.phase, "property_id": req.property_id,
              "proposed": len(resp.proposed_slots),
              "booked": bool(resp.booking)},
    )
    return resp
