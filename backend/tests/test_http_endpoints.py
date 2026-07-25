"""Integration tests for HTTP routes that had zero TestClient coverage:

- ``tours_api.py``     GET /tours/staff, /tours/slots, POST /tours/book,
                       GET /tours, DELETE /tours/{id}, /tours/{id}/calendar.ics,
                       POST /tours/chat
- ``concierge_api.py`` POST /concierge/ask, GET /concierge/status,
                       GET /concierge/lease/{id}, /concierge/lease/{id}/pdf
- ``main.py``          POST/GET /applicants/{id}/decision(s) (reviewer audit trail)

Only the underlying business-logic functions were exercised before (direct
calls to ``tours.py``/``tours_chat.py``/``concierge.py``), never the actual
routes -- so a routing/response-model/status-code regression at the HTTP
layer (like the GET/DELETE /tours identity-check gap found in review) could
ship undetected. These tests drive the real FastAPI app via ``TestClient``.
"""

from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient

from models import ApplicantProfile

PROP_ID = "PROP-002"

APPLICANT = ApplicantProfile(
    name="Jamie Ortiz", monthly_income=6500, desired_rent=1700, credit_score=710,
    employment_status="employed", employment_length_months=48, savings_balance=12000,
    current_rent=1500, years_at_current_address=2, references_count=2,
    landlord_reference=True, guarantor_available=False, household_size=1,
)


@pytest.fixture(scope="module")
def client():
    import main

    with TestClient(main.app) as c:
        yield c


@pytest.fixture(scope="module")
def applicant_id(client):
    resp = client.post("/apply", json=APPLICANT.model_dump())
    assert resp.status_code == 200
    return resp.json()["applicant_id"]


def _first_open_slot(client) -> dict:
    """A real open slot for PROP_ID over a wide window, robust to whatever
    "today" happens to be (weekday coverage varies by agent)."""
    today = date.today()
    resp = client.get(
        "/tours/slots",
        params={
            "property_id": PROP_ID,
            "date_from": today.isoformat(),
            "date_to": (today + timedelta(days=14)).isoformat(),
        },
    )
    assert resp.status_code == 200
    slots = resp.json()["slots"]
    assert slots, "expected at least one open slot in a 14-day window"
    return slots[0]


# ---------------------------------------------------------------------------
# tours_api.py
# ---------------------------------------------------------------------------
def test_tours_staff(client):
    resp = client.get("/tours/staff", params={"property_id": PROP_ID})
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == len(body["staff"]) > 0


def test_tours_staff_unknown_property_404s(client):
    resp = client.get("/tours/staff", params={"property_id": "PROP-NOPE"})
    assert resp.status_code == 404


def test_tours_slots(client):
    slot = _first_open_slot(client)
    assert slot["property_id"] == PROP_ID
    assert slot["slot_id"] and slot["agent_id"]


def test_book_and_list_and_cancel_roundtrip(client):
    slot = _first_open_slot(client)
    book_resp = client.post(
        "/tours/book",
        json={
            "property_id": PROP_ID,
            "slot_id": slot["slot_id"],
            "prospect_name": "Taylor Reed",
            "prospect_email": "taylor.reed@example.com",
        },
    )
    assert book_resp.status_code == 200
    booking = book_resp.json()["booking"]
    assert booking["property_id"] == PROP_ID
    assert booking["status"] == "booked"

    # GET /tours requires a scope filter -- unscoped is rejected (422), the
    # earlier version of this endpoint let anyone dump every booking).
    assert client.get("/tours").status_code == 422

    # Scoped by property_id: the booking's own contact info is redacted since
    # this caller didn't prove it's their own (see _redact_other_prospects).
    list_resp = client.get("/tours", params={"property_id": PROP_ID})
    assert list_resp.status_code == 200
    listed = next(t for t in list_resp.json()["tours"] if t["id"] == booking["id"])
    assert listed["prospect_email"] == ""

    # Scoped by the prospect's own email: their own contact info IS returned.
    own_resp = client.get("/tours", params={"email": "taylor.reed@example.com"})
    own = next(t for t in own_resp.json()["tours"] if t["id"] == booking["id"])
    assert own["prospect_email"] == "taylor.reed@example.com"

    ics_resp = client.get(f"/tours/{booking['id']}/calendar.ics")
    assert ics_resp.status_code == 200
    assert "BEGIN:VCALENDAR" in ics_resp.text

    cancel_resp = client.delete(f"/tours/{booking['id']}")
    assert cancel_resp.status_code == 200
    assert cancel_resp.json()["status"] == "cancelled"

    # A truly unknown booking id 404s; re-cancelling an already-cancelled
    # (but real) booking is idempotent, not an error.
    assert client.delete("/tours/TOUR-doesnotexist").status_code == 404
    assert client.delete(f"/tours/{booking['id']}").status_code == 200


def test_tours_chat_smoke(client):
    resp = client.post(
        "/tours/chat",
        json={
            "messages": [{"role": "user", "content": "hi, can I see this place?"}],
            "property_id": PROP_ID,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["reply"]
    assert body["state"]["phase"]


# ---------------------------------------------------------------------------
# concierge_api.py
# ---------------------------------------------------------------------------
def test_concierge_ask(client):
    resp = client.post(
        "/concierge/ask",
        json={"question": "What's the pet policy?", "property_id": PROP_ID},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"]


def test_concierge_ask_unknown_property_404s(client):
    resp = client.post(
        "/concierge/ask",
        json={"question": "What's the rent?", "property_id": "PROP-NOPE"},
    )
    assert resp.status_code == 404


def test_concierge_status(client):
    resp = client.get("/concierge/status")
    assert resp.status_code == 200
    assert "indexed" in resp.json()


def test_concierge_lease(client):
    resp = client.get(f"/concierge/lease/{PROP_ID}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["property_id"] == PROP_ID
    assert body["sections"]


def test_concierge_lease_unknown_property_404s(client):
    assert client.get("/concierge/lease/PROP-NOPE").status_code == 404


def test_concierge_lease_pdf(client):
    resp = client.get(f"/concierge/lease/{PROP_ID}/pdf")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/pdf"
    assert resp.content[:4] == b"%PDF"


# ---------------------------------------------------------------------------
# Reviewer decision workflow (main.py)
# ---------------------------------------------------------------------------
def test_decision_roundtrip(client, applicant_id):
    resp = client.post(
        f"/applicants/{applicant_id}/decision",
        json={"action": "approve", "note": "Meets all criteria", "reviewer": "pm@example.com"},
    )
    assert resp.status_code == 200
    assert resp.json() == {"ok": True, "status": "approve"}

    list_resp = client.get(f"/applicants/{applicant_id}/decisions")
    assert list_resp.status_code == 200
    decisions = list_resp.json()["decisions"]
    assert len(decisions) == 1
    assert decisions[0]["action"] == "approve"
    assert decisions[0]["note"] == "Meets all criteria"


def test_decision_unknown_applicant_404s(client):
    resp = client.post(
        "/applicants/ghost/decision",
        json={"action": "decline"},
    )
    assert resp.status_code == 404


def test_decisions_list_unknown_applicant_404s(client):
    assert client.get("/applicants/ghost/decisions").status_code == 404
