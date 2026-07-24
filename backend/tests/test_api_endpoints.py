"""Integration tests for the previously-uncovered REST endpoints:

- ``apply_api.py``     POST /apply            (manual applicant intake)
- ``dashboard_api.py`` GET  /dashboard/stats  (aggregate home-page stats)
- ``properties_api.py``GET  /properties, /properties/{id}/screen, /candidates
- ``risk_api.py``      POST /risk/chat, /risk/chat/stream  (grounded, never-500)

Driven through the real FastAPI app via ``TestClient`` (startup seeds the temp
DB from conftest). Deterministic applicants are injected by monkeypatching
``store.get_profile`` / ``store.list_applicants`` (as ``test_risk.py`` does)
rather than depending on real data.
"""

import json

import pytest
from fastapi.testclient import TestClient

import risk_chat
import store
from models import ApplicantProfile

# A known-good, seeded property (from data/properties.json).
PROP_ID = "PROP-001"

STRONG = ApplicantProfile(
    name="Strong", monthly_income=9000, desired_rent=1450, credit_score=780,
    employment_status="employed", employment_length_months=60, savings_balance=20000,
    current_rent=1400, years_at_current_address=5, references_count=3,
    landlord_reference=True, guarantor_available=True, household_size=1,
)
WEAK = ApplicantProfile(
    name="Weak", monthly_income=3000, desired_rent=1650, credit_score=560,
    employment_status="unknown", late_payments_12mo=4, evictions_count=1,
    monthly_debt_payments=900, current_rent=900,
)


@pytest.fixture(scope="module")
def client():
    import main

    with TestClient(main.app) as c:
        yield c


def _sse_events(text: str) -> list[dict]:
    return [
        json.loads(line[len("data:"):].strip())
        for line in text.splitlines()
        if line.startswith("data:")
    ]


# ===========================================================================
# apply_api.py  —  POST /apply
#
# Input contract: the request body is an ApplicantProfile JSON object (NOT an
# UploadFile / multipart form). The handler validates it, generates a PDF from
# the form (best-effort), indexes it, persists the applicant, and returns an
# UploadResponse {applicant_id, profile, chunks_indexed, has_pdf}.
# ===========================================================================
def test_apply_happy_path_creates_applicant(client):
    resp = client.post("/apply", json=STRONG.model_dump())
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {
        "applicant_id", "profile", "chunks_indexed", "has_pdf"
    }
    aid = body["applicant_id"]
    assert aid and isinstance(aid, str)
    assert body["profile"]["name"] == "Strong"
    assert body["chunks_indexed"] >= 0
    assert isinstance(body["has_pdf"], bool)
    # The applicant landed in the (temp) store.
    stored = store.get_profile(aid)
    assert stored is not None
    assert stored.name == "Strong"
    assert any(row["id"] == aid for row in store.list_applicants())


def test_upload_rejects_oversized_file(client, monkeypatch):
    """Regression: /upload had no size cap at all — a large POST could be
    read fully into memory before any validation ran."""
    from settings import settings

    monkeypatch.setattr(settings, "max_upload_mb", 1)
    oversized = b"%PDF-1.4\n" + b"0" * (2 * 1024 * 1024)  # 2MB > the 1MB test cap
    resp = client.post(
        "/upload", files={"file": ("big.pdf", oversized, "application/pdf")}
    )
    assert resp.status_code == 413


def test_recommend_failure_is_still_logged_to_monitoring(client, monkeypatch):
    """Regression: /recommend only called store.log_event AFTER a successful
    result, so an exception mid-request skipped it entirely — a real outage
    produced zero monitoring signal instead of a visible spike."""
    import graphrag

    resp = client.post("/apply", json=STRONG.model_dump())
    aid = resp.json()["applicant_id"]

    def _boom(profile, explain=True):
        raise RuntimeError("simulated graphrag failure")

    monkeypatch.setattr(graphrag, "recommend", _boom)
    before = len(store.recent_events(limit=10_000))

    # TestClient re-raises unhandled exceptions rather than returning a 500
    # response (a real ASGI server would 500 the client) — either way, the
    # point under test is that log_event fires before the error propagates.
    with pytest.raises(RuntimeError, match="simulated graphrag failure"):
        client.get(f"/recommend/{aid}")

    events = store.recent_events(limit=10_000)
    assert len(events) == before + 1
    logged = events[0]
    assert logged["endpoint"] == "recommend"
    assert logged["applicant_id"] == aid
    assert logged["source"] == "error"


def test_delete_applicant_removes_uploaded_pdf_from_disk(client):
    """Regression: DELETE /applicants/{id} only removed the DB row, leaving
    the uploaded/generated PDF orphaned on disk indefinitely."""
    from settings import UPLOAD_DIR

    resp = client.post("/apply", json=STRONG.model_dump())
    aid = resp.json()["applicant_id"]
    pdf_path = UPLOAD_DIR / f"{aid}.pdf"
    if resp.json()["has_pdf"]:
        assert pdf_path.exists()

    del_resp = client.delete(f"/applicants/{aid}")
    assert del_resp.status_code == 200
    assert del_resp.json() == {"deleted": aid}
    assert not pdf_path.exists()
    assert store.get_profile(aid) is None

    # Deleting an already-deleted (or unknown) id 404s.
    assert client.delete(f"/applicants/{aid}").status_code == 404


def test_apply_rejects_zero_rent(client):
    bad = STRONG.model_copy(update={"desired_rent": 0})
    resp = client.post("/apply", json=bad.model_dump())
    assert resp.status_code == 400


def test_apply_rejects_out_of_range_credit(client):
    bad = STRONG.model_copy(update={"credit_score": 200})
    resp = client.post("/apply", json=bad.model_dump())
    assert resp.status_code == 400


def test_apply_rejects_malformed_move_in_date(client):
    bad = STRONG.model_copy(update={"desired_move_in": "not-a-date"})
    resp = client.post("/apply", json=bad.model_dump())
    assert resp.status_code == 400


# ===========================================================================
# dashboard_api.py  —  GET /dashboard/stats
# ===========================================================================
def test_dashboard_stats(client):
    resp = client.get("/dashboard/stats")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {
        "applicants_total", "verdicts", "recent_applicants",
        "properties", "traffic", "feedback",
    }
    assert set(body["verdicts"].keys()) == {
        "qualified", "needs_review", "not_qualified"
    }
    assert isinstance(body["applicants_total"], int)
    assert len(body["recent_applicants"]) <= 5

    props = body["properties"]
    assert set(props.keys()) == {
        "total", "rent_min", "rent_max", "pets_allowed", "areas", "by_area"
    }
    assert props["total"] > 0
    assert props["rent_min"] <= props["rent_max"]

    traffic = body["traffic"]
    assert set(traffic.keys()) == {
        "events_total", "avg_latency_ms_by_endpoint", "requests_by_endpoint",
        "outliers_excluded", "faithfulness_violations",
    }
    assert set(body["feedback"].keys()) == {"up", "down"}


# ===========================================================================
# properties_api.py  —  /properties, /screen, /candidates
# ===========================================================================
def test_list_properties(client):
    resp = client.get("/properties")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {"properties", "total", "areas"}
    assert body["total"] == len(body["properties"]) > 0
    # Cheapest first.
    rents = [p["monthly_rent"] for p in body["properties"]]
    assert rents == sorted(rents)
    assert isinstance(body["areas"], list) and body["areas"]
    row = body["properties"][0]
    assert {"id", "name", "monthly_rent", "area"}.issubset(row.keys())


def test_list_properties_filtered(client):
    body = client.get("/properties", params={"max_rent": 1500}).json()
    assert all(p["monthly_rent"] <= 1500 for p in body["properties"])


def test_screen_applicant(client, monkeypatch):
    monkeypatch.setattr(store, "get_profile", lambda aid: STRONG)
    resp = client.get(
        f"/properties/{PROP_ID}/screen", params={"applicant_id": "a-strong"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["property_id"] == PROP_ID
    assert body["applicant_id"] == "a-strong"
    assert set(body.keys()) >= {
        "property_id", "applicant_id", "passes", "verdict", "checks"
    }
    assert body["verdict"] in ("pass", "fail", "review")
    assert isinstance(body["checks"], list)
    for c in body["checks"]:
        assert {"label", "required", "actual", "ok"}.issubset(c.keys())


def test_screen_unknown_applicant_404(client, monkeypatch):
    monkeypatch.setattr(store, "get_profile", lambda aid: None)
    resp = client.get(
        f"/properties/{PROP_ID}/screen", params={"applicant_id": "nope"}
    )
    assert resp.status_code == 404


def test_screen_unknown_property_404(client, monkeypatch):
    monkeypatch.setattr(store, "get_profile", lambda aid: STRONG)
    resp = client.get(
        "/properties/PROP-NOPE/screen", params={"applicant_id": "a-strong"}
    )
    assert resp.status_code == 404


def test_property_candidates(client, monkeypatch):
    applicants = [{"id": "s", "name": "Strong"}, {"id": "w", "name": "Weak"}]
    profiles = {"s": STRONG, "w": WEAK}
    monkeypatch.setattr(store, "list_applicants", lambda: applicants)
    monkeypatch.setattr(store, "get_profile", lambda aid: profiles.get(aid))
    resp = client.get(f"/properties/{PROP_ID}/candidates")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {"property_id", "name", "candidates", "total"}
    assert body["property_id"] == PROP_ID
    assert body["total"] == 2
    assert len(body["candidates"]) == 2
    cand = body["candidates"][0]
    assert {
        "applicant_id", "name", "score", "signal_breakdown",
        "screen_passes", "screen_verdict",
    }.issubset(cand.keys())
    # Ranked: passing-then-score, highest first.
    keys = [(c["screen_passes"], c["score"]) for c in body["candidates"]]
    assert keys == sorted(keys, reverse=True)


def test_property_candidates_unknown_property_404(client):
    assert client.get("/properties/PROP-NOPE/candidates").status_code == 404


# ===========================================================================
# risk_api.py  —  POST /risk/chat  and  /risk/chat/stream
# ===========================================================================
def test_risk_chat_grounded(client, monkeypatch):
    monkeypatch.setattr(store, "get_profile", lambda aid: WEAK)
    resp = client.post(
        "/risk/chat",
        json={"question": "Why is this applicant risky?", "applicant_id": "a1"},
    )
    assert resp.status_code == 200  # never 500
    body = resp.json()
    assert set(body.keys()) == {
        "answer", "scope", "applicant_id", "intent",
        "sources", "artifact", "follow_ups", "source",
    }
    assert body["scope"] == "applicant"
    assert body["source"] in ("rules", "anthropic")
    assert body["answer"].strip()
    # Grounded: deterministic head-derived sources + a real artifact.
    assert body["sources"], "expected grounding sources for an applicant question"
    assert body["artifact"]["kind"] != "none"


def test_risk_chat_unknown_applicant_deflects_not_404(client, monkeypatch):
    monkeypatch.setattr(store, "get_profile", lambda aid: None)
    resp = client.post(
        "/risk/chat",
        json={"question": "Why is this applicant risky?", "applicant_id": "ghost"},
    )
    assert resp.status_code == 200  # NOT a 404 — chat degrades, never breaks
    body = resp.json()
    assert body["artifact"]["kind"] == "none"
    assert body["sources"] == []
    assert body["answer"].strip()


def test_risk_chat_empty_applicant_id_portfolio_scope(client):
    resp = client.post("/risk/chat", json={"question": "How risky is the portfolio?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["scope"] == "portfolio"
    assert body["answer"].strip()


@pytest.mark.parametrize("question", [
    "which applicants are riskiest",
    "who is the riskiest applicant",
    "show me high risk applicants",
    "which applicant has the lowest risk",
])
def test_risk_chat_portfolio_ranking_routes_to_compare(question):
    """Regression: these phrasings used to fall through to "explain" (no
    applicant -> empty result -> deflection) or "general" (generic blurb about
    the assistant), even though risk_agent.portfolio_summary() -- already used
    by the "compare" intent's no-applicant branch -- can answer them."""
    assert risk_chat.route(question) == "compare"
    # A scoped question with the same risk-adjacent words must still explain
    # THAT applicant, not get swept into the portfolio ranking.
    assert risk_chat.route(question, applicant_id="A1") != "general"


def test_risk_chat_portfolio_ranking_answer_not_a_deflection(client):
    resp = client.post("/risk/chat", json={"question": "which applicants are riskiest"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["intent"] == "compare"
    assert body["scope"] == "portfolio"
    lowered = body["answer"].lower()
    assert "don't have any applicant-level" not in lowered
    assert "does not include applicant-level" not in lowered


def test_risk_chat_stream_sse_framing(client, monkeypatch):
    monkeypatch.setattr(store, "get_profile", lambda aid: WEAK)
    resp = client.post(
        "/risk/chat/stream",
        json={"question": "Why is this applicant risky?", "applicant_id": "a1"},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    events = _sse_events(resp.text)
    types = [e["type"] for e in events]
    assert types[0] == "meta"
    assert types[-1] == "done"
    assert "token" in types
    assert events[-1]["source"] in ("rules", "anthropic")
    assert any(e.get("text", "").strip() for e in events if e["type"] == "token")


def test_risk_chat_stream_unknown_applicant_never_500(client, monkeypatch):
    monkeypatch.setattr(store, "get_profile", lambda aid: None)
    resp = client.post(
        "/risk/chat/stream",
        json={"question": "Why is this applicant risky?", "applicant_id": "ghost"},
    )
    assert resp.status_code == 200
    events = _sse_events(resp.text)
    types = [e["type"] for e in events]
    assert types[0] == "meta" and types[-1] == "done"
    assert "token" in types
