"""Integration tests for the Resident Risk HTTP API (``resident_api.py``).

Driven through the real FastAPI app via ``TestClient`` (which triggers startup:
init_db + seeds against the temp DB from conftest). Runs fully offline — the LLM
degrades server-side to the deterministic ("rules") path, so chat/stream answers
are grounded templates. These tests exercise every route in ``resident_api.py``
and assert response-key shapes, the 404 contract, best->worst health ordering,
grounded/never-500 chat, and SSE meta->token->done framing.
"""

import json

import pytest
from fastapi.testclient import TestClient

import residents_risk


@pytest.fixture(scope="module")
def client():
    import main

    with TestClient(main.app) as c:
        yield c


@pytest.fixture(scope="module")
def sample_resident():
    """A real seeded resident (id + property) to drive the id-scoped routes."""
    residents = residents_risk.load_residents()
    assert residents, "expected seeded residents.json to be non-empty"
    return residents[0]


def _sse_events(text: str) -> list[dict]:
    """Parse an SSE body into the list of decoded ``data:`` JSON frames."""
    events = []
    for line in text.splitlines():
        if line.startswith("data:"):
            events.append(json.loads(line[len("data:"):].strip()))
    return events


# ---------------------------------------------------------------------------
# GET /residents  (portfolio table)
# ---------------------------------------------------------------------------
def test_list_residents_shape(client):
    resp = client.get("/residents")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {"residents", "count", "property_id", "source"}
    assert body["source"] in ("model", "heuristic")
    assert body["count"] == len(body["residents"]) > 0
    # Rows are sorted by late_probability, highest first.
    probs = [r["late_probability"] for r in body["residents"]]
    assert probs == sorted(probs, reverse=True)
    row = body["residents"][0]
    assert {
        "resident_id", "property_id", "unit_id", "name", "base_rent",
        "tenure_months", "late_probability", "late_band", "expected_arrears",
        "churn_probability", "churn_status", "serious_probability",
        "serious_band", "current_balance", "top_driver",
    }.issubset(row.keys())
    assert 0.0 <= row["late_probability"] <= 1.0
    assert row["late_band"] in ("low", "medium", "high")


def test_score_many_source_reflects_actual_fallback_not_static_status(monkeypatch):
    """Regression: the response-level "source" used to come from a static
    residents_risk.status()["trained"] check — true whenever the model bundle
    loaded at all, even if predict_bulk silently fell back to the heuristic
    for some heads on THIS request. It must instead reflect what actually
    happened while scoring, so a partial degradation isn't mislabeled "model"."""
    import resident_api
    import residents_risk

    def _head(source: str, **extra) -> dict:
        return {"probability": 0.2, "band": "low", "source": source, **extra}

    def fake_predict_bulk(residents, heads=None):
        out = []
        for i, r in enumerate(residents):
            # First resident's "late" head fell back to heuristic; everything
            # else (for every resident) used the model.
            late_source = "heuristic" if i == 0 else "model"
            out.append({
                "resident_id": r.get("resident_id", ""),
                "late": _head(late_source),
                "arrears": _head("model", expected_balance=0.0),
                "churn": _head("model"),
                "serious": _head("model"),
            })
        return out

    monkeypatch.setattr(residents_risk, "predict_bulk", fake_predict_bulk)
    # Even though the bundle claims fully "trained", one heuristic head above
    # must still win — proves the label no longer comes from this static check.
    monkeypatch.setattr(residents_risk, "status", lambda: {"trained": True})

    residents = residents_risk.load_residents()[:3]
    rows, aggs, source = resident_api._score_many(residents)
    assert len(rows) == len(residents)
    assert source == "heuristic"

    # And the all-model case still reports "model".
    monkeypatch.setattr(
        residents_risk, "predict_bulk",
        lambda residents, heads=None: [
            {
                "resident_id": r.get("resident_id", ""),
                "late": _head("model"), "arrears": _head("model", expected_balance=0.0),
                "churn": _head("model"), "serious": _head("model"),
            }
            for r in residents
        ],
    )
    _, _, source_all_model = resident_api._score_many(residents)
    assert source_all_model == "model"


def test_list_residents_filtered_by_property(client, sample_resident):
    pid = sample_resident["property_id"]
    body = client.get("/residents", params={"property_id": pid}).json()
    assert body["property_id"] == pid
    assert body["count"] > 0
    assert all(r["property_id"] == pid for r in body["residents"])


def test_list_residents_unknown_property_is_empty_not_404(client):
    body = client.get("/residents", params={"property_id": "PROP-NOPE"}).json()
    assert body["count"] == 0
    assert body["residents"] == []


# ---------------------------------------------------------------------------
# GET /residents/properties  (cheap picker, no scoring)
# ---------------------------------------------------------------------------
def test_resident_properties_picker(client):
    resp = client.get("/residents/properties")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {"properties", "count"}
    assert body["count"] == len(body["properties"]) > 0
    opt = body["properties"][0]
    assert set(opt.keys()) == {"property_id", "name", "resident_count"}
    assert opt["resident_count"] > 0
    # Total headcount across the picker equals the full resident set.
    assert sum(p["resident_count"] for p in body["properties"]) == len(
        residents_risk.load_residents()
    )


# ---------------------------------------------------------------------------
# GET /residents/model-card
# ---------------------------------------------------------------------------
def test_resident_model_card(client):
    resp = client.get("/residents/model-card")
    assert resp.status_code == 200
    body = resp.json()
    for key in ("name", "version", "description", "intended_use",
                "heads", "families", "excluded", "limitations", "source"):
        assert key in body
    assert body["source"] in ("model", "heuristic")
    assert isinstance(body["heads"], list) and body["heads"]
    assert all(set(e.keys()) >= {"field", "reason"} for e in body["excluded"])


# ---------------------------------------------------------------------------
# GET /residents/portfolio/summary
# ---------------------------------------------------------------------------
def test_portfolio_summary(client):
    resp = client.get("/residents/portfolio/summary")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {
        "properties", "overall", "property_count", "resident_count",
        "snapshot_date", "source", "generated_at",
    }
    assert body["source"] in ("model", "heuristic")
    assert body["property_count"] == len(body["properties"]) > 0
    assert body["resident_count"] == len(residents_risk.load_residents())
    overall = body["overall"]
    assert overall["resident_count"] == body["resident_count"]
    assert 0.0 <= overall["predicted_late_rate"] <= 1.0
    assert 0.0 <= overall["predicted_late_rate_1m"] <= 1.0
    for band_key in ("late_bands", "serious_bands", "churn_bands"):
        assert set(overall[band_key].keys()) == {
            "low", "medium", "high", "not_applicable"
        }
    prop = body["properties"][0]
    assert "property_id" in prop and "resident_count" in prop


# ---------------------------------------------------------------------------
# GET /properties/{property_id}/residents
# ---------------------------------------------------------------------------
def test_property_residents(client, sample_resident):
    pid = sample_resident["property_id"]
    resp = client.get(f"/properties/{pid}/residents")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {
        "property_id", "residents", "count", "rollup", "source"
    }
    assert body["property_id"] == pid
    assert body["count"] == len(body["residents"]) > 0
    assert all(r["property_id"] == pid for r in body["residents"])
    probs = [r["late_probability"] for r in body["residents"]]
    assert probs == sorted(probs, reverse=True)
    assert body["rollup"]["property_id"] == pid
    assert body["rollup"]["resident_count"] == body["count"]

    # The KPI tile's "Predicted late-rate" reads predicted_late_rate_1m (next
    # MONTH), distinct from the legacy predicted_late_rate (next quarter) --
    # both are real, differently-scoped numbers, never accidentally equal.
    assert 0.0 <= body["rollup"]["predicted_late_rate_1m"] <= 1.0
    assert 0.0 <= body["rollup"]["predicted_late_rate"] <= 1.0

    # Late-payment count breakdown: 4 quarterly checkpoints (q1..q4), each a
    # genuine trained head — a cumulative-count timeline, so never decreasing.
    breakdown = body["rollup"]["late_count_breakdown"]
    assert [p["key"] for p in breakdown] == ["q1", "q2", "q3", "q4"]
    assert [p["label"] for p in breakdown] == ["Q1", "Q2", "Q3", "Q4"]
    values = [p["expected"] for p in breakdown]
    assert values == sorted(values)

    # Expected-arrears breakdown: same 4-quarter shape as late_count_breakdown.
    # Unlike the late-payment COUNT (which can only accumulate), a balance can
    # legitimately fall quarter to quarter as residents pay down arrears, so
    # this one isn't asserted non-decreasing -- just shape and non-negativity.
    arrears = body["rollup"]["arrears_breakdown"]
    assert [p["key"] for p in arrears] == ["q1", "q2", "q3", "q4"]
    assert [p["label"] for p in arrears] == ["Q1", "Q2", "Q3", "Q4"]
    assert all(p["expected"] >= 0 for p in arrears)

    # Severity-bucket distribution: fixed bucket order, non-negative counts
    # that add up to the property's resident count (every resident lands in
    # exactly one worst-delinquency bucket).
    severity = body["rollup"]["severity_buckets"]
    assert [b["bucket"] for b in severity] == ["none", "1-29", "30-59", "60-89", "90+"]
    assert all(b["count"] >= 0 for b in severity)
    assert sum(b["count"] for b in severity) == body["count"]

    # Horizon forecast: 4 cumulative "late by month T" checkpoints; cumulative
    # values never decrease, and each incremental (marginal, per-period) value
    # is non-negative and never exceeds its own cumulative value.
    horizon = body["rollup"]["horizon_forecast"]
    assert [p["horizon"] for p in horizon] == ["late_1m", "late_3m", "late_6m", "late_12m"]
    cum_values = [p["avg_probability"] for p in horizon]
    assert all(v is not None for v in cum_values)
    assert cum_values == sorted(cum_values)
    for p in horizon:
        assert p["incremental_probability"] is not None
        assert p["incremental_probability"] >= 0.0
        assert p["incremental_probability"] <= p["avg_probability"] + 1e-9
        assert set(p["bands"].keys()) == {"low", "medium", "high", "not_applicable"}


def test_property_residents_unknown_property_empty(client):
    body = client.get("/properties/PROP-NOPE/residents").json()
    assert body["count"] == 0
    assert body["residents"] == []
    assert body["rollup"]["resident_count"] == 0
    assert body["rollup"]["late_count_breakdown"] == [
        {"key": "q1", "label": "Q1", "expected": 0.0},
        {"key": "q2", "label": "Q2", "expected": 0.0},
        {"key": "q3", "label": "Q3", "expected": 0.0},
        {"key": "q4", "label": "Q4", "expected": 0.0},
    ]
    assert body["rollup"]["arrears_breakdown"] == [
        {"key": "q1", "label": "Q1", "expected": 0.0},
        {"key": "q2", "label": "Q2", "expected": 0.0},
        {"key": "q3", "label": "Q3", "expected": 0.0},
        {"key": "q4", "label": "Q4", "expected": 0.0},
    ]
    assert body["rollup"]["severity_buckets"] == [
        {"bucket": "none", "count": 0},
        {"bucket": "1-29", "count": 0},
        {"bucket": "30-59", "count": 0},
        {"bucket": "60-89", "count": 0},
        {"bucket": "90+", "count": 0},
    ]
    # No residents -> nothing to plot, mirrors late_count_breakdown's own
    # "[]" contract for the empty-portfolio case.
    assert body["rollup"]["horizon_forecast"] == []


# ---------------------------------------------------------------------------
# GET /residents/health  (best->worst ranking)
# ---------------------------------------------------------------------------
def test_residents_health_ranking(client):
    resp = client.get("/residents/health")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {
        "properties", "count", "healthiest", "needs_attention",
        "snapshot_date", "source",
    }
    props = body["properties"]
    assert body["count"] == len(props) > 0
    scores = [p["score"] for p in props]
    # Healthiest first (scores descending).
    assert scores == sorted(scores, reverse=True)
    for p in props:
        assert 0.0 <= p["score"] <= 100.0
        assert p["grade"] in ("A", "B", "C", "D", "F")
        assert p["resident_count"] > 0
        assert "name" in p
    # Callouts match the ends of the ranking.
    assert body["healthiest"]["property_id"] == props[0]["property_id"]
    assert body["needs_attention"]["property_id"] == props[-1]["property_id"]
    assert body["healthiest"]["score"] >= body["needs_attention"]["score"]


# ---------------------------------------------------------------------------
# GET /residents/{resident_id}  (detail + 404)
# ---------------------------------------------------------------------------
def test_get_resident_detail(client, sample_resident):
    rid = sample_resident["resident_id"]
    resp = client.get(f"/residents/{rid}")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {
        "resident", "predictions", "ledger_stats", "source"
    }
    assert body["resident"]["resident_id"] == rid
    preds = body["predictions"]
    for head in ("late", "arrears", "churn", "serious"):
        assert head in preds
    assert 0.0 <= preds["late"]["probability"] <= 1.0
    assert preds["late"]["band"] in ("low", "medium", "high")
    assert preds["churn"]["band"] in ("low", "medium", "high", "not_applicable")
    assert preds["serious"]["routes_to_review"] is True
    # Ledger stats are derived (not stored) — a few key fields present.
    assert body["ledger_stats"]["ledger_months"] > 0
    assert body["source"] in ("model", "heuristic")


def test_get_resident_unknown_404(client):
    resp = client.get("/residents/RES-DOESNOTEXIST")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /residents/{resident_id}/score
# ---------------------------------------------------------------------------
def test_score_resident(client, sample_resident):
    rid = sample_resident["resident_id"]
    resp = client.post(f"/residents/{rid}/score")
    assert resp.status_code == 200
    body = resp.json()
    for head in ("late", "arrears", "churn", "serious"):
        assert head in body
    assert body["resident_id"] == rid
    assert body["late"]["source"] in ("model", "heuristic")
    assert 0.0 <= body["serious"]["probability"] <= 1.0


def test_score_unknown_resident_404(client):
    assert client.post("/residents/RES-NOPE/score").status_code == 404


# ---------------------------------------------------------------------------
# POST /residents/chat  (never 500; grounded)
# ---------------------------------------------------------------------------
def test_chat_resident_scoped_is_grounded(client, sample_resident):
    rid = sample_resident["resident_id"]
    resp = client.post(
        "/residents/chat",
        json={"question": "Why is this resident risky?", "resident_id": rid},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {
        "answer", "scope", "resident_id", "property_id", "intent",
        "sources", "artifact", "follow_ups", "source",
    }
    assert body["scope"] == "resident"
    assert body["resident_id"] == rid
    assert body["source"] in ("rules", "anthropic")
    assert body["answer"].strip()
    # Grounded: the answer carries head-derived sources + a citation marker,
    # and the artifact carries the resident's head payloads.
    assert body["sources"], "expected grounding sources for a resident question"
    assert "[1]" in body["answer"]
    assert body["artifact"]["kind"] == "resident"


def test_chat_portfolio_health(client):
    resp = client.post(
        "/residents/chat", json={"question": "Which properties are healthiest?"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["intent"] == "property_health"
    assert body["artifact"]["kind"] == "property_health"
    assert body["answer"].strip()


def test_chat_unknown_resident_deflects_not_404(client):
    resp = client.post(
        "/residents/chat",
        json={"question": "Why are they risky?", "resident_id": "RES-NOPE"},
    )
    assert resp.status_code == 200
    body = resp.json()
    # Deflects gracefully: no invented predictions, empty artifact.
    assert body["artifact"]["kind"] == "none"
    assert body["sources"] == []
    assert body["answer"].strip()


# ---------------------------------------------------------------------------
# POST /residents/chat/stream  (SSE meta->token->done)
# ---------------------------------------------------------------------------
def test_chat_stream_sse_framing(client, sample_resident):
    rid = sample_resident["resident_id"]
    resp = client.post(
        "/residents/chat/stream",
        json={"question": "Why is this resident risky?", "resident_id": rid},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    events = _sse_events(resp.text)
    types = [e["type"] for e in events]
    # meta first, done last, at least one token between.
    assert types[0] == "meta"
    assert types[-1] == "done"
    assert "token" in types
    meta = events[0]
    assert meta["intent"] == "explain"
    assert meta["scope"] == "resident"
    assert meta["artifact"]["kind"] == "resident"
    assert "follow_ups" in meta
    assert events[-1]["source"] in ("rules", "anthropic")
    # A token frame carries prose text.
    assert any(e.get("text", "").strip() for e in events if e["type"] == "token")


def test_chat_stream_unknown_resident_never_500(client):
    resp = client.post(
        "/residents/chat/stream",
        json={"question": "Why are they risky?", "resident_id": "RES-NOPE"},
    )
    assert resp.status_code == 200
    events = _sse_events(resp.text)
    types = [e["type"] for e in events]
    assert types[0] == "meta" and types[-1] == "done"
    assert "token" in types
