"""Tests for the Resident Late-Payment Risk feature.

Runs offline (conftest forces the hash embedder + disables LLMs). Covers feature
extraction + structural exclusion, deterministic scoring, band thresholds, the
heuristic fallback (incl. monotonicity), reason-code shape, the eval runner, and
the HTTP endpoints via TestClient.
"""

import pytest
from fastapi.testclient import TestClient

import risk
import store
from evals import risk_eval
from models import ApplicantProfile

# ---------------------------------------------------------------------------
# Sample profiles
# ---------------------------------------------------------------------------
STRONG = ApplicantProfile(
    name="Strong", monthly_income=9000, desired_rent=1800, credit_score=780,
    employment_status="employed", employment_length_months=60, savings_balance=20000,
    current_rent=1750, years_at_current_address=5, references_count=3,
    landlord_reference=True, guarantor_available=True,
)
WEAK = ApplicantProfile(
    name="Weak", monthly_income=3000, desired_rent=1650, credit_score=560,
    employment_status="unknown", late_payments_12mo=4, evictions_count=1,
    monthly_debt_payments=900, current_rent=900,
)


# ---------------------------------------------------------------------------
# Feature extraction + structural exclusion
# ---------------------------------------------------------------------------
def test_extract_features_returns_exactly_allowed_features():
    f = risk.extract_features(STRONG)
    assert list(f.keys()) == risk.FEATURE_ORDER
    assert len(risk.feature_vector(STRONG)) == len(risk.FEATURE_ORDER)


def test_excluded_fields_do_not_affect_features():
    """Changing protected-class proxies / non-predictive fields must not change
    any model input (they are structurally excluded)."""
    base = risk.extract_features(STRONG)
    mutated = STRONG.model_copy(
        update={
            "name": "Totally Different",
            "preferred_area": "Some Neighborhood",
            "current_address": "123 Main St",
            "dependents": 4,
            "household_size": 6,
            "is_student": True,
            "criminal_record": True,
            "smoker": True,
            "has_pets": True,
            "pet_count": 3,
            "employer": "Acme Corp",
            "job_title": "Engineer",
            "bedrooms_wanted": 3,
            "lease_term_wanted": 24,
            "desired_move_in": "2026-09-01",
        }
    )
    assert risk.extract_features(mutated) == base


def test_missing_credit_is_imputed_and_flagged():
    f = risk.extract_features(ApplicantProfile(monthly_income=5000, desired_rent=1500))
    assert f["credit_imputed"] == 1.0
    assert f["credit_score"] == 680.0


# ---------------------------------------------------------------------------
# Predict shape + determinism
# ---------------------------------------------------------------------------
REQUIRED_KEYS = {
    "applicant_id", "name", "probability", "band", "reason_codes",
    "confidence", "range", "source", "model_type", "scored_at",
}


def test_predict_shape():
    r = risk.predict(STRONG, applicant_id="a1", name="Strong")
    assert REQUIRED_KEYS.issubset(r.keys())
    assert 0.0 <= r["probability"] <= 1.0
    assert r["band"] in ("low", "medium", "high")
    assert r["confidence"] in ("low", "high")
    assert r["source"] in ("model", "heuristic")
    assert r["model_type"] in ("xgboost", "histgb", "heuristic")
    assert len(r["range"]) == 2 and r["range"][0] <= r["range"][1]


def test_predict_is_deterministic():
    a = risk.predict(WEAK, applicant_id="w")
    b = risk.predict(WEAK, applicant_id="w")
    assert a["probability"] == b["probability"]
    assert a["band"] == b["band"]


def test_weak_scores_higher_than_strong():
    assert risk.predict(WEAK)["probability"] > risk.predict(STRONG)["probability"]


# ---------------------------------------------------------------------------
# Band thresholds (0.15 / 0.40 boundaries)
# ---------------------------------------------------------------------------
def test_band_thresholds():
    assert risk._band(0.0) == "low"
    assert risk._band(0.1499) == "low"
    assert risk._band(0.15) == "medium"
    assert risk._band(0.3999) == "medium"
    assert risk._band(0.40) == "high"
    assert risk._band(0.95) == "high"


# ---------------------------------------------------------------------------
# Heuristic fallback (force _model() None) + monotonicity
# ---------------------------------------------------------------------------
@pytest.fixture
def force_heuristic(monkeypatch):
    monkeypatch.setattr(risk, "_model", lambda: None)


def test_heuristic_fallback_used(force_heuristic):
    r = risk.predict(WEAK)
    assert r["source"] == "heuristic"
    assert r["model_type"] == "heuristic"
    assert 0.0 < r["probability"] < 1.0  # never exactly 0 or 1


def test_heuristic_monotonic_rent_to_income(force_heuristic):
    low_burden = ApplicantProfile(monthly_income=8000, desired_rent=1600, credit_score=700)
    high_burden = ApplicantProfile(monthly_income=3200, desired_rent=1600, credit_score=700)
    assert risk.predict(high_burden)["probability"] > risk.predict(low_burden)["probability"]


def test_heuristic_monotonic_credit(force_heuristic):
    good = ApplicantProfile(monthly_income=5000, desired_rent=1500, credit_score=800)
    poor = ApplicantProfile(monthly_income=5000, desired_rent=1500, credit_score=560)
    assert risk.predict(poor)["probability"] > risk.predict(good)["probability"]


# ---------------------------------------------------------------------------
# Reason-code shape (both paths)
# ---------------------------------------------------------------------------
def _assert_reason_codes(codes):
    assert 0 < len(codes) <= 4
    for rc in codes:
        assert set(rc.keys()) == {"feature", "label", "direction", "contribution"}
        assert rc["feature"] in risk.FEATURE_ORDER  # only allowed features cited
        assert rc["direction"] in ("increases", "decreases")
        assert isinstance(rc["label"], str) and rc["label"]


def test_reason_codes_model_path():
    _assert_reason_codes(risk.predict(WEAK)["reason_codes"])


def test_reason_codes_heuristic_path(force_heuristic):
    _assert_reason_codes(risk.predict(WEAK)["reason_codes"])


# ---------------------------------------------------------------------------
# Eval runner
# ---------------------------------------------------------------------------
def test_eval_metrics_and_confusion(monkeypatch):
    monkeypatch.setattr(risk_eval, "EVAL_N", 600)
    res = risk_eval.run(seed=7)
    for key in ("auc", "pr_auc", "brier", "ece"):
        assert res[key] is None or 0.0 <= res[key] <= 1.0
    assert res["auc"] is not None and res["auc"] < 0.999  # not trivially separable
    c = res["confusion"]
    total = (
        c["actual_late"]["pred_late"] + c["actual_late"]["pred_ontime"]
        + c["actual_ontime"]["pred_late"] + c["actual_ontime"]["pred_ontime"]
    )
    assert total == res["n"] == 600
    assert res["threshold"] == 0.40
    assert len(res["calibration"]) == 10
    assert all(set(s) == {"name", "n", "positive_rate", "auc", "brier", "ece", "flag"}
               for s in res["slices"])


# ---------------------------------------------------------------------------
# HTTP endpoints via TestClient
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def client():
    import main

    with TestClient(main.app) as c:
        yield c


def test_get_risk_known_applicant(client, monkeypatch):
    monkeypatch.setattr(store, "get_profile", lambda aid: STRONG)
    resp = client.get("/risk/anyid")
    assert resp.status_code == 200
    body = resp.json()
    assert REQUIRED_KEYS.issubset(body.keys())
    assert body["applicant_id"] == "anyid"


def test_get_risk_unknown_applicant_404(client, monkeypatch):
    monkeypatch.setattr(store, "get_profile", lambda aid: None)
    assert client.get("/risk/nope").status_code == 404


def test_post_risk_score(client):
    resp = client.post("/risk/score", json={"profile": WEAK.model_dump()})
    assert resp.status_code == 200
    assert REQUIRED_KEYS.issubset(resp.json().keys())


def test_get_risk_batch_sorted_desc(client, monkeypatch):
    applicants = [{"id": "s", "name": "Strong"}, {"id": "w", "name": "Weak"}]
    profiles = {"s": STRONG, "w": WEAK}
    monkeypatch.setattr(store, "list_applicants", lambda: applicants)
    monkeypatch.setattr(store, "get_profile", lambda aid: profiles.get(aid))
    body = client.get("/risk").json()
    assert set(body.keys()) == {"rows", "total", "scored", "avg_probability", "high_risk_pct"}
    probs = [r["probability"] for r in body["rows"]]
    assert probs == sorted(probs, reverse=True)
    assert body["scored"] == 2
    for row in body["rows"]:
        assert set(row.keys()) == {"applicant_id", "name", "probability", "band", "top_driver"}


def test_risk_list_scales_are_documented_and_distinct(client, monkeypatch):
    """avg_probability is a 0-1 PROBABILITY; high_risk_pct is already a 0-100
    PERCENTAGE. The two look interchangeable and are not: the UI multiplied
    both by 100, rendering "33.3" as "3330%" on the Elevated tile."""
    applicants = [{"id": "s", "name": "Strong"}, {"id": "w", "name": "Weak"}]
    profiles = {"s": STRONG, "w": WEAK}
    monkeypatch.setattr(store, "list_applicants", lambda: applicants)
    monkeypatch.setattr(store, "get_profile", lambda aid: profiles.get(aid))
    body = client.get("/risk").json()

    assert 0.0 <= body["avg_probability"] <= 1.0
    assert 0.0 <= body["high_risk_pct"] <= 100.0

    # high_risk_pct is the percentage of scored rows in the "high" band.
    high = sum(1 for r in body["rows"] if r["band"] == "high")
    expected = round(100.0 * high / body["scored"], 1)
    assert body["high_risk_pct"] == pytest.approx(expected, abs=0.05)


def test_model_card_keys(client):
    body = client.get("/risk/model-card").json()
    for key in ("name", "version", "description", "intended_use", "features",
                "excluded", "metrics", "bands", "limitations", "source"):
        assert key in body
    assert body["features"] == risk.FEATURE_ORDER
    assert all(set(e.keys()) == {"field", "reason"} for e in body["excluded"])


def test_calibrated_probability_is_never_exactly_0_or_1():
    """Isotonic calibration saturates to exactly 0.0/1.0 beyond its fitted
    range. A live high-burden/low-credit applicant scored exactly 1.0 on the
    deployed instance, which the UI renders as a 100% certainty that a real
    person will pay late -- not something a calibrated estimate can support."""
    extreme = ApplicantProfile(
        name="Stress Test",
        monthly_income=2100,
        desired_rent=1950,
        credit_score=500,
        employment_status="unemployed",
        evictions_count=3,
        late_payments_12mo=9,
        bankruptcies_count=2,
        monthly_debt_payments=900,
        savings_balance=0,
    )
    for profile in (extreme, STRONG, WEAK):
        p = risk.predict(profile)["probability"]
        assert 0.0 < p < 1.0, f"degenerate probability {p}"
