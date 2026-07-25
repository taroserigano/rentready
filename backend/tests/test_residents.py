"""Tests for the Residents stack — synthetic generation, multi-HEAD risk
scoring, property/portfolio health, and the residents chat agent.

Runs fully offline and deterministic (conftest forces the hash embedder + a
throwaway DB). The residents model artifact may or may not be present; tests
that must exercise the transparent heuristic force ``_model()`` to None, and
tests that assert head SHAPES accept either the model or the heuristic source.

Covers: RNG-seeded generation determinism (byte-identical ledgers), ledger
realism (length == min(tenure, 60), contiguity, no snapshot leakage, internal
consistency), every HEAD's payload contract, legacy aliases + name passthrough
+ heads-subset filter, structural feature exclusion, serve-time monotone clamps
+ heuristic monotonicity, predict_bulk == predict_resident, property/portfolio
health (score/grade/drivers/ordering), the never-raises contract on degenerate
residents, and the chat router / tools / answer / SSE stream (grounded + rules
fallback when the LLM is disabled).
"""

import re
from datetime import date

import pytest

import generate_residents as gen
import llm as llm_module
import residents_chat as rc
import residents_risk as rr
from models import LedgerEntry, Resident
from settings import settings as app_settings


# ---------------------------------------------------------------------------
# Offline determinism: force the residents-chat LLM off so the rules path is
# taken deterministically (conftest patches eligibility/graphrag, NOT
# residents_chat, whose get_langchain_llm is bound at import in its namespace).
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _no_chat_llm(monkeypatch):
    monkeypatch.setattr(rc, "get_langchain_llm", lambda: None)


@pytest.fixture
def force_heuristic(monkeypatch):
    """Force every head onto its transparent heuristic and keep the prediction
    cache from leaking model-based results across tests."""
    rr._clear_prediction_caches()
    monkeypatch.setattr(rr, "_model", lambda: None)
    yield
    rr._clear_prediction_caches()


def test_model_prunes_only_the_invalid_head_not_the_whole_bundle(monkeypatch, tmp_path):
    """Regression: _model() used to return None for the ENTIRE bundle if any
    ONE head's feature_order drifted from code (or was missing its required
    model key), reverting all 19 heads to heuristics over a single stale
    head. It must now prune just that head's model key — every other head
    stays servable, and the bad head's own metrics/feature_order survive for
    the model card, only its predictor is gone."""
    import joblib

    required = {
        "binary": "calibrated_model", "multiclass": "calibrated_model",
        "count": "regressor", "regression": "regressor", "survival": "hazard_model",
    }
    good_spec = rr.HEADS[0]
    bad_spec = rr.HEADS[1]
    good_key = required[good_spec["kind"]]
    bad_key = required[bad_spec["kind"]]

    fake_bundle = {
        "schema": rr.BUNDLE_SCHEMA,
        "heads": {
            good_spec["name"]: {
                "feature_order": good_spec["feature_order"],
                good_key: "FAKE_MODEL_OBJECT",
                "metrics": {"auc": 0.81},
            },
            bad_spec["name"]: {
                "feature_order": ["totally", "wrong", "contract"],  # drifted
                bad_key: "FAKE_MODEL_OBJECT",
                "metrics": {"auc": 0.77},  # must survive the prune
            },
        },
    }

    artifact = tmp_path / "fake_bundle.joblib"
    artifact.write_bytes(b"placeholder")  # only needs .exists() to be True
    monkeypatch.setattr(rr, "ARTIFACT_PATH", artifact)
    monkeypatch.setattr(joblib, "load", lambda path: fake_bundle)
    rr._model.cache_clear()
    try:
        bundle = rr._model()
        assert bundle is not None  # the whole bundle is NOT invalidated

        good_sub = bundle["heads"][good_spec["name"]]
        assert good_key in good_sub  # untouched, still servable

        bad_sub = bundle["heads"][bad_spec["name"]]
        assert bad_key not in bad_sub  # stripped -> heuristic for this head only
        assert bad_sub["metrics"] == {"auc": 0.77}  # informational data kept
    finally:
        rr._model.cache_clear()


# ---------------------------------------------------------------------------
# Resident builders (in-memory; resident_id="" so nothing is cached)
# ---------------------------------------------------------------------------
def _entry(period, status="paid", days_late=0, balance_after=0.0,
           amount_paid=1000.0, rent_charged=1000.0):
    return {
        "period": period,
        "rent_charged": rent_charged,
        "amount_paid": amount_paid,
        "paid_date": None,
        "days_late": days_late,
        "late_fee": 0.0 if status == "paid" else 60.0,
        "on_time": status == "paid",
        "status": status,
        "balance_after": balance_after,
        "notices_sent": 0 if status == "paid" else 1,
        "notice_responded": 0,
    }


def _months(n, status="paid", days_late=0, balance_after=0.0, amount_paid=1000.0):
    """n contiguous ledger months ending at the snapshot month (2026-07)."""
    out = []
    y, m = 2026, 7
    for _ in range(n):
        out.append(_entry(f"{y:04d}-{m:02d}", status, days_late, balance_after, amount_paid))
        m -= 1
        if m == 0:
            m, y = 12, y - 1
    return list(reversed(out))


def make_resident(ledger, resident_id="", **overrides):
    base = dict(
        resident_id=resident_id, property_id="PROP-041", name="Test Person",
        unit_id="PROP-041-U001", unit_bedrooms=2, base_rent=1000.0,
        monthly_income=4000.0, other_income_monthly=0.0, income_verified=True,
        autopay_enrolled=False, deposit_held=1000.0,
        move_in_date="2024-07-01", lease_start="2024-07-01", lease_end="2026-12-01",
        lease_term_months=12, prior_renewals=0,
        maintenance_requests_12mo=1, complaints_12mo=0, portal_logins_90d=8,
        ledger=ledger,
    )
    base.update(overrides)
    return base


CLEAN = make_resident(_months(24, "paid"))
# A clearly-worse resident: every month missed, growing balance, thin income.
BAD = make_resident(
    [_entry(e["period"], "missed", 30, 1000.0 * (i + 1), 0.0)
     for i, e in enumerate(_months(24))],
    monthly_income=2500.0, autopay_enrolled=False, income_verified=False,
)


# ---------------------------------------------------------------------------
# Shared constants sanity
# ---------------------------------------------------------------------------
def test_snapshot_is_pinned_no_wallclock():
    assert rr.RESIDENT_SNAPSHOT == date(2026, 7, 1)
    assert rr.HISTORY_MONTHS == 60
    assert rr.FUTURE_HORIZON_MONTHS == 15
    assert gen.FUTURE_MONTHS == rr.FUTURE_HORIZON_MONTHS


# ===========================================================================
# 1. GENERATION DETERMINISM
# ===========================================================================
def test_generate_is_byte_identical_across_runs():
    r1, l1 = gen.generate(n_per_property=2)
    r2, l2 = gen.generate(n_per_property=2)
    assert gen.ledger_signature(r1) == gen.ledger_signature(r2)
    assert r1 == r2          # every serialized field, not just the ledger
    assert l1 == l2          # labels are deterministic too


def test_generated_residents_have_display_name_and_ids():
    residents, _ = gen.generate(n_per_property=2)
    assert residents, "expected some residents"
    for r in residents:
        assert r["name"], "every resident needs a display name"
        assert r["resident_id"].startswith("RES-")
        assert r["property_id"] in rr.RESIDENT_PROPERTY_IDS


# ===========================================================================
# 2. LEDGER REALISM
# ===========================================================================
def test_ledger_length_equals_min_tenure_60():
    residents, _ = gen.generate(n_per_property=3)
    for r in residents:
        tenure = rr._months_between(rr._to_date(r["move_in_date"]), rr.RESIDENT_SNAPSHOT)
        assert len(r["ledger"]) == min(tenure, rr.HISTORY_MONTHS)


def test_short_tenure_yields_short_ledger():
    residents, _ = gen.generate(n_per_property=4)
    # The shortest-tenure resident must have a correspondingly short ledger.
    shortest = min(
        residents,
        key=lambda r: rr._months_between(rr._to_date(r["move_in_date"]), rr.RESIDENT_SNAPSHOT),
    )
    tenure = rr._months_between(rr._to_date(shortest["move_in_date"]), rr.RESIDENT_SNAPSHOT)
    assert len(shortest["ledger"]) == min(tenure, rr.HISTORY_MONTHS)
    assert len(shortest["ledger"]) < rr.HISTORY_MONTHS  # genuinely short


def test_ledger_contiguous_ends_at_snapshot_no_leakage():
    residents, _ = gen.generate(n_per_property=2)
    for r in residents:
        led = r["ledger"]
        months = [tuple(int(x) for x in e["period"].split("-")) for e in led]
        # contiguous month-over-month
        for a, b in zip(months, months[1:]):
            nxt = gen._add_months(date(a[0], a[1], 1), 1)
            assert (nxt.year, nxt.month) == b
        # ends exactly at the snapshot month; nothing dated after the snapshot
        assert months[-1] == (rr.RESIDENT_SNAPSHOT.year, rr.RESIDENT_SNAPSHOT.month)
        for (y, m) in months:
            assert date(y, m, 1) <= rr.RESIDENT_SNAPSHOT


def test_ledger_entries_internally_consistent():
    residents, _ = gen.generate(n_per_property=3)
    for r in residents:
        for e in r["ledger"]:
            # validates the committed schema (balance>=0, amounts>=0, enum status)
            le = LedgerEntry(**e)
            assert le.balance_after >= 0.0
            assert le.on_time == (le.status == "paid")
            if le.status == "paid":
                assert le.days_late == 0 and le.late_fee == 0.0
            else:
                assert le.days_late >= 1


# ===========================================================================
# 3. HEADS — every head produces a valid, kind-correct payload
# ===========================================================================
def _assert_head_valid(name, h):
    spec = rr.HEADS_BY_NAME[name]
    assert h["kind"] == spec["kind"]
    assert h["family"] == spec["family"]
    assert h["confidence"] in ("low", "high")
    assert "reason_codes" in h
    if h.get("band") == "not_applicable":
        # eligibility-gated + ineligible: values blanked, never raised.
        if spec["kind"] == "binary":
            assert h["probability"] is None
        return
    kind = spec["kind"]
    if kind == "binary":
        assert 0.0 <= h["probability"] <= 1.0
        assert h["band"] in ("low", "medium", "high")
        assert len(h["range"]) == 2 and h["range"][0] <= h["range"][1]
    elif kind in ("count", "regression"):
        assert h["expected"] >= 0.0
        assert len(h["interval"]) == 2 and h["interval"][0] <= h["interval"][1]
        if spec["family"] == "arrears":
            assert h["expected_balance"] >= 0.0
    elif kind == "multiclass":
        probs = h["class_probs"]
        assert set(probs.keys()) == set(rr.DELINQ_BUCKETS)
        assert abs(sum(probs.values()) - 1.0) < 0.01
        assert h["predicted_bucket"] in rr.DELINQ_BUCKETS
    elif kind == "survival":
        curve = h["survival_curve"]
        assert curve and all(0.0 <= s <= 1.0 for s in curve)
        # survival is non-increasing
        assert all(curve[i] >= curve[i + 1] - 1e-9 for i in range(len(curve) - 1))
        assert h["expected_months"] >= 0.0
        assert h["median_months"] is None or isinstance(h["median_months"], int)


def _cure_eligible_resident():
    # Carries a live balance -> cure heads applicable; also 30+ severity active.
    led = _months(18, "partial", 20, 500.0, amount_paid=500.0)
    return make_resident(led, base_rent=1000.0)


def _churn_eligible_resident():
    # Lease ends within 6 months of the snapshot -> churn heads applicable.
    return make_resident(_months(24, "paid"), lease_end="2026-10-01")


def test_every_head_valid_on_representative_residents():
    for res in (CLEAN, BAD, _cure_eligible_resident(), _churn_eligible_resident()):
        pred = rr.predict_resident(res)
        assert set(pred["heads"].keys()) == set(rr.HEAD_NAMES)
        for name, h in pred["heads"].items():
            _assert_head_valid(name, h)


def test_every_head_valid_on_heuristic_path(force_heuristic):
    for res in (CLEAN, BAD, _cure_eligible_resident(), _churn_eligible_resident()):
        pred = rr.predict_resident(res)
        for name, h in pred["heads"].items():
            _assert_head_valid(name, h)
            # heuristics never return degenerate certainties for applicable binaries
            if rr.HEADS_BY_NAME[name]["kind"] == "binary" and h.get("probability") is not None:
                assert 0.0 < h["probability"] < 1.0


def test_cure_heads_applicable_when_in_arrears():
    pred = rr.predict_resident(_cure_eligible_resident())
    assert pred["heads"]["p_cure_6m"]["band"] != "not_applicable"
    assert pred["heads"]["p_cure_6m"]["probability"] is not None
    assert pred["heads"]["months_to_cure"]["kind"] == "survival"


def test_cure_not_applicable_when_no_balance():
    pred = rr.predict_resident(CLEAN)
    assert pred["heads"]["p_cure_6m"]["band"] == "not_applicable"
    assert pred["heads"]["p_cure_6m"]["probability"] is None


def test_cure_eligibility_matches_the_training_label_gate():
    """Regression: serving used `current_balance > 0.0` while the training
    labeler (generate_residents.CURE_EPS gate) required `> CURE_EPS` ($1) to
    ever emit a cure-head label — a resident with a balance in (0, CURE_EPS]
    was scored by a model that saw zero training rows in that range. The two
    gates must use the identical threshold."""
    assert gen.CURE_EPS == rr.CURE_EPS
    just_under = {"current_balance": rr.CURE_EPS}
    just_over = {"current_balance": rr.CURE_EPS + 0.01}
    assert rr._elig_cure(just_under) is False
    assert rr._elig_cure(just_over) is True


def test_churn_applicable_within_horizon_else_not():
    near = rr.predict_resident(_churn_eligible_resident())
    assert near["churn"]["band"] != "not_applicable"
    assert near["churn"]["probability"] is not None
    far = rr.predict_resident(make_resident(_months(24), lease_end="2030-01-01"))
    assert far["churn"]["band"] == "not_applicable"
    assert far["churn"]["probability"] is None


def test_legacy_aliases_present_and_shaped():
    pred = rr.predict_resident(BAD)
    assert set(("late", "arrears", "churn", "serious")).issubset(pred.keys())
    assert 0.0 <= pred["late"]["probability"] <= 1.0
    assert pred["arrears"]["expected_balance"] >= 0.0
    assert pred["serious"]["routes_to_review"] is True
    # legacy alias map resolves to the real head names
    assert rr.LEGACY_ALIAS == {"late": "late_3m", "arrears": "arrears_3m",
                               "churn": "churn", "serious": "serious"}


def test_name_and_ids_passthrough():
    pred = rr.predict_resident(make_resident(_months(12), name="Zebra Q", property_id="PROP-008"))
    assert pred["name"] == "Zebra Q"
    assert pred["property_id"] == "PROP-008"
    assert pred["snapshot_date"] == rr.RESIDENT_SNAPSHOT.isoformat()


def test_heads_subset_filter_includes_alias_heads():
    pred = rr.predict_resident(CLEAN, heads=["late_12m"])
    got = set(pred["heads"].keys())
    # requested head plus the always-present alias heads
    assert got == {"late_12m"} | rr._ALIAS_HEADS
    # the legacy top-level aliases are still populated
    assert pred["late"]["probability"] is not None


def test_families_block_matches_registry():
    pred = rr.predict_resident(CLEAN)
    assert pred["families"] == {fam: list(names) for fam, names in rr.FAMILIES.items()}


def test_accepts_pydantic_resident_instance():
    r = Resident(
        resident_id="RES-TEST", property_id="PROP-041", unit_id="U1",
        base_rent=1000.0, lease_start="2024-07-01", lease_end="2026-12-01",
        move_in_date="2024-07-01",
        ledger=[LedgerEntry(**e) for e in _months(12, "paid")],
    )
    pred = rr.predict_resident(r)
    assert 0.0 <= pred["late"]["probability"] <= 1.0


# ===========================================================================
# 4. STRUCTURAL EXCLUSION
# ===========================================================================
def test_excluded_fields_do_not_change_features():
    base = make_resident(_months(18, "paid_late", 5))
    mutated = dict(base)
    mutated.update(
        property_id="PROP-999", name="Totally Different", unit_id="PROP-999-U9",
        unit_bedrooms=5, deposit_held=99999.0,
    )
    assert rr.extract_resident_features(mutated) == rr.extract_resident_features(base)


def test_excluded_fields_do_not_change_score():
    base = make_resident(_months(18, "paid_late", 5))
    mutated = dict(base, property_id="PROP-999", name="Other", unit_id="ZZ", unit_bedrooms=4)
    pb, pm = rr.predict_resident(base), rr.predict_resident(mutated)
    assert pb["late"]["probability"] == pm["late"]["probability"]
    assert pb["serious"]["probability"] == pm["serious"]["probability"]
    assert pb["heads"] == pm["heads"]


def test_excluded_features_documented_on_model_card():
    card = rr.model_card()
    assert all(set(e.keys()) == {"field", "reason"} for e in card["excluded"])
    assert card["excluded"] == rr.EXCLUDED_FEATURES


def test_reason_codes_cite_only_allowed_features():
    pred = rr.predict_resident(BAD)
    superset = set(rr.FEATURE_ORDER_ARREARS) | set(rr.FEATURE_ORDER_CHURN)
    for h in pred["heads"].values():
        for rc_item in h.get("reason_codes", []):
            assert rc_item["feature"] in superset
            assert rc_item["direction"] in ("increases", "decreases")


# ===========================================================================
# 5. MONOTONICITY & SERVE-TIME CLAMPS
# ===========================================================================
def test_worse_resident_scores_higher_late_and_serious():
    good, bad = rr.predict_resident(CLEAN), rr.predict_resident(BAD)
    assert bad["late"]["probability"] >= good["late"]["probability"]
    assert bad["serious"]["probability"] >= good["serious"]["probability"]


def test_worse_resident_scores_higher_heuristic(force_heuristic):
    good, bad = rr.predict_resident(CLEAN), rr.predict_resident(BAD)
    assert bad["late"]["probability"] > good["late"]["probability"]
    assert bad["serious"]["probability"] > good["serious"]["probability"]


def test_late_horizon_probabilities_non_decreasing():
    for res in (CLEAN, BAD, _cure_eligible_resident()):
        h = rr.predict_resident(res)["heads"]
        probs = [h[n]["probability"] for n in ("late_1m", "late_3m", "late_6m", "late_12m")]
        assert all(probs[i] <= probs[i + 1] + 1e-9 for i in range(len(probs) - 1)), probs


def test_severity_threshold_probabilities_non_increasing():
    for res in (CLEAN, BAD, _cure_eligible_resident()):
        h = rr.predict_resident(res)["heads"]
        probs = [h[n]["probability"] for n in ("p_30d_12m", "p_60d_12m", "p_90d_12m")]
        assert all(probs[i] >= probs[i + 1] - 1e-9 for i in range(len(probs) - 1)), probs


def test_band_thresholds_match_edges():
    lo, hi = rr._BAND_EDGES["late_3m"]
    assert rr._band("late_3m", lo - 1e-6) == "low"
    assert rr._band("late_3m", lo) == "medium"
    assert rr._band("late_3m", hi - 1e-6) == "medium"
    assert rr._band("late_3m", hi) == "high"


# ===========================================================================
# 6. predict_bulk == predict_resident
# ===========================================================================
def test_bulk_matches_per_resident_numerically():
    residents = rr.load_residents()[:12]
    assert residents, "committed residents.json should be loadable"
    bulk = rr.predict_bulk(residents, heads=rr.BULK_HEADS)
    assert len(bulk) == len(residents)
    for r, b in zip(residents, bulk):
        single = rr.predict_resident(r, with_reasons=False, heads=rr.BULK_HEADS)
        for alias in ("late", "serious"):
            assert b[alias]["probability"] == single[alias]["probability"]
        assert b["arrears"]["expected_balance"] == single["arrears"]["expected_balance"]
        for name in rr.BULK_HEADS:
            hb, hs = b["heads"][name], single["heads"][name]
            assert hb.get("probability") == hs.get("probability")
            assert hb.get("expected") == hs.get("expected")


def test_bulk_empty_list_returns_empty():
    assert rr.predict_bulk([]) == []
    assert rr.predict_residents([]) == []


# ===========================================================================
# 7. PROPERTY / PORTFOLIO HEALTH
# ===========================================================================
def test_property_health_shape_and_range():
    pid = rr.RESIDENT_PROPERTY_IDS[0]
    h = rr.property_health(pid)
    assert h["property_id"] == pid
    assert 0.0 <= h["score"] <= 100.0
    assert h["grade"] in ("A", "B", "C", "D", "F")
    assert h["resident_count"] > 0
    assert set(h["components"].keys()) == set(rr._HEALTH_WEIGHTS.keys())
    for comp in h["components"].values():
        assert 0.0 <= comp["value"] <= 1.0
    assert isinstance(h["drivers"], list)
    assert h["top_driver"]


def test_portfolio_health_sorted_best_to_worst():
    ranked = rr.portfolio_health()
    assert len(ranked) == len(rr.RESIDENT_PROPERTY_IDS)
    scores = [h["score"] for h in ranked]
    assert scores == sorted(scores, reverse=True)
    for h in ranked:
        assert 0.0 <= h["score"] <= 100.0
        assert h["grade"] in ("A", "B", "C", "D", "F")


def test_grade_boundaries():
    # Graded off the ROUNDED (displayed) score: A>=85, B>=75, C>=65, D>=55, else F.
    assert rr._grade(85) == "A" and rr._grade(84.6) == "A" and rr._grade(84) == "B"
    assert rr._grade(75) == "B" and rr._grade(74) == "C"
    assert rr._grade(65) == "C" and rr._grade(64) == "D"
    assert rr._grade(55) == "D" and rr._grade(54) == "F"
    assert rr._grade(0) == "F" and rr._grade(100) == "A"


def test_property_health_no_residents_is_perfect():
    h = rr.property_health("PROP-DOES-NOT-EXIST")
    assert h["resident_count"] == 0
    assert h["score"] == 100.0
    assert h["grade"] == "A"


# ===========================================================================
# 8. NEVER-RAISES on degenerate residents
# ===========================================================================
@pytest.mark.parametrize("res", [
    make_resident([]),                                             # empty ledger
    make_resident([], move_in_date="2026-07-01"),                  # zero tenure
    {"resident_id": "", "ledger": []},                             # missing everything
    {"resident_id": "", "ledger": [], "base_rent": 0, "monthly_income": 0},
    make_resident(_months(1, "missed", 30, 1000.0, 0.0)),          # single trouble month
])
def test_predict_never_raises_on_degenerate(res):
    pred = rr.predict_resident(res)
    assert 0.0 <= pred["late"]["probability"] <= 1.0
    assert pred["arrears"]["expected_balance"] >= 0.0
    assert set(pred["heads"].keys())  # full head catalog present
    for name, h in pred["heads"].items():
        _assert_head_valid(name, h)


def test_predict_never_raises_on_degenerate_heuristic(force_heuristic):
    pred = rr.predict_resident(make_resident([]))
    assert 0.0 < pred["late"]["probability"] < 1.0


def test_predict_residents_skips_bad_entries():
    out = rr.predict_residents([make_resident(_months(6)), make_resident([])])
    assert len(out) == 2


# ===========================================================================
# 9. RESIDENTS CHAT — router / tools / answer / stream
# ===========================================================================
@pytest.mark.parametrize("question,intent", [
    ("Why is this resident risky?", "explain"),
    ("What's driving their risk?", "explain"),
    ("How likely are they to pay late next quarter?", "horizon"),
    ("What's the chance they pay late next year?", "horizon"),
    ("How often will they pay late?", "frequency"),
    ("How many months will they be late?", "frequency"),
    ("How bad could it get? 30 days late?", "severity"),
    ("What's their worst delinquency going to be?", "severity"),
    ("What is their expected balance?", "arrears"),
    ("How much will they owe?", "arrears"),
    ("Will they clear their balance?", "cure"),
    ("Will they catch up on what they owe?", "cure"),
    ("Will they renew their lease?", "retention"),
    ("Are they going to move out?", "retention"),
    ("Which properties are healthiest?", "property_health"),
    ("Which apartments need attention?", "property_health"),
    ("which residents are most at risk", "at_risk_residents"),
    ("who is behind on rent", "at_risk_residents"),
    ("show me residents at risk", "at_risk_residents"),
    ("riskiest residents", "at_risk_residents"),
    ("who is likely to churn", "at_risk_residents"),
    ("How does this resident compare to the portfolio?", "compare"),
    ("Is age used by the model?", "governance"),
    ("Do you use race or location?", "governance"),
    ("What features does the model use?", "governance"),
    ("hello there", "general"),
    # Regression: "quater" (missing 'r') used to fall through every regex and
    # default to property_health instead of routing on the time horizon typed.
    ("tell me prediction for next quater", "horizon"),
    ("what about next moth", "horizon"),
])
def test_route_maps_questions_to_intents(question, intent):
    assert rc.route(question) == intent


def test_quarter_typo_still_picks_the_3_month_head():
    """Regression: _parse_horizon re-scans the raw question for "quarter" to
    pick which head answers the question — the typo fix must apply to that
    text too, not just inside route(), or the intent would be right but the
    answer would still cite the wrong (12-month) head."""
    assert rc._parse_horizon(rc._fix_typos("prediction for next quater")) == "late_3m"


@pytest.mark.parametrize("question", rc._FOLLOWUPS["property_health"])
def test_property_health_followups_route_back_to_property_health(question):
    """The app's own canned property_health follow-up chips must round-trip back
    to the property_health intent (regression: _PROPERTY_HEALTH_RE was widened to
    catch singular 'property', 'needs the most attention', 'health score', and
    the 'lowest-scoring … dragging … down' phrasings)."""
    assert rc.route(question) == "property_health"


def test_route_scope_defaults():
    assert rc.route("tell me about them", resident_id="RES-0001") == "explain"
    assert rc.route("tell me about it", property_id="PROP-041") == "property_health"
    assert rc.route("hi") == "general"


def test_score_resident_unknown_returns_none():
    assert rc.score_resident(None) is None
    assert rc.score_resident("RES-DOES-NOT-EXIST") is None


def test_answer_grounded_numbers_come_from_heads():
    rid = rr.load_residents()[0]["resident_id"]
    pred = rr.predict_resident(rr.get_resident(rid))
    a = rc.answer("Why is this resident risky?", resident_id=rid)
    assert a["source"] == "rules"
    assert a["intent"] == "explain"
    # every percentage the answer states is one computed from a head payload
    late12_pct = rc._pct(pred["heads"]["late_12m"]["probability"])
    serious_pct = rc._pct(pred["heads"]["serious"]["probability"])
    assert late12_pct in a["answer"]
    assert serious_pct in a["answer"]
    assert "[1]" in a["answer"]  # grounded citation
    # the artifact carries the exact head payloads the answer cites
    assert a["artifact"]["kind"] == "resident"
    assert a["artifact"]["resident_id"] == rid


def test_answer_unknown_resident_deflects_gracefully():
    a = rc.answer("Why are they risky?", resident_id="RES-NOPE")
    assert a["source"] == "rules"
    assert "don't have predictions" in a["answer"]
    assert a["artifact"]["kind"] == "none"


def test_answer_governance_is_grounded_and_safe():
    a = rc.answer("Is age or race ever used?")
    assert a["intent"] == "governance"
    assert a["source"] == "rules"
    assert "NOT used" in a["answer"]
    # never recommends an automated adverse action
    lowered = a["answer"].lower()
    assert "evict" not in lowered and "deny" not in lowered


def test_answer_property_health_ranking():
    a = rc.answer("Which properties are healthiest?")
    assert a["intent"] == "property_health"
    assert a["artifact"]["kind"] == "property_health"
    assert a["artifact"]["healthiest"] is not None
    assert a["artifact"]["count"] == len(rr.RESIDENT_PROPERTY_IDS)


def test_answer_at_risk_residents_ranking():
    """Regression: portfolio-wide questions like 'which residents are most at
    risk' used to fall through to the generic 'general' intent and claim no
    resident data was available, even though a ranking is exactly what
    residents_risk.predict_bulk/load_residents already support."""
    a = rc.answer("which residents are most at risk")
    assert a["intent"] == "at_risk_residents"
    assert a["source"] == "rules"
    assert "don't have enough information" not in a["answer"].lower()
    assert "no actual resident data" not in a["answer"].lower()
    # Every percentage cited must come from a real head prediction, not be
    # invented -- spot check the top-ranked resident's own late_12m probability.
    residents = rr.load_residents()
    preds = rr.predict_bulk(residents, heads=rr.BULK_HEADS)
    best = max(
        zip(residents, preds),
        key=lambda rp: (rp[1] or {}).get("late", {}).get("probability") or 0.0,
    )
    top_pct = rc._pct(best[1]["late"]["probability"])
    assert top_pct in a["answer"]
    assert best[0]["name"] in a["answer"]
    assert len(a["sources"]) == 5


def test_answer_at_risk_residents_scoped_to_property():
    property_id = rr.load_residents()[0]["property_id"]
    a = rc.answer("who is behind on rent", property_id=property_id)
    assert a["intent"] == "at_risk_residents"
    assert all(
        s.get("resident_id", "").startswith("RES-") for s in a["sources"]
    )


def test_answer_at_risk_residents_churn_framed_uses_churn_metric():
    """Regression: a churn/renewal-framed ranking question ('at risk of
    churning') used to be routed to the same at_risk_residents intent but
    silently ranked and labeled everyone by their LATE-PAYMENT probability
    instead -- the right intent, the wrong metric, cited with total
    confidence. It must rank/label by the churn head instead."""
    a = rc.answer("which residents are at risk of churning?")
    assert a["intent"] == "at_risk_residents"
    assert "churn" in a["answer"].lower()
    assert "paying late" not in a["answer"].lower()

    residents = rr.load_residents()
    preds = rr.predict_bulk(residents, heads=rr.BULK_HEADS)
    best = max(
        zip(residents, preds),
        key=lambda rp: (rp[1] or {}).get("churn", {}).get("probability") or 0.0,
    )
    top_pct = rc._pct(best[1]["churn"]["probability"])
    assert top_pct in a["answer"]
    assert best[0]["name"] in a["answer"]


def test_answer_at_risk_residents_late_framed_still_uses_late_metric():
    """Regression guard for the fix above: a plain late-payment framing must
    keep using the late head, not accidentally flip to churn."""
    a = rc.answer("which residents are most likely to fall behind on rent?")
    assert a["intent"] == "at_risk_residents"
    assert "paying late" in a["answer"].lower()
    assert "non-renewal" not in a["answer"].lower()


def test_answer_never_raises_on_garbage():
    a = rc.answer("", resident_id=None, property_id=None)
    assert isinstance(a["answer"], str) and a["answer"]
    assert a["source"] == "rules"


def test_answer_stream_meta_token_done_framing():
    rid = rr.load_residents()[0]["resident_id"]
    events = list(rc.answer_stream("Why is this resident risky?", resident_id=rid))
    types = [e["type"] for e in events]
    assert types[0] == "meta"
    assert types[-1] == "done"
    assert "token" in types
    meta = events[0]
    assert set(("scope", "resident_id", "intent", "follow_ups", "artifact")).issubset(meta.keys())
    assert events[-1]["source"] == "rules"
    # the streamed token text equals the deterministic grounded answer
    token_text = "".join(e["text"] for e in events if e["type"] == "token")
    assert token_text
    assert "[1]" in token_text


def test_answer_stream_matches_answer_artifact():
    rid = rr.load_residents()[0]["resident_id"]
    a = rc.answer("How likely are they to pay late next year?", resident_id=rid)
    meta = next(e for e in rc.answer_stream("How likely are they to pay late next year?",
                                            resident_id=rid) if e["type"] == "meta")
    assert meta["artifact"] == a["artifact"]
    assert meta["intent"] == a["intent"]


# ===========================================================================
# 10. late_count_3m HEAD (quarterly late-payment forecast) — unit tests
# ===========================================================================
def test_late_count_3m_registered_in_head_catalog():
    spec = rr.HEADS_BY_NAME["late_count_3m"]
    assert spec["family"] == "frequency"
    assert spec["kind"] == "count"
    assert spec["feature_order"] == rr.FEATURE_ORDER_BASE
    assert spec["band_edges"] is None  # served as expected+interval, not a band
    assert "late_count_3m" in rr.HEAD_NAMES


def test_late_count_3m_valid_on_representative_residents():
    for res in (CLEAN, BAD, rr.load_residents()[0]):
        pred = rr.predict_resident(res)
        _assert_head_valid("late_count_3m", pred["heads"]["late_count_3m"])


def test_late_count_3m_valid_on_heuristic_path(force_heuristic):
    for res in (CLEAN, BAD):
        pred = rr.predict_resident(res)
        _assert_head_valid("late_count_3m", pred["heads"]["late_count_3m"])


def test_late_count_3m_worse_resident_scores_higher(force_heuristic):
    """A resident who is missing every recent payment must never get a LOWER
    expected quarterly late-count than a spotless one (direction, not an exact
    value, since the DGP/model may not match the heuristic bit-for-bit)."""
    clean = rr.predict_resident(CLEAN)["heads"]["late_count_3m"]["expected"]
    bad = rr.predict_resident(BAD)["heads"]["late_count_3m"]["expected"]
    assert bad > clean


def test_late_count_3m_never_raises_on_degenerate_heuristic(force_heuristic):
    pred = rr.predict_resident(make_resident([]))
    h = pred["heads"]["late_count_3m"]
    assert h["expected"] >= 0.0
    assert len(h["interval"]) == 2


def test_late_count_3m_included_in_bulk_and_matches_single():
    residents = rr.load_residents()[:12]
    bulk = rr.predict_bulk(residents, heads=["late_count_3m"])
    for r, b in zip(residents, bulk):
        single = rr.predict_resident(r, with_reasons=False, heads=["late_count_3m"])
        assert b["heads"]["late_count_3m"]["expected"] == single["heads"]["late_count_3m"]["expected"]


# ===========================================================================
# 10b. late_count_6m / late_count_9m HEADS (mid-year quarterly checkpoints)
# ===========================================================================
@pytest.mark.parametrize("head", ["late_count_6m", "late_count_9m"])
def test_late_count_mid_year_registered_in_head_catalog(head):
    spec = rr.HEADS_BY_NAME[head]
    assert spec["family"] == "frequency"
    assert spec["kind"] == "count"
    assert spec["feature_order"] == rr.FEATURE_ORDER_BASE
    assert spec["band_edges"] is None  # served as expected+interval, not a band
    assert head in rr.HEAD_NAMES


@pytest.mark.parametrize("head", ["late_count_6m", "late_count_9m"])
def test_late_count_mid_year_valid_on_representative_residents(head):
    for res in (CLEAN, BAD, rr.load_residents()[0]):
        pred = rr.predict_resident(res)
        _assert_head_valid(head, pred["heads"][head])


@pytest.mark.parametrize("head", ["late_count_6m", "late_count_9m"])
def test_late_count_mid_year_valid_on_heuristic_path(head, force_heuristic):
    for res in (CLEAN, BAD):
        pred = rr.predict_resident(res)
        _assert_head_valid(head, pred["heads"][head])


@pytest.mark.parametrize("head", ["late_count_6m", "late_count_9m"])
def test_late_count_mid_year_worse_resident_scores_higher(head, force_heuristic):
    """A resident who is missing every recent payment must never get a LOWER
    expected count than a spotless one (direction, not an exact value)."""
    clean = rr.predict_resident(CLEAN)["heads"][head]["expected"]
    bad = rr.predict_resident(BAD)["heads"][head]["expected"]
    assert bad > clean


@pytest.mark.parametrize("head", ["late_count_6m", "late_count_9m"])
def test_late_count_mid_year_never_raises_on_degenerate_heuristic(head, force_heuristic):
    pred = rr.predict_resident(make_resident([]))
    h = pred["heads"][head]
    assert h["expected"] >= 0.0
    assert len(h["interval"]) == 2


@pytest.mark.parametrize("head", ["late_count_6m", "late_count_9m"])
def test_late_count_mid_year_included_in_bulk_and_matches_single(head):
    residents = rr.load_residents()[:12]
    bulk = rr.predict_bulk(residents, heads=[head])
    for r, b in zip(residents, bulk):
        single = rr.predict_resident(r, with_reasons=False, heads=[head])
        assert b["heads"][head]["expected"] == single["heads"][head]["expected"]


@pytest.mark.parametrize("heads", [
    ("late_count_3m", "late_count_6m"),
    ("late_count_6m", "late_count_9m"),
    ("late_count_9m", "late_count_12m"),
])
def test_late_count_quarterly_checkpoints_trend_upward(heads):
    """The 4 quarterly checkpoints are a genuine cumulative-count timeline —
    a resident's expected count at a longer horizon should never be lower
    than at a shorter one (same property of the ground truth as arrears)."""
    shorter, longer = heads
    for res in (CLEAN, BAD, rr.load_residents()[0]):
        pred = rr.predict_resident(res)
        assert pred["heads"][longer]["expected"] >= pred["heads"][shorter]["expected"] - 1e-6


# ===========================================================================
# 11. QUARTERLY LATE-PAYMENT FORECAST (property/portfolio aggregate)
# ===========================================================================
@pytest.mark.parametrize("head", ["late_count_3m", "late_count_6m", "late_count_9m", "late_count_12m"])
def test_property_late_forecast_matches_sum_of_bulk_predictions(head):
    pid = rr.RESIDENT_PROPERTY_IDS[0]
    residents = [r for r in rr.load_residents() if r.get("property_id") == pid]
    assert residents, "fixture property should have residents"
    fc = rr.property_late_forecast(pid, head=head)
    assert fc["property_id"] == pid
    assert fc["resident_count"] == len(residents)

    # The aggregate must be the exact SUM of each resident's own prediction —
    # not a re-derived or approximated number.
    preds = rr.predict_bulk(residents, heads=[head])
    expected_sum = sum(p["heads"][head]["expected"] for p in preds)
    lo_sum = sum(p["heads"][head]["interval"][0] for p in preds)
    hi_sum = sum(p["heads"][head]["interval"][1] for p in preds)
    assert fc["expected"] == pytest.approx(round(expected_sum, 1), abs=0.15)
    assert fc["interval"][0] == pytest.approx(round(lo_sum, 1), abs=0.15)
    assert fc["interval"][1] == pytest.approx(round(hi_sum, 1), abs=0.15)
    assert fc["interval"][0] <= fc["expected"] <= fc["interval"][1]

    # top_contributors is sorted descending and drawn from the real residents.
    contributors = fc["top_contributors"]
    assert len(contributors) <= 5
    assert [c["expected"] for c in contributors] == sorted(
        (c["expected"] for c in contributors), reverse=True
    )
    resident_ids = {r["resident_id"] for r in residents}
    assert all(c["resident_id"] in resident_ids for c in contributors)


def test_portfolio_late_forecast_matches_sum_of_bulk_predictions():
    residents = rr.load_residents()
    fc = rr.portfolio_late_forecast()
    assert fc["resident_count"] == len(residents)
    preds = rr.predict_bulk(residents, heads=["late_count_3m"])
    expected_sum = sum(p["heads"]["late_count_3m"]["expected"] for p in preds)
    assert fc["expected"] == pytest.approx(round(expected_sum, 1), abs=1.0)
    assert fc["interval"][0] <= fc["expected"] <= fc["interval"][1]


def test_late_forecast_rejects_unsupported_head():
    # property_late_forecast never raises (guarded), but an unsupported head
    # is still rejected internally -- caught and degraded to the safe-empty
    # shape rather than silently returning a wrong number.
    pid = rr.RESIDENT_PROPERTY_IDS[0]
    fc = rr.property_late_forecast(pid, head="not_a_real_head")
    assert fc == {"property_id": pid, "resident_count": 0, "expected": 0.0,
                  "interval": [0.0, 0.0], "top_contributors": []}


def test_property_late_forecast_unknown_property_is_empty():
    fc = rr.property_late_forecast("PROP-DOES-NOT-EXIST")
    assert fc["property_id"] == "PROP-DOES-NOT-EXIST"
    assert fc["resident_count"] == 0
    assert fc["expected"] == 0.0
    assert fc["interval"] == [0.0, 0.0]
    assert fc["top_contributors"] == []


def test_late_forecast_never_raises_on_load_failure(monkeypatch):
    monkeypatch.setattr(rr, "load_residents", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    fc = rr.property_late_forecast("PROP-041")
    assert fc == {"property_id": "PROP-041", "resident_count": 0, "expected": 0.0,
                  "interval": [0.0, 0.0], "top_contributors": []}
    fc2 = rr.portfolio_late_forecast()
    assert fc2["resident_count"] == 0


# ===========================================================================
# 12. RESIDENTS CHAT — late_forecast intent (integration)
# ===========================================================================
@pytest.mark.parametrize("question", [
    "how many late payments could anticipate next quarter",
    "how many late payments do you expect next quarter",
    "how many residents will be late next quarter",
])
def test_late_forecast_questions_route_to_frequency(question):
    """These are exactly the phrasings a resident manager would type with no
    resident selected — route() must still classify them as 'frequency' (the
    per-resident intent name); it is _ResidentPlan that upgrades a resident-less
    'frequency' question to the property/portfolio aggregate."""
    assert rc.route(question) == "frequency"


def test_answer_late_forecast_property_scope():
    pid = rr.RESIDENT_PROPERTY_IDS[0]
    a = rc.answer("How many late payments could we anticipate next quarter?",
                  property_id=pid)
    assert a["intent"] == "late_forecast"
    assert a["source"] == "rules"
    assert a["artifact"]["kind"] == "late_forecast"
    assert a["artifact"]["scope"] == "property"
    assert a["artifact"]["property_id"] == pid

    fc = rr.property_late_forecast(pid)
    assert rc._num(fc["expected"]) in a["answer"]
    assert "[1]" in a["answer"]
    # the guard names eviction/denial only to DISCLAIM them ("never a basis
    # for eviction...") — never as an endorsed recommendation.
    assert "never a basis for eviction" in a["answer"].lower()
    assert "recommend eviction" not in a["answer"].lower()


def test_answer_late_forecast_portfolio_scope():
    a = rc.answer("How many late payments do you expect next quarter?")
    assert a["intent"] == "late_forecast"
    assert a["artifact"]["scope"] == "portfolio"
    fc = rr.portfolio_late_forecast()
    assert a["artifact"]["resident_count"] == fc["resident_count"]
    assert rc._num(fc["expected"]) in a["answer"]
    assert "[1]" in a["answer"]


def test_answer_late_forecast_empty_property_is_graceful():
    a = rc.answer("How many late payments next quarter?", property_id="PROP-DOES-NOT-EXIST")
    assert a["intent"] == "late_forecast"
    assert a["artifact"]["kind"] == "none"
    assert a["sources"] == []
    # no fabricated citation when there is nothing to cite
    assert "[1]" not in a["answer"]


def test_answer_frequency_resident_scope_still_includes_quarterly_count():
    """A resident-scoped frequency question must not regress: it should now
    ALSO surface the quarterly (late_count_3m) figure alongside the existing
    12-month one, not replace it."""
    rid = rr.load_residents()[0]["resident_id"]
    pred = rr.predict_resident(rr.get_resident(rid))
    a = rc.answer("How many late payments have they had, and how many next quarter?",
                  resident_id=rid)
    assert a["intent"] == "frequency"
    q_expected = rc._num(pred["heads"]["late_count_3m"]["expected"])
    assert q_expected in a["answer"]
    assert "late_count_3m" in a["artifact"]["heads"]


def test_answer_stream_late_forecast_meta_token_done():
    events = list(rc.answer_stream("How many late payments next quarter?",
                                   property_id=rr.RESIDENT_PROPERTY_IDS[0]))
    types = [e["type"] for e in events]
    assert types[0] == "meta" and types[-1] == "done"
    assert events[0]["intent"] == "late_forecast"
    token_text = "".join(e["text"] for e in events if e["type"] == "token")
    assert "[1]" in token_text


def test_answer_late_forecast_never_raises_on_garbage():
    a = rc.answer("how many late payments", resident_id=None, property_id=None)
    assert isinstance(a["answer"], str) and a["answer"]
    a2 = rc.answer("how many late payments", resident_id="RES-DOES-NOT-EXIST",
                   property_id="PROP-DOES-NOT-EXIST")
    assert isinstance(a2["answer"], str) and a2["answer"]


# ===========================================================================
# 13. LIVE LLM INTEGRATION (opt-in — hits the real Claude API)
#
# Every other test in this module runs through the `no_llm` autouse fixture
# (conftest.py), which forces the deterministic "rules" path so the suite is
# offline/fast/repeatable. This one test deliberately restores the REAL LLM
# for the residents chat agent to verify the full synthesis path end-to-end:
# that Claude, given only the numbered late_forecast context, states the
# correct aggregate number and cites it — not the routing/templating layer,
# which every test above already covers deterministically.
#
# Skips gracefully (not a failure) when no API key is configured, e.g. a CI
# runner without secrets. Run explicitly with:
#   pytest tests/test_residents.py -k test_live_llm
# ===========================================================================
@pytest.mark.skipif(
    not app_settings.anthropic_api_key,
    reason="no ANTHROPIC_API_KEY configured (.env); live-LLM test skipped",
)
def test_live_llm_late_forecast_grounds_the_real_number(monkeypatch):
    # Undo the module-level no_llm patch for just this test.
    monkeypatch.setattr(rc, "get_langchain_llm", llm_module.get_langchain_llm)

    pid = rr.RESIDENT_PROPERTY_IDS[0]
    fc = rr.property_late_forecast(pid)
    a = rc.answer("How many late payments could we anticipate next quarter?",
                  property_id=pid)

    assert a["intent"] == "late_forecast"
    if a["source"] != "anthropic":
        pytest.skip("LLM call did not succeed (network/proxy); rules fallback took over")

    # The real model must state the aggregate figure computed from the heads —
    # never invent its own number — and cite the source it came from.
    assert re.search(rf"\b{re.escape(rc._num(fc['expected']))}\b", a["answer"])
    assert "[1]" in a["answer"]
    # never an ENDORSED adverse action (naming eviction only to disclaim it,
    # e.g. "never a basis for eviction", is fine and expected).
    lowered = a["answer"].lower()
    assert "recommend eviction" not in lowered
    assert "should evict" not in lowered
    assert "you should deny" not in lowered


@pytest.mark.skipif(
    not app_settings.anthropic_api_key,
    reason="no ANTHROPIC_API_KEY configured (.env); live-LLM test skipped",
)
def test_live_llm_late_forecast_portfolio_scope_grounds_the_real_number(monkeypatch):
    """Same grounding contract as the property-scoped test above, but with NO
    property_id — the portfolio-wide aggregate path (``portfolio_late_forecast``),
    which the property-scoped test never exercises."""
    monkeypatch.setattr(rc, "get_langchain_llm", llm_module.get_langchain_llm)

    fc = rr.portfolio_late_forecast()
    a = rc.answer("How many late payments can we expect across the whole portfolio next quarter?")

    assert a["intent"] == "late_forecast"
    assert a["scope"] == "portfolio"
    if a["source"] != "anthropic":
        pytest.skip("LLM call did not succeed (network/proxy); rules fallback took over")

    assert re.search(rf"\b{re.escape(rc._num(fc['expected']))}\b", a["answer"])
    assert "[1]" in a["answer"]
    lowered = a["answer"].lower()
    assert "recommend eviction" not in lowered
    assert "should evict" not in lowered


@pytest.mark.skipif(
    not app_settings.anthropic_api_key,
    reason="no ANTHROPIC_API_KEY configured (.env); live-LLM test skipped",
)
def test_live_llm_late_forecast_multiturn_switches_horizon(monkeypatch):
    """A follow-up question in the SAME conversation switches the window from
    quarterly to 12-month — this is also the exact regression scenario a user
    hit live (asking a 12-month question got the quarterly number silently, or
    a fabricated one, before the horizon-aware fix). The two grounded numbers
    must differ (proving the second turn actually recomputed on the new
    window, rather than repeating the first answer) and each must match its
    own head's real aggregate.

    NOTE: the follow-up restates "late payments" rather than just saying "what
    about 12 months instead?" — routing is a per-turn regex classifier with no
    memory of the PRIOR intent, so a bare horizon-only follow-up with no
    frequency keyword of its own currently falls through to property_health
    instead of continuing the late-forecast thread. That's a real, separate
    conversational-continuity gap (worth its own fix), not something this test
    should paper over by asserting on it as if it worked."""
    monkeypatch.setattr(rc, "get_langchain_llm", llm_module.get_langchain_llm)

    pid = rr.RESIDENT_PROPERTY_IDS[0]
    fc_q = rr.property_late_forecast(pid, head="late_count_3m")
    fc_y = rr.property_late_forecast(pid, head="late_count_12m")
    assert rc._num(fc_q["expected"]) != rc._num(fc_y["expected"]), (
        "test fixture needs a property where the two windows print differently"
    )

    a1 = rc.answer("How many late payments could we anticipate next quarter?", property_id=pid)
    assert a1["intent"] == "late_forecast"
    if a1["source"] != "anthropic":
        pytest.skip("LLM call did not succeed (network/proxy); rules fallback took over")

    history = [
        {"role": "user", "content": "How many late payments could we anticipate next quarter?"},
        {"role": "assistant", "content": a1["answer"]},
    ]
    a2 = rc.answer(
        "How many late payments could we anticipate over the next 12 months instead?",
        property_id=pid, history=history,
    )
    assert a2["intent"] == "late_forecast"
    if a2["source"] != "anthropic":
        pytest.skip("LLM call did not succeed (network/proxy); rules fallback took over")

    assert re.search(rf"\b{re.escape(rc._num(fc_q['expected']))}\b", a1["answer"])
    assert re.search(rf"\b{re.escape(rc._num(fc_y['expected']))}\b", a2["answer"])


@pytest.mark.skipif(
    not app_settings.anthropic_api_key,
    reason="no ANTHROPIC_API_KEY configured (.env); live-LLM test skipped",
)
def test_live_llm_late_forecast_grounds_at_the_low_end_too(monkeypatch):
    """Grounding must hold at the low end of the distribution too, not just the
    typical case the first live-LLM test happens to cover: picks whichever real
    property currently has the SMALLEST expected quarterly count and confirms
    the model still states that (small) number accurately rather than rounding
    it away or dramatizing it."""
    monkeypatch.setattr(rc, "get_langchain_llm", llm_module.get_langchain_llm)

    forecasts = {pid: rr.property_late_forecast(pid) for pid in rr.RESIDENT_PROPERTY_IDS}
    pid = min(forecasts, key=lambda k: forecasts[k]["expected"])
    fc = forecasts[pid]

    a = rc.answer("How many late payments could we anticipate next quarter?", property_id=pid)
    assert a["intent"] == "late_forecast"
    if a["source"] != "anthropic":
        pytest.skip("LLM call did not succeed (network/proxy); rules fallback took over")

    assert re.search(rf"\b{re.escape(rc._num(fc['expected']))}\b", a["answer"])
    assert "[1]" in a["answer"]
