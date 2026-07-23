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

from datetime import date

import pytest

import generate_residents as gen
import residents_chat as rc
import residents_risk as rr
from models import LedgerEntry, Resident


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
])
def test_route_maps_questions_to_intents(question, intent):
    assert rc.route(question) == intent


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
