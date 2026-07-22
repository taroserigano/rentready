"""Resident-Risk GOLDEN-SET evaluation (hand-authored cases + portfolio invariants).

Unlike ``residents_eval.py`` (statistical AUC/calibration on a re-generated
held-out cohort), this is example-by-example verification: every case below is
a DETERMINISTICALLY constructed resident (no RNG) with a human-reviewable
rationale for what the correct answer should be, scored through the exact
production path (``residents_risk.predict_resident`` / ``extract_resident_features``).

Three layers:
  1. SINGLE_CASES   — one resident each; assert band/probability/schema facts
                       that follow directly from the documented feature policy,
                       band edges, and eligibility gates in residents_risk.py.
  2. COMPARATIVE_PAIRS — two residents differing in exactly ONE input; assert
                       the DIRECTION of the effect (e.g. autopay lowers risk)
                       without claiming an exact probability (the DGP has
                       stochastic noise; the model may not match the heuristic
                       bit-for-bit — direction is the falsifiable claim).
  3. Cross-cutting invariants applied to EVERY case, plus a full sweep over the
     REAL committed data/residents.json (250 residents) for schema/consistency
     checks that must hold portfolio-wide, not just on hand-picked examples.

Never raises: every check is caught and reported as a finding, not a crash.

Usage:  python backend/evals/residents_golden_eval.py
"""

import json
import re
import sys
from datetime import date, timedelta
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import residents_risk as rr  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

SNAPSHOT = rr.RESIDENT_SNAPSHOT  # 2026-07-01, pinned — never datetime.now()
NAME_RE = re.compile(r"^[A-Za-z]+(?:[ '\-][A-Za-z]+)+$")


def _add_months(d: date, n: int) -> date:
    m = d.month - 1 + n
    return date(d.year + m // 12, m % 12 + 1, 1)


# ---------------------------------------------------------------------------
# Deterministic ledger builder — no RNG, fully auditable from the spec alone.
# ---------------------------------------------------------------------------
def build_ledger(base_rent: float, specs: list) -> list:
    """``specs``: oldest -> newest, each a (status, days_late, paid_fraction)
    tuple (trailing elements optional). Last entry lands the month BEFORE
    SNAPSHOT, matching the production DGP's leakage guard."""
    n = len(specs)
    ledger = []
    balance = 0.0
    for i, spec in enumerate(specs):
        status = spec[0]
        days_late = spec[1] if len(spec) > 1 else 0
        frac = spec[2] if len(spec) > 2 else 1.0
        months_ago = n - i  # oldest -> largest months_ago; newest -> 1
        period = _add_months(SNAPSHOT, -months_ago)

        if status == "paid":
            amount_paid, days_late = base_rent, 0
        elif status == "paid_late":
            amount_paid = base_rent
            days_late = days_late or 5
        elif status == "partial":
            amount_paid = round(base_rent * frac, 2)
        elif status == "missed":
            amount_paid, days_late = 0.0, (days_late or 30)
        else:
            raise ValueError(f"unknown ledger status {status!r}")

        late_fee = 50.0 if status in ("paid_late", "partial", "missed") else 0.0
        balance = round(max(0.0, balance + base_rent - amount_paid), 2)
        ledger.append({
            "period": period.strftime("%Y-%m"),
            "rent_charged": base_rent,
            "amount_paid": amount_paid,
            "paid_date": None if status == "missed" else period.isoformat(),
            "days_late": days_late,
            "late_fee": late_fee,
            "on_time": status == "paid",
            "status": status,
            "balance_after": balance,
            "notices_sent": 1 if status == "missed" else 0,
            "notice_responded": 0,
        })
    return ledger


def build_resident(case: dict) -> dict:
    base_rent = case["base_rent"]
    ledger = build_ledger(base_rent, case["ledger_specs"])
    n = len(case["ledger_specs"])
    tenure_months = case.get("tenure_months_override", n)
    move_in = _add_months(SNAPSHOT, -tenure_months)
    lease_end = _add_months(SNAPSHOT, case["months_to_lease_end"])
    income = case.get("monthly_income", round(base_rent / 0.28, 2))
    return {
        "resident_id": case["id"],
        "property_id": case.get("property_id", "PROP-041"),
        "name": case.get("name", "Golden Case"),
        "unit_id": case.get("unit_id", f"GOLD-{case['id']}"),
        "unit_bedrooms": 1,
        "base_rent": base_rent,
        "lease_start": move_in.isoformat(),
        "lease_end": lease_end.isoformat(),
        "lease_term_months": case.get("lease_term_months", 12),
        "renewal_offer_sent": case.get("renewal_offer_sent", False),
        "autopay_enrolled": case.get("autopay_enrolled", False),
        "deposit_held": base_rent,
        "move_in_date": move_in.isoformat(),
        "prior_renewals": case.get("prior_renewals", 0),
        "monthly_income": income,
        "other_income_monthly": case.get("other_income_monthly", 0.0),
        "income_verified": case.get("income_verified", True),
        "maintenance_requests_12mo": case.get("maintenance_requests_12mo", 1),
        "complaints_12mo": case.get("complaints_12mo", 0),
        "portal_logins_90d": case.get("portal_logins_90d", 10),
        "ledger": ledger,
        "dgp_version": "golden-v1",
    }


# ---------------------------------------------------------------------------
# Golden cases. Each: build params + `checks` = [(name, fn(pred, feats), why)].
# fn returns True/False; exceptions count as failures (never crash the run).
# ---------------------------------------------------------------------------
CLEAN_24MO = [("paid",)] * 24
CLEAN_36MO = [("paid",)] * 36

SINGLE_CASES = [
    {
        "id": "model_tenant",
        "rationale": "36mo perfect payment history, low rent burden, autopay on, "
                     "mtle=8mo (churn6 gate closed, churn12 gate open), zero balance.",
        "base_rent": 1200.0, "monthly_income": 6000.0, "autopay_enrolled": True,
        "months_to_lease_end": 8, "ledger_specs": CLEAN_36MO,
        "checks": [
            ("late_3m=low", lambda p, f: p["heads"]["late_3m"]["band"] == "low",
             "clean history + low burden should never band as elevated risk"),
            ("serious=low", lambda p, f: p["heads"]["serious"]["band"] == "low",
             "no delinquency signal anywhere in the ledger"),
            ("balance=0", lambda p, f: f["current_balance"] == 0.0, "fully paid every month"),
            ("churn6=not_applicable", lambda p, f: p["heads"]["churn"]["band"] == "not_applicable",
             "mtle=8 > CHURN_HORIZON_MONTHS(6) -> churn6 gate must be closed"),
            ("churn12=applicable", lambda p, f: p["heads"]["churn_12m"]["band"] != "not_applicable",
             "mtle=8 <= CHURN_12M_HORIZON_MONTHS(12) -> churn12 gate must be open"),
            ("cure=not_applicable", lambda p, f: p["heads"]["p_cure_6m"]["band"] == "not_applicable",
             "elig_cure requires current_balance>0; this resident owes nothing"),
        ],
    },
    {
        "id": "chronic_delinquent",
        "rationale": "24mo alternating missed/paid_late, high rent burden, autopay off, "
                     "large outstanding balance, mtle=2mo (both churn gates open).",
        "base_rent": 1200.0, "monthly_income": 2200.0, "autopay_enrolled": False,
        "months_to_lease_end": 2,
        "ledger_specs": [("missed",), ("paid_late", 35), ("partial", 0, 0.5), ("missed",)] * 6,
        "checks": [
            ("late_3m=high", lambda p, f: p["heads"]["late_3m"]["band"] == "high",
             "persistent recent trouble + high burden should clear the 0.40 high-band edge"),
            ("serious=high", lambda p, f: p["heads"]["serious"]["band"] == "high",
             "multiple missed/30+ day-late months should clear the 0.25 serious edge"),
            ("balance>0", lambda p, f: f["current_balance"] > 0.0, "missed/partial months accrue arrears"),
            ("churn6=applicable", lambda p, f: p["heads"]["churn"]["band"] != "not_applicable",
             "mtle=2 <= 6 -> churn6 gate open"),
            ("cure=applicable", lambda p, f: p["heads"]["p_cure_6m"]["band"] != "not_applicable",
             "current_balance>0 -> cure gate open"),
        ],
    },
    {
        "id": "new_tenant_thin_history",
        "rationale": "Moved in 1 month ago, on time so far, mtle=11mo. Must not crash on a "
                     "1-entry ledger and must correctly flag low-confidence.",
        "base_rent": 1400.0, "monthly_income": 5000.0, "autopay_enrolled": True,
        "months_to_lease_end": 11, "ledger_specs": [("paid",)],
        "checks": [
            ("no_crash_1m", lambda p, f: p["heads"]["late_1m"]["probability"] is not None, "must score, not None"),
            ("no_crash_12m", lambda p, f: p["heads"]["late_12m"]["probability"] is not None, "must score, not None"),
            ("low_confidence", lambda p, f: p["late"]["confidence"] == "low",
             "tenure_months=1 < 12 -> _confidence() must report low"),
            ("churn6=not_applicable", lambda p, f: p["heads"]["churn"]["band"] == "not_applicable",
             "mtle=11 > 6 -> churn6 gate closed"),
            ("churn12=applicable", lambda p, f: p["heads"]["churn_12m"]["band"] != "not_applicable",
             "mtle=11 <= 12 -> churn12 gate open"),
        ],
    },
    {
        "id": "far_from_lease_end",
        "rationale": "Clean 24mo history but mtle=20mo — both churn horizons must gate closed.",
        "base_rent": 1200.0, "monthly_income": 5000.0, "autopay_enrolled": True,
        "months_to_lease_end": 20, "ledger_specs": CLEAN_24MO,
        "checks": [
            ("churn6=not_applicable", lambda p, f: p["heads"]["churn"]["band"] == "not_applicable"
             and p["heads"]["churn"]["probability"] is None, "mtle=20 > 6"),
            ("churn12=not_applicable", lambda p, f: p["heads"]["churn_12m"]["band"] == "not_applicable"
             and p["heads"]["churn_12m"]["probability"] is None, "mtle=20 > 12"),
        ],
    },
    {
        "id": "old_trouble_now_clean",
        "rationale": "24mo ledger: oldest 4 months paid_late (21-24mo ago; PAID IN FULL so no "
                     "lasting arrears — balance is genuinely $0 today), most recent 20mo spotless. "
                     "Trouble is outside every short-horizon window (3/6/12mo) and left no debt. "
                     "Tests whether the model over-weights stale, fully-resolved history.",
        "base_rent": 1200.0, "monthly_income": 5000.0, "autopay_enrolled": True,
        "months_to_lease_end": 8,
        "ledger_specs": [("paid_late", 20), ("paid_late", 25), ("paid_late", 18), ("paid_late", 22)]
                        + [("paid",)] * 20,
        "checks": [
            ("late_count_12mo=0", lambda p, f: f["late_count_12mo"] == 0.0,
             "all 4 troubled months are >12mo in the past"),
            ("late_count_24mo=4", lambda p, f: f["late_count_24mo"] == 4.0,
             "24mo window must still count them"),
            ("on_time_streak=20", lambda p, f: f["on_time_streak_months"] == 20.0,
             "the most recent 20 consecutive months are clean"),
            ("balance=0", lambda p, f: f["current_balance"] == 0.0,
             "all 4 troubled months were paid in full (late, not missed) -> no lasting arrears"),
            ("late_3m_not_high", lambda p, f: p["heads"]["late_3m"]["band"] != "high",
             "FLAG (not a hard failure): 20 clean months + $0 balance should at least avoid 'high'"),
            ("late_12m_not_high", lambda p, f: p["heads"]["late_12m"]["band"] != "high",
             "FLAG: same reasoning at the 12mo horizon"),
        ],
    },
    {
        "id": "isolated_blip_high_income",
        "rationale": "24mo mostly clean except ONE minor late payment 2mo ago; high income, low "
                     "burden, autopay on. A single 5-day-late blip must not push band to 'high'.",
        "base_rent": 1200.0, "monthly_income": 7500.0, "autopay_enrolled": True,
        "months_to_lease_end": 9,
        "ledger_specs": [("paid",)] * 22 + [("paid_late", 5), ("paid",)],
        "checks": [
            ("late_3m_not_high", lambda p, f: p["heads"]["late_3m"]["band"] != "high",
             "over-sensitivity guard: one minor blip + strong fundamentals should not be 'high'"),
        ],
    },
    {
        "id": "zero_income_on_file",
        "rationale": "No income on file at all (monthly_income=0, other=0, unverified). Must not "
                     "divide-by-zero; rent_to_income must fall back to the documented neutral 0.30.",
        "base_rent": 1200.0, "monthly_income": 0.0, "other_income_monthly": 0.0,
        "income_verified": False, "months_to_lease_end": 8, "ledger_specs": CLEAN_24MO,
        "checks": [
            ("no_crash", lambda p, f: p["heads"]["late_3m"]["probability"] is not None, "must score"),
            ("neutral_burden", lambda p, f: f["rent_to_income"] == rr._NEUTRAL["rent_to_income"],
             "income<=0 branch must impute the neutral value, not raise or return 0/inf"),
        ],
    },
    {
        "id": "long_tenure_capped_ledger",
        "rationale": "True tenure 84mo (7yr) but ledger is capped at HISTORY_MONTHS(60) per the "
                     "'tenure=ledger-length realism fix' — tenure_months must reflect the TRUE "
                     "84mo tenure from lease dates, decoupled from the 60-entry ledger length.",
        "base_rent": 1200.0, "monthly_income": 5000.0, "autopay_enrolled": True,
        "months_to_lease_end": 8, "tenure_months_override": 84,
        "ledger_specs": CLEAN_24MO[:0] + [("paid",)] * 60,
        "checks": [
            ("tenure=84", lambda p, f: f["tenure_months"] == 84.0,
             "tenure_months is derived from move_in_date/lease_start, not ledger length"),
            ("ledger_len=60", lambda p, f: True, "sanity only; checked on the resident dict below"),
        ],
    },
]

# ---------------------------------------------------------------------------
# Comparative pairs — identical except ONE input; assert the DIRECTION only.
# ---------------------------------------------------------------------------
_MODERATE_LATE = [("paid",)] * 18 + [("paid_late", 10), ("paid",), ("paid_late", 8), ("paid",)] + [("paid",)] * 2

COMPARATIVE_PAIRS = [
    {
        "id": "autopay_effect",
        "rationale": "Identical moderate-lateness ledgers; only autopay_enrolled differs. "
                     "Autopay should not RAISE late-payment risk relative to no autopay.",
        "base": {"base_rent": 1200.0, "monthly_income": 3800.0, "months_to_lease_end": 8,
                  "ledger_specs": _MODERATE_LATE},
        "a": {"autopay_enrolled": False}, "b": {"autopay_enrolled": True},
        "metric": lambda p: p["heads"]["late_3m"]["probability"],
        "assertion": "a >= b", "why": "autopay=off should score >= autopay=on (never lower risk)",
    },
    {
        "id": "rent_burden_effect",
        "rationale": "Identical clean 24mo ledgers; only monthly_income differs (drives "
                     "rent_to_income). Higher burden should not LOWER late/serious risk.",
        "base": {"base_rent": 1200.0, "months_to_lease_end": 8, "ledger_specs": CLEAN_24MO,
                  "autopay_enrolled": True},
        "a": {"monthly_income": 6800.0},   # burden ~0.18 (affordable)
        "b": {"monthly_income": 1850.0},   # burden ~0.65 (burdened)
        "metric": lambda p: p["heads"]["late_3m"]["probability"],
        "assertion": "a <= b", "why": "affordable-rent twin should score <= burdened twin",
    },
    {
        "id": "arrears_trend_effect",
        "rationale": "Both currently owe money (cure-eligible); one is steadily paying it down, "
                     "the other is steadily falling further behind. Cure probability should be "
                     "higher for the improving twin.",
        "base": {"base_rent": 1200.0, "monthly_income": 3600.0, "months_to_lease_end": 8,
                  "autopay_enrolled": False},
        "a": {"ledger_specs": [("missed",), ("partial", 0, 0.5)] + [("partial", 0, 0.85)] * 4 + [("paid",)] * 18},
        "b": {"ledger_specs": [("paid",)] * 18 + [("partial", 0, 0.7)] * 3 + [("missed",), ("missed",), ("missed",)]},
        "metric": lambda p: p["heads"]["p_cure_6m"].get("probability"),
        "assertion": "a >= b", "why": "improving/recently-caught-up twin should out-cure the worsening twin",
        "skip_if_none": True,
    },
    {
        "id": "property_id_excluded",
        "rationale": "Identical resident, ONLY property_id differs. property_id is structurally "
                     "excluded from FEATURE_ORDER (fair-housing / location-proxy policy) so "
                     "predictions must be BIT-IDENTICAL, not just directionally similar.",
        "base": {"base_rent": 1200.0, "monthly_income": 5000.0, "months_to_lease_end": 8,
                  "ledger_specs": CLEAN_24MO, "autopay_enrolled": True},
        "a": {"property_id": "PROP-041"}, "b": {"property_id": "PROP-033"},
        "metric": lambda p: p["heads"]["late_3m"]["probability"],
        "assertion": "a == b", "why": "property_id must never influence any prediction",
    },
    {
        "id": "name_excluded",
        "rationale": "Identical resident, ONLY display name differs. name is documented as "
                     "DISPLAY ONLY / never a feature, so predictions must be bit-identical.",
        "base": {"base_rent": 1200.0, "monthly_income": 5000.0, "months_to_lease_end": 8,
                  "ledger_specs": CLEAN_24MO, "autopay_enrolled": True},
        "a": {"name": "Alex Rivera"}, "b": {"name": "Jordan Nguyen"},
        "metric": lambda p: p["heads"]["late_3m"]["probability"],
        "assertion": "a == b", "why": "name must never influence any prediction",
    },
]


# ---------------------------------------------------------------------------
# Runner — single cases
# ---------------------------------------------------------------------------
def _run_single_cases() -> list:
    out = []
    for case in SINGLE_CASES:
        resident = build_resident(case)
        try:
            feats = rr.extract_resident_features(resident, SNAPSHOT)
            pred = rr.predict_resident(resident, SNAPSHOT)
            error = None
        except Exception as exc:  # noqa: BLE001 — a crash IS a finding, not a fatal error
            feats, pred, error = {}, {}, f"{type(exc).__name__}: {exc}"

        results = []
        if error:
            results.append({"check": "no_crash", "passed": False, "detail": error})
        else:
            for name, fn, why in case["checks"]:
                try:
                    ok = bool(fn(pred, feats))
                    detail = ""
                except Exception as exc:  # noqa: BLE001
                    ok, detail = False, f"raised {type(exc).__name__}: {exc}"
                results.append({"check": name, "passed": ok, "why": why, "detail": detail})
            if case["id"] == "long_tenure_capped_ledger":
                ok = len(resident["ledger"]) == 60
                results.append({"check": "ledger_len=60", "passed": ok, "why": "HISTORY_MONTHS cap",
                                 "detail": f"got {len(resident['ledger'])}"})

        results.extend(_generic_invariants(pred) if not error else [])
        out.append({
            "id": case["id"], "rationale": case["rationale"], "error": error,
            "checks": results, "n_pass": sum(1 for c in results if c["passed"]),
            "n_total": len(results),
        })
    return out


def _generic_invariants(pred: dict) -> list:
    """Checks applied to EVERY scored case: monotonicity, band<->probability
    consistency, and the serious-always-routes-to-review contract."""
    out = []
    heads = pred.get("heads", {})

    # Late horizons must be non-decreasing (structurally enforced by
    # _apply_monotone_clamps — verifying the enforcement actually held).
    prev = None
    ok = True
    for name in ("late_1m", "late_3m", "late_6m", "late_12m"):
        p = heads.get(name, {}).get("probability")
        if p is None:
            continue
        if prev is not None and p < prev - 1e-9:
            ok = False
        prev = p
    out.append({"check": "monotone_late_horizons", "passed": ok,
                 "why": "late_1m<=late_3m<=late_6m<=late_12m must hold"})

    prev = None
    ok = True
    for name in ("p_30d_12m", "p_60d_12m", "p_90d_12m"):
        p = heads.get(name, {}).get("probability")
        if p is None:
            continue
        if prev is not None and p > prev + 1e-9:
            ok = False
        prev = p
    out.append({"check": "monotone_severity_thresholds", "passed": ok,
                 "why": "p_30d>=p_60d>=p_90d must hold"})

    # Band must match what _band() recomputes from the served probability.
    band_ok = True
    mismatches = []
    for name, h in heads.items():
        p = h.get("probability")
        if p is None or h.get("band") == "not_applicable":
            continue
        want = rr._band(name, p)
        if want != h.get("band"):
            band_ok = False
            mismatches.append(f"{name}: p={p} served_band={h.get('band')} expected={want}")
    out.append({"check": "band_matches_probability", "passed": band_ok,
                 "why": "served band must equal _band(head, probability)",
                 "detail": "; ".join(mismatches)})

    out.append({"check": "serious_routes_to_review", "passed": pred["serious"].get("routes_to_review") is True,
                 "why": "serious must always route to a human reviewer"})
    return out


# ---------------------------------------------------------------------------
# Runner — comparative pairs
# ---------------------------------------------------------------------------
def _run_comparative_pairs() -> list:
    out = []
    for spec in COMPARATIVE_PAIRS:
        case_a = {**spec["base"], **spec["a"], "id": f"{spec['id']}_a"}
        case_b = {**spec["base"], **spec["b"], "id": f"{spec['id']}_b"}
        try:
            pred_a = rr.predict_resident(build_resident(case_a), SNAPSHOT)
            pred_b = rr.predict_resident(build_resident(case_b), SNAPSHOT)
            va, vb = spec["metric"](pred_a), spec["metric"](pred_b)
            if (va is None or vb is None) and spec.get("skip_if_none"):
                out.append({"id": spec["id"], "skipped": True, "reason": "metric not applicable to either twin"})
                continue
            op = spec["assertion"]
            if op == "a >= b":
                ok = va >= vb
            elif op == "a <= b":
                ok = va <= vb
            elif op == "a == b":
                ok = abs(va - vb) < 1e-9
            else:
                raise ValueError(op)
            out.append({"id": spec["id"], "rationale": spec["rationale"], "why": spec["why"],
                         "a_value": va, "b_value": vb, "assertion": op, "passed": bool(ok)})
        except Exception as exc:  # noqa: BLE001
            out.append({"id": spec["id"], "passed": False, "error": f"{type(exc).__name__}: {exc}"})
    return out


# ---------------------------------------------------------------------------
# Portfolio-wide invariants — the REAL committed 250 residents, not synthetic.
# ---------------------------------------------------------------------------
def _run_portfolio_invariants() -> dict:
    residents = rr.load_residents()
    n = len(residents)
    findings = {
        "n_residents": n,
        "tenure_ledger_mismatch": [], "churn_none_inconsistent": [],
        "band_mismatch": [], "monotonicity_violations": [], "malformed_name": [],
        "negative_balance": [], "unit_property_mismatch": [], "not_routed_to_review": [],
        "out_of_range_probability": [],
    }
    for r in residents:
        rid = r.get("resident_id", "?")
        try:
            feats = rr.extract_resident_features(r, SNAPSHOT)
            pred = rr.predict_resident(r, SNAPSHOT)
        except Exception as exc:  # noqa: BLE001
            findings.setdefault("scoring_crashed", []).append(f"{rid}: {type(exc).__name__}: {exc}")
            continue

        ledger_len = len(r.get("ledger") or [])
        tenure = feats.get("tenure_months", 0)
        if ledger_len != min(tenure, rr.HISTORY_MONTHS):
            findings["tenure_ledger_mismatch"].append(
                f"{rid}: ledger_len={ledger_len} tenure_months={tenure} expected_len={min(tenure, rr.HISTORY_MONTHS)}")

        churn = pred["churn"]
        mtle = feats.get("months_to_lease_end", 99)
        should_apply = 0 < mtle <= rr.CHURN_HORIZON_MONTHS
        is_applicable = churn.get("band") != "not_applicable"
        has_prob = churn.get("probability") is not None
        if is_applicable != should_apply or has_prob != should_apply:
            findings["churn_none_inconsistent"].append(
                f"{rid}: mtle={mtle} band={churn.get('band')} prob={churn.get('probability')}")

        for name, h in pred.get("heads", {}).items():
            p = h.get("probability")
            if p is None:
                continue
            if not (0.0 <= p <= 1.0):
                findings["out_of_range_probability"].append(f"{rid}/{name}: p={p}")
            if h.get("band") != "not_applicable" and rr._band(name, p) != h.get("band"):
                findings["band_mismatch"].append(f"{rid}/{name}: p={p} band={h.get('band')}")

        heads = pred.get("heads", {})
        prev = None
        for name in ("late_1m", "late_3m", "late_6m", "late_12m"):
            p = heads.get(name, {}).get("probability")
            if p is None:
                continue
            if prev is not None and p < prev - 1e-9:
                findings["monotonicity_violations"].append(f"{rid}: {name} broke non-decreasing late order")
            prev = p

        name_val = r.get("name", "")
        if not NAME_RE.match(name_val or ""):
            findings["malformed_name"].append(f"{rid}: name={name_val!r}")

        if feats.get("current_balance", 0.0) < 0:
            findings["negative_balance"].append(f"{rid}: balance={feats.get('current_balance')}")

        unit_id, property_id = r.get("unit_id", ""), r.get("property_id", "")
        if property_id and not unit_id.startswith(property_id):
            findings["unit_property_mismatch"].append(f"{rid}: unit_id={unit_id} property_id={property_id}")

        if pred["serious"].get("routes_to_review") is not True:
            findings["not_routed_to_review"].append(rid)

    findings["clean"] = all(
        not v for k, v in findings.items() if isinstance(v, list)
    )
    return findings


# ---------------------------------------------------------------------------
# Report + persistence
# ---------------------------------------------------------------------------
def run() -> dict:
    single = _run_single_cases()
    comparative = _run_comparative_pairs()
    portfolio = _run_portfolio_invariants()

    single_pass = sum(c["n_pass"] for c in single)
    single_total = sum(c["n_total"] for c in single)
    comp_scored = [c for c in comparative if not c.get("skipped")]
    comp_pass = sum(1 for c in comp_scored if c.get("passed"))

    results = {
        "generated_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "snapshot": SNAPSHOT.isoformat(),
        "single_cases": single,
        "comparative_pairs": comparative,
        "portfolio_invariants": portfolio,
        "score": {
            "single_cases": f"{single_pass}/{single_total}",
            "comparative_pairs": f"{comp_pass}/{len(comp_scored)}",
            "portfolio_clean": portfolio["clean"],
        },
    }
    _persist(results)
    return results


def _persist(results: dict) -> None:
    try:
        RESULTS.mkdir(exist_ok=True)
        (RESULTS / "residents_golden_latest.json").write_text(json.dumps(results, indent=2))
    except OSError:
        pass


if __name__ == "__main__":
    res = run()
    print(f"snapshot={res['snapshot']}")
    print("\n== SINGLE CASES ==")
    for c in res["single_cases"]:
        flag = "" if c["n_pass"] == c["n_total"] else "  <-- FAIL"
        print(f"  {c['id']:28s} {c['n_pass']}/{c['n_total']}{flag}")
        for chk in c["checks"]:
            if not chk["passed"]:
                print(f"      FAIL {chk['check']}: {chk.get('why', '')} {chk.get('detail', '')}")
    print("\n== COMPARATIVE PAIRS ==")
    for c in res["comparative_pairs"]:
        if c.get("skipped"):
            print(f"  {c['id']:28s} SKIPPED ({c['reason']})")
            continue
        flag = "" if c.get("passed") else "  <-- FAIL"
        print(f"  {c['id']:28s} a={c.get('a_value')} b={c.get('b_value')} {c.get('assertion')}{flag}")
    print("\n== PORTFOLIO INVARIANTS (n={}) ==".format(res["portfolio_invariants"]["n_residents"]))
    for k, v in res["portfolio_invariants"].items():
        if isinstance(v, list):
            status = "OK" if not v else f"{len(v)} VIOLATIONS"
            print(f"  {k:28s} {status}")
            for line in v[:5]:
                print(f"      {line}")
            if len(v) > 5:
                print(f"      ... and {len(v) - 5} more")
    print(f"\nSCORE: {res['score']}")
