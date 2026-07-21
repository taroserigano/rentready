"""Resident risk — multi-target scoring, reason codes, graceful fallback.

DECISION-SUPPORT ONLY. For CURRENT residents (not applicants), this estimates
four forward-looking outcomes over the next quarter, from models trained on
SYNTHETIC data, to help a *person* plan proactive outreach and retention. It
never evicts, denies, prices, conditions a lease, or takes any automated
action. Serious-delinquency always routes to a human (``routes_to_review``).

The public entrypoint is ``predict_resident(resident, snapshot) -> dict``. It
NEVER raises: an XGBoost bundle is used per-target when present, otherwise a
pure-Python transparent heuristic — the output shape is identical either way,
and each of the four sub-results degrades to its heuristic independently
(mirrors ``risk.predict`` / ``graphrag.graph_ask``).

The four targets:
  late    — P(any late payment next quarter)          calibrated classifier
  arrears — expected $ balance at end of next quarter  regressor + PI
  churn   — P(does not renew) for leases ending soon   classifier ("n/a" else)
  serious — P(serious delinquency next quarter)        classifier -> review

Feature policy (governing): only legitimate payment / tenancy factors enter the
models. Protected-class proxies (location, familial, age, ...) are STRUCTURALLY
excluded — they never appear in any FEATURE_ORDER or input vector. property_id /
neighborhood are kept on the record ONLY to audit fairness, never as features.
"""

from datetime import date, datetime, timezone
from functools import lru_cache
from pathlib import Path

ARTIFACT_PATH = Path(__file__).resolve().parent / "artifacts" / "residents_model.joblib"

# --------------------------------------------------------------------------
# Shared constants (single source of truth for gen / train / eval / serve)
# --------------------------------------------------------------------------
HISTORY_MONTHS = 60          # 5 years of ledger serialized per resident
LABEL_HORIZON_MONTHS = 3     # "next quarter" label window
CHURN_HORIZON_MONTHS = 6     # churn labelled only for leases ending within this
RESIDENT_SNAPSHOT = date(2026, 7, 1)  # pinned; NO datetime.now() in gen/scoring
DGP_VERSION = "resident-dgp-v1"
SEED = 42

# 10 properties chosen deterministically from data/properties.json, stratified
# across rent tiers ($900–$2350) and DISTINCT neighborhoods/cities so fairness
# slices have real diversity. Frozen — regenerating with SEED reproduces the
# committed data/residents.json exactly.
RESIDENT_PROPERTY_IDS = [
    "PROP-041",  # $900  Deep Ellum, Dallas
    "PROP-037",  # $925  Pearl District, San Antonio
    "PROP-044",  # $1050 Montrose, Houston
    "PROP-020",  # $1100 Bouldin Creek, Austin
    "PROP-006",  # $1250 North Austin, Austin
    "PROP-013",  # $1600 South Congress, Austin
    "PROP-008",  # $1750 Mueller, Austin
    "PROP-024",  # $1950 Crestview, Austin
    "PROP-018",  # $2325 Hyde Park, Austin
    "PROP-033",  # $2350 Alamo Heights, San Antonio
]

TARGETS = ("late", "arrears", "churn", "serious")

# --------------------------------------------------------------------------
# Feature policy — the input contract. Anything not here can never reach a model.
# --------------------------------------------------------------------------
FEATURE_ORDER_BASE = [
    "late_count_3mo",
    "late_count_6mo",
    "late_count_12mo",
    "late_count_24mo",
    "missed_count_12mo",
    "partial_count_12mo",
    "nsf_count_12mo",
    "max_days_late_12mo",
    "avg_days_late_12mo",
    "on_time_streak_months",
    "recency_weighted_lateness",
    "balance_trend_6mo",
    "current_balance_ratio",
    "late_fees_12mo",
    "rent_to_income",
    "income_verified",
    "autopay_enrolled",
    "tenure_months",
    "prior_renewals",
    "notice_response_rate",
    "notices_sent_12mo",
    "maintenance_requests_12mo",
    "complaints_12mo",
    "portal_logins_90d",
    "months_since_last_late",
]

# Per-target input contracts = shared base + target-specific additions.
FEATURE_ORDER = {
    "late": list(FEATURE_ORDER_BASE),
    "arrears": FEATURE_ORDER_BASE + ["current_balance", "avg_monthly_shortfall_6mo"],
    "churn": FEATURE_ORDER_BASE
    + ["months_to_lease_end", "lease_term_months", "renewal_offer_sent"],
    "serious": list(FEATURE_ORDER_BASE),
}

# Documented for the model card: fields deliberately kept OUT of every model.
EXCLUDED_FEATURES = [
    {"field": "property_id", "reason": "Location proxy for protected classes (redlining). Kept only to audit fairness across properties, never a model input."},
    {"field": "neighborhood / city", "reason": "Location proxy for protected classes. Used only for fairness slicing."},
    {"field": "unit_id / name", "reason": "Identity; not predictive."},
    {"field": "household_size / dependents", "reason": "Familial status — protected under fair-housing law."},
    {"field": "age / DOB / is_student", "reason": "Age — protected."},
    {"field": "race / national_origin / sex / disability", "reason": "Protected classes; never collected or used."},
]

# --------------------------------------------------------------------------
# Bands per target. Elevated routes to human outreach/review, never automated
# action. serious is tighter on the high side (favor recall -> review).
# --------------------------------------------------------------------------
_BAND_EDGES = {
    "late": (0.15, 0.40),
    "serious": (0.10, 0.25),
    "churn": (0.20, 0.50),
}


def _bands(target: str) -> list:
    lo, hi = _BAND_EDGES[target]
    return [
        {"band": "low", "min": 0.0, "max": lo},
        {"band": "medium", "min": lo, "max": hi},
        {"band": "high", "min": hi, "max": 1.0},
    ]


BANDS = {t: _bands(t) for t in _BAND_EDGES}


def _band(target: str, p: float) -> str:
    lo, hi = _BAND_EDGES[target]
    if p < lo:
        return "low"
    if p < hi:
        return "medium"
    return "high"


# Neutral imputation values. Missing history is imputed to neutral and lowers
# confidence; it NEVER pushes risk up. Zeros here are genuine "clean history".
_NEUTRAL = {
    "late_count_3mo": 0.0,
    "late_count_6mo": 0.0,
    "late_count_12mo": 0.0,
    "late_count_24mo": 0.0,
    "missed_count_12mo": 0.0,
    "partial_count_12mo": 0.0,
    "nsf_count_12mo": 0.0,
    "max_days_late_12mo": 0.0,
    "avg_days_late_12mo": 0.0,
    "on_time_streak_months": 12.0,
    "recency_weighted_lateness": 0.0,
    "balance_trend_6mo": 0.0,
    "current_balance_ratio": 0.0,
    "late_fees_12mo": 0.0,
    "rent_to_income": 0.30,
    "income_verified": 1.0,
    "autopay_enrolled": 0.0,
    "tenure_months": 24.0,
    "prior_renewals": 1.0,
    "notice_response_rate": 1.0,
    "notices_sent_12mo": 0.0,
    "maintenance_requests_12mo": 1.0,
    "complaints_12mo": 0.0,
    "portal_logins_90d": 8.0,
    "months_since_last_late": 60.0,
    "current_balance": 0.0,
    "avg_monthly_shortfall_6mo": 0.0,
    "months_to_lease_end": 6.0,
    "lease_term_months": 12.0,
    "renewal_offer_sent": 0.0,
}

MONTHS_CAP = 60  # cap for months_since_last_late / on_time_streak


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _sigmoid(z: float) -> float:
    import math

    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


# --------------------------------------------------------------------------
# Date helpers (pure; snapshot-relative, never datetime.now())
# --------------------------------------------------------------------------
def _to_date(s) -> date:
    if isinstance(s, date):
        return s
    return date.fromisoformat(str(s)[:10])


def _months_between(a: date, b: date) -> int:
    """Whole months from ``a`` to ``b`` (b - a), can be negative."""
    return (b.year - a.year) * 12 + (b.month - a.month)


# --------------------------------------------------------------------------
# Feature extraction — trailing-window stats from the committed ledger.
# Short-tenure residents simply have fewer entries (genuine small counts);
# truly-missing inputs (no income on file, etc.) are neutral-imputed.
# --------------------------------------------------------------------------
def _as_dict(resident) -> dict:
    if hasattr(resident, "model_dump"):
        return resident.model_dump()
    return dict(resident)


def _is_trouble(e: dict) -> bool:
    return str(e.get("status", "paid")) != "paid"


def extract_resident_features(resident, snapshot: date = RESIDENT_SNAPSHOT) -> dict:
    """Derive the full feature superset (all targets) for one resident as of
    ``snapshot``. Returns a dict; per-target vectors are sliced by FEATURE_ORDER.
    Only ledger + immutable facts are read — never a protected attribute."""
    r = _as_dict(resident)
    snapshot = _to_date(snapshot)
    led = list(r.get("ledger") or [])
    n = len(led)

    def window(k: int) -> list:
        return led[-k:] if n else []

    def count_trouble(k: int) -> float:
        return float(sum(1 for e in window(k) if _is_trouble(e)))

    f: dict = {}
    f["late_count_3mo"] = count_trouble(3)
    f["late_count_6mo"] = count_trouble(6)
    f["late_count_12mo"] = count_trouble(12)
    f["late_count_24mo"] = count_trouble(24)

    last12 = window(12)
    f["missed_count_12mo"] = float(sum(1 for e in last12 if e.get("status") == "missed"))
    f["partial_count_12mo"] = float(sum(1 for e in last12 if e.get("status") == "partial"))
    # NSF-like: no money received at all in the month (bounced / no funds).
    f["nsf_count_12mo"] = float(sum(1 for e in last12 if float(e.get("amount_paid", 0)) <= 0.0))

    days = [int(e.get("days_late", 0)) for e in last12]
    late_days = [d for d in days if d > 0]
    f["max_days_late_12mo"] = float(max(days)) if days else 0.0
    f["avg_days_late_12mo"] = float(sum(late_days) / len(late_days)) if late_days else 0.0

    # Trailing consecutive on-time months (from newest backwards).
    streak = 0
    for e in reversed(led):
        if _is_trouble(e):
            break
        streak += 1
    f["on_time_streak_months"] = float(min(streak, MONTHS_CAP))

    # Recency-weighted lateness over last 24 months (exp decay, newest heaviest).
    last24 = window(24)
    rw = 0.0
    for age, e in enumerate(reversed(last24)):
        if _is_trouble(e):
            rw += 0.85 ** age
    f["recency_weighted_lateness"] = round(rw, 4)

    base_rent = float(r.get("base_rent") or 0.0)
    bal_last = float(led[-1]["balance_after"]) if n else 0.0
    bal_6ago = float(led[-7]["balance_after"]) if n >= 7 else (float(led[0]["balance_after"]) if n else 0.0)
    denom = base_rent if base_rent > 0 else 1.0
    f["balance_trend_6mo"] = round((bal_last - bal_6ago) / denom, 4)
    f["current_balance_ratio"] = round(bal_last / denom, 4)
    f["current_balance"] = round(bal_last, 2)
    f["late_fees_12mo"] = float(sum(float(e.get("late_fee", 0)) for e in last12))

    last6 = window(6)
    shortfalls = [max(0.0, float(e.get("rent_charged", 0)) - float(e.get("amount_paid", 0))) for e in last6]
    f["avg_monthly_shortfall_6mo"] = round(sum(shortfalls) / len(shortfalls), 2) if shortfalls else 0.0

    # Income / rent burden.
    income = float(r.get("monthly_income") or 0.0) + float(r.get("other_income_monthly") or 0.0)
    if income > 0 and base_rent > 0:
        f["rent_to_income"] = round(_clamp(base_rent / income, 0.0, 3.0), 4)
    else:
        f["rent_to_income"] = _NEUTRAL["rent_to_income"]
    f["income_verified"] = 1.0 if r.get("income_verified") else 0.0
    f["autopay_enrolled"] = 1.0 if r.get("autopay_enrolled") else 0.0

    # Tenure (capped at history length; short-tenure -> lower confidence).
    move_in = r.get("move_in_date") or r.get("lease_start")
    if move_in:
        tenure = _months_between(_to_date(move_in), snapshot)
    else:
        tenure = n
    f["tenure_months"] = float(max(0, min(tenure, 600)))

    f["prior_renewals"] = float(r.get("prior_renewals") or 0)

    # Notice engagement (last 24 months of the ledger).
    sent = sum(int(e.get("notices_sent", 0)) for e in last24)
    resp = sum(int(e.get("notice_responded", 0)) for e in last24)
    f["notice_response_rate"] = round(resp / sent, 4) if sent > 0 else _NEUTRAL["notice_response_rate"]
    f["notices_sent_12mo"] = float(sum(int(e.get("notices_sent", 0)) for e in last12))

    f["maintenance_requests_12mo"] = float(r.get("maintenance_requests_12mo") or 0)
    f["complaints_12mo"] = float(r.get("complaints_12mo") or 0)
    f["portal_logins_90d"] = float(r.get("portal_logins_90d") or 0)

    # Months since most recent trouble month (capped). Cap when never late.
    msl = MONTHS_CAP
    for age, e in enumerate(reversed(led)):
        if _is_trouble(e):
            msl = min(age, MONTHS_CAP)
            break
    f["months_since_last_late"] = float(msl)

    # Churn-specific lease timing.
    lease_end = r.get("lease_end")
    mtle = _months_between(snapshot, _to_date(lease_end)) if lease_end else _NEUTRAL["months_to_lease_end"]
    f["months_to_lease_end"] = float(mtle)
    f["lease_term_months"] = float(r.get("lease_term_months") or _NEUTRAL["lease_term_months"])
    f["renewal_offer_sent"] = 1.0 if r.get("renewal_offer_sent") else 0.0

    return f


def feature_vector(features: dict, target: str) -> list:
    return [float(features.get(k, _NEUTRAL.get(k, 0.0))) for k in FEATURE_ORDER[target]]


def _confidence(resident, features: dict) -> str:
    """Low when key predictive history is thin — short tenure, few ledger
    months, or no income on file. Never a function of a protected attribute."""
    r = _as_dict(resident)
    missing = 0
    if features.get("tenure_months", 0) < 12:
        missing += 1
    if len(r.get("ledger") or []) < 12:
        missing += 1
    income = float(r.get("monthly_income") or 0.0) + float(r.get("other_income_monthly") or 0.0)
    if income <= 0:
        missing += 1
    return "low" if missing >= 1 else "high"


def churn_eligible(features: dict) -> bool:
    """Churn is labelled/served only for leases ending within the horizon."""
    mtle = features.get("months_to_lease_end", 99)
    return 0 < mtle <= CHURN_HORIZON_MONTHS


# --------------------------------------------------------------------------
# Reason-code templates + up/down mix (mirrors risk._reason_codes).
# --------------------------------------------------------------------------
def _n(v: float) -> str:
    return f"{v:.1f}".rstrip("0").rstrip(".")


REASON_TEMPLATES = {
    "late_count_3mo": lambda v: f"{round(v)} late month(s) in the last 3 months" if v else "On time the last 3 months",
    "late_count_6mo": lambda v: f"{round(v)} late month(s) in the last 6 months" if v else "On time the last 6 months",
    "late_count_12mo": lambda v: f"{round(v)} late month(s) in the last 12 months" if v else "No late months in the last year",
    "late_count_24mo": lambda v: f"{round(v)} late month(s) in the last 24 months" if v else "No late months in two years",
    "missed_count_12mo": lambda v: f"{round(v)} missed payment(s) in the last year" if v else "No missed payments in the last year",
    "partial_count_12mo": lambda v: f"{round(v)} partial payment(s) in the last year" if v else "No partial payments in the last year",
    "nsf_count_12mo": lambda v: f"{round(v)} month(s) with no payment received" if v else "Payment received every month",
    "max_days_late_12mo": lambda v: f"Up to {round(v)} days late in the last year" if v else "Never late in the last year",
    "avg_days_late_12mo": lambda v: f"About {_n(v)} days late on average when late" if v else "No lateness to average",
    "on_time_streak_months": lambda v: f"{round(v)} straight on-time month(s)",
    "recency_weighted_lateness": lambda v: (f"Recent lateness pattern (weighted {_n(v)})" if v > 0.05 else "No recent lateness pattern"),
    "balance_trend_6mo": lambda v: ("Balance rising over 6 months" if v > 0.05 else ("Balance falling over 6 months" if v < -0.05 else "Balance stable over 6 months")),
    "current_balance_ratio": lambda v: (f"Outstanding balance is {_n(v)}x monthly rent" if v > 0.02 else "No outstanding balance"),
    "current_balance": lambda v: (f"${round(v):,} outstanding balance" if v > 1 else "No outstanding balance"),
    "avg_monthly_shortfall_6mo": lambda v: (f"About ${round(v):,}/mo underpaid recently" if v > 1 else "No recent underpayment"),
    "late_fees_12mo": lambda v: (f"${round(v):,} in late fees over the last year" if v > 1 else "No late fees in the last year"),
    "rent_to_income": lambda v: f"Rent is {round(v * 100)}% of monthly income",
    "income_verified": lambda v: ("Income on file is verified" if v >= 0.5 else "Income not verified"),
    "autopay_enrolled": lambda v: ("Enrolled in autopay" if v >= 0.5 else "Not enrolled in autopay"),
    "tenure_months": lambda v: f"{round(v)} months of tenancy",
    "prior_renewals": lambda v: (f"{round(v)} prior lease renewal(s)" if v else "No prior renewals"),
    "notice_response_rate": lambda v: f"Responds to {round(v * 100)}% of notices",
    "notices_sent_12mo": lambda v: (f"{round(v)} notice(s) sent in the last year" if v else "No notices sent in the last year"),
    "maintenance_requests_12mo": lambda v: f"{round(v)} maintenance request(s) in the last year",
    "complaints_12mo": lambda v: (f"{round(v)} complaint(s) in the last year" if v else "No complaints in the last year"),
    "portal_logins_90d": lambda v: f"{round(v)} resident-portal login(s) in 90 days",
    "months_since_last_late": lambda v: (f"Last late payment {round(v)} months ago" if v < MONTHS_CAP else "No late payments on record"),
    "months_to_lease_end": lambda v: f"Lease ends in about {round(v)} month(s)",
    "lease_term_months": lambda v: f"{round(v)}-month lease term",
    "renewal_offer_sent": lambda v: ("Renewal offer already sent" if v >= 0.5 else "No renewal offer sent yet"),
}


def _label(feature: str, value: float) -> str:
    tmpl = REASON_TEMPLATES.get(feature)
    if tmpl is None:
        return feature.replace("_", " ")
    try:
        return tmpl(value)
    except Exception:  # noqa: BLE001 — labels must never break scoring
        return feature.replace("_", " ")


def _reason_codes(contribs: dict, features: dict, feature_order: list, cap: int = 4) -> list:
    """Signed contributions (TreeSHAP or heuristic) -> plain-English ReasonCode
    dicts, top ``cap`` by |contribution| but nudged to a mix of up/down drivers."""
    ranked = sorted(
        (
            {
                "feature": feat,
                "label": _label(feat, features.get(feat, 0.0)),
                "direction": "increases" if c > 0 else "decreases",
                "contribution": round(float(c), 4),
            }
            for feat, c in contribs.items()
            if feat in feature_order and abs(c) > 1e-9
        ),
        key=lambda r: -abs(r["contribution"]),
    )
    if len(ranked) <= cap:
        return ranked
    top = ranked[:cap]
    for direction in ("increases", "decreases"):
        if all(r["direction"] != direction for r in top):
            extra = next((r for r in ranked[cap:] if r["direction"] == direction), None)
            if extra is not None:
                top[-1] = extra
                top.sort(key=lambda r: -abs(r["contribution"]))
    return top


# --------------------------------------------------------------------------
# Transparent per-target heuristics (never return exactly 0 or 1).
# Signed weights mirror the DGP direction. ("dev", w, neutral) => w*(v-neutral);
# ("raw", w) => w*v; ("flag_low", w) => w*(1-v).
# --------------------------------------------------------------------------
_HEURISTICS = {
    "late": (
        -1.2,
        {
            "late_count_6mo": ("raw", 0.55),
            "late_count_12mo": ("raw", 0.18),
            "missed_count_12mo": ("raw", 0.35),
            "partial_count_12mo": ("raw", 0.25),
            "recency_weighted_lateness": ("raw", 0.6),
            "current_balance_ratio": ("raw", 0.5),
            "on_time_streak_months": ("dev", -0.03, _NEUTRAL["on_time_streak_months"]),
            "months_since_last_late": ("dev", -0.02, _NEUTRAL["months_since_last_late"]),
            "rent_to_income": ("dev", 2.2, _NEUTRAL["rent_to_income"]),
            "autopay_enrolled": ("flag", -0.35, None),
            "income_verified": ("flag_low", 0.25, None),
            "late_fees_12mo": ("raw", 0.0008),
        },
    ),
    "serious": (
        -2.1,
        {
            "missed_count_12mo": ("raw", 0.7),
            "nsf_count_12mo": ("raw", 0.4),
            "max_days_late_12mo": ("dev", 0.03, 0.0),
            "current_balance_ratio": ("raw", 1.0),
            "partial_count_12mo": ("raw", 0.3),
            "recency_weighted_lateness": ("raw", 0.5),
            "rent_to_income": ("dev", 2.0, _NEUTRAL["rent_to_income"]),
            "on_time_streak_months": ("dev", -0.02, _NEUTRAL["on_time_streak_months"]),
            "autopay_enrolled": ("flag", -0.25, None),
        },
    ),
    "churn": (
        -0.7,
        {
            "months_to_lease_end": ("dev", -0.12, _NEUTRAL["months_to_lease_end"]),
            "renewal_offer_sent": ("flag", -0.5, None),
            "prior_renewals": ("raw", -0.35),
            "rent_to_income": ("dev", 1.6, _NEUTRAL["rent_to_income"]),
            "late_count_12mo": ("raw", 0.12),
            "recency_weighted_lateness": ("raw", 0.35),
            "complaints_12mo": ("raw", 0.25),
            "portal_logins_90d": ("dev", -0.015, _NEUTRAL["portal_logins_90d"]),
            "tenure_months": ("dev", -0.004, _NEUTRAL["tenure_months"]),
        },
    ),
}


def _heuristic_binary(target: str, features: dict) -> tuple:
    bias, weights = _HEURISTICS[target]
    contribs: dict = {}
    z = bias
    for feat, spec in weights.items():
        v = float(features.get(feat, _NEUTRAL.get(feat, 0.0)))
        kind = spec[0]
        if kind == "dev":
            c = spec[1] * (v - spec[2])
        elif kind == "raw":
            c = spec[1] * v
        elif kind == "flag":
            c = spec[1] * v
        elif kind == "flag_low":
            c = spec[1] * (1.0 - v)
        else:
            c = 0.0
        contribs[feat] = c
        z += c
    p = _clamp(_sigmoid(z), 0.02, 0.98)
    return p, contribs


# Arrears heuristic: current balance carried forward + expected shortfall over
# the horizon, damped by recent on-time behavior.
def _heuristic_arrears(features: dict) -> tuple:
    bal = float(features.get("current_balance", 0.0))
    shortfall = float(features.get("avg_monthly_shortfall_6mo", 0.0))
    late6 = float(features.get("late_count_6mo", 0.0))
    autopay = float(features.get("autopay_enrolled", 0.0))
    # Fraction of the horizon we expect trouble to persist, from recent lateness.
    persist = _clamp(late6 / 6.0, 0.0, 1.0) * (0.6 if autopay < 0.5 else 0.35)
    expected = bal * (0.6 + 0.4 * persist) + shortfall * LABEL_HORIZON_MONTHS * persist
    expected = max(0.0, expected)
    contribs = {
        "current_balance": bal * (0.6 + 0.4 * persist),
        "avg_monthly_shortfall_6mo": shortfall * LABEL_HORIZON_MONTHS * persist,
        "late_count_6mo": late6 * 5.0,
        "autopay_enrolled": -20.0 * autopay,
    }
    return expected, contribs


# --------------------------------------------------------------------------
# Model loading (lru_cache; None on ANY failure or feature-contract mismatch)
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _model():
    """Load the persisted multi-target bundle, or None on any failure. Refuses a
    bundle whose per-target feature_order doesn't match this code's contract."""
    try:
        import joblib

        if not ARTIFACT_PATH.exists():
            return None
        bundle = joblib.load(ARTIFACT_PATH)
        if not isinstance(bundle, dict):
            return None
        for t in TARGETS:
            sub = bundle.get(t)
            if not isinstance(sub, dict):
                return None
            if sub.get("feature_order") != FEATURE_ORDER[t]:
                return None  # trained against a different contract — refuse it
            key = "regressor" if t == "arrears" else "calibrated_model"
            if key not in sub:
                return None
        return bundle
    except Exception as exc:  # noqa: BLE001 — degrade to heuristics
        print(f"residents_risk: model load failed ({type(exc).__name__}: {exc}); using heuristics.")
        return None


def _tree_contribs(sub: dict, features: dict, feature_order: list):
    """TreeSHAP signed contributions from the raw booster, or None if absent."""
    booster = sub.get("booster")
    if booster is None:
        return None
    import pandas as pd
    import xgboost as xgb

    row = pd.DataFrame([[float(features.get(k, _NEUTRAL.get(k, 0.0))) for k in feature_order]], columns=feature_order)
    contribs = booster.predict(xgb.DMatrix(row), pred_contribs=True)[0]
    return {feature_order[i]: float(contribs[i]) for i in range(len(feature_order))}


# --------------------------------------------------------------------------
# Per-target prediction (each degrades to its heuristic independently)
# --------------------------------------------------------------------------
def _predict_bin(target: str, features: dict, bundle, low_conf: bool) -> dict:
    fo = FEATURE_ORDER[target]
    edge_half = 0.08
    if bundle is not None:
        sub = bundle.get(target, {})
        try:
            import pandas as pd

            edge_half = float(sub.get("range_half_width", edge_half))
            row = pd.DataFrame([[float(features.get(k, _NEUTRAL.get(k, 0.0))) for k in fo]], columns=fo)
            p = float(sub["calibrated_model"].predict_proba(row)[0][1])
            contribs = _tree_contribs(sub, features, fo)
            if contribs is None:
                _, contribs = _heuristic_binary(target, features)
            reasons = _reason_codes(contribs, features, fo)
            return _bin_result(target, p, reasons, low_conf, edge_half, "model", sub.get("model_type", "xgboost"))
        except Exception as exc:  # noqa: BLE001 — degrade to heuristic
            print(f"residents_risk: {target} model predict failed ({type(exc).__name__}: {exc}); heuristic.")
    p, contribs = _heuristic_binary(target, features)
    reasons = _reason_codes(contribs, features, fo)
    return _bin_result(target, p, reasons, low_conf, edge_half, "heuristic", "heuristic")


def _bin_result(target, p, reasons, low_conf, edge_half, source, model_type) -> dict:
    half = edge_half + (0.07 if low_conf else 0.0)
    res = {
        "probability": round(_clamp(p), 4),
        "band": _band(target, p),
        "range": [round(_clamp(p - half), 4), round(_clamp(p + half), 4)],
        "reason_codes": reasons,
        "confidence": "low" if low_conf else "high",
        "source": source,
        "model_type": model_type,
    }
    if target == "serious":
        res["routes_to_review"] = True
    return res


def _predict_arrears(features: dict, bundle, low_conf: bool) -> dict:
    fo = FEATURE_ORDER["arrears"]
    if bundle is not None:
        sub = bundle.get("arrears", {})
        try:
            import pandas as pd

            row = pd.DataFrame([[float(features.get(k, _NEUTRAL.get(k, 0.0))) for k in fo]], columns=fo)
            pred = float(sub["regressor"].predict(row)[0])
            pred = max(0.0, pred)
            std = float(sub.get("residual_std", max(50.0, 0.25 * pred + 25.0)))
            half = 1.28 * std * (1.4 if low_conf else 1.0)
            contribs = _tree_contribs(sub, features, fo)
            if contribs is None:
                _, contribs = _heuristic_arrears(features)
            reasons = _reason_codes(contribs, features, fo)
            return {
                "expected_balance": round(pred, 2),
                "interval": [round(max(0.0, pred - half), 2), round(pred + half, 2)],
                "reason_codes": reasons,
                "confidence": "low" if low_conf else "high",
                "source": "model",
                "model_type": sub.get("model_type", "xgboost"),
            }
        except Exception as exc:  # noqa: BLE001
            print(f"residents_risk: arrears model predict failed ({type(exc).__name__}: {exc}); heuristic.")
    pred, contribs = _heuristic_arrears(features)
    std = max(50.0, 0.4 * pred + 40.0)
    half = 1.28 * std * (1.4 if low_conf else 1.0)
    reasons = _reason_codes(contribs, features, fo)
    return {
        "expected_balance": round(pred, 2),
        "interval": [round(max(0.0, pred - half), 2), round(pred + half, 2)],
        "reason_codes": reasons,
        "confidence": "low" if low_conf else "high",
        "source": "heuristic",
        "model_type": "heuristic",
    }


def _predict_churn(features: dict, bundle, low_conf: bool) -> dict:
    mtle = float(features.get("months_to_lease_end", 99))
    if not churn_eligible(features):
        return {
            "probability": None,
            "band": "not_applicable",
            "months_to_lease_end": round(mtle, 1),
            "reason_codes": [],
            "confidence": "low" if low_conf else "high",
            "source": "not_applicable",
        }
    fo = FEATURE_ORDER["churn"]
    if bundle is not None:
        sub = bundle.get("churn", {})
        try:
            import pandas as pd

            row = pd.DataFrame([[float(features.get(k, _NEUTRAL.get(k, 0.0))) for k in fo]], columns=fo)
            p = float(sub["calibrated_model"].predict_proba(row)[0][1])
            contribs = _tree_contribs(sub, features, fo)
            if contribs is None:
                _, contribs = _heuristic_binary("churn", features)
            reasons = _reason_codes(contribs, features, fo)
            return {
                "probability": round(_clamp(p), 4),
                "band": _band("churn", p),
                "months_to_lease_end": round(mtle, 1),
                "reason_codes": reasons,
                "confidence": "low" if low_conf else "high",
                "source": "model",
            }
        except Exception as exc:  # noqa: BLE001
            print(f"residents_risk: churn model predict failed ({type(exc).__name__}: {exc}); heuristic.")
    p, contribs = _heuristic_binary("churn", features)
    reasons = _reason_codes(contribs, features, fo)
    return {
        "probability": round(_clamp(p), 4),
        "band": _band("churn", p),
        "months_to_lease_end": round(mtle, 1),
        "reason_codes": reasons,
        "confidence": "low" if low_conf else "high",
        "source": "heuristic",
    }


# --------------------------------------------------------------------------
# Public entrypoint — NEVER raises
# --------------------------------------------------------------------------
def predict_resident(resident, snapshot: date = RESIDENT_SNAPSHOT) -> dict:
    """Score one resident on all four targets. Returns the multi-target result
    dict. Each sub-result degrades to its heuristic independently; never raises."""
    scored_at = datetime.now(timezone.utc).isoformat()
    r = _as_dict(resident)
    try:
        features = extract_resident_features(r, snapshot)
    except Exception as exc:  # noqa: BLE001 — extraction must never break scoring
        print(f"residents_risk: feature extraction failed ({type(exc).__name__}: {exc}); neutral features.")
        features = dict(_NEUTRAL)
    low_conf = _confidence(r, features) == "low"
    bundle = _model()

    def guarded(fn, *a):
        try:
            return fn(*a)
        except Exception as exc:  # noqa: BLE001 — a broken target never sinks the rest
            print(f"residents_risk: sub-prediction failed ({type(exc).__name__}: {exc}).")
            return None

    late = guarded(_predict_bin, "late", features, bundle, low_conf) or _bin_result("late", 0.3, [], low_conf, 0.1, "heuristic", "heuristic")
    serious = guarded(_predict_bin, "serious", features, bundle, low_conf) or _bin_result("serious", 0.1, [], low_conf, 0.1, "heuristic", "heuristic")
    arrears = guarded(_predict_arrears, features, bundle, low_conf) or {"expected_balance": 0.0, "interval": [0.0, 0.0], "reason_codes": [], "confidence": "low", "source": "heuristic", "model_type": "heuristic"}
    churn = guarded(_predict_churn, features, bundle, low_conf) or {"probability": None, "band": "not_applicable", "months_to_lease_end": features.get("months_to_lease_end"), "reason_codes": [], "confidence": "low", "source": "not_applicable"}

    return {
        "resident_id": r.get("resident_id", ""),
        "property_id": r.get("property_id", ""),
        "snapshot_date": _to_date(snapshot).isoformat(),
        "late": late,
        "arrears": arrears,
        "churn": churn,
        "serious": serious,
        "scored_at": scored_at,
    }


def predict_residents(residents: list, snapshot: date = RESIDENT_SNAPSHOT) -> list:
    """Score a list of residents. Never raises (per-resident guarded)."""
    out = []
    for r in residents or []:
        try:
            out.append(predict_resident(r, snapshot))
        except Exception as exc:  # noqa: BLE001
            print(f"residents_risk: predict_residents skipped one ({type(exc).__name__}: {exc}).")
    return out


def top_driver(sub_result: dict) -> str:
    """Strongest 'increases' reason label from a sub-result (for portfolio rows)."""
    ups = [rc for rc in (sub_result or {}).get("reason_codes", []) if rc.get("direction") == "increases"]
    if not ups:
        return ""
    return max(ups, key=lambda rc: abs(rc.get("contribution", 0)))["label"]


# --------------------------------------------------------------------------
# Cached data loader (mirrors graph.load_properties)
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_residents() -> list:
    """Load the committed residents dataset (list of resident dicts). Empty list
    on any failure (missing file, corrupt JSON) — callers stay graceful."""
    import json

    from settings import DATA_DIR

    path = DATA_DIR / "residents.json"
    try:
        data = json.loads(path.read_text())
        return list(data.get("residents", []))
    except (OSError, ValueError) as exc:
        print(f"residents_risk: could not load residents.json ({type(exc).__name__}: {exc}).")
        return []


def get_resident(resident_id: str):
    for r in load_residents():
        if r.get("resident_id") == resident_id:
            return r
    return None


# --------------------------------------------------------------------------
# Startup + status + model card
# --------------------------------------------------------------------------
def ensure_model() -> dict:
    """Best-effort: train the bundle if missing. Never raises. Clears caches."""
    try:
        if ARTIFACT_PATH.exists():
            _model.cache_clear()
            return {"trained": True, "action": "already_present"}
        import train_residents

        train_residents.train()
        _model.cache_clear()
        return {"trained": ARTIFACT_PATH.exists(), "action": "trained"}
    except Exception as exc:  # noqa: BLE001 — startup must never fail
        print(f"residents_risk.ensure_model: training skipped ({type(exc).__name__}: {exc}).")
        _model.cache_clear()
        return {"trained": False, "action": "failed", "error": type(exc).__name__}


def status() -> dict:
    """Health/status summary. Never raises."""
    try:
        bundle = _model()
        if bundle is None:
            return {
                "trained": False,
                "source": "heuristic",
                "targets": list(TARGETS),
                "artifact": str(ARTIFACT_PATH),
            }
        return {
            "trained": True,
            "source": "model",
            "targets": list(TARGETS),
            "metrics": {t: bundle.get(t, {}).get("metrics", {}) for t in TARGETS},
            "dgp_version": bundle.get("dgp_version"),
            "seed": bundle.get("seed"),
            "generated_at": bundle.get("generated_at"),
            "artifact": str(ARTIFACT_PATH),
        }
    except Exception as exc:  # noqa: BLE001
        return {"trained": False, "source": "heuristic", "error": type(exc).__name__}


def model_card() -> dict:
    """The resident-risk model card. Never raises."""
    bundle = _model()

    def target_card(t: str, kind: str, desc: str) -> dict:
        sub = (bundle or {}).get(t, {})
        card = {
            "target": t,
            "kind": kind,
            "description": desc,
            "features": list(FEATURE_ORDER[t]),
            "metrics": sub.get("metrics", {}),
        }
        if t in _BAND_EDGES:
            card["bands"] = BANDS[t]
        return card

    return {
        "name": "Resident Risk (multi-target)",
        "version": DGP_VERSION,
        "trained_at": (bundle or {}).get("generated_at"),
        "description": (
            "Four gradient-boosted models estimating next-quarter outcomes for "
            "current residents from a 5-year rent ledger, trained on SYNTHETIC "
            "data. Reason codes come from TreeSHAP (model) or transparent weights "
            "(heuristic fallback)."
        ),
        "intended_use": (
            "Decision-support for proactive outreach and retention ONLY. NOT for "
            "eviction, denial, pricing, lease conditioning, or any automated "
            "action. Serious-delinquency routes to a human reviewer."
        ),
        "targets": [
            target_card("late", "classifier", "P(any late payment next quarter)."),
            target_card("arrears", "regressor", "Expected $ balance at the end of next quarter."),
            target_card("churn", "classifier", "P(non-renewal) for leases ending within 6 months; not applicable otherwise."),
            target_card("serious", "classifier", "P(serious delinquency: 30+ days late or a full month in arrears). Routes to review."),
        ],
        "excluded": EXCLUDED_FEATURES,
        "limitations": [
            "Trained on SYNTHETIC data — not validated on real residents.",
            "Estimates, not guarantees; individual outcomes vary.",
            "Missing history is neutral-imputed and lowers confidence, never raising risk.",
            "Never uses race, national origin, sex, familial status, disability, age, or location.",
            "property/neighborhood are used only to audit fairness, never as model inputs.",
            "Not a consumer report; not a substitute for lawful, human-in-the-loop decisions.",
        ],
        "source": "model" if bundle else "heuristic",
    }


if __name__ == "__main__":
    import pprint

    pprint.pp(status())
