"""Resident risk — multi-HEAD scoring, reason codes, graceful fallback (v2).

DECISION-SUPPORT ONLY. For CURRENT residents (not applicants), this estimates a
COMPREHENSIVE catalog of forward-looking outcomes, from models trained on
SYNTHETIC data, to help a *person* plan proactive outreach and retention. It
never evicts, denies, prices, conditions a lease, or takes any automated
action. Serious-delinquency always routes to a human (``routes_to_review``).

The public entrypoint is ``predict_resident(resident, snapshot) -> dict``. It
NEVER raises: a per-head XGBoost bundle is used when present, otherwise a
pure-Python transparent heuristic — the output shape is identical either way,
and EACH head degrades to its heuristic independently.

v2 head catalog (see ``HEADS``), grouped by family:
  late       — P(any late) at horizons 1 / 3 / 6 / 12 months   (calibrated clf)
  frequency  — expected # late / missed months next 12mo        (count)
  severity   — max days late, P(30/60/90-day), delinquency bucket, serious
  arrears    — expected $ balance at 3 / 12 months, peak balance (regression)
  cure       — P(cure) + months-to-cure                         (clf + survival)
  retention  — churn (non-renewal) at <=6mo and <=12mo horizons  (clf)
plus the four LEGACY aliases (late/arrears/churn/serious == late_3m/arrears_3m/
churn/serious) so the existing API and frontend keep working unchanged.

Feature policy (governing): only legitimate payment / tenancy factors enter the
models. Protected-class proxies (location, familial, age, ...) are STRUCTURALLY
excluded — they never appear in any feature_order or input vector. property_id /
neighborhood / display name are kept on the record ONLY to audit fairness or for
display, never as features.
"""

from datetime import date, datetime, timezone
from functools import lru_cache
from pathlib import Path

ARTIFACT_PATH = Path(__file__).resolve().parent / "artifacts" / "residents_model.joblib"
BUNDLE_SCHEMA = "residents-heads-v2"

# --------------------------------------------------------------------------
# Shared constants (single source of truth for gen / train / eval / serve)
# --------------------------------------------------------------------------
HISTORY_MONTHS = 60           # 5 years of ledger serialized per resident
LABEL_HORIZON_MONTHS = 3      # legacy "next quarter" window (late_3m / arrears_3m / serious)
FUTURE_HORIZON_MONTHS = 15    # v2 simulated future window (clean 12mo labels + slack)
CHURN_HORIZON_MONTHS = 6      # churn (legacy) labelled only for leases ending within this
CHURN_12M_HORIZON_MONTHS = 12  # churn_12m horizon
CURE_HORIZON_MONTHS = 12      # months-to-cure survival horizon
CURE_EPS = 1.0                # balance <= $1 counts as "cured"
RESIDENT_SNAPSHOT = date(2026, 7, 1)  # pinned; NO datetime.now() in gen/scoring
DGP_VERSION = "resident-dgp-v2"
SEED = 42

# 10 properties chosen deterministically from data/properties.json, stratified
# across rent tiers ($900-$2350) and DISTINCT neighborhoods/cities so fairness
# slices have real diversity. Frozen.
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

# Legacy target names still referenced across the app (aliases into the heads).
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
    # --- v2 additions (all ledger/lease/income-derived; fair-housing safe) ---
    "trouble_month_rate",
    "longest_late_streak_12mo",
    "count_30plus_12mo",
    "count_60plus_12mo",
    "balance_trend_3mo",
    "max_balance_12mo",
    "months_in_arrears_12mo",
    "months_since_balance_zero",
]

# Per-family input contracts = shared base + family-specific additions.
_ARREARS_EXTRA = ["current_balance", "avg_monthly_shortfall_6mo"]
_CHURN_EXTRA = ["months_to_lease_end", "lease_term_months", "renewal_offer_sent"]

FEATURE_ORDER_ARREARS = FEATURE_ORDER_BASE + _ARREARS_EXTRA
FEATURE_ORDER_CURE = FEATURE_ORDER_BASE + _ARREARS_EXTRA
FEATURE_ORDER_CHURN = FEATURE_ORDER_BASE + _CHURN_EXTRA

# --------------------------------------------------------------------------
# Shared XGBoost monotone-constraint direction, keyed to FEATURE_ORDER_BASE.
# +1 = prediction must be non-decreasing in the feature; -1 = non-increasing;
# 0 = unconstrained. This is the SINGLE source of truth train_residents.py
# reads from (via a per-head feature_order lookup) for every count/regression
# head it constrains — do not define a second/parallel constraint mapping.
# --------------------------------------------------------------------------
MONOTONE_BASE = {
    "late_count_3mo": 1,
    "late_count_6mo": 1,
    "late_count_12mo": 1,
    "late_count_24mo": 1,
    "missed_count_12mo": 1,
    "partial_count_12mo": 1,
    "nsf_count_12mo": 1,
    "max_days_late_12mo": 1,
    "avg_days_late_12mo": 1,
    "on_time_streak_months": -1,
    "recency_weighted_lateness": 1,
    "balance_trend_6mo": 0,
    "current_balance_ratio": 0,
    "late_fees_12mo": 1,
    "rent_to_income": 1,
    "income_verified": 0,
    "autopay_enrolled": -1,
    "tenure_months": 0,
    "prior_renewals": 0,
    "notice_response_rate": 0,
    "notices_sent_12mo": 1,
    "maintenance_requests_12mo": 0,
    "complaints_12mo": 0,
    "portal_logins_90d": 0,
    "months_since_last_late": -1,
    "trouble_month_rate": 1,
    "longest_late_streak_12mo": 1,
    "count_30plus_12mo": 1,
    "count_60plus_12mo": 1,
    "balance_trend_3mo": 0,
    "max_balance_12mo": 1,
    "months_in_arrears_12mo": 1,
    "months_since_balance_zero": -1,
}

# Documented for the model card: fields deliberately kept OUT of every model.
EXCLUDED_FEATURES = [
    {"field": "property_id", "reason": "Location proxy for protected classes (redlining). Kept only to audit fairness across properties, never a model input."},
    {"field": "neighborhood / city", "reason": "Location proxy for protected classes. Used only for fairness slicing."},
    {"field": "unit_id / name", "reason": "Identity / display only; not predictive and never a model input."},
    {"field": "household_size / dependents", "reason": "Familial status — protected under fair-housing law."},
    {"field": "age / DOB / is_student", "reason": "Age — protected."},
    {"field": "race / national_origin / sex / disability", "reason": "Protected classes; never collected or used."},
]

# --------------------------------------------------------------------------
# Delinquency-bucket ordinal labels (multiclass head).
# --------------------------------------------------------------------------
DELINQ_BUCKETS = ["none", "1-29", "30-59", "60-89", "90+"]


def _delinq_bucket(max_days: float) -> int:
    d = float(max_days)
    if d <= 0:
        return 0
    if d < 30:
        return 1
    if d < 60:
        return 2
    if d < 90:
        return 3
    return 4


# --------------------------------------------------------------------------
# Eligibility predicates (feature-derived; churn/cure heads are gated).
# --------------------------------------------------------------------------
def _elig_cure(features: dict) -> bool:
    # Must match generate_residents.py's training-label gate (CURE_EPS) exactly
    # — a resident with a balance in (0, CURE_EPS] would otherwise be served a
    # prediction from a model that never saw such a row during training (every
    # training example with that balance range was excluded as "already cured").
    return float(features.get("current_balance", 0.0)) > CURE_EPS


def _elig_churn6(features: dict) -> bool:
    mtle = float(features.get("months_to_lease_end", 99))
    return 0 < mtle <= CHURN_HORIZON_MONTHS


def _elig_churn12(features: dict) -> bool:
    mtle = float(features.get("months_to_lease_end", 99))
    return 0 < mtle <= CHURN_12M_HORIZON_MONTHS


# --------------------------------------------------------------------------
# HEAD REGISTRY — single source of truth driving gen / train / serve / eval.
# --------------------------------------------------------------------------
def _H(name, family, kind, objective, feature_order, label_window, learnable,
       band_edges=None, eligibility=None, calibration="isotonic",
       legacy_alias=None, low_confidence=False, routes_to_review=False,
       num_class=None, horizon=None):
    return {
        "name": name, "family": family, "kind": kind, "objective": objective,
        "feature_order": list(feature_order), "label_window": label_window,
        "learnable": learnable, "band_edges": band_edges, "eligibility": eligibility,
        "calibration": calibration, "legacy_alias": legacy_alias,
        "low_confidence": low_confidence, "routes_to_review": routes_to_review,
        "num_class": num_class, "horizon": horizon,
    }


HEADS = [
    # LATE family — cumulative event probabilities → monotone-clamped at serve.
    _H("late_1m", "late", "binary", "binary:logistic", FEATURE_ORDER_BASE,
       "P(any late payment in the next 1 month)", "reliable (short horizon; fewer positives)",
       band_edges=(0.10, 0.30)),
    _H("late_3m", "late", "binary", "binary:logistic", FEATURE_ORDER_BASE,
       "P(any late payment in the next 3 months / quarter)", "reliable",
       # Recalibrated against the actual served-probability distribution: the
       # model's floor sits around ~0.12-0.17 (even a spotless 36mo tenant
       # scores ~0.24), so a 0.15 edge left ~0% of residents ever banding
       # "low" (every property, no exceptions). (0.24, 0.42) gives every
       # property a real low/medium/elevated mix -- verified against the
       # committed dataset (min 9 low / 3 high per property).
       band_edges=(0.24, 0.42), legacy_alias="late"),
    _H("late_6m", "late", "binary", "binary:logistic", FEATURE_ORDER_BASE,
       "P(any late payment in the next 6 months)", "reliable",
       band_edges=(0.25, 0.55)),
    _H("late_12m", "late", "binary", "binary:logistic", FEATURE_ORDER_BASE,
       "P(any late payment in the next 12 months / year)",
       "reliable; low-confidence for short-tenure residents", band_edges=(0.35, 0.65)),
    # FREQUENCY family — overdispersed counts → tweedie.
    _H("late_count_3m", "frequency", "count", "reg:tweedie", FEATURE_ORDER_BASE,
       "Expected number of late months over the next 3 months / quarter (0-3)",
       "reliable point estimate; short horizon keeps the interval tighter than "
       "the 12mo count", calibration="empirical_pi"),
    _H("late_count_6m", "frequency", "count", "reg:tweedie", FEATURE_ORDER_BASE,
       "Expected number of late months over the next 6 months (0-6)",
       "reliable point estimate", calibration="empirical_pi"),
    _H("late_count_9m", "frequency", "count", "reg:tweedie", FEATURE_ORDER_BASE,
       "Expected number of late months over the next 9 months (0-9)",
       "reliable point estimate; interval widened for streak overdispersion",
       calibration="empirical_pi"),
    _H("late_count_12m", "frequency", "count", "reg:tweedie", FEATURE_ORDER_BASE,
       "Expected number of late months over the next 12 months (0-12)",
       "reliable point estimate; interval widened for streak overdispersion",
       calibration="empirical_pi"),
    _H("missed_count_12m", "frequency", "count", "reg:tweedie", FEATURE_ORDER_BASE,
       "Expected number of fully-missed months over the next 12 months",
       "reliable; sparser than late_count so wider interval", calibration="empirical_pi"),
    # SEVERITY family.
    _H("max_days_late_12m", "severity", "regression", "reg:tweedie", FEATURE_ORDER_BASE,
       "Expected worst days-late reached over the next 12 months",
       "moderate; zero-inflated (tweedie handles the mass at 0)", calibration="empirical_pi"),
    _H("p_30d_12m", "severity", "binary", "binary:logistic", FEATURE_ORDER_BASE,
       "P(reaching 30+ days late at any point in the next 12 months)", "reliable",
       band_edges=(0.15, 0.40)),
    _H("p_60d_12m", "severity", "binary", "binary:logistic", FEATURE_ORDER_BASE,
       "P(reaching 60+ days late in the next 12 months)",
       "LOW-POWER: rare event, wide CI", band_edges=(0.08, 0.20), low_confidence=True),
    _H("p_90d_12m", "severity", "binary", "binary:logistic", FEATURE_ORDER_BASE,
       "P(reaching 90+ days late in the next 12 months)",
       "LOW-POWER: very rare, treat as directional only", band_edges=(0.05, 0.15),
       low_confidence=True),
    _H("delinquency_bucket_12m", "severity", "multiclass", "multi:softprob", FEATURE_ORDER_BASE,
       "Worst delinquency bucket in the next 12 months {none,1-29,30-59,60-89,90+}",
       "ordinal; shares statistical strength across buckets", calibration="softprob",
       num_class=5),
    # SERIOUS (legacy; routes to review). Folded into the severity family.
    _H("serious", "severity", "binary", "binary:logistic", FEATURE_ORDER_BASE,
       "P(serious delinquency next quarter: 30+ days late OR a full month in arrears)",
       "reliable; always routed to a human reviewer", band_edges=(0.10, 0.25),
       legacy_alias="serious", routes_to_review=True),
    # ARREARS family — zero-inflated $ → tweedie + empirical PI.
    _H("arrears_3m", "arrears", "regression", "reg:tweedie", FEATURE_ORDER_ARREARS,
       "Expected $ balance at the end of the next 3 months (quarter)", "reliable",
       calibration="empirical_pi", legacy_alias="arrears"),
    _H("arrears_6m", "arrears", "regression", "reg:tweedie", FEATURE_ORDER_ARREARS,
       "Expected $ balance at the end of the next 6 months", "reliable",
       calibration="empirical_pi"),
    _H("arrears_9m", "arrears", "regression", "reg:tweedie", FEATURE_ORDER_ARREARS,
       "Expected $ balance at the end of the next 9 months", "moderate (longer horizon)",
       calibration="empirical_pi"),
    _H("arrears_12m", "arrears", "regression", "reg:tweedie", FEATURE_ORDER_ARREARS,
       "Expected $ balance at the end of the next 12 months", "moderate (longer horizon)",
       calibration="empirical_pi"),
    _H("peak_balance_12m", "arrears", "regression", "reg:tweedie", FEATURE_ORDER_ARREARS,
       "Expected peak $ balance reached over the next 12 months", "moderate",
       calibration="empirical_pi"),
    # CURE family — eligibility: only residents currently carrying a balance.
    _H("p_cure_6m", "cure", "binary", "binary:logistic", FEATURE_ORDER_CURE,
       "P(existing balance is cleared within 6 months) | currently in arrears",
       "reliable on the eligible (in-arrears) subset only", band_edges=(0.34, 0.67),
       eligibility=_elig_cure),
    _H("months_to_cure", "cure", "survival", "binary:logistic", FEATURE_ORDER_CURE,
       "Months until an existing balance clears (discrete-time hazard, censored at 12) | in arrears",
       "directional; heavy censoring on the small in-arrears subset",
       calibration="hazard", eligibility=_elig_cure, horizon=CURE_HORIZON_MONTHS),
    # RETENTION family — churn (non-renewal), eligibility-gated by lease timing.
    _H("churn", "retention", "binary", "binary:logistic", FEATURE_ORDER_CHURN,
       "P(non-renewal) for leases ending within 6 months", "reliable on eligible leases",
       band_edges=(0.20, 0.50), eligibility=_elig_churn6, legacy_alias="churn"),
    _H("churn_12m", "retention", "binary", "binary:logistic", FEATURE_ORDER_CHURN,
       "P(non-renewal) for leases ending within 12 months", "reliable on eligible leases",
       band_edges=(0.20, 0.50), eligibility=_elig_churn12),
]

HEADS_BY_NAME = {h["name"]: h for h in HEADS}
HEAD_NAMES = [h["name"] for h in HEADS]
BINARY_HEAD_NAMES = [h["name"] for h in HEADS if h["kind"] == "binary"]

# family -> ordered head names (drives the families{} block + frontend grouping).
FAMILIES: dict = {}
for _h in HEADS:
    FAMILIES.setdefault(_h["family"], []).append(_h["name"])

# Legacy alias -> head name (late/arrears/churn/serious).
LEGACY_ALIAS = {h["legacy_alias"]: h["name"] for h in HEADS if h["legacy_alias"]}

# --------------------------------------------------------------------------
# Bands per binary head. Elevated routes to human outreach/review, never
# automated action. serious is tighter on the high side (favor recall -> review).
# --------------------------------------------------------------------------
_BAND_EDGES = {h["name"]: h["band_edges"] for h in HEADS if h["band_edges"]}


def _bands_for(edges) -> list:
    lo, hi = edges
    return [
        {"band": "low", "min": 0.0, "max": lo},
        {"band": "medium", "min": lo, "max": hi},
        {"band": "high", "min": hi, "max": 1.0},
    ]


BANDS = {name: _bands_for(edges) for name, edges in _BAND_EDGES.items()}


def _band(head: str, p: float) -> str:
    lo, hi = _BAND_EDGES.get(head, (0.15, 0.40))
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
    # v2 additions — neutral == clean history.
    "trouble_month_rate": 0.0,
    "longest_late_streak_12mo": 0.0,
    "count_30plus_12mo": 0.0,
    "count_60plus_12mo": 0.0,
    "balance_trend_3mo": 0.0,
    "max_balance_12mo": 0.0,
    "months_in_arrears_12mo": 0.0,
    "months_since_balance_zero": 0.0,
}

MONTHS_CAP = 60  # cap for months_since_last_late / on_time_streak / months_since_balance_zero


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
# --------------------------------------------------------------------------
def _as_dict(resident) -> dict:
    if hasattr(resident, "model_dump"):
        return resident.model_dump()
    return dict(resident)


def _is_trouble(e: dict) -> bool:
    return str(e.get("status", "paid")) != "paid"


def extract_resident_features(resident, snapshot: date = RESIDENT_SNAPSHOT) -> dict:
    """Derive the full feature superset (all heads) for one resident as of
    ``snapshot``. Returns a dict; per-head vectors are sliced by feature_order.
    Only ledger + immutable facts are read — never a protected attribute, never
    the display name."""
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
    f["nsf_count_12mo"] = float(sum(1 for e in last12 if float(e.get("amount_paid", 0)) <= 0.0))

    days = [int(e.get("days_late", 0)) for e in last12]
    late_days = [d for d in days if d > 0]
    f["max_days_late_12mo"] = float(max(days)) if days else 0.0
    f["avg_days_late_12mo"] = float(sum(late_days) / len(late_days)) if late_days else 0.0

    streak = 0
    for e in reversed(led):
        if _is_trouble(e):
            break
        streak += 1
    f["on_time_streak_months"] = float(min(streak, MONTHS_CAP))

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

    income = float(r.get("monthly_income") or 0.0) + float(r.get("other_income_monthly") or 0.0)
    if income > 0 and base_rent > 0:
        f["rent_to_income"] = round(_clamp(base_rent / income, 0.0, 3.0), 4)
    else:
        f["rent_to_income"] = _NEUTRAL["rent_to_income"]
    f["income_verified"] = 1.0 if r.get("income_verified") else 0.0
    f["autopay_enrolled"] = 1.0 if r.get("autopay_enrolled") else 0.0

    move_in = r.get("move_in_date") or r.get("lease_start")
    if move_in:
        tenure = _months_between(_to_date(move_in), snapshot)
    else:
        tenure = n
    f["tenure_months"] = float(max(0, min(tenure, 600)))

    f["prior_renewals"] = float(r.get("prior_renewals") or 0)

    sent = sum(int(e.get("notices_sent", 0)) for e in last24)
    resp = sum(int(e.get("notice_responded", 0)) for e in last24)
    f["notice_response_rate"] = round(resp / sent, 4) if sent > 0 else _NEUTRAL["notice_response_rate"]
    f["notices_sent_12mo"] = float(sum(int(e.get("notices_sent", 0)) for e in last12))

    f["maintenance_requests_12mo"] = float(r.get("maintenance_requests_12mo") or 0)
    f["complaints_12mo"] = float(r.get("complaints_12mo") or 0)
    f["portal_logins_90d"] = float(r.get("portal_logins_90d") or 0)

    msl = MONTHS_CAP
    for age, e in enumerate(reversed(led)):
        if _is_trouble(e):
            msl = min(age, MONTHS_CAP)
            break
    f["months_since_last_late"] = float(msl)

    # ---- v2 additions (ledger-derived only) --------------------------------
    troubles_all = sum(1 for e in led if _is_trouble(e))
    f["trouble_month_rate"] = round(troubles_all / n, 4) if n else 0.0

    run = best = 0
    for e in last12:
        if _is_trouble(e):
            run += 1
            best = max(best, run)
        else:
            run = 0
    f["longest_late_streak_12mo"] = float(best)

    f["count_30plus_12mo"] = float(sum(1 for e in last12 if int(e.get("days_late", 0)) >= 30))
    f["count_60plus_12mo"] = float(sum(1 for e in last12 if int(e.get("days_late", 0)) >= 60))

    bal_3ago = float(led[-4]["balance_after"]) if n >= 4 else (float(led[0]["balance_after"]) if n else 0.0)
    f["balance_trend_3mo"] = round((bal_last - bal_3ago) / denom, 4)

    f["max_balance_12mo"] = round(max((float(e.get("balance_after", 0)) for e in last12), default=0.0), 2)
    f["months_in_arrears_12mo"] = float(sum(1 for e in last12 if float(e.get("balance_after", 0)) > 0.0))

    msz = 0
    for e in reversed(led):
        if float(e.get("balance_after", 0)) > 0.0:
            msz += 1
        else:
            break
    f["months_since_balance_zero"] = float(min(msz, MONTHS_CAP))

    # Churn-specific lease timing.
    lease_end = r.get("lease_end")
    mtle = _months_between(snapshot, _to_date(lease_end)) if lease_end else _NEUTRAL["months_to_lease_end"]
    f["months_to_lease_end"] = float(mtle)
    f["lease_term_months"] = float(r.get("lease_term_months") or _NEUTRAL["lease_term_months"])
    f["renewal_offer_sent"] = 1.0 if r.get("renewal_offer_sent") else 0.0

    return f


def feature_vector(features: dict, head: str) -> list:
    fo = HEADS_BY_NAME[head]["feature_order"]
    return [float(features.get(k, _NEUTRAL.get(k, 0.0))) for k in fo]


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
    """Legacy helper: churn is labelled/served only for leases ending within 6mo."""
    return _elig_churn6(features)


# --------------------------------------------------------------------------
# Reason-code templates + up/down mix.
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
    # v2 additions
    "trouble_month_rate": lambda v: (f"Late in {round(v * 100)}% of months on record" if v > 0.01 else "On time in essentially every month"),
    "longest_late_streak_12mo": lambda v: (f"Longest recent late streak: {round(v)} month(s)" if v else "No back-to-back late months recently"),
    "count_30plus_12mo": lambda v: (f"{round(v)} month(s) 30+ days late in the last year" if v else "Never 30+ days late in the last year"),
    "count_60plus_12mo": lambda v: (f"{round(v)} month(s) 60+ days late in the last year" if v else "Never 60+ days late in the last year"),
    "balance_trend_3mo": lambda v: ("Balance rising over 3 months" if v > 0.05 else ("Balance falling over 3 months" if v < -0.05 else "Balance stable over 3 months")),
    "max_balance_12mo": lambda v: (f"Peaked at ${round(v):,} owed in the last year" if v > 1 else "Never carried a balance in the last year"),
    "months_in_arrears_12mo": lambda v: (f"{round(v)} month(s) carrying a balance in the last year" if v else "No months in arrears in the last year"),
    "months_since_balance_zero": lambda v: (f"Carried a balance for {round(v)} straight month(s)" if v else "Currently at a zero balance"),
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
# Transparent per-head heuristics (never return exactly 0 or 1).
# --------------------------------------------------------------------------
_HEURISTICS = {
    "late_3m": (
        -1.2,
        {
            "late_count_6mo": ("raw", 0.55),
            "late_count_12mo": ("raw", 0.18),
            "missed_count_12mo": ("raw", 0.35),
            "partial_count_12mo": ("raw", 0.25),
            "recency_weighted_lateness": ("raw", 0.6),
            "current_balance_ratio": ("raw", 0.5),
            "trouble_month_rate": ("raw", 0.8),
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
            "count_30plus_12mo": ("raw", 0.4),
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
_HEURISTICS["churn_12m"] = _HEURISTICS["churn"]

# Late-horizon exponent map (monotone: exp>1 lowers, exp<1 raises prob).
_LATE_HORIZON_EXP = {"late_1m": 1.7, "late_3m": 1.0, "late_6m": 0.72, "late_12m": 0.5}
# Severity-threshold exponent map off the serious heuristic.
_SEV_EXP = {"p_30d_12m": 0.7, "p_60d_12m": 1.3, "p_90d_12m": 2.0}


def _heuristic_logodds(spec_key: str, features: dict) -> tuple:
    bias, weights = _HEURISTICS[spec_key]
    contribs: dict = {}
    z = bias
    for feat, spec in weights.items():
        v = float(features.get(feat, _NEUTRAL.get(feat, 0.0)))
        kind = spec[0]
        if kind == "dev":
            c = spec[1] * (v - spec[2])
        elif kind in ("raw", "flag"):
            c = spec[1] * v
        elif kind == "flag_low":
            c = spec[1] * (1.0 - v)
        else:
            c = 0.0
        contribs[feat] = c
        z += c
    return z, contribs


def _heuristic_arrears(features: dict, horizon_months: int = LABEL_HORIZON_MONTHS, scale: float = 1.0) -> tuple:
    bal = float(features.get("current_balance", 0.0))
    shortfall = float(features.get("avg_monthly_shortfall_6mo", 0.0))
    late6 = float(features.get("late_count_6mo", 0.0))
    autopay = float(features.get("autopay_enrolled", 0.0))
    persist = _clamp(late6 / 6.0, 0.0, 1.0) * (0.6 if autopay < 0.5 else 0.35)
    expected = bal * (0.6 + 0.4 * persist) + shortfall * horizon_months * persist
    expected = max(0.0, expected) * scale
    contribs = {
        "current_balance": bal * (0.6 + 0.4 * persist),
        "avg_monthly_shortfall_6mo": shortfall * horizon_months * persist,
        "late_count_6mo": late6 * 5.0,
        "autopay_enrolled": -20.0 * autopay,
    }
    return expected, contribs


def _heuristic_head(head: str, features: dict):
    """Return (value, contribs) for any head using transparent weights. value is
    a probability (binary), expected count/amount (count/regression), a class
    probability list (multiclass), or a per-month survival curve (survival)."""
    spec = HEADS_BY_NAME[head]
    fam = spec["family"]

    if head in _LATE_HORIZON_EXP:
        z, contribs = _heuristic_logodds("late_3m", features)
        p3 = _clamp(_sigmoid(z), 0.02, 0.98)
        return _clamp(p3 ** _LATE_HORIZON_EXP[head], 0.02, 0.98), contribs
    if head in _SEV_EXP:
        z, contribs = _heuristic_logodds("serious", features)
        ps = _clamp(_sigmoid(z), 0.02, 0.98)
        return _clamp(ps ** _SEV_EXP[head], 0.01, 0.98), contribs
    if head in ("serious", "churn", "churn_12m"):
        z, contribs = _heuristic_logodds(head, features)
        return _clamp(_sigmoid(z), 0.02, 0.98), contribs

    if head == "late_count_3m":
        rate = float(features.get("trouble_month_rate", 0.0))
        recent = float(features.get("late_count_3mo", 0.0))
        expected = max(0.0, 0.6 * rate * 3.0 + 0.4 * recent)
        return expected, {"trouble_month_rate": rate * 1.8, "late_count_3mo": recent * 0.4}
    if head == "late_count_6m":
        rate = float(features.get("trouble_month_rate", 0.0))
        recent = float(features.get("late_count_6mo", 0.0))
        expected = max(0.0, 0.6 * rate * 6.0 + 0.4 * recent)
        return expected, {"trouble_month_rate": rate * 3.6, "late_count_6mo": recent * 0.4}
    if head == "late_count_9m":
        rate = float(features.get("trouble_month_rate", 0.0))
        # No late_count_9mo feature exists (the ledger contract only has
        # 3/6/12/24mo windows) -- the 6mo window is the closest real feature.
        recent = float(features.get("late_count_6mo", 0.0))
        expected = max(0.0, 0.6 * rate * 9.0 + 0.4 * recent)
        return expected, {"trouble_month_rate": rate * 5.4, "late_count_6mo": recent * 0.4}
    if head == "late_count_12m":
        rate = float(features.get("trouble_month_rate", 0.0))
        recent = float(features.get("late_count_12mo", 0.0))
        expected = max(0.0, 0.6 * rate * 12.0 + 0.4 * recent)
        return expected, {"trouble_month_rate": rate * 7.2, "late_count_12mo": recent * 0.4}
    if head == "missed_count_12m":
        missed = float(features.get("missed_count_12mo", 0.0))
        nsf = float(features.get("nsf_count_12mo", 0.0))
        expected = max(0.0, 0.7 * missed + 0.3 * nsf)
        return expected, {"missed_count_12mo": missed * 0.7, "nsf_count_12mo": nsf * 0.3}
    if head == "max_days_late_12m":
        mdl = float(features.get("max_days_late_12mo", 0.0))
        rate = float(features.get("trouble_month_rate", 0.0))
        expected = max(0.0, 0.7 * mdl + 20.0 * rate)
        return expected, {"max_days_late_12mo": mdl * 0.7, "trouble_month_rate": rate * 20.0}
    if fam == "arrears":
        # (horizon_months, scale) per checkpoint — scale interpolates between
        # arrears_3m's 1.0 and arrears_12m's 1.35 for the 6/9-month points;
        # peak gets its own higher scale since it's a max, not an endpoint.
        horizon, scale = {
            "arrears_3m": (3, 1.0), "arrears_6m": (6, 1.12), "arrears_9m": (9, 1.23),
            "arrears_12m": (12, 1.35), "peak_balance_12m": (12, 1.5),
        }.get(head, (12, 1.35))
        expected, contribs = _heuristic_arrears(features, horizon, scale)
        if head == "peak_balance_12m":
            expected = max(expected, float(features.get("max_balance_12mo", 0.0)),
                           float(features.get("current_balance", 0.0)))
        return expected, contribs
    if head == "p_cure_6m":
        bal_ratio = float(features.get("current_balance_ratio", 0.0))
        autopay = float(features.get("autopay_enrolled", 0.0))
        rwl = float(features.get("recency_weighted_lateness", 0.0))
        z = 0.9 - 1.1 * bal_ratio + 0.6 * autopay - 0.5 * rwl
        contribs = {
            "current_balance_ratio": -1.1 * bal_ratio,
            "autopay_enrolled": 0.6 * autopay,
            "recency_weighted_lateness": -0.5 * rwl,
        }
        return _clamp(_sigmoid(z), 0.02, 0.98), contribs
    if head == "delinquency_bucket_12m":
        z, contribs = _heuristic_logodds("serious", features)
        ps = _clamp(_sigmoid(z), 0.02, 0.98)
        mdl = float(features.get("max_days_late_12mo", 0.0))
        w = [1.0 - ps, ps * 0.5, ps * 0.3 + (0.2 if mdl >= 30 else 0.0),
             ps * 0.15 + (0.15 if mdl >= 60 else 0.0), ps * 0.05 + (0.1 if mdl >= 90 else 0.0)]
        tot = sum(w) or 1.0
        return [x / tot for x in w], contribs
    if head == "months_to_cure":
        bal_ratio = float(features.get("current_balance_ratio", 0.0))
        autopay = float(features.get("autopay_enrolled", 0.0))
        h = _clamp(0.35 + 0.25 * autopay - 0.25 * bal_ratio, 0.05, 0.9)
        curve = []
        s = 1.0
        for _ in range(CURE_HORIZON_MONTHS):
            s *= (1.0 - h)
            curve.append(round(s, 4))
        return curve, {"current_balance_ratio": -0.25 * bal_ratio, "autopay_enrolled": 0.25 * autopay}

    return 0.3, {}


# --------------------------------------------------------------------------
# Model loading (lru_cache; None on ANY failure or contract mismatch)
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _model():
    """Load the persisted multi-head bundle, or None if the file itself is
    missing/unreadable/wrong-schema. Each head is then validated
    INDEPENDENTLY — a feature_order drift or missing required key in one head
    prunes just that head from the returned bundle (so it falls back to its
    own heuristic), rather than invalidating the whole bundle and reverting
    all 19 heads to heuristics over one stale head."""
    try:
        import joblib

        if not ARTIFACT_PATH.exists():
            return None
        bundle = joblib.load(ARTIFACT_PATH)
        if not isinstance(bundle, dict) or bundle.get("schema") != BUNDLE_SCHEMA:
            return None
        heads = bundle.get("heads")
        if not isinstance(heads, dict):
            return None
        required = {
            "binary": "calibrated_model", "multiclass": "calibrated_model",
            "count": "regressor", "regression": "regressor", "survival": "hazard_model",
        }
        pruned_heads = dict(heads)
        for spec in HEADS:
            name = spec["name"]
            sub = pruned_heads.get(name)
            if not isinstance(sub, dict):
                continue  # already absent; per-head predict code treats this as heuristic
            key = required[spec["kind"]]
            invalid = sub.get("feature_order") != spec["feature_order"]
            if not invalid and key not in sub and spec["kind"] != "survival":
                # survival may legitimately lack hazard_model (too few
                # eligible rows at training time) -> heuristic, same as
                # always; every OTHER kind missing its key is a real defect.
                invalid = True
            if invalid:
                # Strip only the servable model key — every per-head predict
                # function checks `key in sub` (or catches the resulting
                # KeyError) and falls back to its heuristic, so this head
                # degrades on its own instead of invalidating the whole
                # bundle. metrics/feature_order/etc. stay intact so the model
                # card can still report this head's training-time numbers.
                print(f"residents_risk: head {name!r} failed validation "
                      f"(contract drift or missing {key!r}); heuristic for "
                      f"this head only.")
                stripped = dict(sub)
                stripped.pop(key, None)
                pruned_heads[name] = stripped
        return {**bundle, "heads": pruned_heads}
    except Exception as exc:  # noqa: BLE001 — degrade to heuristics
        print(f"residents_risk: model load failed ({type(exc).__name__}: {exc}); using heuristics.")
        return None


def _tree_contribs(booster, features: dict, feature_order: list):
    """TreeSHAP signed contributions from a single-output raw booster, or None."""
    if booster is None:
        return None
    try:
        import pandas as pd
        import xgboost as xgb

        row = pd.DataFrame([[float(features.get(k, _NEUTRAL.get(k, 0.0))) for k in feature_order]], columns=feature_order)
        contribs = booster.predict(xgb.DMatrix(row), pred_contribs=True)[0]
        if getattr(contribs, "ndim", 1) != 1:  # multiclass -> not single-output
            return None
        return {feature_order[i]: float(contribs[i]) for i in range(len(feature_order))}
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------
# Per-head prediction — each degrades to its heuristic independently.
# --------------------------------------------------------------------------
def _row(features: dict, feature_order: list):
    import pandas as pd

    return pd.DataFrame(
        [[float(features.get(k, _NEUTRAL.get(k, 0.0))) for k in feature_order]],
        columns=feature_order,
    )


def _predict_head(spec: dict, features: dict, bundle, low_conf: bool, with_reasons: bool = True) -> dict:
    """Predict one head. Returns the head payload dict. Never raises. When
    ``with_reasons`` is False, ALL TreeSHAP / heuristic reason-code computation is
    skipped (reason_codes=[]) — the fast bulk path."""
    kind = spec["kind"]
    elig = spec["eligibility"]
    low = low_conf or spec["low_confidence"]

    if elig is not None and not elig(features):
        payload = {
            "kind": kind, "family": spec["family"], "band": "not_applicable",
            "reason_codes": [], "confidence": "low" if low else "high",
            "source": "not_applicable", "model_type": "not_applicable",
        }
        if kind == "binary":
            payload["probability"] = None
            payload["range"] = []
        elif kind in ("count", "regression"):
            payload["expected"] = None
            payload["interval"] = []
            if spec["family"] == "arrears":
                payload["expected_balance"] = None
        elif kind == "survival":
            payload["median_months"] = None
            payload["expected_months"] = None
            payload["survival_curve"] = []
        if spec["family"] == "retention":
            payload["months_to_lease_end"] = round(float(features.get("months_to_lease_end", 0.0)), 1)
        return payload

    sub = (bundle or {}).get("heads", {}).get(spec["name"]) if bundle else None

    if kind in ("binary", "multiclass"):
        return _predict_classifier(spec, features, sub, low, with_reasons)
    if kind in ("count", "regression"):
        return _predict_regression(spec, features, sub, low, with_reasons)
    if kind == "survival":
        return _predict_survival(spec, features, sub, low, with_reasons)
    return {"kind": kind, "source": "heuristic", "confidence": "low", "reason_codes": []}


def _head_reasons(name, features, fo, booster, with_reasons) -> list:
    """Reason codes for one head, or [] when reasons are disabled (bulk path).
    TreeSHAP if a booster is present, else transparent heuristic contribs."""
    if not with_reasons:
        return []
    contribs = _tree_contribs(booster, features, fo) if booster is not None else None
    if contribs is None:
        _, contribs = _heuristic_head(name, features)
    return _reason_codes(contribs, features, fo)


def _predict_classifier(spec, features, sub, low, with_reasons=True) -> dict:
    name, fo, kind = spec["name"], spec["feature_order"], spec["kind"]
    edge_half = 0.08
    source, model_type = "heuristic", "heuristic"
    if sub is not None:
        try:
            edge_half = float(sub.get("range_half_width", edge_half))
            proba = sub["calibrated_model"].predict_proba(_row(features, fo))[0]
            reasons = _head_reasons(name, features, fo, sub.get("booster"), with_reasons)
            source, model_type = "model", sub.get("model_type", "xgboost")
            if kind == "multiclass":
                return _multiclass_payload(spec, proba, reasons, low, source, model_type)
            return _binary_payload(spec, float(proba[1]), reasons, low, edge_half, source, model_type)
        except Exception as exc:  # noqa: BLE001 — degrade to heuristic
            print(f"residents_risk: {name} model predict failed ({type(exc).__name__}: {exc}); heuristic.")
    val, contribs = _heuristic_head(name, features)
    reasons = _reason_codes(contribs, features, fo) if with_reasons else []
    if kind == "multiclass":
        return _multiclass_payload(spec, val, reasons, low, source, model_type)
    return _binary_payload(spec, val, reasons, low, edge_half, source, model_type)


def _binary_payload(spec, p, reasons, low, edge_half, source, model_type) -> dict:
    p = _clamp(p)
    half = edge_half + (0.07 if low else 0.0)
    payload = {
        "kind": "binary", "family": spec["family"],
        "probability": round(p, 4),
        "band": _band(spec["name"], p),
        "range": [round(_clamp(p - half), 4), round(_clamp(p + half), 4)],
        "reason_codes": reasons,
        "confidence": "low" if low else "high",
        "source": source, "model_type": model_type,
    }
    if spec["routes_to_review"]:
        payload["routes_to_review"] = True
    if spec["family"] == "retention":
        payload["months_to_lease_end"] = round(float(spec.get("_mtle", 0.0)), 1)
    return payload


def _multiclass_payload(spec, proba, reasons, low, source, model_type) -> dict:
    probs = [float(x) for x in proba]
    if len(probs) < len(DELINQ_BUCKETS):
        probs = probs + [0.0] * (len(DELINQ_BUCKETS) - len(probs))
    idx = max(range(len(probs)), key=lambda i: probs[i])
    return {
        "kind": "multiclass", "family": spec["family"],
        "class_probs": {DELINQ_BUCKETS[i]: round(probs[i], 4) for i in range(len(DELINQ_BUCKETS))},
        "predicted_bucket": DELINQ_BUCKETS[idx],
        "reason_codes": reasons,
        "confidence": "low" if low else "high",
        "source": source, "model_type": model_type,
    }


def _predict_regression(spec, features, sub, low, with_reasons=True) -> dict:
    name, fo = spec["name"], spec["feature_order"]
    source, model_type = "heuristic", "heuristic"
    if sub is not None:
        try:
            pred = max(0.0, float(sub["regressor"].predict(_row(features, fo))[0]))
            q = sub.get("residual_quantiles") or [-0.5 * pred, 0.5 * pred]
            lo = max(0.0, pred + float(q[0]))
            hi = max(lo, pred + float(q[1]))
            if low:
                span = (hi - lo) * 0.2
                lo = max(0.0, lo - span)
                hi = hi + span
            reasons = _head_reasons(name, features, fo, sub.get("booster"), with_reasons)
            source, model_type = "model", sub.get("model_type", "xgboost")
            return _regression_payload(spec, pred, [lo, hi], reasons, low, source, model_type)
        except Exception as exc:  # noqa: BLE001
            print(f"residents_risk: {name} model predict failed ({type(exc).__name__}: {exc}); heuristic.")
    pred, contribs = _heuristic_head(name, features)
    pred = max(0.0, float(pred))
    if spec["kind"] == "count":
        half = max(1.0, 0.6 * pred + 1.0) * (1.3 if low else 1.0)
    else:
        std = max(50.0, 0.4 * pred + 40.0)
        half = 1.28 * std * (1.4 if low else 1.0)
    reasons = _reason_codes(contribs, features, fo) if with_reasons else []
    return _regression_payload(spec, pred, [max(0.0, pred - half), pred + half], reasons, low, source, model_type)


def _regression_payload(spec, pred, interval, reasons, low, source, model_type) -> dict:
    is_count = spec["kind"] == "count"
    val = round(pred, 3) if is_count else round(pred, 2)
    payload = {
        "kind": spec["kind"], "family": spec["family"],
        "expected": val,
        "interval": [round(interval[0], 2), round(interval[1], 2)],
        "reason_codes": reasons,
        "confidence": "low" if low else "high",
        "source": source, "model_type": model_type,
    }
    if spec["family"] == "arrears":
        payload["expected_balance"] = round(pred, 2)  # legacy-alias-friendly key
    return payload


def _predict_survival(spec, features, sub, low, with_reasons=True) -> dict:
    name, fo = spec["name"], spec["feature_order"]
    horizon = spec["horizon"] or CURE_HORIZON_MONTHS
    source, model_type = "heuristic", "heuristic"
    curve = None
    reasons: list = []
    if sub is not None and "hazard_model" in sub:
        try:
            import pandas as pd

            haz = sub["hazard_model"]
            haz_fo = sub.get("hazard_feature_order", fo + ["horizon_month"])
            base = {k: float(features.get(k, _NEUTRAL.get(k, 0.0))) for k in fo}
            rows = []
            for m in range(1, horizon + 1):
                r = dict(base)
                r["horizon_month"] = float(m)
                rows.append([r[k] for k in haz_fo])
            hazards = haz.predict_proba(pd.DataFrame(rows, columns=haz_fo))[:, 1]
            curve = []
            s = 1.0
            for h in hazards:
                s *= (1.0 - float(h))
                curve.append(round(s, 4))
            reasons = _head_reasons(name, features, fo, sub.get("booster"), with_reasons)
            source, model_type = "model", sub.get("model_type", "xgboost")
        except Exception as exc:  # noqa: BLE001
            print(f"residents_risk: {name} survival predict failed ({type(exc).__name__}: {exc}); heuristic.")
            curve = None
    if curve is None:
        curve, contribs = _heuristic_head(name, features)
        reasons = _reason_codes(contribs, features, fo) if with_reasons else []
    # survival_curve[k] = P(still in arrears after k+1 months). Median = first month it drops <= 0.5.
    median = None
    for i, s in enumerate(curve):
        if s <= 0.5:
            median = i + 1
            break
    return {
        "kind": "survival", "family": spec["family"],
        "median_months": median,
        "expected_months": round(sum(curve), 2),  # expected months still in arrears (area under S)
        "survival_curve": curve,
        "reason_codes": reasons,
        "confidence": "low" if low else "high",
        "source": source, "model_type": model_type,
    }


# --------------------------------------------------------------------------
# Serve-time monotone clamps across related heads.
# --------------------------------------------------------------------------
def _reband(h: dict, name: str, p: float) -> None:
    h["probability"] = p
    h["band"] = _band(name, p)
    half = (h["range"][1] - h["range"][0]) / 2 if len(h.get("range", [])) == 2 else 0.08
    h["range"] = [round(_clamp(p - half), 4), round(_clamp(p + half), 4)]


def _apply_monotone_clamps(heads: dict) -> None:
    # Cumulative late probabilities must be non-decreasing in horizon.
    prev = None
    for name in ("late_1m", "late_3m", "late_6m", "late_12m"):
        h = heads.get(name)
        if not h or h.get("probability") is None:
            continue
        if prev is not None and h["probability"] < prev:
            _reband(h, name, prev)
        prev = h["probability"]
    # Threshold-severity probabilities must be non-increasing: p30 >= p60 >= p90.
    prev = None
    for name in ("p_30d_12m", "p_60d_12m", "p_90d_12m"):
        h = heads.get(name)
        if not h or h.get("probability") is None:
            continue
        if prev is not None and h["probability"] > prev:
            _reband(h, name, prev)
        prev = h["probability"]


# --------------------------------------------------------------------------
# Legacy-alias projection (exact v1 shapes for late / arrears / churn / serious).
# --------------------------------------------------------------------------
def _legacy_late(h: dict) -> dict:
    return {
        "probability": h["probability"], "band": h["band"], "range": h["range"],
        "reason_codes": h["reason_codes"], "confidence": h["confidence"],
        "source": h["source"], "model_type": h["model_type"],
    }


def _legacy_serious(h: dict) -> dict:
    d = _legacy_late(h)
    d["routes_to_review"] = True
    return d


def _legacy_arrears(h: dict) -> dict:
    return {
        "expected_balance": h.get("expected_balance", h.get("expected") or 0.0) or 0.0,
        "interval": h.get("interval", []),
        "reason_codes": h["reason_codes"], "confidence": h["confidence"],
        "source": h["source"], "model_type": h["model_type"],
    }


def _legacy_churn(h: dict) -> dict:
    return {
        "probability": h.get("probability"),
        "band": h.get("band", "not_applicable"),
        "months_to_lease_end": h.get("months_to_lease_end"),
        "reason_codes": h.get("reason_codes", []),
        "confidence": h.get("confidence", "high"),
        "source": h.get("source", "not_applicable"),
    }


# --------------------------------------------------------------------------
# Public entrypoint — NEVER raises
# --------------------------------------------------------------------------
# Heads always needed so the legacy top-level aliases are always present.
_ALIAS_HEADS = frozenset({"late_3m", "serious", "arrears_3m", "churn"})
# Minimal head set the bulk rows + property-health components need. late_1m is
# included so the late-horizon monotone clamp on late_3m matches the full detail
# path exactly (keeps row/health late probabilities identical to /residents/{id}).
BULK_HEADS = ["late_1m", "late_3m", "serious", "churn", "arrears_3m", "p_30d_12m"]

# Per-resident prediction cache. Data + model + snapshot are STATIC and
# deterministic for a running server, so committed residents (with an id) are
# memoized by (resident_id, snapshot, with_reasons, heads-key). Cleared on
# (re)train via ``_clear_prediction_caches`` / ``ensure_model``.
_PRED_CACHE: dict = {}
_PORTFOLIO_CACHE: dict = {}


def _clear_prediction_caches() -> None:
    _PRED_CACHE.clear()
    _PORTFOLIO_CACHE.clear()


def predict_resident(resident, snapshot: date = RESIDENT_SNAPSHOT,
                     with_reasons: bool = True, heads: list = None) -> dict:
    """Score one resident. Returns a backward-compatible superset: legacy
    top-level late/arrears/churn/serious + heads{} + families{}. Never raises.

    Fast bulk path:
      * ``with_reasons=False`` SKIPS all TreeSHAP / heuristic reason-code work
        (reason_codes=[]) — the dominant cost when scoring the whole portfolio.
      * ``heads`` (a list of head names) computes ONLY those heads (plus the four
        alias heads), instead of all 19.
    Committed residents (those with a ``resident_id``) are cached deterministically."""
    r = _as_dict(resident)
    rid = r.get("resident_id", "")
    snap_iso = _to_date(snapshot).isoformat()
    want = None if heads is None else (frozenset(heads) | _ALIAS_HEADS)
    cache_key = (rid, snap_iso, bool(with_reasons), want)
    if rid and cache_key in _PRED_CACHE:
        return _PRED_CACHE[cache_key]

    scored_at = datetime.now(timezone.utc).isoformat()
    try:
        features = extract_resident_features(r, snapshot)
    except Exception as exc:  # noqa: BLE001 — extraction must never break scoring
        print(f"residents_risk: feature extraction failed ({type(exc).__name__}: {exc}); neutral features.")
        features = dict(_NEUTRAL)
    low_conf = _confidence(r, features) == "low"
    bundle = _model()
    mtle = round(float(features.get("months_to_lease_end", 0.0)), 1)

    heads_out: dict = {}
    for base_spec in HEADS:
        if want is not None and base_spec["name"] not in want:
            continue
        spec = dict(base_spec)
        spec["_mtle"] = mtle
        try:
            heads_out[spec["name"]] = _predict_head(spec, features, bundle, low_conf, with_reasons)
        except Exception as exc:  # noqa: BLE001 — a broken head never sinks the rest
            print(f"residents_risk: head {spec['name']} failed ({type(exc).__name__}: {exc}); neutral.")
            heads_out[spec["name"]] = {
                "kind": spec["kind"], "family": spec["family"], "source": "heuristic",
                "confidence": "low", "reason_codes": [],
                **({"probability": 0.3, "band": "low", "range": [0.2, 0.4], "model_type": "heuristic"}
                   if spec["kind"] == "binary" else {}),
            }

    _apply_monotone_clamps(heads_out)

    result = {
        "resident_id": rid,
        "property_id": r.get("property_id", ""),
        "name": r.get("name", ""),
        "snapshot_date": snap_iso,
        "late": _legacy_late(heads_out["late_3m"]),
        "arrears": _legacy_arrears(heads_out["arrears_3m"]),
        "churn": _legacy_churn(heads_out["churn"]),
        "serious": _legacy_serious(heads_out["serious"]),
        "heads": heads_out,
        "families": {fam: list(names) for fam, names in FAMILIES.items()},
        "scored_at": scored_at,
    }
    if rid:
        _PRED_CACHE[cache_key] = result
    return result


def predict_residents(residents: list, snapshot: date = RESIDENT_SNAPSHOT) -> list:
    """Score a list of residents. Never raises (per-resident guarded)."""
    out = []
    for r in residents or []:
        try:
            out.append(predict_resident(r, snapshot))
        except Exception as exc:  # noqa: BLE001
            print(f"residents_risk: predict_residents skipped one ({type(exc).__name__}: {exc}).")
    return out


def predict_bulk(residents: list, snapshot: date = RESIDENT_SNAPSHOT,
                 heads: list = None) -> list:
    """VECTORIZED bulk scoring for portfolio views. Runs ONE model call per head
    across ALL residents (not one call per resident) — the fix for the per-call
    predict_proba overhead that made whole-portfolio scoring slow. Never computes
    reason codes (bulk path). Returns the same superset shape as
    ``predict_resident`` (legacy aliases + heads{} + families{}) for the requested
    heads, and populates the per-resident cache so later reads are instant.

    Falls back to per-resident ``predict_resident`` on any batch failure."""
    residents = [_as_dict(r) for r in residents or []]
    if not residents:
        return []
    snap_iso = _to_date(snapshot).isoformat()
    want = _ALIAS_HEADS if heads is None else (frozenset(heads) | _ALIAS_HEADS)
    want_specs = [h for h in HEADS if h["name"] in want]
    bundle = _model()
    scored_at = datetime.now(timezone.utc).isoformat()

    try:
        import numpy as np
        import pandas as pd

        feats = [extract_resident_features(r, snapshot) for r in residents]
        lows = [_confidence(residents[i], feats[i]) == "low" for i in range(len(residents))]

        # One batched model call per head over all residents.
        batched: dict = {}
        for spec in want_specs:
            name, fo, kind = spec["name"], spec["feature_order"], spec["kind"]
            sub = (bundle or {}).get("heads", {}).get(name) if bundle else None
            key = "regressor" if kind in ("count", "regression") else "calibrated_model"
            if sub is not None and key in sub:
                try:
                    X = pd.DataFrame(
                        [[float(f.get(k, _NEUTRAL.get(k, 0.0))) for k in fo] for f in feats],
                        columns=fo,
                    )
                    if kind in ("count", "regression"):
                        vals = np.clip(sub["regressor"].predict(X), 0.0, None)
                    elif kind == "multiclass":
                        vals = sub["calibrated_model"].predict_proba(X)
                    else:
                        vals = sub["calibrated_model"].predict_proba(X)[:, 1]
                    batched[name] = (sub, vals)
                    continue
                except Exception as exc:  # noqa: BLE001 — this head degrades to heuristic
                    print(f"residents_risk: bulk {name} predict failed ({type(exc).__name__}: {exc}); heuristic.")
            batched[name] = (None, None)

        out = []
        for i, r in enumerate(residents):
            f, low = feats[i], lows[i]
            mtle = round(float(f.get("months_to_lease_end", 0.0)), 1)
            heads_out: dict = {}
            for spec in want_specs:
                s = dict(spec)
                s["_mtle"] = mtle
                heads_out[spec["name"]] = _bulk_head_payload(s, f, batched.get(spec["name"]), i, low)
            _apply_monotone_clamps(heads_out)
            result = {
                "resident_id": r.get("resident_id", ""),
                "property_id": r.get("property_id", ""),
                "name": r.get("name", ""),
                "snapshot_date": snap_iso,
                "late": _legacy_late(heads_out["late_3m"]),
                "arrears": _legacy_arrears(heads_out["arrears_3m"]),
                "churn": _legacy_churn(heads_out["churn"]),
                "serious": _legacy_serious(heads_out["serious"]),
                "heads": heads_out,
                "families": {fam: list(names) for fam, names in FAMILIES.items()},
                "scored_at": scored_at,
            }
            rid = result["resident_id"]
            if rid:
                _PRED_CACHE[(rid, snap_iso, False, want)] = result
            out.append(result)
        return out
    except Exception as exc:  # noqa: BLE001 — never raise; fall back to per-resident path
        print(f"residents_risk: predict_bulk failed ({type(exc).__name__}: {exc}); per-resident fallback.")
        return [predict_resident(r, snapshot, with_reasons=False, heads=heads) for r in residents]


def _bulk_head_payload(spec, features, batch, i, low) -> dict:
    """Assemble one head payload from a precomputed BATCHED value (no reasons).
    Mirrors the eligibility / kind branches of ``_predict_head``."""
    kind = spec["kind"]
    elig = spec["eligibility"]
    lo = low or spec["low_confidence"]

    if elig is not None and not elig(features):
        payload = {"kind": kind, "family": spec["family"], "band": "not_applicable",
                   "reason_codes": [], "confidence": "low" if lo else "high",
                   "source": "not_applicable", "model_type": "not_applicable"}
        if kind == "binary":
            payload["probability"] = None
            payload["range"] = []
        elif kind in ("count", "regression"):
            payload["expected"] = None
            payload["interval"] = []
            if spec["family"] == "arrears":
                payload["expected_balance"] = None
        if spec["family"] == "retention":
            payload["months_to_lease_end"] = round(float(features.get("months_to_lease_end", 0.0)), 1)
        return payload

    sub, vals = (batch or (None, None))
    if sub is not None and vals is not None:
        if kind == "binary":
            edge_half = float(sub.get("range_half_width", 0.08))
            return _binary_payload(spec, float(vals[i]), [], lo, edge_half, "model", sub.get("model_type", "xgboost"))
        if kind == "multiclass":
            return _multiclass_payload(spec, vals[i], [], lo, "model", sub.get("model_type", "xgboost"))
        # count / regression
        pred = max(0.0, float(vals[i]))
        q = sub.get("residual_quantiles") or [-0.5 * pred, 0.5 * pred]
        plo = max(0.0, pred + float(q[0]))
        phi = max(plo, pred + float(q[1]))
        if lo:
            span = (phi - plo) * 0.2
            plo = max(0.0, plo - span)
            phi = phi + span
        return _regression_payload(spec, pred, [plo, phi], [], lo, "model", sub.get("model_type", "xgboost"))

    # Heuristic fallback (no model / batch failed) — no reasons.
    val, _ = _heuristic_head(spec["name"], features)
    if kind == "binary":
        return _binary_payload(spec, val, [], lo, 0.08, "heuristic", "heuristic")
    if kind == "multiclass":
        return _multiclass_payload(spec, val, [], lo, "heuristic", "heuristic")
    pred = max(0.0, float(val))
    half = (max(1.0, 0.6 * pred + 1.0) if kind == "count" else 1.28 * max(50.0, 0.4 * pred + 40.0)) * (1.3 if lo else 1.0)
    return _regression_payload(spec, pred, [max(0.0, pred - half), pred + half], [], lo, "heuristic", "heuristic")


def top_driver(sub_result: dict) -> str:
    """Strongest 'increases' reason label from a sub-result (for portfolio rows)."""
    ups = [rc for rc in (sub_result or {}).get("reason_codes", []) if rc.get("direction") == "increases"]
    if not ups:
        return ""
    return max(ups, key=lambda rc: abs(rc.get("contribution", 0)))["label"]


def heuristic_top_driver(features: dict, head: str = "late_3m") -> str:
    """Cheap top-driver label for ``head`` from transparent heuristic weights —
    NO TreeSHAP. Lets bulk rows keep a driver label without the per-head SHAP
    cost. Defaults to the late-payment head; pass ``head="churn"`` for a
    churn/retention-framed ranking."""
    try:
        _, contribs = _heuristic_head(head, features)
        return top_driver({"reason_codes": _reason_codes(contribs, features, FEATURE_ORDER_BASE)})
    except Exception:  # noqa: BLE001
        return ""


# --------------------------------------------------------------------------
# Property / portfolio health (regional-director best->worst ranking)
# --------------------------------------------------------------------------
_HEALTH_WEIGHTS = {
    "on_time": 0.30,        # 1 - predicted next-quarter late rate
    "not_serious": 0.25,    # 1 - serious-flag rate
    "retention": 0.20,      # 1 - churn-risk rate (among eligible)
    "collection": 0.15,     # 1 - normalized total expected arrears / rent roll
    "not_chronic": 0.10,    # 1 - mean P(30+ days late next 12mo)
}
_HEALTH_LABELS = {
    "on_time": "on-time payment rate",
    "not_serious": "serious-delinquency flags",
    "retention": "renewal / churn risk",
    "collection": "arrears vs rent roll",
    "not_chronic": "chronic-delinquency risk",
}


def _grade(score: float) -> str:
    # Graded off the rounded (displayed) score so the letter always matches
    # what's shown, e.g. a displayed "85" is never a hair under an A cutoff.
    s = round(score)
    if s >= 85:
        return "A"
    if s >= 75:
        return "B"
    if s >= 65:
        return "C"
    if s >= 55:
        return "D"
    return "F"


def _health_from_preds(property_id: str, residents: list, preds: list) -> dict:
    n = len(preds)
    if n == 0:
        comps = {k: {"value": 1.0, "weight": w, "contribution": round(100 * w, 1)}
                 for k, w in _HEALTH_WEIGHTS.items()}
        return {"property_id": property_id, "score": 100.0, "grade": "A",
                "resident_count": 0, "components": comps, "drivers": [], "top_driver": ""}

    late_rate = sum((p["late"].get("probability") or 0.0) for p in preds) / n
    serious_flag_rate = sum(1 for p in preds if p["serious"].get("band") == "high") / n
    elig = [p for p in preds if p["churn"].get("probability") is not None]
    churn_risk_rate = (sum(1 for p in elig if p["churn"].get("band") == "high") / len(elig)) if elig else 0.0
    total_arrears = sum((p["arrears"].get("expected_balance") or 0.0) for p in preds)
    rent_roll = sum(float((r or {}).get("base_rent") or 0.0) for r in residents) or 1.0
    collection = 1.0 - _clamp(total_arrears / rent_roll, 0.0, 1.0)
    mean_p30 = sum((p["heads"].get("p_30d_12m", {}).get("probability") or 0.0) for p in preds) / n

    values = {
        "on_time": _clamp(1.0 - late_rate),
        "not_serious": _clamp(1.0 - serious_flag_rate),
        "retention": _clamp(1.0 - churn_risk_rate),
        "collection": _clamp(collection),
        "not_chronic": _clamp(1.0 - mean_p30),
    }
    score = round(100.0 * sum(_HEALTH_WEIGHTS[k] * values[k] for k in _HEALTH_WEIGHTS), 1)
    components = {
        k: {"value": round(values[k], 4), "weight": _HEALTH_WEIGHTS[k],
            "contribution": round(100.0 * _HEALTH_WEIGHTS[k] * values[k], 1)}
        for k in _HEALTH_WEIGHTS
    }
    drags = sorted(_HEALTH_WEIGHTS.keys(), key=lambda k: -(_HEALTH_WEIGHTS[k] * (1.0 - values[k])))
    drivers = [
        {"component": k, "label": _HEALTH_LABELS[k],
         "lost_points": round(100.0 * _HEALTH_WEIGHTS[k] * (1.0 - values[k]), 1)}
        for k in drags if (1.0 - values[k]) > 1e-6
    ][:3]
    return {
        "property_id": property_id, "score": score, "grade": _grade(score),
        "resident_count": n, "components": components, "drivers": drivers,
        "top_driver": drivers[0]["label"] if drivers else "healthy across the board",
    }


def property_health(property_id: str, snapshot: date = RESIDENT_SNAPSHOT) -> dict:
    """Composite health (0-100 + letter grade) for one property, from its
    residents' predictions. Higher = healthier. Never raises."""
    try:
        residents = [r for r in load_residents() if r.get("property_id") == property_id]
        # Fast bulk path: only the heads health needs, no reason codes.
        preds = predict_bulk(residents, snapshot, heads=BULK_HEADS)
        return _health_from_preds(property_id, residents, preds)
    except Exception as exc:  # noqa: BLE001
        print(f"residents_risk.property_health failed ({type(exc).__name__}: {exc}).")
        return {"property_id": property_id, "score": 0.0, "grade": "F",
                "resident_count": 0, "components": {}, "drivers": [], "top_driver": ""}


def portfolio_health(snapshot: date = RESIDENT_SNAPSHOT) -> list:
    """Ranked property-health list, healthiest first. Memoized by snapshot (data
    + model are static). Uses the fast bulk path (no reasons, needed heads only).
    Never raises."""
    snap_iso = _to_date(snapshot).isoformat()
    if snap_iso in _PORTFOLIO_CACHE:
        return _PORTFOLIO_CACHE[snap_iso]
    try:
        by_prop: dict = {}
        for r in load_residents():
            by_prop.setdefault(r.get("property_id"), []).append(r)
        ordered = [p for p in RESIDENT_PROPERTY_IDS if p in by_prop]
        ordered += [p for p in by_prop if p not in ordered]
        out = []
        for pid in ordered:
            residents = by_prop[pid]
            preds = predict_bulk(residents, snapshot, heads=BULK_HEADS)
            out.append(_health_from_preds(pid, residents, preds))
        out.sort(key=lambda h: -h["score"])
        _PORTFOLIO_CACHE[snap_iso] = out
        return out
    except Exception as exc:  # noqa: BLE001
        print(f"residents_risk.portfolio_health failed ({type(exc).__name__}: {exc}).")
        return []


# --------------------------------------------------------------------------
# Late-payment count forecast — sums a genuine frequency head (``late_count_3m``
# for the quarterly view, ``late_count_6m``/``late_count_9m`` for the mid-year
# checkpoints, ``late_count_12m`` for the annual one; never a proration of one
# from the other) across a scope's residents.
# --------------------------------------------------------------------------
_LATE_FORECAST_HEADS = {"late_count_3m", "late_count_6m", "late_count_9m", "late_count_12m"}


def _late_forecast_from_residents(residents: list, snapshot: date, head: str = "late_count_3m") -> dict:
    if head not in _LATE_FORECAST_HEADS:
        raise ValueError(f"unsupported late-forecast head: {head!r}")
    if not residents:
        return {"resident_count": 0, "expected": 0.0, "interval": [0.0, 0.0], "top_contributors": []}
    preds = predict_bulk(residents, snapshot, heads=[head])
    total_expected = total_lo = total_hi = 0.0
    rows = []
    for r, pred in zip(residents, preds):
        h = (pred.get("heads") or {}).get(head) or {}
        exp = float(h.get("expected") or 0.0)
        iv = h.get("interval") or [exp, exp]
        lo, hi = (float(iv[0]), float(iv[1])) if len(iv) == 2 else (exp, exp)
        total_expected += exp
        total_lo += lo
        total_hi += hi
        rows.append({"resident_id": r.get("resident_id", ""), "name": r.get("name", ""),
                     "expected": round(exp, 2)})
    rows.sort(key=lambda x: -x["expected"])
    return {
        "resident_count": len(residents),
        "expected": round(total_expected, 1),
        "interval": [round(total_lo, 1), round(total_hi, 1)],
        "top_contributors": rows[:5],
    }


def property_late_forecast(property_id: str, snapshot: date = RESIDENT_SNAPSHOT,
                            head: str = "late_count_3m") -> dict:
    """Expected total late payments for one property over the requested window
    (quarter via ``late_count_3m``, 12 months via ``late_count_12m``) — the SUM
    of every resident's own estimate, a genuine model output for that window
    rather than a proration of the other one. Never raises."""
    try:
        residents = [r for r in load_residents() if r.get("property_id") == property_id]
        out = _late_forecast_from_residents(residents, snapshot, head=head)
        out["property_id"] = property_id
        return out
    except Exception as exc:  # noqa: BLE001
        print(f"residents_risk.property_late_forecast failed ({type(exc).__name__}: {exc}).")
        return {"property_id": property_id, "resident_count": 0, "expected": 0.0,
                "interval": [0.0, 0.0], "top_contributors": []}


def portfolio_late_forecast(snapshot: date = RESIDENT_SNAPSHOT, head: str = "late_count_3m") -> dict:
    """Same aggregate forecast, across every resident in the portfolio."""
    try:
        return _late_forecast_from_residents(load_residents(), snapshot, head=head)
    except Exception as exc:  # noqa: BLE001
        print(f"residents_risk.portfolio_late_forecast failed ({type(exc).__name__}: {exc}).")
        return {"resident_count": 0, "expected": 0.0, "interval": [0.0, 0.0], "top_contributors": []}


# --------------------------------------------------------------------------
# Property/portfolio-level late-payment LIKELIHOOD aggregate -- averages each
# resident's own probability (not a proration) across every horizon, for a
# "how likely, next quarter / next year" question with no resident selected.
# --------------------------------------------------------------------------
_HORIZON_FORECAST_HEADS = ("late_1m", "late_3m", "late_6m", "late_12m")


def _horizon_forecast_from_residents(residents: list, snapshot: date) -> dict:
    if not residents:
        return {"resident_count": 0, "horizons": {}}
    preds = predict_bulk(residents, snapshot, heads=list(_HORIZON_FORECAST_HEADS))
    horizons: dict = {}
    for head in _HORIZON_FORECAST_HEADS:
        probs = []
        bands = {"low": 0, "medium": 0, "high": 0, "not_applicable": 0}
        for pred in preds:
            h = (pred.get("heads") or {}).get(head) or {}
            band = h.get("band", "not_applicable")
            bands[band] = bands.get(band, 0) + 1
            p = h.get("probability")
            if p is not None:
                probs.append(float(p))
        avg = round(sum(probs) / len(probs), 4) if probs else None
        horizons[head] = {"avg_probability": avg, "bands": bands}
    return {"resident_count": len(residents), "horizons": horizons}


def property_horizon_forecast(property_id: str, snapshot: date = RESIDENT_SNAPSHOT) -> dict:
    """Average late-payment probability for one property, every horizon —
    each resident's own estimate averaged, not summed. Never raises."""
    try:
        residents = [r for r in load_residents() if r.get("property_id") == property_id]
        out = _horizon_forecast_from_residents(residents, snapshot)
        out["property_id"] = property_id
        return out
    except Exception as exc:  # noqa: BLE001
        print(f"residents_risk.property_horizon_forecast failed ({type(exc).__name__}: {exc}).")
        return {"property_id": property_id, "resident_count": 0, "horizons": {}}


def portfolio_horizon_forecast(snapshot: date = RESIDENT_SNAPSHOT) -> dict:
    """Same aggregate, across every resident in the portfolio."""
    try:
        return _horizon_forecast_from_residents(load_residents(), snapshot)
    except Exception as exc:  # noqa: BLE001
        print(f"residents_risk.portfolio_horizon_forecast failed ({type(exc).__name__}: {exc}).")
        return {"resident_count": 0, "horizons": {}}


# --------------------------------------------------------------------------
# Property/portfolio-level SEVERITY aggregate -- avg expected worst-days-late,
# avg 30/60/90-day probabilities, and the delinquency-bucket distribution.
# --------------------------------------------------------------------------
_SEVERITY_PROB_HEADS = ("p_30d_12m", "p_60d_12m", "p_90d_12m")


def _severity_forecast_from_residents(residents: list, snapshot: date) -> dict:
    if not residents:
        return {"resident_count": 0, "avg_max_days_late": None, "probabilities": {}, "bucket_counts": {}}
    heads = ["max_days_late_12m", "delinquency_bucket_12m", *_SEVERITY_PROB_HEADS]
    preds = predict_bulk(residents, snapshot, heads=heads)
    days_vals = []
    bucket_counts = {"none": 0, "1-29": 0, "30-59": 0, "60-89": 0, "90+": 0}
    prob_sums: dict = {h: [] for h in _SEVERITY_PROB_HEADS}
    for pred in preds:
        hh = pred.get("heads") or {}
        md = (hh.get("max_days_late_12m") or {}).get("expected")
        if md is not None:
            days_vals.append(float(md))
        bucket = (hh.get("delinquency_bucket_12m") or {}).get("predicted_bucket")
        if bucket in bucket_counts:
            bucket_counts[bucket] += 1
        for h in _SEVERITY_PROB_HEADS:
            p = (hh.get(h) or {}).get("probability")
            if p is not None:
                prob_sums[h].append(float(p))
    probabilities = {h: (round(sum(vals) / len(vals), 4) if vals else None) for h, vals in prob_sums.items()}
    avg_days = round(sum(days_vals) / len(days_vals), 1) if days_vals else None
    return {"resident_count": len(residents), "avg_max_days_late": avg_days,
            "probabilities": probabilities, "bucket_counts": bucket_counts}


def property_severity_forecast(property_id: str, snapshot: date = RESIDENT_SNAPSHOT) -> dict:
    """Average expected worst-days-late, 30/60/90-day risk, and delinquency-
    bucket distribution for one property. Never raises."""
    try:
        residents = [r for r in load_residents() if r.get("property_id") == property_id]
        out = _severity_forecast_from_residents(residents, snapshot)
        out["property_id"] = property_id
        return out
    except Exception as exc:  # noqa: BLE001
        print(f"residents_risk.property_severity_forecast failed ({type(exc).__name__}: {exc}).")
        return {"property_id": property_id, "resident_count": 0, "avg_max_days_late": None,
                "probabilities": {}, "bucket_counts": {}}


def portfolio_severity_forecast(snapshot: date = RESIDENT_SNAPSHOT) -> dict:
    """Same aggregate, across every resident in the portfolio."""
    try:
        return _severity_forecast_from_residents(load_residents(), snapshot)
    except Exception as exc:  # noqa: BLE001
        print(f"residents_risk.portfolio_severity_forecast failed ({type(exc).__name__}: {exc}).")
        return {"resident_count": 0, "avg_max_days_late": None, "probabilities": {}, "bucket_counts": {}}


# --------------------------------------------------------------------------
# Property/portfolio-level ARREARS aggregate -- SUM of expected balances
# across residents (a genuine total dollar exposure), for each quarterly
# checkpoint (3/6/9/12mo) plus the 12mo peak.
# --------------------------------------------------------------------------
_ARREARS_FORECAST_HEADS = ("arrears_3m", "arrears_6m", "arrears_9m", "arrears_12m", "peak_balance_12m")


def _arrears_forecast_from_residents(residents: list, snapshot: date, head: str = "arrears_12m") -> dict:
    if head not in _ARREARS_FORECAST_HEADS:
        raise ValueError(f"unsupported arrears-forecast head: {head!r}")
    if not residents:
        return {"resident_count": 0, "expected": 0.0, "interval": [0.0, 0.0], "top_contributors": []}
    preds = predict_bulk(residents, snapshot, heads=[head])
    total_expected = total_lo = total_hi = 0.0
    rows = []
    for r, pred in zip(residents, preds):
        h = (pred.get("heads") or {}).get(head) or {}
        exp = float(h.get("expected") or 0.0)
        iv = h.get("interval") or [exp, exp]
        lo, hi = (float(iv[0]), float(iv[1])) if len(iv) == 2 else (exp, exp)
        total_expected += exp
        total_lo += lo
        total_hi += hi
        rows.append({"resident_id": r.get("resident_id", ""), "name": r.get("name", ""),
                     "expected": round(exp, 2)})
    rows.sort(key=lambda x: -x["expected"])
    return {"resident_count": len(residents), "expected": round(total_expected, 2),
            "interval": [round(total_lo, 2), round(total_hi, 2)], "top_contributors": rows[:5]}


def property_arrears_forecast(property_id: str, snapshot: date = RESIDENT_SNAPSHOT,
                              head: str = "arrears_12m") -> dict:
    """Expected total arrears balance for one property (SUM of every
    resident's own estimate). Never raises."""
    try:
        residents = [r for r in load_residents() if r.get("property_id") == property_id]
        out = _arrears_forecast_from_residents(residents, snapshot, head=head)
        out["property_id"] = property_id
        return out
    except Exception as exc:  # noqa: BLE001
        print(f"residents_risk.property_arrears_forecast failed ({type(exc).__name__}: {exc}).")
        return {"property_id": property_id, "resident_count": 0, "expected": 0.0,
                "interval": [0.0, 0.0], "top_contributors": []}


def portfolio_arrears_forecast(snapshot: date = RESIDENT_SNAPSHOT, head: str = "arrears_12m") -> dict:
    """Same aggregate, across every resident in the portfolio."""
    try:
        return _arrears_forecast_from_residents(load_residents(), snapshot, head=head)
    except Exception as exc:  # noqa: BLE001
        print(f"residents_risk.portfolio_arrears_forecast failed ({type(exc).__name__}: {exc}).")
        return {"resident_count": 0, "expected": 0.0, "interval": [0.0, 0.0], "top_contributors": []}


# --------------------------------------------------------------------------
# Property/portfolio-level CURE aggregate -- among residents CURRENTLY in
# arrears (cure is not_applicable otherwise), avg chance of clearing + timing.
# --------------------------------------------------------------------------
def _cure_forecast_from_residents(residents: list, snapshot: date) -> dict:
    if not residents:
        return {"resident_count": 0, "eligible_count": 0, "avg_probability": None, "avg_months_to_cure": None}
    # months_to_cure is a survival-kind head with no vectorized bulk path (its
    # batched value is a per-period hazard array, not a scalar) -- requesting
    # it via predict_bulk fails the WHOLE batch and silently falls back to
    # scoring every resident one at a time. Request only the bulk-safe binary
    # head here, then resolve months-to-cure per resident but ONLY for the
    # (typically few) residents actually in arrears.
    preds = predict_bulk(residents, snapshot, heads=["p_cure_6m"])
    probs, months = [], []
    for r, pred in zip(residents, preds):
        hh = pred.get("heads") or {}
        cure = hh.get("p_cure_6m") or {}
        if cure.get("band") == "not_applicable":
            continue
        p = cure.get("probability")
        if p is not None:
            probs.append(float(p))
        full = predict_resident(r, snapshot, with_reasons=False, heads=["months_to_cure"])
        mtc = (full.get("heads") or {}).get("months_to_cure", {}).get("median_months")
        if mtc is not None:
            months.append(float(mtc))
    avg = round(sum(probs) / len(probs), 4) if probs else None
    med = round(sum(months) / len(months), 1) if months else None
    return {"resident_count": len(residents), "eligible_count": len(probs),
            "avg_probability": avg, "avg_months_to_cure": med}


def property_cure_forecast(property_id: str, snapshot: date = RESIDENT_SNAPSHOT) -> dict:
    """Among residents currently carrying a balance at one property, average
    chance of clearing it within 6 months + typical time to clear. Never raises."""
    try:
        residents = [r for r in load_residents() if r.get("property_id") == property_id]
        out = _cure_forecast_from_residents(residents, snapshot)
        out["property_id"] = property_id
        return out
    except Exception as exc:  # noqa: BLE001
        print(f"residents_risk.property_cure_forecast failed ({type(exc).__name__}: {exc}).")
        return {"property_id": property_id, "resident_count": 0, "eligible_count": 0,
                "avg_probability": None, "avg_months_to_cure": None}


def portfolio_cure_forecast(snapshot: date = RESIDENT_SNAPSHOT) -> dict:
    """Same aggregate, across every resident in the portfolio."""
    try:
        return _cure_forecast_from_residents(load_residents(), snapshot)
    except Exception as exc:  # noqa: BLE001
        print(f"residents_risk.portfolio_cure_forecast failed ({type(exc).__name__}: {exc}).")
        return {"resident_count": 0, "eligible_count": 0, "avg_probability": None, "avg_months_to_cure": None}


# --------------------------------------------------------------------------
# Property/portfolio-level RETENTION aggregate -- among residents whose lease
# ends within the renewal-risk horizon (churn is not_applicable otherwise),
# avg non-renewal probability + how many are flagged high risk.
# --------------------------------------------------------------------------
def _retention_forecast_from_residents(residents: list, snapshot: date) -> dict:
    if not residents:
        return {"resident_count": 0, "eligible_count": 0, "avg_probability": None, "high_risk_count": 0}
    preds = predict_bulk(residents, snapshot, heads=["churn", "churn_12m"])
    probs, high = [], 0
    for pred in preds:
        hh = pred.get("heads") or {}
        c12 = hh.get("churn_12m") or {}
        if c12.get("band") == "not_applicable":
            continue
        p = c12.get("probability")
        if p is not None:
            probs.append(float(p))
        if c12.get("band") == "high":
            high += 1
    avg = round(sum(probs) / len(probs), 4) if probs else None
    return {"resident_count": len(residents), "eligible_count": len(probs),
            "avg_probability": avg, "high_risk_count": high}


def property_retention_forecast(property_id: str, snapshot: date = RESIDENT_SNAPSHOT) -> dict:
    """Among residents at one property whose lease ends within the renewal-
    risk horizon, average non-renewal probability + count flagged high risk.
    Never raises."""
    try:
        residents = [r for r in load_residents() if r.get("property_id") == property_id]
        out = _retention_forecast_from_residents(residents, snapshot)
        out["property_id"] = property_id
        return out
    except Exception as exc:  # noqa: BLE001
        print(f"residents_risk.property_retention_forecast failed ({type(exc).__name__}: {exc}).")
        return {"property_id": property_id, "resident_count": 0, "eligible_count": 0,
                "avg_probability": None, "high_risk_count": 0}


def portfolio_retention_forecast(snapshot: date = RESIDENT_SNAPSHOT) -> dict:
    """Same aggregate, across every resident in the portfolio."""
    try:
        return _retention_forecast_from_residents(load_residents(), snapshot)
    except Exception as exc:  # noqa: BLE001
        print(f"residents_risk.portfolio_retention_forecast failed ({type(exc).__name__}: {exc}).")
        return {"resident_count": 0, "eligible_count": 0, "avg_probability": None, "high_risk_count": 0}


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
    """Best-effort: train the bundle if missing/stale. Never raises. Clears caches."""
    try:
        if ARTIFACT_PATH.exists() and _model() is not None:
            return {"trained": True, "action": "already_present"}
        import train_residents

        train_residents.train()
        _model.cache_clear()
        _clear_prediction_caches()
        return {"trained": _model() is not None, "action": "trained"}
    except Exception as exc:  # noqa: BLE001 — startup must never fail
        print(f"residents_risk.ensure_model: training skipped ({type(exc).__name__}: {exc}).")
        _model.cache_clear()
        _clear_prediction_caches()
        return {"trained": False, "action": "failed", "error": type(exc).__name__}


def status() -> dict:
    """Health/status summary. Never raises."""
    try:
        bundle = _model()
        head_specs = [
            {"name": h["name"], "family": h["family"], "kind": h["kind"],
             "label_window": h["label_window"]}
            for h in HEADS
        ]
        if bundle is None:
            return {"trained": False, "source": "heuristic", "schema": BUNDLE_SCHEMA,
                    "heads": head_specs, "targets": list(TARGETS), "artifact": str(ARTIFACT_PATH)}
        return {
            "trained": True, "source": "model", "schema": BUNDLE_SCHEMA,
            "heads": head_specs, "targets": list(TARGETS),
            "metrics": {h["name"]: bundle.get("heads", {}).get(h["name"], {}).get("metrics", {}) for h in HEADS},
            "dgp_version": bundle.get("dgp_version"), "seed": bundle.get("seed"),
            "generated_at": bundle.get("generated_at"), "artifact": str(ARTIFACT_PATH),
        }
    except Exception as exc:  # noqa: BLE001
        return {"trained": False, "source": "heuristic", "error": type(exc).__name__}


def model_card() -> dict:
    """The resident-risk model card (v2). Never raises."""
    bundle = _model()

    def head_card(h: dict) -> dict:
        sub = (bundle or {}).get("heads", {}).get(h["name"], {})
        card = {
            "name": h["name"], "family": h["family"], "kind": h["kind"],
            "objective": h["objective"], "label_window": h["label_window"],
            "learnable": h["learnable"], "features": list(h["feature_order"]),
            "metrics": sub.get("metrics", {}), "low_confidence": h["low_confidence"],
        }
        if h["band_edges"]:
            card["bands"] = BANDS.get(h["name"])
        if h["eligibility"] is not None:
            card["eligibility"] = "current_balance>0 (in arrears)" if h["family"] == "cure" else "lease ending within horizon"
        return card

    return {
        "name": "Resident Risk (multi-head)", "version": DGP_VERSION, "schema": BUNDLE_SCHEMA,
        "trained_at": (bundle or {}).get("generated_at"),
        "model_type": "xgboost" if bundle else "heuristic",
        "description": (
            "A catalog of XGBoost gradient-boosted heads estimating late-payment, frequency, "
            "severity, arrears, cure, and retention outcomes for current residents from "
            "a 5-year rent ledger, trained on SYNTHETIC data. Reason codes come from "
            "TreeSHAP (model) or transparent weights (heuristic fallback). Legacy "
            "late/arrears/churn/serious remain as aliases."
        ),
        "intended_use": (
            "Decision-support for proactive outreach and retention ONLY. NOT for eviction, "
            "denial, pricing, lease conditioning, or any automated action. Serious-"
            "delinquency routes to a human reviewer."
        ),
        "families": {fam: list(names) for fam, names in FAMILIES.items()},
        "heads": [head_card(h) for h in HEADS],
        "deferred_families": [
            {"family": "engagement",
             "reason": "Future maintenance / complaint / login volume is NOT learnable: the DGP stores only static snapshot counts, with no forward engagement panel. Omitted until the DGP simulates one."},
            {"family": "move_out_timing",
             "reason": "Intra-lease move-out timing beyond the lease-end date is not modeled by the DGP; served as the lease-end date (rule-derived), not a learned head."},
        ],
        "excluded": EXCLUDED_FEATURES,
        "limitations": [
            "Trained on SYNTHETIC data — not validated on real residents.",
            "Estimates, not guarantees; individual outcomes vary.",
            "Rare-tail heads (60+/90+ day delinquency) are low-power — treat as directional.",
            "Long-horizon (12-month) heads are lower-confidence for short-tenure residents.",
            "Missing history is neutral-imputed and lowers confidence, never raising risk.",
            "Never uses race, national origin, sex, familial status, disability, age, or location.",
            "property/neighborhood/display-name are used only for audit or display, never as model inputs.",
            "Not a consumer report; not a substitute for lawful, human-in-the-loop decisions.",
        ],
        "source": "model" if bundle else "heuristic",
    }


if __name__ == "__main__":
    import pprint

    pprint.pp(status())
