"""Resident Late-Payment Risk — scoring, reason codes, graceful fallback.

DECISION-SUPPORT ONLY. This produces an estimated probability that a resident
pays rent late, from a model trained on SYNTHETIC data, to help a *person*
review an application. It never approves, denies, prices, or conditions a
lease. No protected attribute (race, national origin, sex, familial status,
disability, age) and no location field is a model input — see
EXCLUDED_FEATURES. Note the honest caveat: dollar-denominated features can
still carry residual property-scale information, so fairness is verified by
SLICING rather than by the absence of a column alone.

The public entrypoint is ``predict(profile) -> dict`` (the ``RiskResult``
shape). It NEVER raises: an XGBoost artifact is used when present, otherwise a
pure-Python transparent heuristic — the output shape is identical either way
(mirrors ``graphrag.graph_ask`` / ``concierge.answer`` / ``strength.compute``).

Feature policy (governing): only legitimate financial / payment factors enter
the model. Protected-class proxies and non-predictive fields are STRUCTURALLY
excluded — they never appear in ``FEATURE_ORDER`` or the input vector.
"""

from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from models import ApplicantProfile

ARTIFACT_PATH = Path(__file__).resolve().parent / "artifacts" / "risk_model.joblib"

# --------------------------------------------------------------------------
# Feature policy
# --------------------------------------------------------------------------
# Allowed model inputs — every one is a legitimate financial / payment factor
# derived from ApplicantProfile. This list IS the input contract: anything not
# here can never reach the model.
FEATURE_ORDER = [
    "rent_to_income",
    "dti",
    "credit_score",
    "credit_imputed",
    "savings_runway_months",
    "employment_length_months",
    "has_verified_income_source",
    "years_at_current_address",
    "rent_jump",
    "late_payments_12mo",
    "evictions_count",
    "bankruptcies_count",
    "references_count",
    "has_landlord_reference",
    "has_guarantor",
]

# Documented for the model card: fields deliberately kept OUT of the model.
EXCLUDED_FEATURES = [
    {"field": "name", "reason": "Identity / potential protected-class proxy; not predictive."},
    {"field": "preferred_area", "reason": "Location proxy for protected classes (redlining risk)."},
    {"field": "current_address", "reason": "Location proxy for protected classes."},
    {"field": "dependents", "reason": "Familial status — protected under fair housing law."},
    {"field": "household_size", "reason": "Familial status proxy — protected."},
    {"field": "co_applicants", "reason": "Familial / household-composition proxy."},
    {"field": "is_student", "reason": "Age proxy — protected."},
    {"field": "criminal_record", "reason": "Disparate-impact risk; excluded from the model."},
    {"field": "smoker", "reason": "Lifestyle; not a payment factor."},
    {"field": "has_pets / pet_count / pet_types", "reason": "Lifestyle; not a payment factor."},
    {"field": "reason_for_moving", "reason": "Free text; not a payment factor."},
    {"field": "employer / job_title", "reason": "Potential proxy; raw employer text not predictive."},
    {"field": "employment_status (raw)", "reason": "Only a verified-income boolean is derived; raw category excluded."},
    {"field": "bedrooms_wanted / unit & amenity prefs", "reason": "Preferences; not payment factors."},
    {"field": "lease_term_wanted", "reason": "Preference; not a payment factor."},
    {"field": "desired_move_in", "reason": "Logistics; not a payment factor."},
]

# Bands (tunable module constants). Elevated routes to human review, never to
# automated rejection.
BAND_LOW_MAX = 0.15
BAND_HIGH_MIN = 0.40

# Neutral imputation values (also the DGP medians in train_risk). Missing data
# is imputed to neutral and lowers confidence; it never pushes risk up.
_NEUTRAL = {
    "rent_to_income": 0.30,
    "dti": 0.15,
    "credit_score": 680.0,
    "savings_runway_months": 3.0,
    "employment_length_months": 24.0,
    "years_at_current_address": 2.0,
    "rent_jump": 1.0,
}

# Employment statuses that imply a verifiable, stable income source. Derived to
# a single boolean — the raw category never enters the model.
_VERIFIED_EMPLOYMENT = {
    "employed", "full_time", "full-time", "fulltime", "salaried", "w2",
    "part_time", "part-time", "self_employed", "self-employed", "contractor",
    "retired", "military", "pensioner",
}

DGP_VERSION = "risk-dgp-v1"


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _sigmoid(z: float) -> float:
    import math

    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


# --------------------------------------------------------------------------
# Feature extraction (allowed features only)
# --------------------------------------------------------------------------
def _income(profile: ApplicantProfile) -> float:
    return float(profile.monthly_income or 0) + float(profile.other_income_monthly or 0)


def extract_features(profile: ApplicantProfile) -> dict:
    """Derive the allowed features from a profile, neutral-imputing anything
    missing. Returns a dict keyed by FEATURE_ORDER — the ONLY data that reaches
    the model. Excluded fields are structurally absent (never read here)."""
    income = _income(profile)
    rent = float(profile.desired_rent or 0)

    f: dict = {}

    # Rent-to-income (rent / total income).
    f["rent_to_income"] = (
        _clamp(rent / income, 0.0, 3.0) if income > 0 and rent > 0 else _NEUTRAL["rent_to_income"]
    )

    # Debt-to-income.
    f["dti"] = (
        _clamp(float(profile.monthly_debt_payments or 0) / income, 0.0, 3.0)
        if income > 0
        else _NEUTRAL["dti"]
    )

    # Credit score (+ imputed flag, median-impute 680).
    if profile.credit_score is None:
        f["credit_score"] = _NEUTRAL["credit_score"]
        f["credit_imputed"] = 1.0
    else:
        f["credit_score"] = float(profile.credit_score)
        f["credit_imputed"] = 0.0

    # Savings runway in months of rent.
    if profile.savings_balance is not None and rent > 0:
        f["savings_runway_months"] = _clamp(float(profile.savings_balance) / rent, 0.0, 60.0)
    else:
        f["savings_runway_months"] = _NEUTRAL["savings_runway_months"]

    # Employment tenure.
    f["employment_length_months"] = (
        float(profile.employment_length_months)
        if profile.employment_length_months is not None
        else _NEUTRAL["employment_length_months"]
    )

    # Verified income source (boolean derived from employment_status, NOT the
    # raw category).
    status = (profile.employment_status or "").strip().lower()
    f["has_verified_income_source"] = 1.0 if status in _VERIFIED_EMPLOYMENT else 0.0

    # Address tenure.
    f["years_at_current_address"] = (
        float(profile.years_at_current_address)
        if profile.years_at_current_address is not None
        else _NEUTRAL["years_at_current_address"]
    )

    # Rent jump (requested vs current rent; neutral 1.0 if unknown).
    if profile.current_rent and profile.current_rent > 0 and rent > 0:
        f["rent_jump"] = _clamp(rent / float(profile.current_rent), 0.0, 5.0)
    else:
        f["rent_jump"] = _NEUTRAL["rent_jump"]

    # Screening / reference history (real values; defaults are genuine zeros).
    f["late_payments_12mo"] = float(profile.late_payments_12mo or 0)
    f["evictions_count"] = float(profile.evictions_count or 0)
    f["bankruptcies_count"] = float(profile.bankruptcies_count or 0)
    f["references_count"] = float(profile.references_count or 0)
    f["has_landlord_reference"] = 1.0 if profile.landlord_reference else 0.0
    f["has_guarantor"] = 1.0 if profile.guarantor_available else 0.0

    return {k: f[k] for k in FEATURE_ORDER}


def feature_vector(profile: ApplicantProfile) -> list:
    """The allowed features as a plain list in FEATURE_ORDER."""
    f = extract_features(profile)
    return [f[k] for k in FEATURE_ORDER]


def _confidence(profile: ApplicantProfile, f: dict) -> str:
    """Low when key predictive inputs (income / credit / employment) are
    missing — never a function of any protected attribute."""
    missing = 0
    if f["credit_imputed"] >= 1.0:
        missing += 1
    if _income(profile) <= 0:
        missing += 1
    if profile.employment_length_months is None:
        missing += 1
    return "low" if missing >= 1 else "high"


# Keep calibrated probabilities strictly inside (0, 1). See the call site in
# score() -- isotonic calibration saturates to exactly 0.0/1.0 outside its
# fitted range, and "100% certain to pay late" is not a defensible output for
# a decision-support estimate. Same epsilon as residents_risk._clamp_prob.
_PROB_EPS = 0.01


def _clamp_prob(p) -> float:
    return min(1.0 - _PROB_EPS, max(_PROB_EPS, float(p)))


# --------------------------------------------------------------------------
# Bands
# --------------------------------------------------------------------------
def _band(p: float) -> str:
    if p < BAND_LOW_MAX:
        return "low"
    if p < BAND_HIGH_MIN:
        return "medium"
    return "high"


BANDS = [
    {"band": "low", "min": 0.0, "max": BAND_LOW_MAX},
    {"band": "medium", "min": BAND_LOW_MAX, "max": BAND_HIGH_MIN},
    {"band": "high", "min": BAND_HIGH_MIN, "max": 1.0},
]


# --------------------------------------------------------------------------
# Reason-code templates (style of signals._TEMPLATES). Plain-English, factual,
# FCRA "key-factors" tone. Only allowed features appear here.
# --------------------------------------------------------------------------
def _pct(v: float) -> str:
    return f"{round(v * 100)}%"


REASON_TEMPLATES = {
    "rent_to_income": lambda v: f"Rent is {_pct(v)} of monthly income",
    "dti": lambda v: f"Debt payments are {_pct(v)} of monthly income",
    "credit_score": lambda v: f"Credit score of {round(v)}",
    "credit_imputed": lambda v: "Credit score was not provided",
    "savings_runway_months": lambda v: f"Savings cover about {v:.1f} months of rent",
    "employment_length_months": lambda v: f"{round(v)} months at current job",
    "has_verified_income_source": lambda v: (
        "Verified income source" if v >= 0.5 else "Income source not verified"
    ),
    "years_at_current_address": lambda v: f"{v:.1f} years at current address",
    "rent_jump": lambda v: f"Requested rent is {_pct(v)} of current rent",
    "late_payments_12mo": lambda v: (
        f"{round(v)} late payment(s) in the last 12 months" if v else "No recent late payments"
    ),
    "evictions_count": lambda v: (
        f"{round(v)} prior eviction(s)" if v else "No prior evictions"
    ),
    "bankruptcies_count": lambda v: (
        f"{round(v)} prior bankruptcy(ies)" if v else "No prior bankruptcies"
    ),
    "references_count": lambda v: f"{round(v)} reference(s) provided",
    "has_landlord_reference": lambda v: (
        "Landlord reference provided" if v >= 0.5 else "No landlord reference"
    ),
    "has_guarantor": lambda v: (
        "Guarantor available" if v >= 0.5 else "No guarantor"
    ),
}


def _label(feature: str, value: float) -> str:
    tmpl = REASON_TEMPLATES.get(feature)
    if tmpl is None:
        return feature.replace("_", " ")
    try:
        return tmpl(value)
    except Exception:  # noqa: BLE001 — labels must never break scoring
        return feature.replace("_", " ")


def _reason_codes(contribs: dict, features: dict, cap: int = 4) -> list:
    """Turn signed per-feature contributions (TreeSHAP or heuristic) into
    plain-English ReasonCode dicts. Positive contribution -> raises risk.

    Picks the top ``cap`` by |contribution|, but nudges toward a mix of
    up/down drivers so the card never reads as one-sided.
    """
    ranked = sorted(
        (
            {
                "feature": feat,
                "label": _label(feat, features.get(feat, 0.0)),
                "direction": "increases" if c > 0 else "decreases",
                "contribution": round(float(c), 4),
            }
            for feat, c in contribs.items()
            if feat in FEATURE_ORDER and abs(c) > 1e-9
        ),
        key=lambda r: -abs(r["contribution"]),
    )
    if len(ranked) <= cap:
        return ranked

    top = ranked[:cap]
    # Guarantee at least one "decreases" if the top slice is all "increases"
    # (and vice-versa), so reviewers see mitigating factors too.
    for direction in ("increases", "decreases"):
        if all(r["direction"] != direction for r in top):
            extra = next((r for r in ranked[cap:] if r["direction"] == direction), None)
            if extra is not None:
                top[-1] = extra
                top.sort(key=lambda r: -abs(r["contribution"]))
    return top


# --------------------------------------------------------------------------
# Transparent heuristic fallback (never returns exactly 0 or 1)
# --------------------------------------------------------------------------
# Signed weights mirror the DGP: rent-to-income +, dti +, credit -, savings -,
# tenure -, unstable income +, late payments (strong) +, evictions +,
# bankruptcies +, address tenure -, guarantor -, rent jump +.
_HEURISTIC_BIAS = -1.35
_HEURISTIC_WEIGHTS = {
    "rent_to_income": ("dev", 2.6, _NEUTRAL["rent_to_income"]),
    "dti": ("dev", 2.0, _NEUTRAL["dti"]),
    "credit_score": ("dev", -0.010, _NEUTRAL["credit_score"]),
    "savings_runway_months": ("dev", -0.08, _NEUTRAL["savings_runway_months"]),
    "employment_length_months": ("dev", -0.010, _NEUTRAL["employment_length_months"]),
    "has_verified_income_source": ("flag_low", 0.6, None),  # unstable income raises risk
    "years_at_current_address": ("dev", -0.05, _NEUTRAL["years_at_current_address"]),
    "rent_jump": ("dev", 1.0, _NEUTRAL["rent_jump"]),
    "late_payments_12mo": ("raw", 0.9, None),
    "evictions_count": ("raw", 1.2, None),
    "bankruptcies_count": ("raw", 0.8, None),
    "has_guarantor": ("flag", -0.5, None),
}


def _heuristic(profile: ApplicantProfile) -> tuple:
    """Pure-Python transparent risk. Returns (probability, contribs) where
    contribs is a signed per-feature dict for reason codes."""
    f = extract_features(profile)
    contribs: dict = {}
    z = _HEURISTIC_BIAS
    for feat, (kind, w, neutral) in _HEURISTIC_WEIGHTS.items():
        v = f[feat]
        if kind == "dev":
            c = w * (v - neutral)
        elif kind == "raw":
            c = w * v
        elif kind == "flag":  # boolean present -> contribution
            c = w * v
        elif kind == "flag_low":  # contribution when boolean is 0 (e.g. unverified)
            c = w * (1.0 - v)
        else:
            c = 0.0
        contribs[feat] = c
        z += c
    p = _clamp(_sigmoid(z), 0.02, 0.98)  # never exactly 0 or 1
    return p, contribs


# --------------------------------------------------------------------------
# Model loading (lru_cache; None on ANY failure)
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _model():
    """Load the persisted risk bundle. Returns the bundle dict or None on any
    failure (missing artifact, joblib/xgboost import error, corrupt file)."""
    try:
        import joblib

        if not ARTIFACT_PATH.exists():
            return None
        bundle = joblib.load(ARTIFACT_PATH)
        if not isinstance(bundle, dict) or "calibrated_model" not in bundle:
            return None
        if bundle.get("feature_order") != FEATURE_ORDER:
            # Trained against a different feature contract — refuse it.
            return None
        return bundle
    except Exception as exc:  # noqa: BLE001 — degrade to heuristic
        print(f"risk: model load failed ({type(exc).__name__}: {exc}); using heuristic.")
        return None


def _model_reason_codes(bundle: dict, features: dict) -> list:
    """TreeSHAP reason codes from the raw booster (last column = bias)."""
    booster = bundle.get("booster")
    if booster is None:
        # Fallback model (e.g. HistGB) with no TreeSHAP booster: explain with the
        # transparent heuristic's signed contributions (allowed features only).
        return None
    import pandas as pd
    import xgboost as xgb

    row = pd.DataFrame([[features[k] for k in FEATURE_ORDER]], columns=FEATURE_ORDER)
    contribs = booster.predict(xgb.DMatrix(row), pred_contribs=True)[0]
    signed = {FEATURE_ORDER[i]: float(contribs[i]) for i in range(len(FEATURE_ORDER))}
    return _reason_codes(signed, features)


def _range(p: float, low_conf: bool, bundle) -> list:
    """A plausibility interval around p, from the calibration-bin spread. Wider
    when confidence is low."""
    half = 0.08
    if bundle is not None:
        half = float(bundle.get("range_half_width", half))
    if low_conf:
        half += 0.07
    return [round(_clamp(p - half), 4), round(_clamp(p + half), 4)]


# --------------------------------------------------------------------------
# Public entrypoint — NEVER raises
# --------------------------------------------------------------------------
def predict(profile: ApplicantProfile, applicant_id: str = "", name: str = "") -> dict:
    """Score one profile. Returns a RiskResult dict. Falls back from model to
    heuristic on any error; never raises."""
    features = extract_features(profile)
    low_conf = _confidence(profile, features) == "low"
    scored_at = datetime.now(timezone.utc).isoformat()

    bundle = _model()
    if bundle is not None:
        try:
            import pandas as pd

            row = pd.DataFrame([[features[k] for k in FEATURE_ORDER]], columns=FEATURE_ORDER)
            # Clamp away from 0/1: isotonic calibration is a step function and
            # returns EXACTLY 0.0 or 1.0 for raw scores past the range it was
            # fitted on. Serving 1.0 tells a reviewer an applicant is CERTAIN
            # to pay late, which no calibrated model can support -- and this is
            # reachable in practice (a low-credit, high-burden profile scored
            # exactly 1.0 on the live deployment). Mirrors residents_risk's
            # _clamp_prob so both models behave the same way.
            p = _clamp_prob(bundle["calibrated_model"].predict_proba(row)[0][1])
            reasons = _model_reason_codes(bundle, features)
            if reasons is None:  # no booster -> heuristic explanation
                _, contribs = _heuristic(profile)
                reasons = _reason_codes(contribs, features)
            return {
                "applicant_id": applicant_id,
                "name": name or profile.name,
                "probability": round(_clamp(p), 4),
                "band": _band(p),
                "reason_codes": reasons,
                "confidence": "low" if low_conf else "high",
                "range": _range(p, low_conf, bundle),
                "source": "model",
                "model_type": bundle.get("model_type", "xgboost"),
                "scored_at": scored_at,
            }
        except Exception as exc:  # noqa: BLE001 — degrade to heuristic
            print(f"risk: model predict failed ({type(exc).__name__}: {exc}); using heuristic.")

    p, contribs = _heuristic(profile)
    return {
        "applicant_id": applicant_id,
        "name": name or profile.name,
        "probability": round(_clamp(p), 4),
        "band": _band(p),
        "reason_codes": _reason_codes(contribs, features),
        "confidence": "low" if low_conf else "high",
        "range": _range(p, low_conf, None),
        "source": "heuristic",
        "model_type": "heuristic",
        "scored_at": scored_at,
    }


def top_driver(result: dict) -> str:
    """The strongest 'increases' reason label (for RiskRow.top_driver)."""
    ups = [r for r in result.get("reason_codes", []) if r.get("direction") == "increases"]
    if not ups:
        return ""
    return max(ups, key=lambda r: abs(r.get("contribution", 0)))["label"]


# --------------------------------------------------------------------------
# Startup + status
# --------------------------------------------------------------------------
def ensure_model() -> dict:
    """Best-effort: if the artifact is missing, train it. Wrapped so it can be
    called from startup without ever raising. Clears the load cache afterward."""
    try:
        if ARTIFACT_PATH.exists():
            _model.cache_clear()
            return {"trained": True, "action": "already_present"}
        import train_risk

        train_risk.train()
        _model.cache_clear()
        return {"trained": ARTIFACT_PATH.exists(), "action": "trained"}
    except Exception as exc:  # noqa: BLE001 — startup must never fail
        print(f"risk.ensure_model: training skipped ({type(exc).__name__}: {exc}).")
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
                "model_type": "heuristic",
                "features": len(FEATURE_ORDER),
                "artifact": ARTIFACT_PATH.name,
            }
        return {
            "trained": True,
            "source": "model",
            "model_type": bundle.get("model_type", "xgboost"),
            "features": len(FEATURE_ORDER),
            "n_train": bundle.get("n_train"),
            "metrics": bundle.get("metrics", {}),
            "dgp_version": bundle.get("dgp_version"),
            "generated_at": bundle.get("generated_at"),
            "artifact": ARTIFACT_PATH.name,
        }
    except Exception as exc:  # noqa: BLE001
        return {"trained": False, "source": "heuristic", "model_type": "heuristic",
                "error": type(exc).__name__}


def model_card() -> dict:
    """The RiskModelCard dict — describes intended use, features, exclusions,
    metrics, bands, limitations. Never raises."""
    bundle = _model()
    metrics = bundle.get("metrics", {}) if bundle else {}
    return {
        "name": "Resident Late-Payment Risk",
        "version": DGP_VERSION,
        "trained_at": (bundle or {}).get("generated_at"),
        "model_type": bundle.get("model_type", "xgboost") if bundle else "heuristic",
        "description": (
            "XGBoost gradient-boosted model estimating the probability that a resident "
            "pays rent late, trained on synthetic data. Reason codes come from "
            "TreeSHAP (model) or transparent weights (heuristic fallback)."
        ),
        "intended_use": (
            "Decision-support for a human reviewer. NOT for automated approval, "
            "denial, pricing, or lease conditioning. Elevated scores route to a "
            "person for review."
        ),
        "features": list(FEATURE_ORDER),
        "excluded": EXCLUDED_FEATURES,
        "metrics": metrics,
        "bands": BANDS,
        "limitations": [
            "Trained on SYNTHETIC data — not validated on real residents.",
            "An estimate, not a guarantee; individual outcomes vary.",
            "Missing data is neutral-imputed and lowers confidence, never raising risk.",
            "Never uses race, national origin, sex, familial status, disability, age, or location.",
            "Not a consumer report; not a substitute for FCRA-compliant screening.",
        ],
        "source": "model" if bundle else "heuristic",
    }
