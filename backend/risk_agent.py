"""Risk Chat — deterministic tools over ``risk.py`` (NO LLM).

These functions are the *only* place the risk chat agent touches the model.
Every number the agent ever states — a probability, band, reason code, or
governance fact — originates here, from ``risk.predict`` / ``risk.model_card``.
The LLM (when present) only rewrites the grounded context into prose; it can
never invent or recompute a figure.

Pure and side-effect free (aside from reading the store): safe to call from the
router, the deterministic templates, and the eval harness alike. Consumers must
treat a ``None`` result as "no applicant / unknown id" and deflect gracefully.

Phase 1 implements: ``score_applicant``, ``explain_reasons``,
``model_governance`` and the ``artifact`` builders. Phase 2 will add the
what-if / counterfactual / portfolio / compare tools (see TODO seams below).
"""

from __future__ import annotations

import math
import re

import risk
import store
from models import ApplicantProfile


# ---------------------------------------------------------------------------
# Applicant loading + scoring
# ---------------------------------------------------------------------------
def _load_profile(applicant_id: str | None) -> tuple[ApplicantProfile | None, str]:
    """Load a stored applicant's profile by id, plus its display name. Returns
    ``(None, "")`` for a missing/blank/unknown id — never raises."""
    if not applicant_id:
        return None, ""
    try:
        profile = store.get_profile(applicant_id)
    except Exception as exc:  # noqa: BLE001 — a store hiccup must degrade, not 500
        print(f"risk_agent: profile load failed ({type(exc).__name__}: {exc}).")
        return None, ""
    if profile is None:
        return None, ""
    return profile, (profile.name or "")


def score_applicant(applicant_id: str | None) -> dict | None:
    """Score a stored applicant. Returns a RiskResult dict (identical shape to
    ``GET /risk/{id}``) or ``None`` if the id is unknown/blank. Never raises."""
    profile, name = _load_profile(applicant_id)
    if profile is None:
        return None
    return risk.predict(profile, applicant_id=applicant_id or "", name=name)


def explain_reasons(applicant_id: str | None) -> dict | None:
    """Score + explain one applicant: the RiskResult, its strongest upward
    driver, and the band thresholds/definitions. ``None`` if unknown. The band
    numbers come straight from ``risk`` constants so the answer can cite them
    without recomputing."""
    result = score_applicant(applicant_id)
    if result is None:
        return None
    return {
        "result": result,
        "top_driver": risk.top_driver(result),
        "band_thresholds": {
            "low_max": risk.BAND_LOW_MAX,
            "high_min": risk.BAND_HIGH_MIN,
            "bands": risk.BANDS,
        },
    }


# ---------------------------------------------------------------------------
# Model governance / feature policy
# ---------------------------------------------------------------------------
# Term -> feature resolution for governance ("is X used?", "what's excluded?").
# USED phrases are matched FIRST and are deliberately specific, so a query about
# "years at current address" (a used tenure feature) is not stolen by the bare
# "address" location-proxy exclusion below.
_USED_SYNONYMS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"rent[\s-]?to[\s-]?income|rent to income ratio", re.I), "rent_to_income"),
    (re.compile(r"\bdti\b|debt[\s-]?to[\s-]?income|debt payment|monthly debt", re.I), "dti"),
    (re.compile(r"credit score|credit rating|\bfico\b|\bcredit\b", re.I), "credit_score"),
    (re.compile(r"savings|runway|cash reserve|reserves", re.I), "savings_runway_months"),
    (re.compile(r"employment length|job tenure|time (?:at|on) (?:the )?job|months employed|length of employment", re.I), "employment_length_months"),
    (re.compile(r"verified income|income source|income verif|\bincome\b|salary|earnings", re.I), "has_verified_income_source"),
    (re.compile(r"years at (?:the )?(?:current )?address|address tenure|time at (?:current )?address|how long .*address", re.I), "years_at_current_address"),
    (re.compile(r"rent jump|rent increase|rent change", re.I), "rent_jump"),
    (re.compile(r"late payment|payment history|paid late", re.I), "late_payments_12mo"),
    (re.compile(r"eviction", re.I), "evictions_count"),
    (re.compile(r"bankruptc", re.I), "bankruptcies_count"),
    (re.compile(r"reference count|number of references|\breferences?\b", re.I), "references_count"),
    (re.compile(r"landlord reference", re.I), "has_landlord_reference"),
    (re.compile(r"guarantor|co[\s-]?signer|cosigner", re.I), "has_guarantor"),
]

# EXCLUDED phrases -> (canonical field label, reason). Real EXCLUDED_FEATURES
# fields plus the protected-class attributes the model never uses (per the
# model card's limitations). Location-proxy address terms are location-y words
# only, so they don't collide with the used "years at current address" tenure.
_EXCLUDED_SYNONYMS: list[tuple[re.Pattern, str, str]] = [
    (re.compile(r"\brace\b|ethnicit|national origin", re.I), "race / national origin",
     "Protected class under fair housing law; never used."),
    (re.compile(r"\bsex\b|gender", re.I), "sex / gender",
     "Protected class under fair housing law; never used."),
    (re.compile(r"disab", re.I), "disability",
     "Protected class under fair housing law; never used."),
    (re.compile(r"religion|faith", re.I), "religion",
     "Protected class under fair housing law; never used."),
    (re.compile(r"familial status|marital status|married", re.I), "familial status",
     "Protected class under fair housing law; never used."),
    (re.compile(r"\bage\b|birth ?date|date of birth|\bdob\b|is_?student|\bstudent\b", re.I), "age / student status",
     "Age proxy — protected; excluded from the model."),
    (re.compile(r"\blocation\b|neighborhood|preferred area|redlin|\bzip\b|where (?:they|you|i) (?:live|reside)|home address|street address", re.I), "location / address",
     "Location proxy for protected classes (redlining risk); excluded."),
    (re.compile(r"dependent|\bchildren\b|\bkids\b", re.I), "dependents",
     "Familial status — protected under fair housing law."),
    (re.compile(r"household size|household", re.I), "household_size",
     "Familial status proxy — protected."),
    (re.compile(r"co[\s-]?applicant", re.I), "co_applicants",
     "Familial / household-composition proxy."),
    (re.compile(r"criminal|conviction|\bfelony\b|\bcrime\b", re.I), "criminal_record",
     "Disparate-impact risk; excluded from the model."),
    (re.compile(r"smoker|smoking|\bsmoke\b", re.I), "smoker",
     "Lifestyle; not a payment factor."),
    (re.compile(r"\bpets?\b|\bdog\b|\bcat\b", re.I), "has_pets / pet_count / pet_types",
     "Lifestyle; not a payment factor."),
    (re.compile(r"reason for moving|why (?:they|you) (?:are )?moving", re.I), "reason_for_moving",
     "Free text; not a payment factor."),
    (re.compile(r"employer|job title|\bcompany\b", re.I), "employer / job_title",
     "Potential proxy; raw employer text not predictive."),
    (re.compile(r"lease term", re.I), "lease_term_wanted",
     "Preference; not a payment factor."),
    (re.compile(r"move[\s-]?in|amenit|bedrooms? wanted", re.I), "unit & amenity preferences",
     "Preferences / logistics; not payment factors."),
]

# A short, uniform rationale for the allowed (used) features.
_USED_REASON = "Legitimate financial / payment factor used by the model."


def model_governance(query: str = "") -> dict:
    """Governance view of the feature policy. Returns the model card plus a
    ``feature_lookup`` resolving any field(s) named in ``query`` to used
    (in ``FEATURE_ORDER``) or excluded (with its documented reason), and an
    ``is_used`` flag for the primary resolved field (``None`` if the query
    names no specific field). Never raises."""
    q = query or ""
    feature_lookup: list[dict] = []
    seen: set[str] = set()

    # Used features first (specific phrasing wins over the location proxy).
    for rx, feat in _USED_SYNONYMS:
        if feat in seen:
            continue
        if rx.search(q):
            feature_lookup.append({"field": feat, "status": "used", "reason": _USED_REASON})
            seen.add(feat)

    for rx, field, reason in _EXCLUDED_SYNONYMS:
        if field in seen:
            continue
        if rx.search(q):
            feature_lookup.append({"field": field, "status": "excluded", "reason": reason})
            seen.add(field)

    is_used: bool | None = None
    if feature_lookup:
        is_used = feature_lookup[0]["status"] == "used"

    return {
        "card": risk.model_card(),
        "feature_lookup": feature_lookup,
        "is_used": is_used,
    }


# ---------------------------------------------------------------------------
# WHAT-IF field whitelist
# ---------------------------------------------------------------------------
# The ONLY ApplicantProfile fields a what-if / counterfactual may perturb. Each
# maps to the FEATURE_ORDER feature(s) it can move. By construction every target
# feature is an ALLOWED model input — so a what-if can NEVER touch an excluded /
# protected feature (dependents, location, criminal record, …). Any override key
# not in this map is REJECTED and never applied. The import-time assertion below
# is the structural guarantee.
_WHATIF_FIELD_FEATURES: dict[str, tuple[str, ...]] = {
    "monthly_income": ("rent_to_income", "dti"),
    "other_income_monthly": ("rent_to_income", "dti"),
    "desired_rent": ("rent_to_income", "savings_runway_months", "rent_jump"),
    "credit_score": ("credit_score", "credit_imputed"),
    "savings_balance": ("savings_runway_months",),
    "monthly_debt_payments": ("dti",),
    "employment_length_months": ("employment_length_months",),
    "employment_status": ("has_verified_income_source",),
    "years_at_current_address": ("years_at_current_address",),
    "current_rent": ("rent_jump",),
    "late_payments_12mo": ("late_payments_12mo",),
    "evictions_count": ("evictions_count",),
    "bankruptcies_count": ("bankruptcies_count",),
    "references_count": ("references_count",),
    "landlord_reference": ("has_landlord_reference",),
    "guarantor_available": ("has_guarantor",),
}

# The whitelist proper — membership test for score_whatif.
WHATIF_FIELDS = frozenset(_WHATIF_FIELD_FEATURES)

# IMPORT-TIME GUARANTEE: every whitelisted profile field is a real profile field,
# is NOT one of the documented excluded fields, and maps ONLY to features that
# exist in risk.FEATURE_ORDER. If any of these break, the module refuses to load
# rather than silently allow a what-if to move a protected/excluded feature.
_EXCLUDED_FIELD_NAMES = {e["field"] for e in risk.EXCLUDED_FEATURES}
for _wf, _feats in _WHATIF_FIELD_FEATURES.items():
    assert _wf in ApplicantProfile.model_fields, (
        f"WHATIF_FIELDS: '{_wf}' is not an ApplicantProfile field"
    )
    assert _wf not in _EXCLUDED_FIELD_NAMES, (
        f"WHATIF_FIELDS: '{_wf}' is a documented EXCLUDED field — cannot be a lever"
    )
    for _feat in _feats:
        assert _feat in risk.FEATURE_ORDER, (
            f"WHATIF_FIELDS: '{_wf}' maps to '{_feat}', which is not in FEATURE_ORDER"
        )
del _wf, _feats, _feat, _EXCLUDED_FIELD_NAMES


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------
def _feature_label(feature: str, value) -> str:
    """A plain-English label for a feature value, from ``risk.REASON_TEMPLATES``
    (the model's own key-factor phrasing). Never raises."""
    tmpl = risk.REASON_TEMPLATES.get(feature)
    if tmpl is None:
        return feature.replace("_", " ")
    try:
        return tmpl(float(value))
    except Exception:  # noqa: BLE001 — a label must never break a tool
        return feature.replace("_", " ")


def _band_of(p: float) -> str:
    """Band for a probability using the public risk thresholds."""
    if p < risk.BAND_LOW_MAX:
        return "low"
    if p < risk.BAND_HIGH_MIN:
        return "medium"
    return "high"


_BAND_RANK = {"low": 0, "medium": 1, "high": 2}


def _percentile(sorted_vals: list[float], q: float) -> float:
    """Linear-interpolated percentile of an already-sorted list (0..100)."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    k = (len(sorted_vals) - 1) * (q / 100.0)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return float(sorted_vals[int(k)])
    return float(sorted_vals[lo]) * (hi - k) + float(sorted_vals[hi]) * (k - lo)


# ---------------------------------------------------------------------------
# What-if override parsing (deterministic; reuses concierge's money-regex style)
# ---------------------------------------------------------------------------
# A number with an optional leading $ and a trailing k / month / year unit,
# mirroring concierge._MAX_RENT_RE ("$6,000", "6000", "6k", "36 months").
_NUM_UNIT_RE = re.compile(
    r"\$?\s*(\d[\d,]*(?:\.\d+)?)\s*(k\b|months?\b|mos?\b|years?\b|yrs?\b)?",
    re.IGNORECASE,
)

# Cue phrase -> (profile field, value kind). Ordered so more specific cues
# ("debt", "credit") are tried before the broad "income" / "rent" cues. Every
# field here is in WHATIF_FIELDS, so the parser can only ever emit allowed keys.
_OVERRIDE_CUES: list[tuple[re.Pattern, str, str]] = [
    (re.compile(r"credit(?:\s+score)?|\bfico\b", re.I), "credit_score", "int"),
    (re.compile(r"monthly debt|debt payments?|\bdebt\b|\bdti\b", re.I), "monthly_debt_payments", "money"),
    (re.compile(r"savings|cash reserve|reserves|\bcash\b", re.I), "savings_balance", "money"),
    (re.compile(r"employ\w*|job tenure|\btenure\b|months? (?:at|on) (?:the )?job", re.I), "employment_length_months", "months"),
    (re.compile(r"\bincome\b|\bsalary\b|earn\w*|\bwages?\b|\bpay\b", re.I), "monthly_income", "money"),
    (re.compile(r"\brent\b", re.I), "desired_rent", "money"),
]


def _parse_overrides(question: str) -> dict:
    """Parse a natural-language what-if into a ``{profile_field: value}`` dict of
    ALLOWED levers. For each cue found, the first number appearing at/after the
    cue is bound to that field. Returns ``{}`` when nothing parseable is present
    (the caller then explains what can be varied). Never raises."""
    q = question or ""
    overrides: dict = {}
    for cue_re, field, kind in _OVERRIDE_CUES:
        if field in overrides:
            continue
        cue = cue_re.search(q)
        if cue is None:
            continue
        num = _NUM_UNIT_RE.search(q, cue.end())
        if num is None:
            continue
        try:
            raw = float(num.group(1).replace(",", ""))
        except ValueError:
            continue
        unit = (num.group(2) or "").strip().lower()
        if kind == "money":
            if unit == "k":
                raw *= 1000.0
            overrides[field] = round(raw, 2)
        elif kind == "int":
            overrides[field] = int(round(raw))
        elif kind == "months":
            if unit.startswith("year") or unit.startswith("yr"):
                raw *= 12.0
            overrides[field] = int(round(raw))
    return overrides


# ---------------------------------------------------------------------------
# score_whatif — re-score a modified COPY; never persists
# ---------------------------------------------------------------------------
def score_whatif(applicant_id: str | None, overrides: dict | None) -> dict | None:
    """Re-score an applicant under hypothetical overrides. Keys NOT in
    ``WHATIF_FIELDS`` are REJECTED (returned under ``rejected``, never applied),
    so a what-if can never move an excluded / protected feature. Builds a
    modified profile COPY (``model_copy``) — the stored profile is never mutated
    or persisted. ``None`` for an unknown id. Never raises."""
    profile, name = _load_profile(applicant_id)
    if profile is None:
        return None

    applied: dict = {}
    rejected: dict = {}
    for key, value in (overrides or {}).items():
        if key in WHATIF_FIELDS:
            applied[key] = value
        else:
            rejected[key] = value

    base = risk.predict(profile, applicant_id=applicant_id or "", name=name)
    modified = profile.model_copy(update=applied) if applied else profile
    whatif = (
        risk.predict(modified, applicant_id=applicant_id or "", name=name)
        if applied
        else base
    )

    base_feats = risk.extract_features(profile)
    mod_feats = risk.extract_features(modified)
    features_changed = [
        {
            "feature": feat,
            "label": _feature_label(feat, mod_feats[feat]),
            "from": base_feats[feat],
            "to": mod_feats[feat],
        }
        for feat in risk.FEATURE_ORDER
        if base_feats[feat] != mod_feats[feat]
    ]

    return {
        "base": base,
        "whatif": whatif,
        "overrides": applied,
        "rejected": rejected,
        "delta_probability": round(whatif["probability"] - base["probability"], 4),
        "band_changed": base["band"] != whatif["band"],
        "features_changed": features_changed,
    }


# ---------------------------------------------------------------------------
# counterfactual — greedy, bounded, direction-guarded lever search
# ---------------------------------------------------------------------------
# A FIXED list of levers. Each is an ALLOWED profile field, its primary target
# feature, and an ``improve`` function returning a realistic, bounded, better
# value (or ``None`` when no improvement is possible / applicable). Every lever
# only moves a field in the IMPROVING direction, so re-scoring can only lower
# risk — the search is direction-guarded on top of that.
def _imp_credit(p: ApplicantProfile):
    cur = float(p.credit_score) if p.credit_score is not None else 680.0
    tgt = min(800.0, cur + 100.0)
    return int(tgt) if tgt > cur else None


def _imp_income(p: ApplicantProfile):
    inc = float(p.monthly_income or 0)
    if inc <= 0:
        floor = float(p.desired_rent or 0) * 3.0
        return round(floor, 2) if floor > 0 else None
    return round(inc * 1.25, 2)  # a ~25% raise


def _imp_debt(p: ApplicantProfile):
    debt = float(p.monthly_debt_payments or 0)
    return round(debt * 0.5, 2) if debt > 0 else None  # halve monthly debt


def _imp_savings(p: ApplicantProfile):
    cur = float(p.savings_balance or 0)
    rent = float(p.desired_rent or 0)
    add = 3.0 * rent if rent > 0 else 3000.0  # +~3 months of runway
    return round(cur + add, 2)


def _imp_rent(p: ApplicantProfile):
    rent = float(p.desired_rent or 0)
    return round(rent * 0.85, 2) if rent > 0 else None  # a cheaper unit (-15%)


def _imp_employment(p: ApplicantProfile):
    cur = float(p.employment_length_months) if p.employment_length_months is not None else 24.0
    return int(cur + 12)  # a year longer at the job


def _imp_address(p: ApplicantProfile):
    cur = float(p.years_at_current_address) if p.years_at_current_address is not None else 2.0
    return round(cur + 2.0, 1)  # two more years of address tenure


def _imp_late(p: ApplicantProfile):
    return 0 if (p.late_payments_12mo or 0) > 0 else None  # clear recent lates


def _imp_guarantor(p: ApplicantProfile):
    return True if not p.guarantor_available else None


def _imp_landlord(p: ApplicantProfile):
    return True if not p.landlord_reference else None


# (profile field, primary feature, improve fn) — allowed fields ONLY.
_CF_LEVERS: list[tuple[str, str, object]] = [
    ("credit_score", "credit_score", _imp_credit),
    ("monthly_income", "rent_to_income", _imp_income),
    ("monthly_debt_payments", "dti", _imp_debt),
    ("savings_balance", "savings_runway_months", _imp_savings),
    ("desired_rent", "rent_to_income", _imp_rent),
    ("employment_length_months", "employment_length_months", _imp_employment),
    ("years_at_current_address", "years_at_current_address", _imp_address),
    ("late_payments_12mo", "late_payments_12mo", _imp_late),
    ("guarantor_available", "has_guarantor", _imp_guarantor),
    ("landlord_reference", "has_landlord_reference", _imp_landlord),
]

_CF_MAX_STEPS = 3


def counterfactual(applicant_id: str | None, target_band: str = "medium") -> dict | None:
    """Greedy, bounded, direction-guarded search for changes that would move an
    applicant to ``target_band`` or lower. Each round screens every remaining
    lever ALONE (re-scoring a modified copy), commits the single biggest drop,
    and repeats — capped at three committed steps. Direction-guarded: a lever is
    only ever considered if it strictly lowers the estimate, so risk can never be
    driven up. ``None`` for an unknown id. Never raises / never persists."""
    profile, name = _load_profile(applicant_id)
    if profile is None:
        return None

    target = target_band if target_band in _BAND_RANK else "medium"
    target_rank = _BAND_RANK[target]
    predict_calls = 0

    def _predict(p: ApplicantProfile) -> dict:
        nonlocal predict_calls
        predict_calls += 1
        return risk.predict(p, applicant_id=applicant_id or "", name=name)

    base = _predict(profile)

    def _meets(result: dict) -> bool:
        return _BAND_RANK[result["band"]] <= target_rank

    working = profile
    working_result = base
    steps: list[dict] = []
    used_fields: set[str] = set()

    while not _meets(working_result) and len(steps) < _CF_MAX_STEPS:
        cur_prob = working_result["probability"]
        cur_feats = risk.extract_features(working)
        best = None
        for field, feature, improve in _CF_LEVERS:
            if field in used_fields:
                continue
            try:
                new_val = improve(working)
            except Exception:  # noqa: BLE001 — a bad lever must not break the search
                new_val = None
            if new_val is None:
                continue
            candidate = working.model_copy(update={field: new_val})
            cand_result = _predict(candidate)
            drop = cur_prob - cand_result["probability"]
            if drop <= 1e-9:  # direction guard — must strictly lower risk
                continue
            if best is None or drop > best["drop"]:
                cand_feats = risk.extract_features(candidate)
                best = {
                    "field": field,
                    "feature": feature,
                    "from": cur_feats[feature],
                    "to": cand_feats[feature],
                    "profile": candidate,
                    "result": cand_result,
                    "drop": drop,
                }
        if best is None:
            break  # no remaining lever lowers the estimate further

        working = best["profile"]
        working_result = best["result"]
        used_fields.add(best["field"])
        steps.append(
            {
                "field": best["field"],
                "label": _feature_label(best["feature"], best["to"]),
                "from": best["from"],
                "to": best["to"],
                "feature": best["feature"],
                "band_after": working_result["band"],
                "probability_after": working_result["probability"],
            }
        )

    return {
        "base": base,
        "target_band": target,
        "achievable": _meets(working_result),
        "steps": steps,
        "final": working_result,
        "predict_calls": predict_calls,
    }


# ---------------------------------------------------------------------------
# Portfolio summary + single-applicant comparison
# ---------------------------------------------------------------------------
def _score_portfolio() -> list[dict]:
    """Score every stored applicant, per-row try/except (mirrors
    ``risk_api.list_risk``). One bad row never fails the batch."""
    results: list[dict] = []
    try:
        applicants = store.list_applicants()
    except Exception as exc:  # noqa: BLE001 — a store hiccup degrades to empty
        print(f"risk_agent: portfolio list failed ({type(exc).__name__}: {exc}).")
        return results
    for a in applicants:
        try:
            profile = store.get_profile(a["id"])
            if profile is None:
                continue
            results.append(
                risk.predict(profile, applicant_id=a["id"], name=a.get("name", ""))
            )
        except Exception as exc:  # noqa: BLE001 — skip a bad row, never fail the batch
            print(f"risk_agent: scoring {a.get('id')} failed ({type(exc).__name__}: {exc}).")
    return results


def portfolio_summary() -> dict:
    """Distribution of estimated risk across the scored portfolio. Never raises."""
    rows = _score_portfolio()
    probs = sorted(r["probability"] for r in rows)
    scored = len(probs)
    band_counts = {"low": 0, "medium": 0, "high": 0}
    for r in rows:
        band_counts[r["band"]] = band_counts.get(r["band"], 0) + 1

    if scored == 0:
        return {
            "scored": 0,
            "avg_probability": 0.0,
            "band_counts": band_counts,
            "high_risk_pct": 0.0,
            "median_probability": 0.0,
            "p_percentiles": {"p25": 0.0, "p50": 0.0, "p75": 0.0, "p90": 0.0},
        }

    avg = round(sum(probs) / scored, 4)
    median = round(_percentile(probs, 50), 4)
    return {
        "scored": scored,
        "avg_probability": avg,
        "band_counts": band_counts,
        "high_risk_pct": round(100.0 * band_counts["high"] / scored, 1),
        "median_probability": median,
        "p_percentiles": {
            "p25": round(_percentile(probs, 25), 4),
            "p50": median,
            "p75": round(_percentile(probs, 75), 4),
            "p90": round(_percentile(probs, 90), 4),
        },
    }


def compare(applicant_id: str | None) -> dict | None:
    """Position one applicant against the scored portfolio: percentile (higher =
    riskier), distance from the average, and 1-based risk rank. ``None`` for an
    unknown id. Never raises."""
    result = score_applicant(applicant_id)
    if result is None:
        return None
    rows = _score_portfolio()
    probs = [r["probability"] for r in rows]
    scored = len(probs)
    p = result["probability"]

    if scored:
        below = sum(1 for x in probs if x < p)
        percentile = round(100.0 * below / scored)
        rank = sum(1 for x in probs if x > p) + 1
        avg = round(sum(probs) / scored, 4)
    else:  # applicant not in the store yet — still comparable to itself
        percentile, rank, avg = 100, 1, round(p, 4)

    return {
        "result": result,
        "portfolio": portfolio_summary(),
        "percentile": percentile,
        "vs_avg": round(p - avg, 4),
        "band_rank": rank,
    }


# ---------------------------------------------------------------------------
# Artifact builders — the discriminated union the frontend renders (keyed on
# ``kind``). Built here so risk_chat / the API just attach the returned dict.
# ---------------------------------------------------------------------------
def _none_artifact() -> dict:
    """No structured artifact for this turn (e.g. general / governance prose)."""
    return {"kind": "none"}


def _score_artifact(result: dict) -> dict:
    """A full RiskResult the frontend renders with RiskGauge / RiskCard."""
    return {"kind": "score", "result": result}


def _reasons_artifact(codes: list) -> dict:
    """The reason codes the frontend renders with ReasonCodes."""
    return {"kind": "reasons", "codes": list(codes or [])}


def _whatif_artifact(result: dict, baseline: float, changes: list) -> dict:
    """A hypothetical re-score (kind="whatif"). ``result`` is a full RiskResult
    (same shape as GET /risk/{id}) so the frontend reuses RiskGauge/RiskCard;
    ``baseline`` is the saved estimate's probability; ``changes`` are the feature
    moves the what-if produced."""
    return {
        "kind": "whatif",
        "result": result,
        "baseline": baseline,
        "changes": list(changes or []),
    }


def _counterfactual_artifact(
    target_band: str, achievable: bool, changes: list, result: dict | None = None
) -> dict:
    """The counterfactual levers (kind="counterfactual"). ``changes`` are the
    committed feature moves; ``result`` (optional) is the projected RiskResult
    after applying them."""
    artifact = {
        "kind": "counterfactual",
        "target_band": target_band,
        "achievable": bool(achievable),
        "changes": list(changes or []),
    }
    if result is not None:
        artifact["result"] = result
    return artifact


def _comparison_artifact(subject: dict, rows: list, percentile: int | None = None) -> dict:
    """A portfolio comparison (kind="comparison"). ``subject`` and ``rows`` are
    RiskComparisonRow dicts ({label, probability, band?, applicant_id?})."""
    artifact = {
        "kind": "comparison",
        "subject": subject,
        "rows": list(rows or []),
    }
    if percentile is not None:
        artifact["percentile"] = int(percentile)
    return artifact
