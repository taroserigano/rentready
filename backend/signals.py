"""Transparent, weighted scoring for property recommendations.

Each signal returns a sub-score in [0, 1], or None when the inputs are
missing. The final score is a weighted average over ONLY the signals that
are present (weights renormalized), so missing data never silently
penalizes a property. The per-signal breakdown is returned too, which is
what makes the score explainable (and what the LLM explains in words).
"""

from rapidfuzz import fuzz

from models import ApplicantProfile

# Soft-preference weights. They sum to 1.0; renormalized per-applicant over
# whichever signals actually have data.
WEIGHTS = {
    "affordability": 0.22,  # rent vs income (30% rule) - the real constraint
    "area": 0.18,           # in the neighborhood they asked for
    "amenities": 0.14,      # coverage of the amenities they want
    "bedrooms": 0.12,       # right number of bedrooms
    "budget_pref": 0.08,    # near their stated desired rent
    "bathrooms": 0.07,      # bathroom count + type (full vs shower-only)
    "transit": 0.06,        # neighborhood walk/transit score
    "square_feet": 0.05,    # meets their space floor
    "parking": 0.03,        # has parking when wanted
    "balcony": 0.03,        # has a balcony when wanted
    "laundry": 0.02,        # in-unit laundry when wanted
}


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _affordability(p: dict, a: ApplicantProfile):
    if a.monthly_income <= 0:
        return None
    target = a.monthly_income / 3.0  # 30%-of-income rule
    ratio = p["monthly_rent"] / target
    # ratio 1.0 -> 1.0, ratio 1.5 (45% of income) -> 0.0, cheaper caps at 1.0
    return _clamp(1 - (ratio - 1) / 0.5)


def _budget_pref(p: dict, a: ApplicantProfile):
    if a.desired_rent <= 0:
        return None
    over = (p["monthly_rent"] - a.desired_rent) / a.desired_rent
    return _clamp(1 - max(0.0, over) / 0.10)  # 10% over desired -> 0


def _area(p: dict, a: ApplicantProfile):
    if not a.preferred_area:
        return None
    want, have = a.preferred_area.lower(), str(p.get("area", "")).lower()
    if want == have:
        return 1.0
    fuzzy = fuzz.token_set_ratio(want, have) / 100
    return fuzzy if fuzzy >= 0.6 else 0.0


def _amenities(p: dict, a: ApplicantProfile):
    wanted = {x.lower() for x in a.wanted_amenities}
    if not wanted:
        return None
    have = {x.lower() for x in p.get("amenities", [])}
    return len(wanted & have) / len(wanted)  # coverage of what they asked for


def _bedrooms(p: dict, a: ApplicantProfile):
    if a.bedrooms_wanted is None:
        return None
    diff = p["bedrooms"] - a.bedrooms_wanted
    if diff == 0:
        return 1.0
    if diff == 1:
        return 0.7   # one extra bedroom: mildly positive
    if diff == -1:
        return 0.3   # one fewer: a real downgrade
    return _clamp(1 - 0.35 * abs(diff))


def _bathrooms(p: dict, a: ApplicantProfile):
    if a.bathrooms_wanted is None:
        return None
    have = p.get("bathrooms", 0) or 0
    count = 1.0 if have >= a.bathrooms_wanted else _clamp(have / a.bathrooms_wanted)
    type_ok = 1.0
    if a.bath_type_wanted == "full" and p.get("bathroom_type") == "shower_only":
        type_ok = 0.5
    return 0.7 * count + 0.3 * type_ok


def _square_feet(p: dict, a: ApplicantProfile):
    if not a.min_square_feet:
        return None
    sqft = p.get("square_feet", 0) or 0
    return 1.0 if sqft >= a.min_square_feet else _clamp(sqft / a.min_square_feet)


def _transit(p: dict, a: ApplicantProfile):
    # Only counts if the applicant cares about location/transit (has a
    # preferred area). Uses the neighborhood's transit score.
    if not a.preferred_area:
        return None
    ts = p.get("transit_score")
    if ts is None:
        return None
    return _clamp(ts / 100)


def _bool_pref(wanted: bool, has) -> float:
    if not wanted:
        return None  # only scored when the applicant explicitly wants it
    return 1.0 if has else 0.0


def _parking(p, a):
    return _bool_pref(a.needs_parking, p.get("parking_type") not in (None, "none"))


def _balcony(p, a):
    return _bool_pref(a.needs_balcony, bool(p.get("has_balcony")))


def _laundry(p, a):
    return _bool_pref(a.needs_in_unit_laundry, bool(p.get("in_unit_laundry")))


SIGNALS = {
    "affordability": _affordability,
    "area": _area,
    "amenities": _amenities,
    "bedrooms": _bedrooms,
    "budget_pref": _budget_pref,
    "bathrooms": _bathrooms,
    "transit": _transit,
    "square_feet": _square_feet,
    "parking": _parking,
    "balcony": _balcony,
    "laundry": _laundry,
}


def score_property(p: dict, a: ApplicantProfile) -> tuple:
    """Return (final_score, breakdown) for one candidate property.

    breakdown maps each present signal to its 0-1 sub-score (rounded).
    """
    subs = {}
    for name, fn in SIGNALS.items():
        val = fn(p, a)
        if val is not None:
            subs[name] = round(float(val), 3)

    active_weight = sum(WEIGHTS[n] for n in subs)
    if active_weight == 0:
        return 0.5, subs  # no usable signals -> neutral
    score = sum(WEIGHTS[n] * subs[n] for n in subs) / active_weight
    return round(score, 4), subs


# Templated explanations for the no-LLM fallback.
_TEMPLATES = {
    "area": lambda p: f"in your preferred area ({p['area']})",
    "affordability": lambda p: "comfortably within your income",
    "budget_pref": lambda p: "right around your target rent",
    "bedrooms": lambda p: f"{p['bedrooms']} bedroom(s) as wanted",
    "bathrooms": lambda p: f"{p.get('bathrooms', 0)} bath ({p.get('bathroom_type','')})",
    "amenities": lambda p: "has amenities you want",
    "transit": lambda p: "good transit/walkability",
    "square_feet": lambda p: f"{p.get('square_feet', 0)} sq ft",
    "parking": lambda p: "includes parking",
    "balcony": lambda p: "has a balcony",
    "laundry": lambda p: "in-unit laundry",
}


def templated_reason(p: dict, subs: dict) -> tuple:
    """Build a (reason, highlights) from the strongest sub-scores."""
    top = sorted(
        ((n, s) for n, s in subs.items() if s >= 0.6 and n in _TEMPLATES),
        key=lambda x: -x[1],
    )[:4]
    highlights = [_TEMPLATES[n](p) for n, _ in top]
    reason = (
        "Good fit: " + ", ".join(highlights)
        if highlights
        else "Meets your budget and basic requirements."
    )
    return reason, highlights
