"""Applicant strength score (F4) — a deterministic 0-100 tenant-quality score.

ADVISORY ONLY: this does NOT change the eligibility verdict (which stays the
authoritative pass/fail). It rewards the rich, already-collected fields the
verdict ignores (savings, tenure, references, clean history) so two "qualified"
applicants can be compared. Pure Python, no LLM — mirrors signals.py's style.
"""

from models import ApplicantProfile


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def compute(profile: ApplicantProfile) -> dict:
    factors = []

    def add(label: str, points: float, mx: float, tip: str = ""):
        factors.append({
            "label": label,
            "points": round(points, 1),
            "max": mx,
            "suggestion": tip if points < mx * 0.6 else "",
        })

    # Credit (25): 620 -> 0, 800 -> full.
    if profile.credit_score is not None:
        add("Credit score", 25 * _clamp((profile.credit_score - 620) / 180),
            25, "A higher credit score strengthens the application.")
    else:
        add("Credit score", 0, 25, "Provide a credit score.")

    # Income cushion (20): rewards income beyond the 3x floor, up to ~5x.
    rent = profile.desired_rent or 0
    income = profile.monthly_income + (profile.other_income_monthly or 0)
    if rent > 0 and income > 0:
        ratio = income / rent
        add("Income cushion", 20 * _clamp((ratio - 3) / 2), 20,
            "Income closer to 4-5x rent reads as very comfortable.")
    else:
        add("Income cushion", 0, 20, "Add income and desired rent.")

    # Savings runway (15): months of rent covered, full at 6 months.
    if profile.savings_balance and rent > 0:
        add("Savings runway", 15 * _clamp((profile.savings_balance / rent) / 6),
            15, "More savings on hand improves standing.")
    else:
        add("Savings runway", 0, 15, "List savings on hand.")

    # Employment tenure (10): full at 24 months.
    if profile.employment_length_months:
        add("Employment tenure", 10 * _clamp(profile.employment_length_months / 24),
            10, "Longer time at the current job helps.")
    else:
        add("Employment tenure", 0, 10, "Add months at current job.")

    # Low debt (10): DTI, full when debt is 0.
    if income > 0:
        dti = (profile.monthly_debt_payments or 0) / income
        add("Low debt load", 10 * _clamp(1 - dti / 0.4), 10,
            "Lower monthly debt payments strengthen affordability.")
    else:
        add("Low debt load", 0, 10)

    # Clean history (10): evictions/bankruptcies/late payments each subtract.
    hist = 10.0
    hist -= 5 * profile.evictions_count
    hist -= 4 * profile.bankruptcies_count
    hist -= 1 * profile.late_payments_12mo
    add("Clean history", _clamp(hist, 0, 10), 10,
        "Address past evictions/bankruptcies/late payments.")

    # References (5): landlord reference + counted references.
    refs = (3 if profile.landlord_reference else 0) + min(profile.references_count, 2)
    add("References", _clamp(refs, 0, 5), 5, "Add a landlord reference.")

    # Guarantor (5).
    add("Guarantor", 5 if profile.guarantor_available else 0, 5,
        "A guarantor can offset weaker areas.")

    score = round(sum(f["points"] for f in factors))
    band = "strong" if score >= 75 else "solid" if score >= 50 else "thin"
    suggestions = [f["suggestion"] for f in factors if f["suggestion"]][:3]
    return {"score": score, "band": band, "factors": factors, "suggestions": suggestions}
