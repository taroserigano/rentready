"""Eligibility rules + a plain-English explanation from Claude.

The rules are deterministic (income vs rent, credit). Claude only *explains*
the verdict in friendly language -- it doesn't decide it -- so the decision
stays transparent and testable.
"""

from models import ApplicantProfile, EligibilityResult
from settings import settings
from llm import get_langchain_llm


def evaluate(profile: ApplicantProfile, explain: bool = True) -> EligibilityResult:
    reasons = []
    ratio = None

    rent = profile.desired_rent or 0
    if rent > 0 and profile.monthly_income > 0:
        ratio = round(profile.monthly_income / rent, 2)

    income_ok = ratio is not None and ratio >= settings.income_rent_multiple
    if ratio is None:
        reasons.append("Income or desired rent missing; cannot check ratio.")
    elif income_ok:
        reasons.append(
            f"Income is {ratio}x rent (needs "
            f"{settings.income_rent_multiple}x)."
        )
    else:
        reasons.append(
            f"Income is only {ratio}x rent (needs "
            f"{settings.income_rent_multiple}x)."
        )

    credit_ok = True
    if profile.credit_score is not None:
        credit_ok = profile.credit_score >= settings.min_credit_score
        reasons.append(
            f"Credit score {profile.credit_score} "
            f"({'meets' if credit_ok else 'below'} minimum "
            f"{settings.min_credit_score})."
        )
    else:
        reasons.append("No credit score provided.")

    if income_ok and credit_ok and profile.credit_score is not None:
        verdict = "qualified"
    elif ratio is None or profile.credit_score is None:
        verdict = "needs_review"
    elif not income_ok:
        verdict = "not_qualified"
    else:
        verdict = "needs_review"

    # --- Conservative history checks -------------------------------------
    # These can only add reasons or downgrade "qualified" to "needs_review".
    # They never upgrade a verdict, and profiles with default values (no
    # evictions, no bankruptcies, no debt) are completely unaffected.
    review_flags = []

    if profile.evictions_count >= 1:
        review_flags.append(
            f"Has {profile.evictions_count} past "
            f"eviction{'s' if profile.evictions_count > 1 else ''}; "
            "a person needs to review this application."
        )

    if profile.bankruptcies_count >= 1:
        review_flags.append(
            f"Has {profile.bankruptcies_count} past "
            f"bankruptc{'ies' if profile.bankruptcies_count > 1 else 'y'}; "
            "a person needs to review this application."
        )

    if (
        profile.monthly_debt_payments > 0
        and profile.monthly_debt_payments > 0.4 * profile.monthly_income
    ):
        review_flags.append(
            "Monthly debt payments are more than 40% of income; "
            "a person needs to review this application."
        )

    if review_flags:
        if profile.guarantor_available:
            # A guarantor softens the wording but never changes the verdict.
            review_flags = [
                flag + " A guarantor is available, which helps."
                for flag in review_flags
            ]
        reasons.extend(review_flags)
        if verdict == "qualified":
            verdict = "needs_review"

    explanation = (
        _explain(profile, verdict, reasons)
        if explain
        else f"Verdict: {verdict.replace('_', ' ')}. " + " ".join(reasons)
    )
    return EligibilityResult(
        verdict=verdict,
        reasons=reasons,
        income_to_rent_ratio=ratio,
        explanation=explanation,
    )


def _explain(profile: ApplicantProfile, verdict: str, reasons: list) -> str:
    llm = get_langchain_llm()
    if llm is None:
        return (
            f"Verdict: {verdict.replace('_', ' ')}. " + " ".join(reasons)
        )
    try:
        prompt = (
            "Explain this rental eligibility decision to the applicant in 2-3 "
            "friendly, plain-English sentences. Do not change the verdict.\n"
            f"Applicant: {profile.name}\nVerdict: {verdict}\n"
            f"Facts: {'; '.join(reasons)}"
        )
        return llm.invoke(prompt).content
    except Exception as exc:  # noqa: BLE001
        print(f"Eligibility explanation LLM failed ({type(exc).__name__}: {exc})")
        return f"Verdict: {verdict.replace('_', ' ')}. " + " ".join(reasons)
