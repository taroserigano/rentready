"""Tests for the deterministic eligibility rules."""

import eligibility
from models import ApplicantProfile


def test_qualified_when_income_and_credit_good():
    profile = ApplicantProfile(
        name="Good Applicant",
        monthly_income=6000,
        desired_rent=1500,
        credit_score=700,
    )
    result = eligibility.evaluate(profile)
    assert result.verdict == "qualified"
    assert result.income_to_rent_ratio == 4.0


def test_not_qualified_when_income_too_low():
    profile = ApplicantProfile(
        monthly_income=2000,
        desired_rent=1500,
        credit_score=700,
    )
    result = eligibility.evaluate(profile)
    assert result.verdict == "not_qualified"


def test_needs_review_when_credit_missing():
    profile = ApplicantProfile(
        monthly_income=6000,
        desired_rent=1500,
        credit_score=None,
    )
    result = eligibility.evaluate(profile)
    assert result.verdict == "needs_review"


def test_needs_review_when_rent_or_income_missing():
    profile = ApplicantProfile(monthly_income=0, desired_rent=0)
    result = eligibility.evaluate(profile)
    assert result.verdict == "needs_review"
    assert result.income_to_rent_ratio is None
