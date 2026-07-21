"""Tests for the transparent scoring module."""

import signals
from models import ApplicantProfile

PROP = {
    "id": "PROP-X",
    "name": "Test Place",
    "area": "South Congress",
    "monthly_rent": 1800,
    "bedrooms": 2,
    "bathrooms": 2.0,
    "bathroom_type": "full",
    "square_feet": 900,
    "has_balcony": True,
    "in_unit_laundry": True,
    "parking_type": "garage",
    "transit_score": 60,
    "amenities": ["Pet Park", "Gym"],
}


def test_perfect_match_scores_high():
    a = ApplicantProfile(
        monthly_income=6000,
        desired_rent=1800,
        preferred_area="South Congress",
        bedrooms_wanted=2,
        bathrooms_wanted=2.0,
        bath_type_wanted="full",
        needs_balcony=True,
        wanted_amenities=["Pet Park"],
    )
    score, subs = signals.score_property(PROP, a)
    assert score > 0.85
    assert subs["area"] == 1.0
    assert subs["bedrooms"] == 1.0


def test_missing_signals_are_not_penalized():
    # Applicant only states income + area; other signals should be absent
    # from the breakdown, not scored as zero.
    a = ApplicantProfile(monthly_income=6000, preferred_area="South Congress")
    score, subs = signals.score_property(PROP, a)
    assert "bedrooms" not in subs
    assert "amenities" not in subs
    assert score > 0.5


def test_affordability_drops_when_rent_too_high():
    cheap = dict(PROP, monthly_rent=1200)
    pricey = dict(PROP, monthly_rent=2600)
    a = ApplicantProfile(monthly_income=4000)
    cheap_score, _ = signals.score_property(cheap, a)
    pricey_score, _ = signals.score_property(pricey, a)
    assert cheap_score > pricey_score


def test_shower_only_penalized_when_full_wanted():
    shower = dict(PROP, bathroom_type="shower_only")
    a = ApplicantProfile(bathrooms_wanted=2.0, bath_type_wanted="full")
    full_score, _ = signals.score_property(PROP, a)
    shower_score, _ = signals.score_property(shower, a)
    assert full_score > shower_score


def test_balcony_required_but_absent_scores_zero_subsignal():
    no_balcony = dict(PROP, has_balcony=False)
    a = ApplicantProfile(needs_balcony=True)
    _, subs = signals.score_property(no_balcony, a)
    assert subs["balcony"] == 0.0
