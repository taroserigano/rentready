"""Tests for the regex/heuristic profile extraction (no LLM)."""

import rag_llamaindex

SAMPLE = """RENTAL APPLICATION
Name: Jordan Rivera
Employment Status: Employed full-time
Monthly Income: $6,200
Credit Score: 712
Desired Rent: $1,800
Bedrooms: 2
Bathrooms: 2 (full bath)
Minimum Size: 900 square feet
Preferred Area: South Congress
Pets: Yes - one small dog
Balcony: Yes
Parking: Yes - covered parking required
Laundry: Yes - in-unit laundry
Lease Term: 12 months
Wanted Amenities: Pet Park, Gym
"""


def test_heuristic_extracts_core_fields():
    profile = rag_llamaindex._heuristic_profile(SAMPLE)
    assert profile.monthly_income == 6200
    assert profile.desired_rent == 1800
    assert profile.credit_score == 712
    assert profile.has_pets is True
    assert profile.bedrooms_wanted == 2


def test_heuristic_extracts_rich_preferences():
    profile = rag_llamaindex._heuristic_profile(SAMPLE)
    assert profile.bathrooms_wanted == 2.0
    assert profile.bath_type_wanted == "full"
    assert profile.min_square_feet == 900
    assert profile.needs_balcony is True
    assert profile.needs_parking is True
    assert profile.needs_in_unit_laundry is True
    assert profile.lease_term_wanted == 12


def test_heuristic_finds_known_amenities():
    profile = rag_llamaindex._heuristic_profile(SAMPLE)
    assert "Pet Park" in profile.wanted_amenities
    assert "Gym" in profile.wanted_amenities


def test_heuristic_handles_empty_text():
    profile = rag_llamaindex._heuristic_profile("")
    assert profile.monthly_income == 0.0
    assert profile.name == "Unknown"
