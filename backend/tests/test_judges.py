"""Offline tests for the LLM-as-judge module.

We test the deterministic, no-LLM parts: the faithfulness tripwire, JSON
parsing/clamping, and that the live suite skips cleanly without a key.
"""

from evals import judges


# ------------------------- deterministic faithfulness ----------------------
def test_flags_claimed_amenity_not_present():
    v = judges.faithfulness_violations(
        "A bright unit that features a gym and a sparkling pool.", []
    )
    assert set(v) == {"Gym", "Pool"}


def test_no_violation_when_amenity_present():
    v = judges.faithfulness_violations(
        "Comes with a gym for your workouts.", ["Gym"]
    )
    assert v == []


def test_negated_amenity_is_not_a_violation():
    # Honestly stating an absence must not count as a hallucination.
    assert judges.faithfulness_violations("There is no gym on-site.", []) == []
    assert judges.faithfulness_violations("It has no pool.", []) == []


def test_multiword_amenity_detected():
    v = judges.faithfulness_violations(
        "Residents enjoy a rooftop deck with skyline views.", []
    )
    assert v == ["Rooftop Deck"]


def test_empty_reason_is_clean():
    assert judges.faithfulness_violations("", ["Gym"]) == []


# ------------------------------ json parsing -------------------------------
def test_first_json_obj_extracts_object_from_prose():
    raw = 'Sure! Here is the result:\n{"score": 4, "reason": "ok"} thanks'
    assert judges._first_json_obj(raw) == {"score": 4, "reason": "ok"}


def test_first_json_obj_handles_garbage():
    assert judges._first_json_obj("not json at all") == {}


def test_clamp_score_bounds_and_rounding():
    assert judges._clamp_score(9) == 5
    assert judges._clamp_score(0) == 1
    assert judges._clamp_score("3") == 3
    assert judges._clamp_score(4.6) == 5
    assert judges._clamp_score(None) is None
    assert judges._clamp_score("nope") is None


# ------------------------------ graceful skip ------------------------------
def test_judge_suite_skips_without_llm(monkeypatch):
    monkeypatch.setattr(judges, "get_langchain_llm", lambda: None)
    out = judges.run_judge_suite()
    assert out["skipped"] is True
