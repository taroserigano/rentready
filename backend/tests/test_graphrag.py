"""Tests for candidate filtering, scoring, and the recommend() flow.

These run offline: no Neo4j (memory fallback) and no LLM (templated reasons,
patched off in conftest).
"""

import graph
import graphrag
from models import ApplicantProfile


def test_memory_candidates_respect_budget_and_pets():
    rows = graph._memory_candidates(
        max_rent=1400, pets_required=True, preferred_area="Downtown"
    )
    assert rows, "expected at least one candidate"
    for r in rows:
        assert r["monthly_rent"] <= 1400
        assert r["pets_allowed"] is True


def test_memory_candidates_prefer_area_first():
    rows = graph._memory_candidates(
        max_rent=5000, pets_required=False, preferred_area="Zilker"
    )
    assert rows[0]["area"] == "Zilker"


def test_recommend_ranks_by_score_and_uses_templates():
    profile = ApplicantProfile(
        monthly_income=6200,
        desired_rent=1800,
        has_pets=True,
        preferred_area="South Congress",
        bedrooms_wanted=2,
        bathrooms_wanted=2.0,
        needs_balcony=True,
        wanted_amenities=["Pet Park"],
    )
    result = graphrag.recommend(profile)
    recs = result["recommendations"]
    assert recs, "expected recommendations"
    # No LLM in tests -> deterministic scorer path.
    assert result["source"] == "scorer"
    # Sorted by score descending.
    scores = [r.score for r in recs]
    assert scores == sorted(scores, reverse=True)
    # Every rec has a templated reason and a transparent breakdown.
    assert all(r.match_reason for r in recs)
    assert all(r.signal_breakdown for r in recs)


def test_graph_ask_without_llm_is_graceful():
    result = graphrag.graph_ask("How many properties are there?")
    assert "answer" in result and "cypher" in result
    assert "ANTHROPIC_API_KEY" in result["answer"]


def test_read_only_cypher_guard():
    assert graphrag.is_read_only_cypher("MATCH (n) RETURN n") is True
    assert graphrag.is_read_only_cypher("MATCH (n) DETACH DELETE n") is False
    assert graphrag.is_read_only_cypher("CREATE (n:Foo)") is False
    assert graphrag.is_read_only_cypher("MATCH (n) SET n.x = 1") is False


def test_clean_cypher_strips_fences_and_prefix():
    assert graphrag._clean_cypher("```cypher\nMATCH (n) RETURN n```") == "MATCH (n) RETURN n"
    assert graphrag._clean_cypher("cypher\nMATCH (n) RETURN n") == "MATCH (n) RETURN n"
    assert graphrag._clean_cypher("MATCH (n) RETURN n") == "MATCH (n) RETURN n"
