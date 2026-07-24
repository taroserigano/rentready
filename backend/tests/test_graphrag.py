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


def test_read_only_cypher_guard_known_gap_documents_why_layer_2_exists():
    """The keyword blocklist is a fast pre-check, not the real safety
    boundary — a write hidden behind an APOC procedure name containing none
    of the blocked keywords slips past it. This is why graph_ask() also runs
    the query under a real Neo4j read-mode transaction (run_read_only_cypher),
    which the server itself enforces regardless of how the write is spelled."""
    sneaky_write = "CALL apoc.refactor.rename.label('Property','Deleted') YIELD committedOperations RETURN committedOperations"
    assert graphrag.is_read_only_cypher(sneaky_write) is True


def test_run_read_only_cypher_uses_read_routing(monkeypatch):
    """graph.run_read_only_cypher must open the query in READ access mode —
    this is what actually stops a write query from executing, not the
    client-side keyword check."""
    from neo4j import RoutingControl

    calls = {}

    class FakeDriver:
        def execute_query(self, cypher, params, database_, routing_):
            calls["cypher"] = cypher
            calls["routing_"] = routing_
            return ([], None, None)

    monkeypatch.setattr(graph, "_driver", lambda: FakeDriver())
    graph.run_read_only_cypher("MATCH (n) RETURN n")
    assert calls["routing_"] is RoutingControl.READ


# ---------------------------------------------------------------------------
# Deterministic Cypher templates (the common filter/search shapes) — no
# Cypher-writing LLM call, so these are pure functions testable offline.
# ---------------------------------------------------------------------------


def test_template_area_and_pets():
    cypher, params, kind, meta = graphrag._build_template(
        "Which properties in South Congress allow pets?"
    )
    assert kind == "property_search"
    assert params["area"] == "South Congress"
    assert "pets_allowed = true" in cypher
    assert meta["want_pets"] is True


def test_template_beds_and_price():
    cypher, params, kind, meta = graphrag._build_template("Cheapest 2-bedroom under $2,000?")
    assert kind == "property_search"
    assert params["min_beds"] == 2
    assert params["max_rent"] == 2000.0


def test_template_area_reverse_lookup_for_amenity():
    cypher, params, kind, meta = graphrag._build_template("Which areas have a gym?")
    assert kind == "areas_with_amenity"
    assert params == {"amenity": "Gym"}
    assert "IN_NEIGHBORHOOD" in cypher


def test_template_multiple_amenities_and_parking():
    cypher, params, kind, meta = graphrag._build_template("Homes with a pool and parking")
    assert kind == "property_search"
    assert params["wanted_amenities"] == ["Pool"]
    assert meta["want_parking"] is True
    assert "collect(" in cypher and "all(" in cypher


def test_no_template_for_unfiltered_question():
    """A question with no recognizable filter falls back to the free-form LLM
    Cypher path, unchanged from before this feature existed."""
    assert graphrag._build_template("How many properties are there?") is None


def test_display_cypher_inlines_param_values():
    cypher = "MATCH (p:Property) WHERE p.monthly_rent <= $max_rent RETURN p"
    rendered = graphrag._display_cypher(cypher, {"max_rent": 2000.0})
    assert "$max_rent" not in rendered
    assert "2000.0" in rendered


def test_template_answer_property_search_lists_matches():
    rows = [{"name": "The Heights", "property_type": "Apartment",
             "monthly_rent": 1300, "bedrooms": 1, "bathrooms": 1}]
    answer = graphrag._template_answer("property_search", rows, {})
    assert "The Heights" in answer and "$1,300/mo" in answer


def test_template_answer_property_search_handles_no_matches():
    assert "No properties" in graphrag._template_answer("property_search", [], {})


def test_template_answer_areas_with_amenity():
    rows = [{"neighborhood": "Zilker"}, {"neighborhood": "Downtown"}]
    answer = graphrag._template_answer("areas_with_amenity", rows, {"amenity": "Gym"})
    assert "Zilker" in answer and "Downtown" in answer and "Gym" in answer


def test_graph_ask_uses_template_source_when_matched(monkeypatch):
    """graph_ask() must skip the Cypher-writing LLM entirely for a recognized
    filter shape — even offline (no Anthropic key), as long as Neo4j itself is
    reachable — and must not fall through to the free-form LLM path."""
    monkeypatch.setattr(graph, "is_available", lambda: True)
    monkeypatch.setattr(
        graph, "run_read_only_cypher",
        lambda cypher, params=None: [
            {"name": "The Heights", "property_type": "Apartment",
             "monthly_rent": 1300, "bedrooms": 1, "bathrooms": 1}
        ],
    )
    result = graphrag.graph_ask("Which properties in South Congress allow pets?")
    assert result["source"] == "template"
    assert "The Heights" in result["answer"]
    assert "$area" not in result["cypher"]  # rendered for display, not the raw placeholder


def test_graph_ask_template_degrades_when_neo4j_unavailable():
    """The autouse conftest fixture forces graph.is_available() False, so a
    matched template must degrade gracefully rather than crash."""
    result = graphrag.graph_ask("Which properties in South Congress allow pets?")
    assert result["source"] == "rules"
    assert "not available" in result["answer"]


def test_graph_ask_coalesces_extended_thinking_content_blocks(monkeypatch):
    """A regression guard: the free-form answer step must extract only the
    "text" block, not stringify a "thinking" block ahead of it."""
    assert graphrag._coalesce([
        {"type": "thinking", "thinking": "reasoning...", "signature": "abc"},
        {"type": "text", "text": "final answer"},
    ]) == "final answer"


# ---------------------------------------------------------------------------
# Property-graph visualization — a small, safe (hand-authored, not
# LLM-generated) subgraph fetched alongside a matched template's answer.
# ---------------------------------------------------------------------------


def test_fetch_subgraph_builds_nodes_and_edges(monkeypatch):
    def fake_run(cypher, params=None):
        assert params["ids"] == ["PROP-001", "PROP-002"]
        return [
            {"pid": "PROP-001", "pname": "The Heights", "nname": "South Congress",
             "amenities": ["Gym", "Pool"]},
            {"pid": "PROP-002", "pname": "Sunset Terrace", "nname": "South Congress",
             "amenities": ["Pool"]},
        ]

    monkeypatch.setattr(graph, "run_read_only_cypher", fake_run)
    result = graphrag._fetch_subgraph(["PROP-001", "PROP-002"])
    types = {n["type"] for n in result["nodes"]}
    assert types == {"Property", "Neighborhood", "Amenity"}
    # The shared neighborhood and shared "Pool" amenity are deduped to ONE
    # node each, not one per property.
    assert sum(1 for n in result["nodes"] if n["type"] == "Neighborhood") == 1
    assert sum(1 for n in result["nodes"] if n["id"] == "a:Pool") == 1
    edge_pairs = {(e["source"], e["target"]) for e in result["edges"]}
    assert ("p:PROP-001", "n:South Congress") in edge_pairs
    assert ("p:PROP-001", "a:Gym") in edge_pairs
    assert ("p:PROP-002", "a:Pool") in edge_pairs
    assert ("p:PROP-001", "a:Pool") in edge_pairs


def test_fetch_subgraph_filters_to_wanted_amenities(monkeypatch):
    """When the search filtered on specific amenities, the diagram stays
    scoped to those -- not every amenity the property happens to have."""
    monkeypatch.setattr(
        graph, "run_read_only_cypher",
        lambda cypher, params=None: [
            {"pid": "PROP-001", "pname": "The Heights", "nname": "South Congress",
             "amenities": ["Gym", "Pool", "Concierge"]},
        ],
    )
    result = graphrag._fetch_subgraph(["PROP-001"], amenity_filter=["Pool"])
    amenity_ids = {n["id"] for n in result["nodes"] if n["type"] == "Amenity"}
    assert amenity_ids == {"a:Pool"}


def test_fetch_subgraph_empty_ids_returns_empty():
    assert graphrag._fetch_subgraph([]) == {"nodes": [], "edges": []}


def test_fetch_subgraph_degrades_on_query_error(monkeypatch):
    def boom(cypher, params=None):
        raise RuntimeError("connection lost")

    monkeypatch.setattr(graph, "run_read_only_cypher", boom)
    assert graphrag._fetch_subgraph(["PROP-001"]) == {"nodes": [], "edges": []}


def test_graph_ask_includes_subgraph_when_template_matches(monkeypatch):
    monkeypatch.setattr(graph, "is_available", lambda: True)

    def fake_run(cypher, params=None):
        if "IN_NEIGHBORHOOD]->(n:Neighborhood) WHERE p.id IN" in cypher:
            return [{"pid": "PROP-001", "pname": "The Heights", "nname": "South Congress",
                     "amenities": []}]
        return [{"id": "PROP-001", "name": "The Heights", "property_type": "Apartment",
                 "monthly_rent": 1300, "bedrooms": 1, "bathrooms": 1}]

    monkeypatch.setattr(graph, "run_read_only_cypher", fake_run)
    result = graphrag.graph_ask("Which properties in South Congress allow pets?")
    assert result["source"] == "template"
    assert result["graph"] is not None
    assert any(n["type"] == "Property" for n in result["graph"]["nodes"])


def test_graph_ask_graph_is_none_when_no_properties_matched(monkeypatch):
    monkeypatch.setattr(graph, "is_available", lambda: True)
    monkeypatch.setattr(graph, "run_read_only_cypher", lambda cypher, params=None: [])
    result = graphrag.graph_ask("Which properties in South Congress allow pets?")
    assert result["graph"] is None


def test_graph_ask_free_form_path_has_no_graph(monkeypatch):
    """The free-form LLM-Cypher path never attempts subgraph extraction --
    an arbitrary query's result shape isn't predictable enough."""
    result = graphrag.graph_ask("How many properties are there?")
    assert result["graph"] is None
