"""Tests for the Property & Lease Concierge.

Runs fully offline: the conftest forces the hash embedder, and we disable the
Anthropic LLM so synthesis takes the deterministic ("rules") path. The lease
knowledge base is ingested into a throwaway Chroma dir per test session so the
shared repo store is never touched.
"""

import pytest

import concierge
import knowledge
import leases
from graph import load_properties


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _no_concierge_llm(monkeypatch):
    """Force the deterministic synthesis path regardless of any .env key."""
    monkeypatch.setattr(concierge, "get_langchain_llm", lambda: None)


@pytest.fixture(scope="module")
def kb(tmp_path_factory):
    """Ingest all leases into an isolated Chroma dir, once per module."""
    tmp = tmp_path_factory.mktemp("knowledge_chroma")
    orig = knowledge.CHROMA_DIR
    knowledge.CHROMA_DIR = tmp
    knowledge._chroma_collection.cache_clear()
    knowledge._index.cache_clear()
    knowledge._corpus.cache_clear()
    result = knowledge.ingest_all()
    yield result
    knowledge.CHROMA_DIR = orig
    knowledge._chroma_collection.cache_clear()
    knowledge._index.cache_clear()
    knowledge._corpus.cache_clear()


def _first_pets_allowed():
    return next(p for p in load_properties() if p.get("pets_allowed"))


def _first_no_pets():
    return next(p for p in load_properties() if not p.get("pets_allowed"))


# ---------------------------------------------------------------------------
# 1. Lease-section generation is consistent with the structured facts
# ---------------------------------------------------------------------------
def test_lease_sections_are_18_titled_sections():
    prop = load_properties()[0]
    sections = leases.lease_sections(prop)
    assert len(sections) == 18
    titles = [t for t, _ in sections]
    assert "Rent" in titles and "Pets" in titles and "Subletting & Assignment" in titles
    assert all(text.strip() for _, text in sections)


def test_lease_rent_and_term_match_facts():
    prop = load_properties()[0]
    sections = dict(leases.lease_sections(prop))
    rent = prop["monthly_rent"]
    assert f"${int(rent):,}" in sections["Rent"]
    assert f"{prop['lease_term_months']} months" in sections["Term"]


def test_lease_deposit_matches_structured_field():
    prop = _first_no_pets()  # e.g. Riverside Lofts
    sections = dict(leases.lease_sections(prop))
    dep = int(prop["security_deposit"])
    assert f"${dep:,}" in sections["Security Deposit"]


def test_lease_pet_policy_reflects_pets_allowed():
    allowed = dict(leases.lease_sections(_first_pets_allowed()))["Pets"]
    denied = dict(leases.lease_sections(_first_no_pets()))["Pets"]
    assert "permitted" in allowed.lower()
    assert "no pets" in denied.lower()
    assert "assistance animal" in denied.lower()


def test_lease_markdown_renders():
    md = leases.lease_markdown(load_properties()[0])
    assert md.startswith("# Residential Lease Agreement")
    assert "## 6. Pets" in md


# ---------------------------------------------------------------------------
# 2. Router classification
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "question,expected",
    [
        ("How much is the rent?", "property"),
        ("Does it have a gym?", "property"),
        ("How many bedrooms are there?", "property"),
        ("What is the square footage?", "property"),
        ("What is the security deposit?", "lease"),
        ("Can I sublet the apartment?", "lease"),
        ("What's the notice period to move out?", "lease"),
        ("What happens if I pay rent late?", "lease"),
        ("How much is the rent and can I sublet?", "both"),
        ("Which pet-friendly homes are under $2000?", "compare"),
        ("What are the cheapest 2-bedroom apartments?", "compare"),
        ("Compare homes in Zilker with a gym", "compare"),
        ("List the most affordable studios", "compare"),
        ("pet friendly under $1500", "compare"),
    ],
)
def test_router_classification(question, expected):
    assert concierge.route(question) == expected


# ---------------------------------------------------------------------------
# 2b. Cross-property comparison
# ---------------------------------------------------------------------------
def test_route_scoped_property_not_compare():
    # A scoped amenity question stays a property question (not a compare scan).
    assert concierge.route("Does it have a gym?", property_id="PROP-001") == "property"


def test_compare_properties_pets_under_budget():
    items = concierge.compare_properties("pet-friendly homes under $1,300")
    assert items
    assert all(c["pets_allowed"] for c in items)
    assert all(c["monthly_rent"] <= 1300 for c in items)
    # sorted ascending by rent, capped at 6
    rents = [c["monthly_rent"] for c in items]
    assert rents == sorted(rents)
    assert len(items) <= 6
    for c in items:
        assert "allows pets" in c["matched"]


def test_compare_properties_no_filter_returns_cheapest():
    items = concierge.compare_properties("list the most affordable apartments")
    assert 0 < len(items) <= 6
    rents = [c["monthly_rent"] for c in items]
    assert rents == sorted(rents)


def test_answer_compare_shape_and_comparison(kb):
    res = concierge.answer("which pet-friendly homes are under $2000?")
    _assert_shape(res, "")
    assert res["route"] == "compare"
    assert res["comparison"], "expected a non-empty comparison"
    assert all(c["pets_allowed"] for c in res["comparison"])
    # sources are grounded property records, one per matched home
    assert res["sources"]
    assert all(s["type"] == "property" for s in res["sources"])


def test_answer_compare_no_match_degrades(kb):
    res = concierge.answer("which homes under $100 have a gym and a pool?")
    _assert_shape(res, "")
    assert res["route"] == "compare"
    assert res["comparison"] == []
    assert "couldn't find" in res["answer"].lower()


# ---------------------------------------------------------------------------
# 3. Hybrid search finds the right lease section (LLM disabled, hash embedder)
# ---------------------------------------------------------------------------
def test_search_finds_pets_section(kb):
    prop = _first_pets_allowed()
    hits = knowledge.search("how much is the pet deposit", property_id=prop["id"], k=4)
    assert hits
    assert any(h["section"] == "Pets" for h in hits)
    assert all(h["property_id"] == prop["id"] for h in hits)


def test_search_finds_sublet_section(kb):
    prop = load_properties()[0]
    hits = knowledge.search("can I sublet my apartment", property_id=prop["id"], k=4)
    assert any(h["section"] == "Subletting & Assignment" for h in hits)


def test_search_finds_deposit_section(kb):
    prop = load_properties()[0]
    hits = knowledge.search("what is the security deposit", property_id=prop["id"], k=4)
    assert any(h["section"] == "Security Deposit" for h in hits)


# ---------------------------------------------------------------------------
# 4. Grounded answers: never raise, correct shape, cited
# ---------------------------------------------------------------------------
def _assert_shape(res, expected_pid=""):
    assert set(res) == {
        "answer", "route", "sources", "source", "property_id", "follow_ups",
        "comparison",
    }
    assert isinstance(res["answer"], str) and res["answer"].strip()
    assert res["route"] in ("property", "lease", "both", "general", "compare")
    assert res["source"] in ("anthropic", "rules")
    assert res["property_id"] == expected_pid
    assert isinstance(res["follow_ups"], list) and len(res["follow_ups"]) <= 3
    assert isinstance(res["comparison"], list)
    for s in res["sources"]:
        assert s["type"] in ("property", "lease")
        assert s["label"] and "snippet" in s
        # deep-link fields for the lease viewer
        assert "section" in s and "property_id" in s
    for c in res["comparison"]:
        assert set(c) >= {
            "id", "name", "area", "property_type", "monthly_rent", "bedrooms",
            "bathrooms", "square_feet", "pets_allowed", "matched",
        }
        assert isinstance(c["matched"], list)


def test_answer_gym_is_property_route_and_correct(kb):
    prop = next(p for p in load_properties() if "Gym" in (p.get("amenities") or []))
    res = concierge.answer("Does it have a gym?", property_id=prop["id"])
    _assert_shape(res, prop["id"])
    assert res["route"] == "property"
    assert res["source"] == "rules"
    assert "yes" in res["answer"].lower()
    assert any(s["type"] == "property" for s in res["sources"])


def test_answer_pet_and_deposit_is_lease_route_cited(kb):
    prop = _first_no_pets()
    res = concierge.answer(
        "What's the pet policy and security deposit?", property_id=prop["id"]
    )
    _assert_shape(res, prop["id"])
    assert res["route"] == "lease"
    assert res["sources"], "expected lease citations"
    assert any(s["type"] == "lease" for s in res["sources"])
    assert "[1]" in res["answer"]  # inline citation present


def test_answer_sublet_is_lease(kb):
    prop = load_properties()[0]
    res = concierge.answer("Can I sublet the apartment?", property_id=prop["id"])
    _assert_shape(res, prop["id"])
    assert res["route"] == "lease"
    assert any(s["section"] if False else s["type"] == "lease" for s in res["sources"])


def test_answer_never_raises_on_junk(kb):
    for q in ["", "   ", "asdkfjhaslkdjf", "???"]:
        res = concierge.answer(q, property_id=None)
        _assert_shape(res, "")


# ---------------------------------------------------------------------------
# 5. Unknown property handling
# ---------------------------------------------------------------------------
def test_property_facts_unknown_returns_none():
    assert concierge.property_facts("PROP-999") is None


def test_answer_unknown_property_degrades(kb):
    # concierge itself never raises even if the id is unknown (the API layer is
    # what returns 404); it simply has no property facts to cite.
    res = concierge.answer("Does it have a gym?", property_id="PROP-999")
    _assert_shape(res, "PROP-999")
