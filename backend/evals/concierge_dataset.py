"""Labeled dataset for the Concierge evaluation suite.

Each item is anchored to real property data (``data/properties.json``) and the
generated lease sections (``leases._SECTION_SPECS``), so the deterministic
answers and retrieval targets are reproducible offline (hash embedder, no LLM).

Fields per item:
  id                    stable identifier
  question              the resident's question
  property_id           scoped property, or None (compare items)
  expected_route        one of property | lease | both | compare | general
  expected_section      lease section that should be retrieved (lease items)
  expected_property_ids ids that must appear in the comparison (compare items)
  must_include          facts that should surface in the answer OR its sources
"""

CONCIERGE_DATASET = [
    # --- property (structured facts) -------------------------------------
    {
        "id": "prop-gym",
        "question": "Does it have a gym?",
        "property_id": "PROP-001",  # Maple Court — has a Gym
        "expected_route": "property",
        "expected_section": None,
        "expected_property_ids": None,
        "must_include": ["gym"],
    },
    {
        "id": "prop-rent",
        "question": "How much is the monthly rent?",
        "property_id": "PROP-002",  # Riverside Lofts — $2,200/mo
        "expected_route": "property",
        "expected_section": None,
        "expected_property_ids": None,
        "must_include": ["$2,200"],
    },
    {
        "id": "prop-laundry",
        "question": "Is there in-unit laundry?",
        "property_id": "PROP-001",  # Maple Court — in-unit laundry
        "expected_route": "property",
        "expected_section": None,
        "expected_property_ids": None,
        "must_include": ["laundry"],
    },
    # --- lease (policy retrieval) ----------------------------------------
    {
        "id": "lease-deposit",
        "question": "What is the security deposit?",
        "property_id": "PROP-002",  # Riverside Lofts — $1,100 deposit
        "expected_route": "lease",
        "expected_section": "Security Deposit",
        "expected_property_ids": None,
        "must_include": ["deposit"],
    },
    {
        "id": "lease-sublet",
        "question": "Can I sublet the apartment?",
        "property_id": "PROP-001",
        "expected_route": "lease",
        "expected_section": "Subletting & Assignment",
        "expected_property_ids": None,
        "must_include": ["sublet"],
    },
    {
        "id": "lease-notice",
        "question": "What notice must the landlord give before entering?",
        "property_id": "PROP-001",
        "expected_route": "lease",
        "expected_section": "Landlord Entry",
        "expected_property_ids": None,
        "must_include": ["notice"],
    },
    {
        "id": "lease-pets",
        "question": "What is the pet policy?",
        "property_id": "PROP-002",  # Riverside Lofts — no pets
        "expected_route": "lease",
        "expected_section": "Pets",
        "expected_property_ids": None,
        "must_include": ["pet"],
    },
    {
        "id": "lease-rent-due",
        "question": "When is rent due, and what is the late fee?",
        "property_id": "PROP-001",
        "expected_route": "lease",
        "expected_section": "Rent",
        "expected_property_ids": None,
        "must_include": ["rent"],
    },
    # --- compare (cross-property scan) -----------------------------------
    {
        "id": "compare-pets-under",
        "question": "Which pet-friendly homes are under $1,300?",
        "property_id": None,
        "expected_route": "compare",
        "expected_section": None,
        # PROP-010 ($1,150) and PROP-004 ($1,300) both allow pets and qualify.
        "expected_property_ids": ["PROP-010", "PROP-004"],
        "must_include": ["pets"],
    },
    {
        "id": "compare-cheapest-2bed",
        "question": "What are the cheapest 2-bedroom homes?",
        "property_id": None,
        "expected_route": "compare",
        "expected_section": None,
        # Oakwood Gardens ($1,650) is the cheapest 2-bed.
        "expected_property_ids": ["PROP-003"],
        "must_include": ["Oakwood Gardens"],
    },
    {
        "id": "compare-gym-neighborhood",
        "question": "Which homes in Zilker have a gym?",
        "property_id": None,
        "expected_route": "compare",
        "expected_section": None,
        # Zilker Park View is the only Zilker home with a Gym.
        "expected_property_ids": ["PROP-011"],
        "must_include": ["Zilker Park View"],
    },
    {
        "id": "compare-cheapest",
        "question": "List the most affordable apartments.",
        "property_id": None,
        "expected_route": "compare",
        "expected_section": None,
        # Elm Street Commons ($900) is the cheapest overall.
        "expected_property_ids": ["PROP-041"],
        "must_include": ["$900"],
    },
]


def item_kind(item: dict) -> str:
    """Which retrieval expectation applies to this item."""
    if item.get("expected_property_ids") is not None:
        return "compare"
    if item.get("expected_section") is not None:
        return "lease"
    return "property"
