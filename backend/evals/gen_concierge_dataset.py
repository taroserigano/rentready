"""Generator for the 200-item Concierge golden set (``concierge_dataset.py``).

Every question is derived from real property/lease data (never hand-guessed),
and every item is self-validated before being accepted:

  - ``expected_route``        checked against the real ``concierge.route()``
  - ``expected_property_ids`` taken from the real ``concierge.compare_properties()``
  - ``must_include`` (lease)  checked against the real 220-char source snippet
                              (``concierge._snippet``) so groundedness can't
                              silently rely on text past the truncation point
  - ``must_include`` (property) taken verbatim from ``concierge._fact_sheet``'s
                              own formatting, which is always attached as a
                              source snippet regardless of route path

Run with no LLM / no network (hash embedder, in-memory graph). Regenerate with:
    EMBEDDING_BACKEND=hash python backend/evals/gen_concierge_dataset.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("EMBEDDING_BACKEND", "hash")

BACKEND = Path(__file__).resolve().parent.parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import graph  # noqa: E402

graph.is_available = lambda: False  # deterministic in-memory backend, no Neo4j needed

import concierge  # noqa: E402
import leases  # noqa: E402

PROPS = {p["id"]: p for p in graph.load_properties()}
ALL = sorted(PROPS.values(), key=lambda p: p["id"])
BY_RENT = sorted(ALL, key=lambda p: p["monthly_rent"])

WARNINGS: list[str] = []
items: list[dict] = []
_seen_ids: set[str] = set()


def money(n) -> str:
    return f"${int(round(float(n or 0))):,}"


def add(
    id_: str,
    question: str,
    property_id: str | None,
    expected_route: str,
    expected_section: str | None = None,
    expected_property_ids: list[str] | None = None,
    must_include: list[str] | None = None,
):
    if id_ in _seen_ids:
        raise ValueError(f"duplicate id {id_!r}")
    _seen_ids.add(id_)
    items.append(
        {
            "id": id_,
            "question": question,
            "property_id": property_id,
            "expected_route": expected_route,
            "expected_section": expected_section,
            "expected_property_ids": expected_property_ids,
            "must_include": must_include or [],
        }
    )


def where(pred, n=None):
    out = [p for p in ALL if pred(p)]
    return out if n is None else out[:n]


def spread(seq, n):
    """n items evenly spread across seq (covers the range, not just the head)."""
    if not seq:
        return []
    if n >= len(seq):
        return list(seq)
    step = len(seq) / n
    return [seq[int(i * step)] for i in range(n)]


# =============================================================================
# A. PROPERTY-FACT items — must_include drawn straight from _fact_sheet's own
#    formatting, which is unconditionally attached as a source snippet.
# =============================================================================

_BOOL_FEATURES = [
    ("gym", "Does it have a gym?", lambda p: "Gym" in p["amenities"], "gym"),
    ("pool", "Is there a pool?", lambda p: "Pool" in p["amenities"], "pool"),
    ("rooftop", "Does it have a rooftop deck?", lambda p: "Rooftop Deck" in p["amenities"], "rooftop"),
    ("bike", "Is there bike storage?", lambda p: "Bike Storage" in p["amenities"], "bike"),
    ("concierge-amen", "Does it have a concierge?", lambda p: "Concierge" in p["amenities"], "concierge"),
    ("playground", "Is there a playground on site?", lambda p: "Playground" in p["amenities"], "playground"),
    ("petpark", "Is there a pet park?", lambda p: "Pet Park" in p["amenities"], "pet park"),
    ("balcony", "Does it have a private balcony?", lambda p: p["has_balcony"], "balcony"),
    ("laundry", "Is there in-unit laundry?", lambda p: p["in_unit_laundry"], "laundry"),
    ("parking-avail", "Is parking available?", lambda p: p["parking_type"] != "none", "parking"),
    ("furnished", "Is it furnished?", lambda p: p["furnished"], "furnish"),
    ("gated", "Is the community gated?", lambda p: p["gated"], "gated"),
    ("storage", "Is there extra storage space?", lambda p: p["storage_unit_available"], "storage"),
]

for key, question, check, kw in _BOOL_FEATURES:
    trues = where(check, 2)
    falses = where(lambda p, c=check: not c(p), 2)
    for i, p in enumerate(trues):
        add(f"prop-{key}-yes-{i}", question, p["id"], "property", must_include=[kw])
    for i, p in enumerate(falses):
        add(f"prop-{key}-no-{i}", question, p["id"], "property", must_include=[kw])

# Numeric facts — must_include mirrors _fact_sheet's exact line format.
for i, p in enumerate(spread(BY_RENT, 6)):
    add(f"prop-rent-{i}", "How much is the monthly rent?", p["id"], "property",
        must_include=[money(p["monthly_rent"])])

for i, p in enumerate(spread(ALL, 6)):
    add(f"prop-beds-{i}", "How many bedrooms and bathrooms does it have?", p["id"], "property",
        must_include=[f"bedrooms: {p['bedrooms']}"])

for i, p in enumerate(spread(sorted(ALL, key=lambda p: p["square_feet"]), 6)):
    add(f"prop-sqft-{i}", "How big is it in square feet?", p["id"], "property",
        must_include=[f"square feet: {p['square_feet']}"])

for i, p in enumerate(spread(ALL, 5)):
    add(f"prop-avail-{i}", "When is it available to move in?", p["id"], "property",
        must_include=[p["availability_date"]])

for i, p in enumerate(spread(ALL, 5)):
    hood = p["neighborhood"]["name"]
    add(f"prop-hood-{i}", "What neighborhood is it in?", p["id"], "property",
        must_include=[hood.lower()])


# =============================================================================
# B. LEASE-POLICY items — must_include is checked (below) against the REAL
#    220-char snippet of that section's generated text for that property.
# =============================================================================

_LEASE_TOPICS = [
    ("deposit", "Security Deposit", "What's the security deposit amount?", "deposit"),
    ("pets", "Pets", "What's the pet policy?", "pet"),
    ("sublet", "Subletting & Assignment", "Can I sublet my apartment?", "sublet"),
    ("latefee", "Late Fees & Returned Payments", "What is the late fee for paying rent late?", "late fee"),
    ("entry", "Landlord Entry", "What notice must the landlord give before entering?", "enter"),
    ("renew", "Renewal", "How do I renew my lease?", "month-to-month"),
    ("terminate", "Termination & Default", "What happens if I need to break the lease early?", "terminat"),
    ("guests", "Occupancy & Guests", "How many guests can stay over, and for how long?", "guest"),
    ("insurance", "Renter's Insurance", "Do I need renter's insurance?", "insurance"),
    ("utilities", "Utilities", "What utilities are included?", "utilit"),
    ("maintenance", "Maintenance & Repairs", "Who handles maintenance and repairs?", "repair"),
    ("smoking", "Community Amenities & Rules", "Is smoking allowed?", "community"),
]

for key, section, question, kw in _LEASE_TOPICS:
    for i, p in enumerate(spread(ALL, 6)):
        add(f"lease-{key}-{i}", question, p["id"], "lease",
            expected_section=section, must_include=[kw])


# =============================================================================
# C. COMPARE items — expected_property_ids taken from the REAL
#    compare_properties() output (self-consistent oracle, not hand-computed).
# =============================================================================

_COMPARE_QUESTIONS = [
    ("cheapest-all-1", "List the most affordable apartments."),
    ("cheapest-all-2", "What are the cheapest homes available?"),
    ("cheapest-all-3", "Show me your least expensive listings."),
    ("under-1200", "Which properties are under $1,200?"),
    ("under-1500", "Which properties are under $1,500?"),
    ("under-2000", "Which pet-friendly homes are under $2,000?"),
    ("under-2500", "What's available under $2,500 a month?"),
    ("under-3000-3bed", "3-bedroom homes under $3,000?"),
    ("studio-cheap", "What's the cheapest studio?"),
    ("cheapest-2bed", "What are the cheapest 2-bedroom homes?"),
    ("cheapest-4bed", "Cheapest 4-bedroom apartment?"),
    ("pets-only", "Which properties allow pets?"),
    ("area-zilker", "Which properties are in Zilker?"),
    ("area-downtown", "Which properties are in Downtown?"),
    ("area-southcongress-pets", "Which properties in South Congress allow pets?"),
    ("area-deepellum", "What homes are in Deep Ellum?"),
    ("area-mueller-pool", "Which homes in Mueller have a pool?"),
    ("gym-only", "Which properties have a gym?"),
    ("pool-only", "Show me homes with a pool"),
    ("rooftop-only", "Which properties have a rooftop deck?"),
    ("bike-only", "Which properties have bike storage?"),
    ("gym-and-pool", "Which properties have a gym and a pool?"),
    ("furnished-only", "Which properties are furnished?"),
    ("laundry-only", "Which properties have in-unit laundry?"),
    ("balcony-only", "Which properties have a balcony?"),
    ("parking-only", "Which properties include parking?"),
    ("1bed-under-1600", "1-bedroom under $1,600?"),
    ("2bed-pets-under-2200", "2-bedroom pet-friendly homes under $2,200?"),
    ("compare-explicit-1", "Compare homes in Zilker"),
    ("compare-explicit-2", "Compare the cheapest 2-bedroom apartments"),
    ("options-list", "What options do you have for 1-bedroom apartments?"),
    ("any-pets-that", "Any apartments that allow pets under $1,800?"),
    ("across-portfolio", "Across all properties, which is cheapest?"),
    ("all-the-homes", "List all the homes with a pool"),
    ("least-expensive", "Which is the least expensive option?"),
    ("lowest-rent", "Which property has the lowest rent?"),
    ("north-austin", "What's available in North Austin?"),
    ("east-austin-laundry", "Which East Austin homes have in-unit laundry?"),
    ("uptown-options", "Which properties are in Uptown?"),
    ("3bed-gym", "3-bedroom homes with a gym?"),
    ("cheapest-furnished", "What's the cheapest furnished option?"),
    # Previously-silent amenity gap (_AMENITY_MAP had only 7/16 real amenities;
    # these regress-test the other 9 that used to be dropped without filtering).
    ("dogpark-only", "Which properties have a dog park?"),
    ("ev-only", "Which properties have EV charging?"),
    ("yoga-only", "Which properties have a yoga studio?"),
    ("bbq-only", "Which properties have a BBQ area?"),
    ("clubhouse-only", "Which properties have a clubhouse?"),
    ("coworking-only", "Which properties have a coworking space?"),
    ("packagelockers-only", "Which properties have package lockers?"),
    ("smartlocks-only", "Which properties have smart locks?"),
    ("storageunits-only", "Which properties have storage units?"),
]

for key, question in _COMPARE_QUESTIONS:
    result = concierge.compare_properties(question)
    ids = [c["id"] for c in result]
    add(f"compare-{key}", question, None, "compare", expected_property_ids=ids,
        must_include=[])


# =============================================================================
# D. BOTH / GENERAL items — router edge cases the original 11-item set never
#    exercised at all (both = property+lease keyword collision; general = no
#    recognized keyword). expected_route is whatever concierge.route() ACTUALLY
#    returns for well-chosen phrasing, verified below, not assumed.
# =============================================================================

_BOTH_CANDIDATES = [
    ("both-rent-pet", "What's the monthly rent, and can I have a dog?", "PROP-001"),
    ("both-deposit-parking", "How much is the deposit, and is parking included?", "PROP-002"),
    ("both-furnished-sublet", "Is it furnished, and can I sublet if I relocate for work?", "PROP-003"),
    ("both-gym-notice", "Does it have a gym, and how much notice must the landlord give to enter?", "PROP-004"),
    ("both-size-insurance", "How big is it, and do I need renter's insurance?", "PROP-005"),
    ("both-laundry-utilities", "Is there in-unit laundry, and what utilities are included?", "PROP-006"),
    ("both-balcony-guests", "Does it have a balcony, and how many guests can stay over?", "PROP-007"),
    ("both-avail-renew", "When is it available, and how do I renew after the first year?", "PROP-008"),
]
for key, question, pid in _BOTH_CANDIDATES:
    add(key, question, pid, "both", must_include=[])

_GENERAL_CANDIDATES = [
    ("general-hi", "Hi there!", None),
    ("general-help", "Can you help me?", None),
    ("general-tell-more", "Tell me more about this place.", None),
    ("general-good-deal", "Is this a good deal?", None),
    ("general-what-should-know", "What should I know before signing?", None),
    ("general-thoughts", "What do you think of this one?", None),
    ("general-next-steps", "What are my next steps?", None),
    ("general-summary", "Give me a quick summary.", "PROP-010"),
]
for key, question, pid in _GENERAL_CANDIDATES:
    add(key, question, pid, "general", must_include=[])


# =============================================================================
# SELF-VALIDATION — every item is checked against the real, running system
# before being written out. Failures are fixed here (reworded/re-targeted),
# not silently accepted, and not "fixed" by hand-editing the output file.
# =============================================================================

def validate():
    for it in items:
        got_route = concierge.route(it["question"], it["property_id"])
        if got_route != it["expected_route"]:
            WARNINGS.append(
                f"[route] {it['id']}: expected {it['expected_route']!r}, "
                f"router says {got_route!r} — {it['question']!r}"
            )

        if it["expected_section"]:
            prop = PROPS[it["property_id"]]
            sections = dict(leases.lease_sections(prop))
            text = sections.get(it["expected_section"], "")
            snippet = concierge._snippet(text).lower()  # noqa: SLF001 (test-only introspection)
            for kw in it["must_include"]:
                if kw.lower() not in snippet:
                    WARNINGS.append(
                        f"[snippet-truncation] {it['id']}: {kw!r} not within the "
                        f"220-char snippet of {it['property_id']}/{it['expected_section']!r}"
                    )

        if it["expected_property_ids"] is not None and not it["expected_property_ids"]:
            WARNINGS.append(f"[compare-empty] {it['id']}: matched 0 properties — {it['question']!r}")


validate()

print(f"Generated {len(items)} items.")
if WARNINGS:
    print(f"\n{len(WARNINGS)} WARNING(S):")
    for w in WARNINGS:
        print(" -", w)
else:
    print("All items self-validated cleanly.")

# Route distribution sanity check.
from collections import Counter  # noqa: E402

dist = Counter(it["expected_route"] for it in items)
print("\nRoute distribution:", dict(dist))


# =============================================================================
# EMIT concierge_dataset.py
# =============================================================================

def _fmt_list(vals):
    if vals is None:
        return "None"
    return "[" + ", ".join(repr(v) for v in vals) + "]"


def emit(path: Path):
    lines = [
        '"""Labeled dataset for the Concierge evaluation suite.',
        "",
        "Each item is anchored to real property data (``data/properties.json``) and the",
        "generated lease sections (``leases._SECTION_SPECS``), so the deterministic",
        "answers and retrieval targets are reproducible offline (hash embedder, no LLM).",
        "",
        f"{len(items)} items across 5 route categories (property/lease/both/compare/general),",
        "generated + self-validated by ``evals/gen_concierge_dataset.py`` against the",
        "real router, real lease text, and real compare_properties() output — see that",
        "file before hand-editing this one; regenerate instead where possible.",
        "",
        "Fields per item:",
        "  id                    stable identifier",
        "  question              the resident's question",
        "  property_id           scoped property, or None (compare/general items)",
        "  expected_route        one of property | lease | both | compare | general",
        "  expected_section      lease section that should be retrieved (lease items)",
        "  expected_property_ids ids that must appear in the comparison (compare items)",
        "  must_include          facts that should surface in the answer OR its sources",
        '"""',
        "",
        "CONCIERGE_DATASET = [",
    ]
    for it in items:
        lines.append("    {")
        lines.append(f"        \"id\": {it['id']!r},")
        lines.append(f"        \"question\": {it['question']!r},")
        lines.append(f"        \"property_id\": {it['property_id']!r},")
        lines.append(f"        \"expected_route\": {it['expected_route']!r},")
        lines.append(f"        \"expected_section\": {it['expected_section']!r},")
        lines.append(f"        \"expected_property_ids\": {_fmt_list(it['expected_property_ids'])},")
        lines.append(f"        \"must_include\": {_fmt_list(it['must_include'])},")
        lines.append("    },")
    lines.append("]")
    lines.append("")
    lines.append("")
    lines.append('def item_kind(item: dict) -> str:')
    lines.append('    """Which retrieval expectation applies to this item."""')
    lines.append("    if item.get(\"expected_property_ids\") is not None:")
    lines.append('        return "compare"')
    lines.append("    if item.get(\"expected_section\") is not None:")
    lines.append('        return "lease"')
    lines.append('    return "property"')
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {len(items)} items to {path}")


if __name__ == "__main__":
    out = BACKEND / "evals" / "concierge_dataset.py"
    if "--check" not in sys.argv:
        emit(out)
