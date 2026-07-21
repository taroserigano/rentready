"""Property & Lease Concierge — the agent.

``answer(question, property_id=None, history=None) -> dict`` runs a
retrieve-then-read agentic flow:

  1. ROUTER   — a deterministic keyword/heuristic classifier picks which
                tool(s) to use: property_facts | lease | both | general.
  2. TOOLS    — property_facts(property_id) returns EXACT structured fields;
                knowledge.search() returns hybrid-retrieved lease passages.
  3. SYNTHESIZE — Claude (via llm.get_langchain_llm) answers using ONLY the
                assembled context with inline [n] citations (source="anthropic").
                On None/any error we build a templated answer from the
                structured facts and/or the top lease passage (source="rules").

Mirrors ``graphrag.graph_ask`` / ``tours_chat.handle``: NEVER raises — the
outermost guard always returns a safe ConciergeAnswer dict.
"""

from __future__ import annotations

import re

from llm import get_langchain_llm
import graph
import knowledge

# ---------------------------------------------------------------------------
# ROUTER — intent classification
# ---------------------------------------------------------------------------
# Property intent: structured facts (amenities, rent, size, availability...).
_PROPERTY_RE = re.compile(
    r"\b("
    r"price|cost|how much|monthly rent|what'?s the rent|what is the rent|rental rate|"
    r"bedroom|bed\b|bathroom|bath\b|square feet|square footage|sq ?ft|size|how big|"
    r"amenit|gym|fitness|pool|swim|rooftop|deck|bike|concierge|playground|"
    r"balcony|laundry|washer|dryer|furnished|gated|storage|"
    r"walk score|transit|neighborhood|located|location|address|"
    r"available|availability|move[- ]?in|year built|floor|"
    r"does it have|do they have|is there a|are there|what amenities"
    r")\b",
    re.IGNORECASE,
)

# Lease intent: policies, obligations, "can I ...", "what happens if ...".
_LEASE_RE = re.compile(
    r"\b("
    r"deposit|sublet|sublease|assign|"
    r"notice|terminat|break the lease|break my lease|end the lease|move out early|"
    r"late fee|late payment|returned payment|nsf|bounced|"
    r"guests?|visitor|occupan|"
    r"can i|am i allowed|allowed to|what happens if|do i have to|how do i|"
    r"pet policy|pets? allowed|pet deposit|pet rent|"
    r"insurance|liability|"
    r"renew|renewal|month[- ]to[- ]month|"
    r"landlord enter|landlord entry|enter the|entry|"
    r"utilit\w*|maintenance|repair|"
    r"lease term|how long is the lease|"
    r"rule|alteration|paint|change the lock|smoking|"
    r"evict|default|policy"
    r")\b",
    re.IGNORECASE,
)


# Comparative / list intent: the user wants to scan ACROSS properties.
_COMPARE_INTENT_RE = re.compile(
    r"("
    r"which\s+propert|which\s+(home|place|apartment|unit|rental|listing|building|complex)|"
    r"\bcompare\b|cheapest|least expensive|most affordable|lowest rent|"
    r"\blist\b|show me|\boptions\b|\bcompar|"
    r"any\s+(homes?|places?|apartments?|units?|rentals?|listings?|properties)\s+that|"
    r"what\s+(properties|homes|apartments|places|rentals|units)|"
    r"across (all )?propert|all (the )?(homes|properties|apartments)"
    r")",
    re.IGNORECASE,
)

# "Singular" phrasing — the user is clearly asking about ONE (scoped) home.
# These veto the filter-phrase compare trigger so scoped Q&A still routes to
# property/lease even with no property_id in a unit test.
_SINGULAR_RE = re.compile(
    r"\b(does it|is it|is there|are there|do they|does this|is this|it has|"
    r"has it|this (apartment|unit|home|place|property))\b",
    re.IGNORECASE,
)

# A price ceiling: "under/below/less than/at most/no more than/up to $N" (or bare N).
_MAX_RENT_RE = re.compile(
    r"(?:under|below|less than|at most|no more than|up to|cheaper than|max(?:imum)?|within)\s*"
    r"\$?\s*(\d[\d,]*)\s*(k)?",
    re.IGNORECASE,
)
# Bedroom count: "2-bed", "2 bedroom", "2 br", "studio" (=0).
_BEDS_RE = re.compile(r"\b(\d+)\s*(?:-|\s)?\s*(?:bed(?:room)?s?|br)\b", re.IGNORECASE)
_STUDIO_RE = re.compile(r"\bstudio\b", re.IGNORECASE)
_PETS_RE = re.compile(
    r"pet[\s-]?friendly|allows?\s+pets|with\s+pets|pets?\s+allowed|dog[\s-]?friendly",
    re.IGNORECASE,
)

# Amenity keyword -> canonical amenity label in the data.
_AMENITY_MAP = [
    (re.compile(r"\b(gym|fitness)\b", re.I), "Gym", "has a gym"),
    (re.compile(r"\b(pool|swim)\b", re.I), "Pool", "has a pool"),
    (re.compile(r"\b(rooftop|roof deck|deck)\b", re.I), "Rooftop Deck", "has a rooftop deck"),
    (re.compile(r"\bconcierge\b", re.I), "Concierge", "has a concierge"),
    (re.compile(r"\bbike\b", re.I), "Bike Storage", "has bike storage"),
    (re.compile(r"\bplayground\b", re.I), "Playground", "has a playground"),
    (re.compile(r"\bpet park\b", re.I), "Pet Park", "has a pet park"),
]

# Neighborhoods mentioned in the contract, plus any others discovered at runtime.
_KNOWN_AREAS = [
    "Downtown", "North Austin", "South Congress", "Zilker", "Mueller", "East Austin",
]


def _all_areas() -> list[str]:
    """Every neighborhood name in the data (deduped, longest first so
    multi-word names like 'North Austin' match before 'Austin')."""
    seen: dict[str, None] = {}
    for name in _KNOWN_AREAS:
        seen.setdefault(name, None)
    try:
        for p in graph.load_properties():
            nm = (p.get("neighborhood") or {}).get("name")
            if nm:
                seen.setdefault(nm, None)
    except Exception:  # noqa: BLE001
        pass
    return sorted(seen, key=len, reverse=True)


def _area_in(question: str) -> str | None:
    q = question or ""
    for area in _all_areas():
        if re.search(r"\b" + re.escape(area) + r"\b", q, re.IGNORECASE):
            return area
    return None


def _has_filter_phrase(question: str) -> bool:
    """A concrete search filter (price / beds / pets / area) is present. A bare
    amenity mention alone is intentionally NOT enough (so scoped "does it have a
    gym?" stays a property question)."""
    q = question or ""
    if _MAX_RENT_RE.search(q):
        return True
    if _BEDS_RE.search(q) or _STUDIO_RE.search(q):
        return True
    if _PETS_RE.search(q):
        return True
    if _area_in(q):
        return True
    return False


def route(question: str, property_id: str | None = None) -> str:
    """Classify the question into which tool(s) to use.

    ``compare`` (cross-property scan) takes precedence when the question has
    explicit comparative/list intent, or when it carries a search filter and no
    single property is scoped.
    """
    q = question or ""
    if _COMPARE_INTENT_RE.search(q):
        return "compare"
    if (
        not property_id
        and _has_filter_phrase(q)
        and not _SINGULAR_RE.search(q)
    ):
        return "compare"

    is_property = bool(_PROPERTY_RE.search(q))
    is_lease = bool(_LEASE_RE.search(q))
    if is_property and is_lease:
        return "both"
    if is_lease:
        return "lease"
    if is_property:
        return "property"
    return "general"


# ---------------------------------------------------------------------------
# TOOLS
# ---------------------------------------------------------------------------
def property_facts(property_id: str | None) -> dict | None:
    """Exact structured fields for a property, or None if unknown/unspecified."""
    if not property_id:
        return None
    for p in graph.load_properties():
        if p.get("id") == property_id:
            return p
    return None


def _money(amount) -> str:
    try:
        v = float(amount or 0)
    except (TypeError, ValueError):
        v = 0.0
    return f"${int(v):,}" if v == int(v) else f"${v:,.2f}"


def _fact_sheet(p: dict) -> str:
    """A compact, LLM-friendly fact sheet drawn straight from the data."""
    n = p.get("neighborhood") or {}
    amenities = ", ".join(p.get("amenities") or []) or "none listed"
    pets = "allowed" if p.get("pets_allowed") else "not allowed (assistance animals excepted)"
    parking = p.get("parking_type") or "none"
    lines = [
        f"Name: {p.get('name','')} (ID {p.get('id','')})",
        f"Type: {p.get('property_type','')}",
        f"Monthly rent: {_money(p.get('monthly_rent'))}",
        f"Bedrooms: {p.get('bedrooms')}; Bathrooms: {p.get('bathrooms')} ({p.get('bathroom_type','')})",
        f"Square feet: {p.get('square_feet')}",
        f"Amenities: {amenities}",
        f"Private balcony: {'yes' if p.get('has_balcony') else 'no'}",
        f"In-unit laundry: {'yes' if p.get('in_unit_laundry') else 'no'}",
        f"Parking: {parking}",
        f"Furnished: {'yes' if p.get('furnished') else 'no'}",
        f"Pets: {pets}",
        f"Lease term: {p.get('lease_term_months')} months",
        f"Security deposit: {_money(p.get('security_deposit'))}",
        f"Neighborhood: {n.get('name','')}, {n.get('city','')} "
        f"(walk score {n.get('walk_score')}, transit score {n.get('transit_score')})",
        f"Availability: {p.get('availability_date','')}",
    ]
    return "\n".join(lines)


# Feature detectors used by the deterministic property answer.
_FEATURE_CHECKS = [
    (re.compile(r"\b(gym|fitness)\b", re.I), lambda p: "Gym" in (p.get("amenities") or []), "a gym"),
    (re.compile(r"\b(pool|swim)\b", re.I), lambda p: "Pool" in (p.get("amenities") or []), "a pool"),
    (re.compile(r"\b(rooftop|deck)\b", re.I), lambda p: "Rooftop Deck" in (p.get("amenities") or []), "a rooftop deck"),
    (re.compile(r"\bbike\b", re.I), lambda p: "Bike Storage" in (p.get("amenities") or []), "bike storage"),
    (re.compile(r"\bconcierge\b", re.I), lambda p: "Concierge" in (p.get("amenities") or []), "a concierge"),
    (re.compile(r"\bplayground\b", re.I), lambda p: "Playground" in (p.get("amenities") or []), "a playground"),
    (re.compile(r"\bpet park\b", re.I), lambda p: "Pet Park" in (p.get("amenities") or []), "a pet park"),
    (re.compile(r"\bbalcon", re.I), lambda p: bool(p.get("has_balcony")), "a private balcony"),
    (re.compile(r"\b(laundry|washer|dryer)\b", re.I), lambda p: bool(p.get("in_unit_laundry")), "in-unit laundry"),
    (re.compile(r"\b(parking|garage)\b", re.I), lambda p: (p.get("parking_type") or "none") != "none", "parking"),
    (re.compile(r"\bfurnished\b", re.I), lambda p: bool(p.get("furnished")), "furnishings"),
    (re.compile(r"\bgated\b", re.I), lambda p: bool(p.get("gated")), "gated access"),
    (re.compile(r"\bstorage\b", re.I), lambda p: bool(p.get("storage_unit_available")), "storage units"),
]


def _deterministic_property_answer(p: dict, question: str) -> str:
    """Answer a property question from exact fields, no LLM."""
    name = p.get("name", "This property")
    q = question or ""

    # "Does it have X?" style — yes/no on a concrete feature.
    if re.search(r"\b(does|do|is there|are there|has|have)\b", q, re.I):
        for rx, check, phrase in _FEATURE_CHECKS:
            if rx.search(q):
                if check(p):
                    return f"Yes — {name} includes {phrase}."
                return f"No — {name} does not list {phrase}."

    # Specific numeric fields.
    if re.search(r"\b(rent|price|cost|how much|monthly)\b", q, re.I):
        return f"{name} rents for {_money(p.get('monthly_rent'))} per month."
    if re.search(r"\b(bedroom|beds?)\b", q, re.I):
        return f"{name} has {p.get('bedrooms')} bedroom(s) and {p.get('bathrooms')} bathroom(s)."
    if re.search(r"\b(square feet|sq ?ft|size|how big)\b", q, re.I):
        return f"{name} is approximately {p.get('square_feet'):,} square feet."
    if re.search(r"\bpets?\b", q, re.I):
        allowed = p.get("pets_allowed")
        return (
            f"Pets are {'allowed' if allowed else 'not allowed'} at {name}"
            + ("." if allowed else " (assistance animals excepted).")
        )
    if re.search(r"\bavailab|move[- ]?in\b", q, re.I):
        return f"{name} is available from {p.get('availability_date','the posted date')}."

    # Generic fact summary.
    n = p.get("neighborhood") or {}
    return (
        f"{name} is a {p.get('bedrooms')}-bed / {p.get('bathrooms')}-bath "
        f"{str(p.get('property_type','residence')).lower()} of about "
        f"{p.get('square_feet'):,} sq ft in {n.get('name','')}, renting for "
        f"{_money(p.get('monthly_rent'))} per month."
    )


# ---------------------------------------------------------------------------
# CROSS-PROPERTY COMPARISON
# ---------------------------------------------------------------------------
def _parse_max_rent(question: str):
    """The lowest price ceiling mentioned, or None. Handles '$2,000', '2000',
    '2k'."""
    best = None
    for m in _MAX_RENT_RE.finditer(question or ""):
        raw = m.group(1).replace(",", "")
        try:
            val = float(raw)
        except ValueError:
            continue
        if m.group(2):  # a trailing 'k'
            val *= 1000
        best = val if best is None else min(best, val)
    return best


def _parse_min_bedrooms(question: str):
    q = question or ""
    m = _BEDS_RE.search(q)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            pass
    if _STUDIO_RE.search(q):
        return 0
    return None


def _compare_item(p: dict, matched: list[str]) -> dict:
    n = p.get("neighborhood") or {}
    return {
        "id": p.get("id", ""),
        "name": p.get("name", ""),
        "area": n.get("name", ""),
        "property_type": p.get("property_type", ""),
        "monthly_rent": p.get("monthly_rent"),
        "bedrooms": p.get("bedrooms"),
        "bathrooms": p.get("bathrooms"),
        "square_feet": p.get("square_feet"),
        "pets_allowed": bool(p.get("pets_allowed")),
        "matched": matched,
    }


def compare_properties(question: str) -> list[dict]:
    """Scan every property, keep the ones matching the filters parsed from the
    question (AND semantics), and return CompareItem dicts sorted by rent asc,
    capped at 6. With no parseable filter, returns the 6 cheapest."""
    q = question or ""

    max_rent = _parse_max_rent(q)
    min_beds = _parse_min_bedrooms(q)
    area = _area_in(q)
    want_pets = bool(_PETS_RE.search(q))
    amenities_wanted = [(label, phrase) for rx, label, phrase in _AMENITY_MAP if rx.search(q)]
    want_balcony = bool(re.search(r"\bbalcon", q, re.I))
    want_laundry = bool(re.search(r"\b(in[\s-]?unit laundry|laundry|washer|dryer)\b", q, re.I))
    want_furnished = bool(re.search(r"\bfurnished\b", q, re.I))
    want_parking = bool(re.search(r"\b(parking|garage)\b", q, re.I))

    any_filter = any([
        max_rent is not None, min_beds is not None, area, want_pets,
        amenities_wanted, want_balcony, want_laundry, want_furnished, want_parking,
    ])

    results: list[dict] = []
    for p in graph.load_properties():
        matched: list[str] = []
        ok = True

        if max_rent is not None:
            if (p.get("monthly_rent") or 0) <= max_rent:
                matched.append(f"under {_money(max_rent)}")
            else:
                ok = False
        if ok and min_beds is not None:
            if (p.get("bedrooms") or 0) >= min_beds:
                matched.append("studio" if min_beds == 0 else f"{min_beds}+ bedrooms")
            else:
                ok = False
        if ok and area:
            if (p.get("neighborhood") or {}).get("name") == area:
                matched.append(f"in {area}")
            else:
                ok = False
        if ok and want_pets:
            if p.get("pets_allowed"):
                matched.append("allows pets")
            else:
                ok = False
        if ok and amenities_wanted:
            have = p.get("amenities") or []
            for label, phrase in amenities_wanted:
                if label in have:
                    matched.append(phrase)
                else:
                    ok = False
                    break
        if ok and want_balcony:
            if p.get("has_balcony"):
                matched.append("has a balcony")
            else:
                ok = False
        if ok and want_laundry:
            if p.get("in_unit_laundry"):
                matched.append("in-unit laundry")
            else:
                ok = False
        if ok and want_furnished:
            if p.get("furnished"):
                matched.append("furnished")
            else:
                ok = False
        if ok and want_parking:
            if (p.get("parking_type") or "none") != "none":
                matched.append("has parking")
            else:
                ok = False

        if ok:
            # With a filter, keep only homes that matched something; with no
            # filter, keep everything and take the cheapest below.
            if any_filter and not matched:
                continue
            results.append(_compare_item(p, matched))

    results.sort(key=lambda c: (c["monthly_rent"] is None, c["monthly_rent"] or 0))
    return results[:6]


def _compare_context(comparison: list[dict]):
    """Numbered context blocks + Source dicts for a comparison — one per home,
    so the LLM stays grounded and citations line up."""
    context_blocks: list[str] = []
    sources: list[dict] = []
    for c in comparison:
        pets = "allowed" if c["pets_allowed"] else "not allowed"
        line = (
            f"{c['name']} (ID {c['id']}): {_money(c['monthly_rent'])}/mo, "
            f"{c['bedrooms']}-bed / {c['bathrooms']}-bath {str(c['property_type']).lower()}, "
            f"{c['square_feet']} sq ft, {c['area']}; pets {pets}."
        )
        context_blocks.append(f"Property record — {c['name']}:\n{line}")
        sources.append(
            {
                "type": "property",
                "label": f"Property record — {c['name']}",
                "snippet": _snippet(line),
                "section": "",
                "property_id": c["id"],
            }
        )
    return context_blocks, sources


_COMPARE_FOLLOWUPS = [
    "Which of these allow pets?",
    "Show the most affordable option",
    "Which is closest downtown?",
    "Which property has the most square footage?",
    "Which properties have a gym or a pool?",
    "Which properties have the shortest lease terms?",
]


def _compare_follow_ups(question: str) -> list[str]:
    q = (question or "").lower()

    def too_similar(cand: str) -> bool:
        for kw in ("pet", "affordable", "cheap", "downtown", "square", "gym",
                   "pool", "lease term", "term"):
            if kw in cand.lower() and kw in q:
                return True
        return False

    out = [c for c in _COMPARE_FOLLOWUPS if not too_similar(c)]
    return out[:3]


def _deterministic_compare_answer(comparison: list[dict]) -> str:
    if not comparison:
        return (
            "I couldn't find any homes matching that — try widening the budget "
            "or area."
        )
    parts = []
    for i, c in enumerate(comparison):
        beds = "studio" if c["bedrooms"] == 0 else f"{c['bedrooms']}bd"
        parts.append(
            f"{c['name']} — {_money(c['monthly_rent'])}, {beds}, {c['area']} [{i+1}]"
        )
    n = len(comparison)
    lead = f"Here {'is' if n == 1 else 'are'} {n} home{'' if n == 1 else 's'} that match: "
    return lead + "; ".join(parts) + "."


# ---------------------------------------------------------------------------
# CONTEXT ASSEMBLY + SOURCES
# ---------------------------------------------------------------------------
def _snippet(text: str, n: int = 220) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def _assemble(question: str, route_: str, prop: dict | None, property_id: str | None):
    """Return (context_blocks, sources) as parallel lists.

    context_blocks[i] is the text for citation [i+1]; sources[i] is the
    matching Source dict (identical numbering front/back).
    """
    context_blocks: list[str] = []
    sources: list[dict] = []

    want_property = route_ in ("property", "both", "general")
    want_lease = route_ in ("lease", "both", "general")

    if want_property and prop is not None:
        sheet = _fact_sheet(prop)
        context_blocks.append(f"Property record — {prop.get('name','')}:\n{sheet}")
        sources.append(
            {
                "type": "property",
                "label": f"Property record — {prop.get('name','')}",
                "snippet": _snippet(sheet.replace("\n", " · ")),
                "section": "",
                "property_id": prop.get("id", ""),
            }
        )

    if want_lease:
        passages = knowledge.search(question, property_id=property_id, k=4)
        for psg in passages:
            label = f"Lease · {psg.get('section','')}"
            if psg.get("property_name"):
                label += f" — {psg['property_name']}"
            context_blocks.append(f"{label}:\n{psg.get('text','')}")
            sources.append(
                {
                    "type": "lease",
                    "label": label,
                    "snippet": _snippet(psg.get("text", "")),
                    # Deep-link fields: the UI opens the full lease at this
                    # section for this property when a citation is clicked.
                    "section": psg.get("section", ""),
                    "property_id": psg.get("property_id", "") or (property_id or ""),
                }
            )

    return context_blocks, sources


# ---------------------------------------------------------------------------
# FOLLOW-UP SUGGESTIONS (deterministic — work with the LLM offline)
# ---------------------------------------------------------------------------
_FOLLOWUPS_LEASE = [
    "How much is the security deposit?",
    "Is subletting allowed?",
    "When is rent due and what's the late fee?",
    "What's the pet policy?",
    "How do I renew or end the lease?",
    "How much notice must the landlord give to enter?",
    "Am I required to carry renter's insurance?",
    "What utilities are included?",
    "How many guests can stay over, and for how long?",
    "What happens if I need to break the lease early?",
    "Is smoking allowed on the property?",
]
_FOLLOWUPS_PROPERTY = [
    "Does it have a gym?",
    "Is parking included?",
    "Is there in-unit laundry?",
    "When is it available to move in?",
    "How big is it?",
    "Is there a pool?",
    "What neighborhood is it in?",
    "Is it furnished?",
    "Is there extra storage space?",
]


def _follow_ups(route_: str, prop: dict | None, question: str) -> list[str]:
    """Three relevant next questions, skipping anything too close to what was
    just asked. Pure/deterministic so the chips work even with the LLM down."""
    q = (question or "").lower()
    pool: list[str] = []
    if route_ in ("lease", "both", "general"):
        pool += _FOLLOWUPS_LEASE
    if route_ in ("property", "both", "general"):
        pool += _FOLLOWUPS_PROPERTY
    if not pool:
        pool = _FOLLOWUPS_PROPERTY + _FOLLOWUPS_LEASE

    def too_similar(cand: str) -> bool:
        # crude overlap: skip a suggestion sharing its key noun with the question
        for kw in ("deposit", "sublet", "rent", "pet", "renew", "gym",
                   "parking", "laundry", "available", "size", "notice", "enter",
                   "insurance", "utilit", "guest", "break the lease", "smoking",
                   "pool", "neighborhood", "furnished", "storage"):
            if kw in cand.lower() and kw in q:
                return True
        return False

    out: list[str] = []
    for cand in pool:
        if cand not in out and not too_similar(cand):
            out.append(cand)
        if len(out) == 3:
            break
    return out


# ---------------------------------------------------------------------------
# SYNTHESIS
# ---------------------------------------------------------------------------
_SYSTEM = (
    "You are a Property & Lease Concierge. Answer the resident's question using "
    "ONLY the numbered context provided — never invent facts, amenities, prices, "
    "or policies. Cite the sources you use inline with bracketed numbers like "
    "[1] or [2] that match the context blocks. Be concise, warm, and specific. "
    "If the context does not contain the answer, say you don't have that in the "
    "documents and suggest contacting the leasing office."
)

_COMPARE_SYSTEM = (
    "You are a Property & Lease Concierge helping a resident compare homes. Each "
    "numbered context block is a real property record. Summarize the matching "
    "homes in 1–3 sentences using ONLY those records — never invent homes, "
    "prices, or amenities. Cite each home you mention with its bracketed number "
    "like [1]. Lead with the most affordable option. If the context is empty, "
    "say you couldn't find a match and suggest widening the budget or area."
)


def _system_for(route_: str) -> str:
    return _COMPARE_SYSTEM if route_ == "compare" else _SYSTEM


def _coalesce(content) -> str:
    """LangChain chunk/message ``.content`` can be a str or a list of blocks."""
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, str):
                parts.append(b)
            elif isinstance(b, dict):
                parts.append(str(b.get("text", "")))
            else:
                parts.append(str(b))
        return "".join(parts)
    return str(content or "")


def _build_messages(question, context_blocks, history, system):
    numbered = "\n\n".join(
        f"[{i+1}] {block}" for i, block in enumerate(context_blocks)
    )
    messages = [("system", system)]
    for turn in (history or [])[-6:]:
        role = turn.get("role") if isinstance(turn, dict) else None
        content = turn.get("content") if isinstance(turn, dict) else None
        if role in ("user", "human") and content:
            messages.append(("human", content))
        elif role in ("assistant", "ai", "bot") and content:
            messages.append(("ai", content))
    messages.append(
        (
            "human",
            f"Context:\n{numbered}\n\nQuestion: {question}\n\n"
            f"Answer using only the context above, with [n] citations.",
        )
    )
    return messages


def _llm_answer(question, context_blocks, history, system=_SYSTEM) -> str | None:
    llm = get_langchain_llm()
    if llm is None:
        return None
    try:
        messages = _build_messages(question, context_blocks, history, system)
        raw = llm.invoke(messages).content
        text = _coalesce(raw).strip()
        return text or None
    except Exception as exc:  # noqa: BLE001
        print(f"concierge: LLM synthesis failed ({type(exc).__name__}); templated.")
        return None


def _deterministic_answer(question, route_, prop, sources, comparison=None) -> str:
    """Templated grounded answer from structured facts and/or top lease passage."""
    if route_ == "compare":
        return _deterministic_compare_answer(comparison or [])

    parts: list[str] = []
    n_sources = len(sources)

    lease_sources = [s for s in sources if s["type"] == "lease"]
    has_property = any(s["type"] == "property" for s in sources)

    if route_ in ("property", "both", "general") and prop is not None:
        cite = "[1]" if has_property else ""
        parts.append(_deterministic_property_answer(prop, question) + (f" {cite}".rstrip()))

    if route_ in ("lease", "both", "general") and lease_sources:
        # Cite up to the two most relevant lease passages.
        base_num = 2 if has_property else 1
        offset = 1 if has_property else 0
        top = lease_sources[:2]
        chunks = []
        for i, s in enumerate(top):
            num = offset + i + 1
            chunks.append(f"{s['snippet']} [{num}]")
        lead = "According to the lease: " if not parts else "From the lease terms: "
        parts.append(lead + " ".join(chunks))

    if not parts:
        if prop is not None:
            return _deterministic_property_answer(prop, question)
        return (
            "I don't have that in the documents I can see. Please contact the "
            "leasing office, or select a specific property so I can check its "
            "lease and details."
        )
    return " ".join(parts).strip()


# ---------------------------------------------------------------------------
# SHARED DETERMINISTIC PASS (used by both answer() and answer_stream())
# ---------------------------------------------------------------------------
class _Plan:
    """The fast, deterministic groundwork shared by the streaming and
    non-streaming paths: routing, retrieval, sources, comparison, follow-ups
    and a templated fallback answer."""

    __slots__ = (
        "route", "prop", "comparison", "context_blocks", "sources",
        "follow_ups", "system",
    )

    def __init__(self, question, property_id):
        self.route = route(question or "", property_id)
        self.prop = property_facts(property_id)
        self.comparison: list[dict] = []
        if self.route == "compare":
            self.comparison = compare_properties(question or "")
            self.context_blocks, self.sources = _compare_context(self.comparison)
            self.follow_ups = _compare_follow_ups(question or "")
        else:
            self.context_blocks, self.sources = _assemble(
                question, self.route, self.prop, property_id
            )
            self.follow_ups = _follow_ups(self.route, self.prop, question or "")
        self.system = _system_for(self.route)

    def deterministic_answer(self, question) -> str:
        return _deterministic_answer(
            question, self.route, self.prop, self.sources, self.comparison
        )


# ---------------------------------------------------------------------------
# PUBLIC ENTRY POINT — never raises
# ---------------------------------------------------------------------------
def answer(question: str, property_id: str | None = None, history=None) -> dict:
    try:
        return _answer(question, property_id, history)
    except Exception as exc:  # noqa: BLE001 — mirror graph_ask: never 500
        print(f"concierge: unexpected error ({type(exc).__name__}: {exc}); safe fallback.")
        r = route(question or "", property_id)
        return {
            "answer": (
                "Sorry, I hit a snag looking that up. Please try rephrasing, or "
                "contact the leasing office."
            ),
            "route": r,
            "sources": [],
            "source": "rules",
            "property_id": property_id or "",
            "follow_ups": _follow_ups(r, None, question or ""),
            "comparison": [],
        }


def _answer(question: str, property_id: str | None, history) -> dict:
    plan = _Plan(question, property_id)

    llm_text = (
        _llm_answer(question, plan.context_blocks, history, plan.system)
        if plan.context_blocks
        else None
    )
    if llm_text is not None:
        source = "anthropic"
        text = llm_text
    else:
        source = "rules"
        text = plan.deterministic_answer(question)

    return {
        "answer": text,
        "route": plan.route,
        "sources": plan.sources,
        "source": source,
        "property_id": property_id or "",
        "follow_ups": plan.follow_ups,
        "comparison": plan.comparison,
    }


# ---------------------------------------------------------------------------
# STREAMING ENTRY POINT — a generator that can never crash the response
# ---------------------------------------------------------------------------
def answer_stream(question: str, property_id: str | None = None, history=None):
    """Yield SSE event dicts: one ``meta`` (deterministic pass, sent up front),
    then ``token`` events streaming the LLM prose, then a final ``done``. On any
    failure it degrades to a single deterministic ``token`` + ``done`` (rules).
    """
    try:
        plan = _Plan(question, property_id)
    except Exception as exc:  # noqa: BLE001 — never crash the stream
        print(f"concierge: stream planning failed ({type(exc).__name__}: {exc}).")
        yield {
            "type": "meta",
            "route": route(question or "", property_id),
            "property_id": property_id or "",
            "sources": [],
            "follow_ups": [],
            "comparison": [],
        }
        yield {
            "type": "token",
            "text": (
                "Sorry, I hit a snag looking that up. Please try rephrasing, or "
                "contact the leasing office."
            ),
        }
        yield {"type": "done", "source": "rules"}
        return

    yield {
        "type": "meta",
        "route": plan.route,
        "property_id": property_id or "",
        "sources": plan.sources,
        "follow_ups": plan.follow_ups,
        "comparison": plan.comparison,
    }

    # Try streaming the LLM prose; fall back to one deterministic token.
    streamed_any = False
    if plan.context_blocks:
        llm = get_langchain_llm()
        if llm is not None:
            try:
                messages = _build_messages(
                    question, plan.context_blocks, history, plan.system
                )
                for chunk in llm.stream(messages):
                    text = _coalesce(getattr(chunk, "content", ""))
                    if text:
                        streamed_any = True
                        yield {"type": "token", "text": text}
                if streamed_any:
                    yield {"type": "done", "source": "anthropic"}
                    return
            except Exception as exc:  # noqa: BLE001 — degrade to deterministic
                print(f"concierge: stream synthesis failed ({type(exc).__name__}).")

    try:
        det = plan.deterministic_answer(question)
    except Exception as exc:  # noqa: BLE001
        print(f"concierge: deterministic fallback failed ({type(exc).__name__}).")
        det = (
            "Sorry, I hit a snag looking that up. Please try rephrasing, or "
            "contact the leasing office."
        )
    yield {"type": "token", "text": det}
    yield {"type": "done", "source": "rules"}
