"""GraphRAG recommendations + natural-language graph Q&A.

Recommendation pattern (the important bit):
  1. Retrieve candidates from Neo4j using only HARD constraints (budget, pets).
  2. Score each candidate with the transparent weighted scorer (signals.py).
     The deterministic score is what RANKS them -- reproducible and testable.
  3. Claude only EXPLAINS the top matches in plain English (it does not
     reorder). Without an LLM we fall back to templated reasons.

graph_ask() is a separate GraphRAG flow: for the common filter/search shapes
(area, pets, price, bedrooms, amenities, parking) a deterministic Cypher
TEMPLATE runs instead -- no Cypher-writing LLM call at all, so it's instant,
free, and can't get case-sensitivity/syntax wrong. Anything else falls back to
the free-form path: Claude writes Cypher, we run it against Neo4j
(langchain-neo4j), and Claude answers from the rows.
"""

import json
import re

from models import ApplicantProfile, PropertyRecommendation
from llm import get_langchain_llm
import concierge
import graph
import signals


def _max_rent(profile: ApplicantProfile) -> float:
    """A generous affordability ceiling for the hard filter.

    Real underwriting fails around 40% of income; we use income/2.5 so
    stretch options still surface, and fall back to desired rent + 25%.
    """
    ceilings = []
    if profile.monthly_income > 0:
        ceilings.append(profile.monthly_income / 2.5)
    if profile.desired_rent > 0:
        ceilings.append(profile.desired_rent * 1.25)
    return max(ceilings) if ceilings else 10**9


def recommend(profile: ApplicantProfile, explain: bool = True) -> dict:
    """Rank properties for an applicant.

    When ``explain`` is False the LLM explanation pass is skipped and
    templated reasons are used — fast, deterministic, and used by the
    what-if simulator so slider drags don't wait on Claude.
    """
    backend = "neo4j" if graph.is_available() else "memory"
    ceiling = _max_rent(profile)
    candidates = graph.query_candidates(
        max_rent=ceiling,
        pets_required=profile.has_pets,
        preferred_area=profile.preferred_area,
    )

    relaxed = False
    if not candidates:
        # Relax the budget once rather than return nothing.
        relaxed = True
        candidates = graph.query_candidates(
            max_rent=ceiling * 1.5,
            pets_required=profile.has_pets,
            preferred_area=profile.preferred_area,
        )

    if not candidates:
        return {
            "recommendations": [],
            "source": "none",
            "graph_backend": backend,
            "relaxed": relaxed,
            "weights": signals.WEIGHTS,
        }

    # Deterministic scoring + ranking.
    scored = []
    for c in candidates:
        score, subs = signals.score_property(c, profile)
        scored.append((c, score, subs))
    scored.sort(key=lambda t: t[1], reverse=True)
    top = scored[:5]

    llm = get_langchain_llm() if explain else None
    explanations = {}
    source = "scorer"
    if llm is not None:
        try:
            explanations = _explain_with_llm(profile, top, llm)
            source = "anthropic"
        except Exception as exc:  # noqa: BLE001
            print(f"LLM explanation failed ({type(exc).__name__}: {exc}); templated.")

    recs = []
    for c, score, subs in top:
        ex = explanations.get(c["id"])
        if ex:
            reason = ex.get("match_reason", "")
            highlights = ex.get("fit_highlights", [])
        else:
            reason, highlights = signals.templated_reason(c, subs)
        recs.append(
            PropertyRecommendation(
                property_id=c["id"],
                name=c["name"],
                area=c.get("area", ""),
                property_type=c.get("property_type", ""),
                monthly_rent=c["monthly_rent"],
                bedrooms=c["bedrooms"],
                bathrooms=float(c.get("bathrooms", 0) or 0),
                bathroom_type=c.get("bathroom_type", "") or "",
                square_feet=int(c.get("square_feet", 0) or 0),
                has_balcony=bool(c.get("has_balcony", False)),
                in_unit_laundry=bool(c.get("in_unit_laundry", False)),
                pets_allowed=bool(c.get("pets_allowed", False)),
                parking_type=c.get("parking_type", "") or "",
                walk_score=c.get("walk_score"),
                transit_score=c.get("transit_score"),
                amenities=c.get("amenities", []),
                photo_url=c.get("photo_url"),
                photo_urls=c.get("photo_urls") or ([c["photo_url"]] if c.get("photo_url") else []),
                match_reason=reason,
                fit_highlights=highlights,
                score=score,
                signal_breakdown=subs,
            )
        )
    return {
        "recommendations": recs,
        "source": source,
        "graph_backend": backend,
        "relaxed": relaxed,
        "weights": signals.WEIGHTS,
    }


_EXPLAIN_SYSTEM = (
    "You are RentReady's leasing assistant. You write short, warm, plain-"
    "English explanations of why a rental fits an applicant.\n"
    "RULES:\n"
    "1. The ranking and scores are FINAL and computed by our system. Do NOT "
    "reorder or change scores.\n"
    "2. Use ONLY the data provided. Never invent amenities, prices, or "
    "features. Lead with the strongest matching signals (highest sub-scores). "
    "Mention at most one weakness if a sub-score is below 0.4, framed kindly.\n"
    "3. Plain English, no jargon, no field names, no numbers in the prose.\n"
    "4. Return ONLY a JSON array, one object per property (same property_ids):\n"
    '[{"property_id": "...", "match_reason": "1-2 sentences", '
    '"fit_highlights": ["short phrase", "short phrase"]}]'
)


def _explain_with_llm(profile: ApplicantProfile, top: list, llm) -> dict:
    payload = [
        {
            "property_id": c["id"],
            "name": c["name"],
            "area": c.get("area"),
            "property_type": c.get("property_type"),
            "monthly_rent": c["monthly_rent"],
            "bedrooms": c["bedrooms"],
            "bathrooms": c.get("bathrooms"),
            "bathroom_type": c.get("bathroom_type"),
            "square_feet": c.get("square_feet"),
            "has_balcony": c.get("has_balcony"),
            "in_unit_laundry": c.get("in_unit_laundry"),
            "pets_allowed": c.get("pets_allowed"),
            "parking_type": c.get("parking_type"),
            "walk_score": c.get("walk_score"),
            "transit_score": c.get("transit_score"),
            "amenities": c.get("amenities", []),
            "score": score,
            "signal_breakdown": subs,
        }
        for c, score, subs in top
    ]
    messages = [
        ("system", _EXPLAIN_SYSTEM),
        (
            "human",
            f"Applicant profile:\n{profile.model_dump_json()}\n\n"
            f"Ranked candidates (already scored; do not reorder):\n"
            f"{json.dumps(payload)}\n\nReturn the JSON array now.",
        ),
    ]
    raw = llm.invoke(messages).content
    if isinstance(raw, list):  # some providers return content blocks
        raw = "".join(str(b) for b in raw)
    items = _parse_json_array(raw)
    out = {}
    valid_ids = {c["id"] for c, _, _ in top}
    for item in items:
        pid = item.get("property_id")
        if pid in valid_ids:
            out[pid] = item
    return out


# --------------------------------------------------------------------------
# Natural-language Q&A over the property graph (langchain-neo4j).
# --------------------------------------------------------------------------
_CYPHER_SYSTEM = (
    "You translate questions into a single Cypher query for Neo4j. "
    "Output ONLY the raw Cypher query. Do NOT use markdown code fences, do "
    "NOT write the word 'cypher', and do NOT add any explanation.\n\n"
    "Graph schema:\n"
    "(:Property {{id, name, property_type, monthly_rent, bedrooms, bathrooms, "
    "bathroom_type, square_feet, has_balcony, in_unit_laundry, parking_type, "
    "pets_allowed, lease_term_months, furnished}})\n"
    "(:Neighborhood {{name, city, walk_score, transit_score}})\n"
    "(:Amenity {{name}})\n"
    "(:Property)-[:IN_NEIGHBORHOOD]->(:Neighborhood)\n"
    "(:Property)-[:OFFERS]->(:Amenity)\n\n"
    "Matching rules:\n"
    "- String values are stored Capitalized (amenities e.g. 'Gym', 'Pool', "
    "'Rooftop Deck', 'Pet Park', 'Bike Storage', 'Concierge'; neighborhoods e.g. "
    "'Downtown', 'East Austin'). Match names CASE-INSENSITIVELY — use "
    "toLower(x.name) = toLower('value') or x.name =~ '(?i)value' — never a "
    "case-sensitive equality, or you will miss rows.\n"
    "- 'pet-friendly' means p.pets_allowed = true. 'studio' means bedrooms = 0. "
    "Rent ceilings compare p.monthly_rent.\n"
)


def _coalesce(content) -> str:
    """Extended-thinking responses are a list of content blocks -- a
    {"type": "thinking", ...} block (no "text" key) plus the real
    {"type": "text", "text": "..."} block. Pulling only "text" (rather than
    str(b) on every block) keeps a stringified thinking payload out of the
    Cypher/answer. Mirrors the identical helper already in concierge.py /
    residents_chat.py / risk_chat.py."""
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


def _clean_cypher(text: str) -> str:
    """Strip markdown fences and a stray leading 'cypher' token."""
    text = text.strip()
    # Remove ``` fences (with optional language label).
    fence = re.search(r"```(?:cypher)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1)
    text = text.strip()
    # Remove a leading bare "cypher" line if present.
    text = re.sub(r"^\s*cypher\s*\n", "", text, flags=re.IGNORECASE)
    return text.strip()


# Cypher keywords that mutate the database. graph-ask must be read-only:
# the model output is untrusted, and a prompt-injected write (e.g.
# "MATCH (n) DETACH DELETE n") would otherwise wipe the graph.
# Any write clause makes the query unsafe. A CALL{} subquery that writes will
# itself contain one of these keywords, so it's covered too.
_WRITE_CLAUSES = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|FOREACH|LOAD\s+CSV)\b",
    re.IGNORECASE,
)


def is_read_only_cypher(cypher: str) -> bool:
    """True if the query contains no write/mutating clauses."""
    return not _WRITE_CLAUSES.search(cypher or "")


# ---------------------------------------------------------------------------
# Deterministic Cypher TEMPLATES for the common filter/search shapes -- the
# same vocabulary concierge.py already parses (reused, not reimplemented, so
# "2k", known areas, and the amenity list stay in exactly one place). Anything
# that doesn't match one of these shapes falls back to the free-form LLM path.
# ---------------------------------------------------------------------------

# "which/what areas/neighborhoods have/offer/with <amenity>" -- a REVERSE
# lookup (amenity -> areas), a different query shape than a property search.
_AREA_AMENITY_RE = re.compile(
    r"\b(which|what)\b.{0,25}\b(areas?|neighbou?rhoods?)\b.{0,25}\b(have|has|offer|offers|with)\b",
    re.IGNORECASE,
)

_WANT_PARKING_RE = re.compile(r"\b(parking|garage)\b", re.IGNORECASE)
_WANT_LAUNDRY_RE = re.compile(r"\b(in[\s-]?unit laundry|laundry|washer|dryer)\b", re.IGNORECASE)
_WANT_FURNISHED_RE = re.compile(r"\bfurnished\b", re.IGNORECASE)
_WANT_BALCONY_RE = re.compile(r"\bbalcon", re.IGNORECASE)

_SEARCH_RETURN = (
    "RETURN p.name AS name, p.property_type AS property_type, "
    "p.monthly_rent AS monthly_rent, p.bedrooms AS bedrooms, "
    "p.bathrooms AS bathrooms ORDER BY p.monthly_rent ASC LIMIT 10"
)


def _match_amenities(q: str) -> list[tuple[str, str]]:
    return [(label, phrase) for rx, label, phrase in concierge._AMENITY_MAP if rx.search(q)]  # noqa: SLF001


def _build_template(question: str):
    """A deterministic (cypher, params, kind, meta) for a recognized filter
    shape, or None to fall back to the free-form LLM Cypher path."""
    q = question or ""
    amenities = _match_amenities(q)

    # Shape 1: "which areas have a gym?" -- amenity -> neighborhoods.
    if amenities and _AREA_AMENITY_RE.search(q):
        label, _phrase = amenities[0]
        cypher = (
            "MATCH (p:Property)-[:OFFERS]->(a:Amenity) "
            "MATCH (p)-[:IN_NEIGHBORHOOD]->(n:Neighborhood) "
            "WHERE toLower(a.name) = toLower($amenity) "
            "RETURN DISTINCT n.name AS neighborhood ORDER BY n.name"
        )
        return cypher, {"amenity": label}, "areas_with_amenity", {"amenity": label}

    # Shape 2: a property search/filter (area, pets, price, beds, amenities, parking).
    max_rent = concierge._parse_max_rent(q)  # noqa: SLF001
    min_beds = concierge._parse_min_bedrooms(q)  # noqa: SLF001
    area = concierge._area_in(q)  # noqa: SLF001
    want_pets = bool(concierge._PETS_RE.search(q))  # noqa: SLF001
    want_parking = bool(_WANT_PARKING_RE.search(q))
    want_laundry = bool(_WANT_LAUNDRY_RE.search(q))
    want_furnished = bool(_WANT_FURNISHED_RE.search(q))
    want_balcony = bool(_WANT_BALCONY_RE.search(q))

    if not any([max_rent is not None, min_beds is not None, area, want_pets,
                amenities, want_parking, want_laundry, want_furnished, want_balcony]):
        return None

    match_parts = ["MATCH (p:Property)"]
    where_parts = []
    params: dict = {}

    if area:
        match_parts.append("MATCH (p)-[:IN_NEIGHBORHOOD]->(n:Neighborhood)")
        where_parts.append("toLower(n.name) = toLower($area)")
        params["area"] = area
    if amenities:
        match_parts.append("OPTIONAL MATCH (p)-[:OFFERS]->(a:Amenity)")
    if max_rent is not None:
        where_parts.append("p.monthly_rent <= $max_rent")
        params["max_rent"] = max_rent
    if min_beds is not None:
        where_parts.append("p.bedrooms >= $min_beds")
        params["min_beds"] = min_beds
    if want_pets:
        where_parts.append("p.pets_allowed = true")
    if want_parking:
        where_parts.append("p.parking_type <> 'none'")
    if want_laundry:
        where_parts.append("p.in_unit_laundry = true")
    if want_furnished:
        where_parts.append("p.furnished = true")
    if want_balcony:
        where_parts.append("p.has_balcony = true")

    if amenities:
        # collect() + all(...) rather than an EXISTS{} subquery -- works on
        # any Cypher version, and AND-combines multiple wanted amenities.
        carry = "p, n" if area else "p"
        params["wanted_amenities"] = [label for label, _ in amenities]
        with_clause = f"WITH {carry}, collect(toLower(a.name)) AS amenity_names"
        where_parts.append("all(x IN $wanted_amenities WHERE toLower(x) IN amenity_names)")
        cypher = (
            " ".join(match_parts) + " " + with_clause
            + " WHERE " + " AND ".join(where_parts) + " " + _SEARCH_RETURN
        )
    else:
        where_clause = (" WHERE " + " AND ".join(where_parts)) if where_parts else ""
        cypher = " ".join(match_parts) + where_clause + " " + _SEARCH_RETURN

    meta = {"area": area, "max_rent": max_rent, "min_beds": min_beds,
            "want_pets": want_pets, "amenities": [label for label, _ in amenities],
            "want_parking": want_parking}
    return cypher, params, "property_search", meta


def _display_cypher(cypher: str, params: dict) -> str:
    """The template query with its bound param VALUES inlined, for display
    only -- execution always uses the safe parameterized form above."""
    text = cypher
    for key, val in params.items():
        if isinstance(val, bool):
            literal = "true" if val else "false"
        elif isinstance(val, (int, float)):
            literal = str(val)
        elif isinstance(val, list):
            literal = "[" + ", ".join("'" + str(v).replace("'", "\\'") + "'" for v in val) + "]"
        else:
            literal = "'" + str(val).replace("'", "\\'") + "'"
        text = text.replace(f"${key}", literal)
    return text


def _template_answer(kind: str, rows: list, meta: dict) -> str:
    if kind == "areas_with_amenity":
        names = [r.get("neighborhood") for r in rows if r.get("neighborhood")]
        amenity = meta.get("amenity", "that amenity")
        if not names:
            return f"No neighborhood currently has a property with {amenity}."
        if len(names) == 1:
            return f"{names[0]} has a property with {amenity}."
        return f"{', '.join(names[:-1])} and {names[-1]} have properties with {amenity}."

    if not rows:
        return "No properties matched that search."
    bits = []
    for r in rows:
        rent = r.get("monthly_rent")
        rent_txt = f"${int(rent):,}/mo" if rent is not None else "rent n/a"
        beds = r.get("bedrooms")
        bed_txt = "studio" if beds == 0 else f"{beds}BR"
        bits.append(f"{r.get('name')} ({r.get('property_type', '')}, {bed_txt}, {rent_txt})")
    if len(rows) == 1:
        return f"One match: {bits[0]}."
    return f"{len(rows)} matching properties: " + "; ".join(bits) + "."


def graph_ask(question: str) -> dict:
    """Answer a natural-language question about the property graph.

    A deterministic Cypher TEMPLATE handles the common filter/search shapes
    (zero LLM calls -- works even offline as long as Neo4j is reachable).
    Anything else falls back to the free-form path: Claude writes Cypher, we
    run it via langchain-neo4j, Claude answers from the rows.
    """
    template = _build_template(question)
    if template is not None:
        cypher, params, kind, meta = template
        if not graph.is_available():
            return {"answer": "Neo4j is not available.", "cypher": "", "source": "rules"}
        try:
            rows = graph.run_read_only_cypher(cypher, params)
        except Exception as exc:  # noqa: BLE001
            return {
                "answer": f"Couldn't run that query: {exc}",
                "cypher": _display_cypher(cypher, params),
                "source": "rules",
            }
        return {
            "answer": _template_answer(kind, rows, meta),
            "cypher": _display_cypher(cypher, params),
            "source": "template",
        }

    llm = get_langchain_llm()
    lc_graph = graph.get_langchain_graph()
    if llm is None:
        return {"answer": "Set ANTHROPIC_API_KEY to ask the graph.", "cypher": "", "source": "rules"}
    if lc_graph is None:
        return {"answer": "Neo4j is not available.", "cypher": "", "source": "rules"}

    try:
        cypher_raw = llm.invoke(
            [("system", _CYPHER_SYSTEM), ("human", question)]
        ).content
    except Exception as exc:  # noqa: BLE001
        # Rate limit / API error / network — degrade like the rest of the app
        # instead of surfacing a 500 (which the browser shows as "Failed to
        # fetch" across origins).
        return {
            "answer": f"The language model is unavailable right now, so I "
            f"couldn't write a query. ({exc})",
            "cypher": "",
            "source": "rules",
        }
    cypher = _clean_cypher(_coalesce(cypher_raw))

    # Safety, layer 1: a fast keyword pre-check gives a friendly rejection for
    # the common case without ever touching the database.
    if not is_read_only_cypher(cypher):
        return {
            "answer": "That question would require modifying the database, "
            "which isn't allowed here. Try a read-only question.",
            "cypher": cypher,
            "source": "rules",
        }

    # Safety, layer 2 (the real enforcement): run it under a Neo4j read-mode
    # transaction. The keyword blocklist above is a string check and can miss
    # a write hidden behind an APOC procedure name (e.g. apoc.refactor.rename.
    # label) that contains none of the blocked keywords — the server itself
    # rejects any write inside a read-mode transaction regardless of how it's
    # spelled, so this is the actual safety boundary.
    try:
        rows = graph.run_read_only_cypher(cypher)
    except Exception as exc:  # noqa: BLE001
        return {"answer": f"Couldn't run that query: {exc}", "cypher": cypher, "source": "rules"}

    try:
        answer = llm.invoke(
            [
                (
                    "system",
                    "Answer the user's question in one or two plain-English "
                    "sentences using ONLY the query results. If empty, say you "
                    "couldn't find anything.",
                ),
                (
                    "human",
                    f"Question: {question}\n\nQuery results (JSON):\n"
                    f"{json.dumps(rows, default=str)}",
                ),
            ]
        ).content
    except Exception as exc:  # noqa: BLE001
        # The query DID run — surface the result count + the Cypher so the
        # feature stays useful even when the model can't summarize.
        n = len(rows) if isinstance(rows, list) else 0
        return {
            "answer": f"Ran the query and found {n} result(s), but the language "
            f"model is unavailable to summarize them right now. ({exc})",
            "cypher": cypher,
            "source": "rules",
        }
    answer = _coalesce(answer)
    return {"answer": answer, "cypher": cypher, "source": "anthropic"}


def _parse_json_array(s: str) -> list:
    match = re.search(r"\[.*\]", s, re.DOTALL)
    if not match:
        return []
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
