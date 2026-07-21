"""GraphRAG recommendations + natural-language graph Q&A.

Recommendation pattern (the important bit):
  1. Retrieve candidates from Neo4j using only HARD constraints (budget, pets).
  2. Score each candidate with the transparent weighted scorer (signals.py).
     The deterministic score is what RANKS them -- reproducible and testable.
  3. Claude only EXPLAINS the top matches in plain English (it does not
     reorder). Without an LLM we fall back to templated reasons.

graph_ask() is a separate GraphRAG flow: Claude writes a Cypher query, we run
it against Neo4j (langchain-neo4j), and Claude answers from the rows.
"""

import json
import re

from models import ApplicantProfile, PropertyRecommendation
from llm import get_langchain_llm
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
    "(:Property)-[:OFFERS]->(:Amenity)\n"
)


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


def graph_ask(question: str) -> dict:
    """Answer a natural-language question about the property graph.

    Claude writes Cypher, we run it via langchain-neo4j, Claude answers from
    the rows. This is GraphRAG where retrieval is a generated graph query.
    """
    llm = get_langchain_llm()
    lc_graph = graph.get_langchain_graph()
    if llm is None:
        return {"answer": "Set ANTHROPIC_API_KEY to ask the graph.", "cypher": ""}
    if lc_graph is None:
        return {"answer": "Neo4j is not available.", "cypher": ""}

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
        }
    if isinstance(cypher_raw, list):
        cypher_raw = "".join(str(b) for b in cypher_raw)
    cypher = _clean_cypher(cypher_raw)

    # Safety: never execute a generated query that could mutate the graph.
    if not is_read_only_cypher(cypher):
        return {
            "answer": "That question would require modifying the database, "
            "which isn't allowed here. Try a read-only question.",
            "cypher": cypher,
        }

    try:
        rows = lc_graph.query(cypher)
    except Exception as exc:  # noqa: BLE001
        return {"answer": f"Couldn't run that query: {exc}", "cypher": cypher}

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
        }
    if isinstance(answer, list):
        answer = "".join(str(b) for b in answer)
    return {"answer": answer, "cypher": cypher}


def _parse_json_array(s: str) -> list:
    match = re.search(r"\[.*\]", s, re.DOTALL)
    if not match:
        return []
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
