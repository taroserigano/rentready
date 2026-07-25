"""Neo4j property graph: connection, seeding, and querying.

Graph shape:
  (Property)-[:LOCATED_IN]->(Area)
  (Property)-[:OFFERS]->(Amenity)

If Neo4j can't be reached, everything falls back to an in-memory list of the
same properties, so the app still works (just without the real graph DB).
"""

import json
import time
from functools import lru_cache

from settings import DATA_DIR, settings

_PROPERTIES_FILE = DATA_DIR / "properties.json"


@lru_cache(maxsize=1)
def load_properties() -> list:
    with open(_PROPERTIES_FILE) as f:
        return json.load(f)


def _is_local_uri(uri: str) -> bool:
    """True for a Neo4j on this machine (localhost / 127.0.0.1 / ::1)."""
    host = uri.split("://", 1)[-1].split("/", 1)[0].rsplit("@", 1)[-1]
    host = host.split(":", 1)[0].strip("[]").lower()
    return host in ("localhost", "127.0.0.1", "::1", "")


@lru_cache(maxsize=1)
def _driver():
    from neo4j import GraphDatabase

    # Timeouts depend on WHERE Neo4j is:
    #
    # * LOCAL -- fail FAST. If it isn't running (the common local case) the
    #   default timeouts make every recommend() that falls back to the
    #   in-memory graph take several seconds.
    # * REMOTE (e.g. a free Neo4j Aura instance) -- a TLS handshake plus the
    #   initial routing-table fetch across the internet regularly exceeds 2s,
    #   so the local values would report a perfectly healthy database as
    #   "not available". Give it real headroom instead.
    local = _is_local_uri(settings.neo4j_uri)
    return GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_username, settings.neo4j_password),
        connection_timeout=2.0 if local else 15.0,
        connection_acquisition_timeout=3.0 if local else 20.0,
        max_transaction_retry_time=2.0 if local else 10.0,
    )


@lru_cache(maxsize=1)
def get_langchain_graph():
    """A langchain-neo4j Neo4jGraph wrapper, used by graph_ask().

    Returns None if Neo4j isn't reachable — or if the wrapper can't be built.

    ``refresh_schema=False``: Neo4jGraph's constructor otherwise introspects
    the schema via ``apoc.meta.data()``, which raises when the APOC plugin
    isn't installed (the common case for a plain Neo4j image). graph_ask()
    supplies its own hard-coded schema in the Cypher prompt and only calls
    ``.query()``, so the APOC-based refresh is unnecessary. We still guard the
    whole construction so any Neo4j/driver error degrades to None (the caller
    then answers "Neo4j is not available") instead of surfacing a 500.
    """
    if not is_available():
        return None
    try:
        from langchain_neo4j import Neo4jGraph

        return Neo4jGraph(
            url=settings.neo4j_uri,
            username=settings.neo4j_username,
            password=settings.neo4j_password,
            database=settings.neo4j_database,
            refresh_schema=False,
        )
    except Exception:
        return None


def run_read_only_cypher(cypher: str, params: dict | None = None) -> list[dict]:
    """Execute LLM-generated Cypher (graph_ask) under a real read-only
    transaction, enforced by the Neo4j SERVER itself — not just a client-side
    keyword check. ``routing_=RoutingControl.READ`` opens a read-mode
    transaction; Neo4j rejects any write clause inside one (including a write
    hidden behind an APOC procedure name that a keyword blocklist would miss)
    with a ``Neo.ClientError.Statement.AccessMode`` error, which callers
    should treat the same as any other "couldn't run that query" failure.
    """
    from neo4j import RoutingControl

    records, _, _ = _driver().execute_query(
        cypher,
        params or {},
        database_=settings.neo4j_database,
        routing_=RoutingControl.READ,
    )
    return [r.data() for r in records]


# Cached availability: (checked_at_monotonic, result).
_AVAILABILITY: tuple[float, bool] | None = None

# A NEGATIVE result is re-checked after this long; a POSITIVE one is cached for
# the process. Rationale: this is called on every recommend()/candidate query,
# so an uncached slow socket probe per call is unacceptable -- but caching
# "offline" forever meant one transient blip (or simply starting Neo4j after
# the app) left the graph disabled until a restart. That was tolerable when
# Neo4j could only be localhost; with a REMOTE instance a momentary network
# hiccup would silently disable the feature for the life of the process.
_UNAVAILABLE_RECHECK_S = 60.0


def _availability_cached() -> bool:
    """The real cache logic, kept separate from the public ``is_available`` so
    it stays testable -- the test suite stubs ``is_available`` wholesale."""
    global _AVAILABILITY
    now = time.monotonic()
    if _AVAILABILITY is not None:
        checked_at, result = _AVAILABILITY
        if result or (now - checked_at) < _UNAVAILABLE_RECHECK_S:
            return result
    result = _probe_connectivity()
    _AVAILABILITY = (now, result)
    return result


def is_available() -> bool:
    """True if Neo4j is reachable. Cheap: cached (see _UNAVAILABLE_RECHECK_S)."""
    return _availability_cached()


def _probe_connectivity() -> bool:
    try:
        _driver().verify_connectivity()
        return True
    except Exception:
        return False


def seed_graph() -> dict:
    """Load properties.json into Neo4j as a graph. No-op if unavailable."""
    properties = load_properties()
    if not is_available():
        return {"backend": "memory", "properties": len(properties)}

    cypher = """
    UNWIND $rows AS row
    MERGE (p:Property {id: row.id})
      SET p.name = row.name,
          p.property_type = row.property_type,
          p.monthly_rent = row.monthly_rent,
          p.bedrooms = row.bedrooms,
          p.bathrooms = row.bathrooms,
          p.bathroom_type = row.bathroom_type,
          p.square_feet = row.square_feet,
          p.has_balcony = row.has_balcony,
          p.in_unit_laundry = row.in_unit_laundry,
          p.parking_type = row.parking_type,
          p.pets_allowed = row.pets_allowed,
          p.lease_term_months = row.lease_term_months,
          p.furnished = row.furnished,
          p.photo_url = row.photo_url,
          p.photo_urls = row.photo_urls
    MERGE (n:Neighborhood {name: row.neighborhood.name})
      SET n.city = row.neighborhood.city,
          n.walk_score = row.neighborhood.walk_score,
          n.transit_score = row.neighborhood.transit_score
    MERGE (p)-[:IN_NEIGHBORHOOD]->(n)
    FOREACH (amenity IN row.amenities |
      MERGE (am:Amenity {name: amenity})
      MERGE (p)-[:OFFERS]->(am)
    )
    """
    with _driver().session(database=settings.neo4j_database) as session:
        session.run("MATCH (n) DETACH DELETE n")
        session.run(cypher, rows=properties)
    return {"backend": "neo4j", "properties": len(properties)}


def query_candidates(
    max_rent: float, pets_required: bool, preferred_area: str
) -> list:
    """Return properties matching the HARD constraints, from Neo4j.

    Hard constraints are only affordability (rent ceiling) and pets; every
    other preference is scored softly later. Falls back to filtering the
    in-memory list if Neo4j is unavailable.
    """
    if not is_available():
        return _memory_candidates(max_rent, pets_required, preferred_area)

    cypher = """
    MATCH (p:Property)-[:IN_NEIGHBORHOOD]->(n:Neighborhood)
    WHERE p.monthly_rent <= $max_rent
      AND ($pets_required = false OR p.pets_allowed = true)
    OPTIONAL MATCH (p)-[:OFFERS]->(am:Amenity)
    WITH p, n, collect(DISTINCT am.name) AS amenities
    RETURN p.id AS id, p.name AS name, n.name AS area, n.city AS city,
           n.walk_score AS walk_score, n.transit_score AS transit_score,
           p.property_type AS property_type, p.monthly_rent AS monthly_rent,
           p.bedrooms AS bedrooms, p.bathrooms AS bathrooms,
           p.bathroom_type AS bathroom_type, p.square_feet AS square_feet,
           p.has_balcony AS has_balcony, p.in_unit_laundry AS in_unit_laundry,
           p.parking_type AS parking_type, p.pets_allowed AS pets_allowed,
           p.lease_term_months AS lease_term_months, p.furnished AS furnished,
           p.photo_url AS photo_url, p.photo_urls AS photo_urls,
           amenities,
           (n.name = $preferred_area) AS area_match
    ORDER BY area_match DESC, p.monthly_rent ASC
    LIMIT 20
    """
    with _driver().session(database=settings.neo4j_database) as session:
        result = session.run(
            cypher,
            max_rent=max_rent,
            pets_required=pets_required,
            preferred_area=preferred_area,
        )
        return [record.data() for record in result]


def _flatten(p: dict, preferred_area: str) -> dict:
    """Turn a properties.json row into the flat candidate shape."""
    n = p.get("neighborhood", {})
    row = {k: v for k, v in p.items() if k != "neighborhood"}
    row.update(
        {
            "area": n.get("name", ""),
            "city": n.get("city", ""),
            "walk_score": n.get("walk_score"),
            "transit_score": n.get("transit_score"),
            "area_match": n.get("name", "") == preferred_area,
        }
    )
    return row


def _memory_candidates(
    max_rent: float, pets_required: bool, preferred_area: str
) -> list:
    rows = []
    for p in load_properties():
        if p["monthly_rent"] > max_rent:
            continue
        if pets_required and not p["pets_allowed"]:
            continue
        rows.append(_flatten(p, preferred_area))
    rows.sort(key=lambda r: (not r["area_match"], r["monthly_rent"]))
    return rows[:20]
