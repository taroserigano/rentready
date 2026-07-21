"""API endpoints for browsing the property inventory.

Serves the full listings from data/properties.json (via graph.load_properties)
as flat rows, with simple server-side filters. Powers the "Properties" tab.
"""

import time

from fastapi import APIRouter, HTTPException, Query

import graph
import signals
import store
from models import ApplicantProfile

router = APIRouter(tags=["properties"])


def _property_by_id(property_id: str) -> dict:
    for p in graph.load_properties():
        if p.get("id") == property_id:
            return p
    raise HTTPException(404, "Unknown property id.")


def _screen(prop: dict, profile: ApplicantProfile) -> dict:
    """Deterministically check an applicant against a listing's OWN
    underwriting criteria (each property advertises its own thresholds).
    Returns itemized checks; no LLM. `ok` is None when undeterminable."""
    rent = prop.get("monthly_rent", 0) or 0
    checks = []

    mult = prop.get("min_income_multiplier") or 0
    if mult and rent:
        need = mult * rent
        checks.append({
            "label": "Income",
            "required": f"{mult}x rent (${need:,.0f}/mo)",
            "actual": f"${profile.monthly_income:,.0f}/mo",
            "ok": profile.monthly_income >= need,
        })

    min_credit = prop.get("min_credit_score")
    if min_credit:
        checks.append({
            "label": "Credit score",
            "required": f"{min_credit}+",
            "actual": str(profile.credit_score) if profile.credit_score is not None else "not provided",
            "ok": None if profile.credit_score is None else profile.credit_score >= min_credit,
        })

    max_occ = prop.get("max_occupants")
    if max_occ:
        checks.append({
            "label": "Occupancy",
            "required": f"up to {max_occ}",
            "actual": f"{profile.household_size} in household",
            "ok": profile.household_size <= max_occ,
        })

    if profile.has_pets or profile.pet_count > 0:
        checks.append({
            "label": "Pets",
            "required": "allowed" if prop.get("pets_allowed") else "not allowed",
            "actual": f"{max(profile.pet_count, 1)} pet(s)",
            "ok": bool(prop.get("pets_allowed")),
        })
        max_pets = prop.get("pets_max_count")
        if max_pets and profile.pet_count:
            checks.append({
                "label": "Pet count",
                "required": f"up to {max_pets}",
                "actual": f"{profile.pet_count}",
                "ok": profile.pet_count <= max_pets,
            })

    if profile.smoker and prop.get("smoking_allowed") is False:
        checks.append({
            "label": "Smoking",
            "required": "not allowed",
            "actual": "applicant smokes",
            "ok": False,
        })

    decided = [c["ok"] for c in checks if c["ok"] is not None]
    if decided and not all(decided):
        verdict = "fail"
    elif any(c["ok"] is None for c in checks):
        verdict = "review"
    else:
        verdict = "pass"
    return {"passes": verdict == "pass", "verdict": verdict, "checks": checks}


def _flatten(p: dict) -> dict:
    """Turn a properties.json row into a flat listing row.

    Same shape as graph._flatten, minus the applicant-specific area_match.
    """
    n = p.get("neighborhood", {})
    row = {k: v for k, v in p.items() if k != "neighborhood"}
    plans = p.get("floor_plans") or []
    row.update(
        {
            "area": n.get("name", ""),
            "city": n.get("city", ""),
            "walk_score": n.get("walk_score"),
            "transit_score": n.get("transit_score"),
            "plan_count": len(plans),
            "max_bedrooms": max(
                (fp["bedrooms"] for fp in plans), default=p.get("bedrooms")
            ),
        }
    )
    return row


@router.get("/properties")
def list_properties(
    area: str = Query(None, description="Exact neighborhood name"),
    max_rent: float = Query(None, description="Monthly rent ceiling"),
    min_bedrooms: int = Query(None, description="At least this many bedrooms"),
    pets_allowed: bool = Query(None, description="Only pet-friendly homes"),
    q: str = Query(None, description="Substring match on property name"),
) -> dict:
    """All listings, flattened and filtered, cheapest first."""
    t0 = time.perf_counter()
    rows = [_flatten(p) for p in graph.load_properties()]
    areas = sorted({r["area"] for r in rows if r["area"]})

    if area:
        rows = [r for r in rows if r["area"] == area]
    if max_rent is not None:
        rows = [r for r in rows if r["monthly_rent"] <= max_rent]
    if min_bedrooms is not None:
        # Match if ANY floor plan has enough bedrooms; fall back to the
        # top-level (base unit) bedrooms when a row has no floor plans.
        rows = [
            r
            for r in rows
            if any(
                fp["bedrooms"] >= min_bedrooms
                for fp in (r.get("floor_plans") or [{"bedrooms": r["bedrooms"]}])
            )
        ]
    if pets_allowed:
        rows = [r for r in rows if r["pets_allowed"]]
    if q:
        needle = q.lower()
        rows = [r for r in rows if needle in r["name"].lower()]

    rows.sort(key=lambda r: r["monthly_rent"])
    store.log_event(
        endpoint="properties",
        latency_ms=(time.perf_counter() - t0) * 1000,
        meta={"total": len(rows)},
    )
    return {"properties": rows, "total": len(rows), "areas": areas}


@router.get("/properties/{property_id}/screen")
def screen_applicant(property_id: str, applicant_id: str) -> dict:
    """Will this applicant qualify for THIS listing? (F1) Deterministic."""
    prop = _property_by_id(property_id)
    profile = store.get_profile(applicant_id)
    if profile is None:
        raise HTTPException(404, "Unknown applicant id.")
    result = _screen(prop, profile)
    store.log_event(endpoint="screen", applicant_id=applicant_id,
                    meta={"property_id": property_id, "verdict": result["verdict"]})
    return {"property_id": property_id, "applicant_id": applicant_id, **result}


@router.get("/properties/{property_id}/candidates")
def property_candidates(property_id: str, limit: int = Query(10)) -> dict:
    """Rank stored applicants for a vacancy (F2, inverse of /recommend).
    Reuses the deterministic scorer, so ranking stays authoritative."""
    t0 = time.perf_counter()
    prop = _property_by_id(property_id)
    flat = _flatten(prop)
    candidates = []
    for row in store.list_applicants():
        profile = store.get_profile(row["id"])
        if profile is None:
            continue
        score, breakdown = signals.score_property(flat, profile)
        screen = _screen(prop, profile)
        candidates.append({
            "applicant_id": row["id"],
            "name": row["name"],
            "score": round(score, 4),
            "signal_breakdown": breakdown,
            "screen_passes": screen["passes"],
            "screen_verdict": screen["verdict"],
        })
    candidates.sort(key=lambda c: (c["screen_passes"], c["score"]), reverse=True)
    store.log_event(endpoint="candidates", latency_ms=(time.perf_counter() - t0) * 1000,
                    meta={"property_id": property_id, "n": len(candidates)})
    return {"property_id": property_id, "name": prop.get("name", ""),
            "candidates": candidates[:limit], "total": len(candidates)}
