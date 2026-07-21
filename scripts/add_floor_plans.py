"""Add floor_plans to every property in data/properties.json.

Deterministic per property id (md5 hash), no randomness. The cheapest plan
mirrors the existing top-level unit fields exactly; larger plans are added
above it for Apartment/Condo/Loft/Studio types.
"""

import hashlib
import json
from datetime import date, timedelta
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "properties.json"

MULTI_PLAN_TYPES = {"Apartment", "Condo", "Loft"}
SINGLE_PLAN_TYPES = {"House", "Townhouse", "Townhome", "Duplex"}

# Per step above the base unit: (rent multiplier, sqft multiplier)
STEP_MULT = {1: (1.35, 1.40), 2: (1.70, 1.80), 3: (2.00, 2.15)}


def plan_name(bedrooms: int) -> str:
    return "Studio" if bedrooms == 0 else f"{bedrooms} Bedroom"


def h(pid: str, salt: str) -> int:
    return int.from_bytes(hashlib.md5(f"{pid}:{salt}".encode()).digest()[:4], "big")


def base_plan(p: dict) -> dict:
    return {
        "name": plan_name(int(p["bedrooms"])),
        "bedrooms": int(p["bedrooms"]),
        "bathrooms": p["bathrooms"],
        "square_feet": p["square_feet"],
        "monthly_rent": p["monthly_rent"],
        "available_units": 1 + h(p["id"], "units0") % 8,  # 1-8: base is available
        "availability_date": p["availability_date"],
    }


def larger_plan(p: dict, step: int) -> dict:
    rent_m, sqft_m = STEP_MULT[step]
    beds = int(p["bedrooms"]) + step
    baths = min(float(p["bathrooms"]) + 0.5 * step, float(beds))
    base_date = date.fromisoformat(p["availability_date"])
    offset = h(p["id"], f"date{step}") % 45  # 0-44 days later
    return {
        "name": plan_name(beds),
        "bedrooms": beds,
        "bathrooms": baths,
        "square_feet": int(round(p["square_feet"] * sqft_m / 10.0)) * 10,
        "monthly_rent": round(p["monthly_rent"] * rent_m / 5.0) * 5,
        "available_units": h(p["id"], f"units{step}") % 9,  # 0-8
        "availability_date": (base_date + timedelta(days=offset)).isoformat(),
    }


def build_plans(p: dict) -> list:
    ptype = p["property_type"]
    plans = [base_plan(p)]
    if ptype in SINGLE_PLAN_TYPES:
        return plans
    if ptype == "Studio":
        # Studio base, optionally a 1 Bedroom above.
        if h(p["id"], "studio_extra") % 2 == 0:
            plans.append(larger_plan(p, 1))
        return plans
    if ptype in MULTI_PLAN_TYPES:
        # 1-3 larger plans, capped so bedrooms stay <= 4.
        extra = 1 + h(p["id"], "extra") % 3
        extra = min(extra, 4 - int(p["bedrooms"]))
        extra = max(extra, 1)  # contract: 2-4 plans
        for step in range(1, extra + 1):
            plans.append(larger_plan(p, step))
        return plans
    return plans


def main() -> None:
    props = json.loads(DATA.read_text())
    for p in props:
        plans = build_plans(p)
        # Sanity: base plan must be cheapest and mirror top-level fields.
        assert plans[0]["monthly_rent"] == p["monthly_rent"]
        assert plans[0]["bedrooms"] == p["bedrooms"]
        assert all(
            pl["monthly_rent"] > plans[0]["monthly_rent"] for pl in plans[1:]
        )
        p["floor_plans"] = plans
    DATA.write_text(json.dumps(props, indent=1) + "\n")
    counts = {}
    for p in props:
        counts.setdefault(p["property_type"], []).append(len(p["floor_plans"]))
    for t, c in sorted(counts.items()):
        print(t, c)


if __name__ == "__main__":
    main()
