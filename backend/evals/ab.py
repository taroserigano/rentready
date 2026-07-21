"""A/B comparison harness: pick a winner between two variants.

This is the core experimentation workflow: hold the DATASET and the JUDGE
fixed, vary the system-under-test (a prompt and/or a model), score both, and
declare a winner with the metric deltas.

We A/B the *recommendation explanation* task. The deterministic scorer picks
the SAME top properties for every variant (fair comparison) -- only the
plain-English explanation differs. Each variant is graded on three axes:

  * groundedness  -- the LLM-as-judge score (quality);
  * latency       -- mean wall-clock per generation call;
  * cost          -- estimated $ from token usage.

A higher groundedness wins; ties break on lower cost. Skipped cleanly with no
ANTHROPIC_API_KEY so it never breaks CI.
"""

import json
import sys
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

import graph  # noqa: E402
import graphrag  # noqa: E402
import pdf_ingest  # noqa: E402
import rag_llamaindex  # noqa: E402
import signals  # noqa: E402
from evals import judges  # noqa: E402
from settings import settings  # noqa: E402

APPLICATIONS_DIR = BACKEND.parent / "data" / "applications"
DEFAULT_SLUGS = ["jordan-rivera", "alex-chen"]

# Approximate Anthropic list prices (USD per 1M tokens) -- for the cost lesson,
# not billing. Update freely; the point is to make cost a first-class metric.
_PRICES = {
    "claude-sonnet-4-6": {"in": 3.0, "out": 15.0},
    "claude-haiku-4-5-20251001": {"in": 0.80, "out": 4.0},
    "_default": {"in": 3.0, "out": 15.0},
}

_DETAILED_SYSTEM = (
    "You are RentReady's enthusiastic leasing assistant. Write a warm, vivid "
    "explanation of why each rental fits the applicant.\n"
    "RULES:\n"
    "1. The ranking and scores are FINAL. Do NOT reorder or change scores.\n"
    "2. Write 2-3 sentences per property and paint a picture of living there. "
    "Lead with the strongest matches and call out several great features.\n"
    "3. Plain English, friendly tone, no field names or numbers in the prose.\n"
    "4. Return ONLY a JSON array, one object per property (same property_ids):\n"
    '[{"property_id": "...", "match_reason": "2-3 sentences", '
    '"fit_highlights": ["short phrase", "short phrase", "short phrase"]}]'
)

# Each variant overrides a model and/or the explanation system prompt. Both
# variants are graded by the SAME (default-model) judge.
VARIANTS = {
    "concise": {
        "id": "concise",
        "label": "Concise prompt (current)",
        "model": settings.anthropic_model,
        "system": graphrag._EXPLAIN_SYSTEM,
    },
    "detailed": {
        "id": "detailed",
        "label": "Detailed/vivid prompt",
        "model": settings.anthropic_model,
        "system": _DETAILED_SYSTEM,
    },
    "haiku_concise": {
        "id": "haiku_concise",
        "label": "Concise prompt on Haiku (cheaper)",
        "model": "claude-haiku-4-5-20251001",
        "system": graphrag._EXPLAIN_SYSTEM,
    },
}


def list_variants() -> list:
    return [
        {"id": v["id"], "label": v["label"], "model": v["model"]}
        for v in VARIANTS.values()
    ]


def _build_llm(model: str):
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(
        model=model,
        api_key=settings.anthropic_api_key,
        temperature=0.0,
        max_tokens=1024,
    )


def _top_candidates(profile, n: int = 2) -> list:
    """The deterministic top-N candidates, shared by both variants."""
    ceiling = graphrag._max_rent(profile)
    candidates = graph.query_candidates(
        max_rent=ceiling,
        pets_required=profile.has_pets,
        preferred_area=profile.preferred_area,
    )
    scored = []
    for c in candidates:
        score, subs = signals.score_property(c, profile)
        scored.append((c, score, subs))
    scored.sort(key=lambda t: t[1], reverse=True)
    return scored[:n]


def _payload(top: list) -> list:
    return [
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
            "amenities": c.get("amenities", []),
            "score": score,
            "signal_breakdown": subs,
        }
        for c, score, subs in top
    ]


def _rec_dict(c: dict, reason: str) -> dict:
    """A recommendation-shaped dict the judge can score (facts + prose)."""
    return {
        "name": c["name"],
        "area": c.get("area", ""),
        "property_type": c.get("property_type", ""),
        "monthly_rent": c["monthly_rent"],
        "bedrooms": c["bedrooms"],
        "bathrooms": c.get("bathrooms"),
        "bathroom_type": c.get("bathroom_type", ""),
        "square_feet": c.get("square_feet"),
        "has_balcony": c.get("has_balcony"),
        "in_unit_laundry": c.get("in_unit_laundry"),
        "pets_allowed": c.get("pets_allowed"),
        "parking_type": c.get("parking_type", ""),
        "walk_score": c.get("walk_score"),
        "transit_score": c.get("transit_score"),
        "amenities": c.get("amenities", []),
        "match_reason": reason,
    }


def _usage_tokens(message) -> tuple:
    usage = getattr(message, "usage_metadata", None) or {}
    return usage.get("input_tokens", 0), usage.get("output_tokens", 0)


def _cost(model: str, in_tok: int, out_tok: int) -> float:
    p = _PRICES.get(model, _PRICES["_default"])
    return round((in_tok * p["in"] + out_tok * p["out"]) / 1_000_000, 6)


def _run_variant(variant: dict, profiles: list, judge_llm) -> dict:
    """Generate explanations with the variant, then judge groundedness."""
    gen_llm = _build_llm(variant["model"])
    scores, latencies, cases = [], [], []
    in_tokens = out_tokens = 0

    for slug, profile, top in profiles:
        payload = _payload(top)
        messages = [
            ("system", variant["system"]),
            (
                "human",
                f"Applicant profile:\n{profile.model_dump_json()}\n\n"
                f"Ranked candidates (already scored; do not reorder):\n"
                f"{json.dumps(payload)}\n\nReturn the JSON array now.",
            ),
        ]
        t0 = time.perf_counter()
        msg = gen_llm.invoke(messages)
        latencies.append(time.perf_counter() - t0)
        ti, to = _usage_tokens(msg)
        in_tokens += ti
        out_tokens += to

        raw = msg.content
        if isinstance(raw, list):
            raw = "".join(str(b) for b in raw)
        items = {
            it.get("property_id"): it for it in graphrag._parse_json_array(raw)
        }

        for c, _score, _subs in top:
            reason = (items.get(c["id"], {}) or {}).get("match_reason", "")
            verdict = judges.judge_recommendation(
                profile.model_dump(), _rec_dict(c, reason), judge_llm
            )
            if verdict["score"] is not None:
                scores.append(verdict["score"])
            cases.append(
                {
                    "slug": slug,
                    "property": c["name"],
                    "score": verdict["score"],
                    "reason": reason,
                    "violations": verdict["violations"],
                }
            )

    def avg(xs):
        return round(sum(xs) / len(xs), 4) if xs else None

    mean_score = avg(scores)
    return {
        "id": variant["id"],
        "label": variant["label"],
        "model": variant["model"],
        "mean_groundedness": mean_score,
        "mean_groundedness_pct": round(mean_score / 5, 4) if mean_score else None,
        "mean_latency_s": avg(latencies),
        "input_tokens": in_tokens,
        "output_tokens": out_tokens,
        "est_cost_usd": _cost(variant["model"], in_tokens, out_tokens),
        "n": len(cases),
        "per_case": cases,
    }


def compare(a_id: str, b_id: str, slugs: list = None) -> dict:
    """Run both variants on the same data + judge, and pick a winner."""
    judge_llm = judges.get_langchain_llm()
    if judge_llm is None:
        return {"skipped": True, "reason": "No ANTHROPIC_API_KEY; A/B needs a model."}
    if a_id not in VARIANTS or b_id not in VARIANTS:
        return {"skipped": True, "reason": f"Unknown variant(s): {a_id}, {b_id}."}

    slugs = slugs or DEFAULT_SLUGS
    # Build the shared, deterministic dataset once (same top properties for both).
    profiles = []
    for slug in slugs:
        pdf = APPLICATIONS_DIR / f"{slug}.pdf"
        if not pdf.exists():
            continue
        text = pdf_ingest.extract_text(str(pdf))
        profile = rag_llamaindex._heuristic_profile(text)
        profiles.append((slug, profile, _top_candidates(profile)))

    a = _run_variant(VARIANTS[a_id], profiles, judge_llm)
    b = _run_variant(VARIANTS[b_id], profiles, judge_llm)

    winner, rationale = _decide(a, b)
    return {
        "skipped": False,
        "dataset_slugs": [s for s, _, _ in profiles],
        "judge_model": getattr(judge_llm, "model", ""),
        "a": a,
        "b": b,
        "winner": winner,
        "rationale": rationale,
        "deltas": {
            "groundedness_pct": _delta(a, b, "mean_groundedness_pct"),
            "latency_s": _delta(a, b, "mean_latency_s"),
            "cost_usd": _delta(a, b, "est_cost_usd"),
        },
    }


def _delta(a: dict, b: dict, key: str):
    av, bv = a.get(key), b.get(key)
    if av is None or bv is None:
        return None
    return round(bv - av, 6)


def _decide(a: dict, b: dict) -> tuple:
    """Higher groundedness wins; ties (within 0.02) break on lower cost."""
    ag = a.get("mean_groundedness_pct") or 0
    bg = b.get("mean_groundedness_pct") or 0
    if abs(ag - bg) < 0.02:
        ac = a.get("est_cost_usd") or 0
        bc = b.get("est_cost_usd") or 0
        if ac == bc:
            return "tie", "Groundedness and cost are effectively equal."
        winner = a if ac < bc else b
        return (
            winner["id"],
            f"Groundedness is a tie; {winner['label']} is cheaper.",
        )
    winner = a if ag > bg else b
    return (
        winner["id"],
        f"{winner['label']} has higher groundedness "
        f"({round(max(ag, bg) * 100)}% vs {round(min(ag, bg) * 100)}%).",
    )


if __name__ == "__main__":
    import pprint

    pprint.pp(compare("concise", "detailed"))
