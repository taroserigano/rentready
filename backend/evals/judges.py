"""LLM-as-judge evaluators with guardrails + a deterministic faithfulness check.

Two complementary layers live here:

1. Deterministic faithfulness (no LLM, runs in CI). Scans a recommendation's
   written ``match_reason`` for community amenities it claims the property HAS
   but that are not in the property's amenity list. This catches the most
   common hallucination -- inventing a feature -- cheaply and reproducibly.

2. LLM-as-judge (needs ANTHROPIC_API_KEY; skipped offline). Claude grades two
   things our deterministic checks can't: how GROUNDED a recommendation
   explanation is in the supplied facts, and whether an eligibility
   explanation is CONSISTENT with the (rule-decided) verdict.

Guardrails on the judge -- this is the "evaluation practice" point:
  * temperature 0 (already set on the shared client) for repeatable grades;
  * JSON-only structured output that we parse, validate, and clamp;
  * reference-anchored prompts -- the judge only sees the facts we pass, so it
    grades against ground truth rather than its own world knowledge;
  * it scores FREE TEXT only -- it never changes a verdict or a ranking;
  * graceful skip when no key is present, so CI stays hermetic.
"""

import json
import re
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

import eligibility  # noqa: E402
import graphrag  # noqa: E402
import pdf_ingest  # noqa: E402
import rag_llamaindex  # noqa: E402
from llm import get_langchain_llm  # noqa: E402

APPLICATIONS_DIR = BACKEND.parent / "data" / "applications"

# Slugs used for the live judge run (kept small to bound cost/latency).
DEFAULT_SLUGS = ["jordan-rivera", "alex-chen", "sam-patel"]

# Community amenities that a recommendation might claim. Structural features
# (balcony/parking/laundry) aren't carried on the recommendation object, so we
# only check this vocabulary -- the LLM judge covers the rest.
_AMENITY_VOCAB = [
    "Gym",
    "Pool",
    "Pet Park",
    "Rooftop Deck",
    "Bike Storage",
    "Concierge",
    "Playground",
]

# "claims to HAVE it" phrasing vs. negation. We only flag POSITIVE possession
# of a feature the property lacks ("features a gym"); absence phrasing
# ("has no gym", "no pool on-site") is correct, not a hallucination.
_POSSESSION = re.compile(
    r"\b(?:has|have|with|includes?|including|offers?|featur\w*|boasts?|"
    r"comes with|on[- ]site|access to|enjoy|amenities)\b"
)
_NEGATION = re.compile(
    r"\b(?:no|not|without|lacks?|missing|don'?t|doesn'?t|isn'?t|aren'?t|never)\b"
)


def faithfulness_violations(reason: str, amenities: list) -> list:
    """Amenities the prose claims the property HAS but that aren't present.

    Deterministic and offline -- a cheap hallucination tripwire that runs in CI.
    For each amenity the property lacks, we flag it only if the words just
    before a mention positively assert possession and aren't negated.
    """
    if not reason:
        return []
    have = {a.lower() for a in (amenities or [])}
    low = reason.lower()
    violations = []
    for amenity in _AMENITY_VOCAB:
        a = amenity.lower()
        if a in have or a not in low:
            continue
        for m in re.finditer(re.escape(a), low):
            window = low[max(0, m.start() - 50):m.start()]
            if _POSSESSION.search(window) and not _NEGATION.search(window):
                violations.append(amenity)
                break
    return violations


def _first_json_obj(s: str) -> dict:
    match = re.search(r"\{.*\}", s or "", re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def _clamp_score(value, lo: int = 1, hi: int = 5):
    try:
        v = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    return max(lo, min(hi, v))


def _invoke(llm, system: str, human: str) -> str:
    raw = llm.invoke([("system", system), ("human", human)]).content
    if isinstance(raw, list):  # some providers return content blocks
        raw = "".join(str(b) for b in raw)
    return raw


_REC_JUDGE_SYSTEM = (
    "You are a strict QA reviewer for an apartment recommender. You are given "
    "an applicant, a property's FACTS, and the assistant's written explanation "
    "of why the property fits. Judge ONLY whether the explanation is GROUNDED "
    "and FAITHFUL to the facts -- not whether the match is good, and not "
    "completeness.\n"
    "IMPORTANT: Only statements the explanation ACTUALLY makes can be "
    "violations. Omissions are NOT violations -- it is fine to leave facts out. "
    "A claim counts as supported if it is consistent with the facts (e.g. "
    "'has a balcony' when has_balcony is true; 'covered parking' when "
    "parking_type says covered).\n"
    "Score 1-5: 5 = every stated claim is supported by the facts; 3 = mostly "
    "grounded with a vague or minor unsupported bit; 1 = invents amenities/"
    "prices/features or contradicts the facts.\n"
    "Return ONLY JSON: "
    '{"score": <1-5 int>, "reason": "<one sentence>", '
    '"violations": ["<false claim the text makes>", ...]}'
)


def judge_recommendation(profile: dict, rec: dict, llm=None) -> dict:
    """Grade how grounded a single recommendation explanation is (1-5)."""
    llm = llm or get_langchain_llm()
    facts = {
        "name": rec.get("name"),
        "area": rec.get("area"),
        "property_type": rec.get("property_type"),
        "monthly_rent": rec.get("monthly_rent"),
        "bedrooms": rec.get("bedrooms"),
        "bathrooms": rec.get("bathrooms"),
        "bathroom_type": rec.get("bathroom_type"),
        "square_feet": rec.get("square_feet"),
        "has_balcony": rec.get("has_balcony"),
        "in_unit_laundry": rec.get("in_unit_laundry"),
        "pets_allowed": rec.get("pets_allowed"),
        "parking_type": rec.get("parking_type"),
        "walk_score": rec.get("walk_score"),
        "transit_score": rec.get("transit_score"),
        "amenities": rec.get("amenities", []),
    }
    human = (
        f"Applicant (JSON):\n{json.dumps(profile)}\n\n"
        f"Property FACTS (the only ground truth):\n{json.dumps(facts)}\n\n"
        f"Assistant explanation:\n\"{rec.get('match_reason', '')}\"\n\n"
        "Return the JSON now."
    )
    obj = _first_json_obj(_invoke(llm, _REC_JUDGE_SYSTEM, human))
    score = _clamp_score(obj.get("score"))
    violations = obj.get("violations") or []
    if not isinstance(violations, list):
        violations = [str(violations)]
    return {
        "score": score,
        "reason": str(obj.get("reason", "")),
        "violations": [str(v) for v in violations],
    }


_ELIG_JUDGE_SYSTEM = (
    "You are auditing a rental eligibility decision. The VERDICT is fixed by "
    "deterministic rules; your job is only to check the written explanation. "
    "Is the explanation CONSISTENT with the verdict and the stated facts (it "
    "must not contradict the verdict or invent facts)?\n"
    "Return ONLY JSON: "
    '{"consistent": <true|false>, "score": <1-5 int>, "reason": "<one sentence>"}'
)


def judge_eligibility(
    profile: dict, verdict: str, explanation: str, reasons: list, llm=None
) -> dict:
    """Check that an eligibility explanation agrees with its verdict."""
    llm = llm or get_langchain_llm()
    human = (
        f"Applicant: {profile.get('name')}\n"
        f"VERDICT (fixed): {verdict}\n"
        f"Facts the rules used: {'; '.join(reasons)}\n\n"
        f"Explanation shown to the applicant:\n\"{explanation}\"\n\n"
        "Return the JSON now."
    )
    obj = _first_json_obj(_invoke(llm, _ELIG_JUDGE_SYSTEM, human))
    return {
        "consistent": bool(obj.get("consistent", False)),
        "score": _clamp_score(obj.get("score")),
        "reason": str(obj.get("reason", "")),
    }


def _mean(xs: list):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def run_judge_suite(slugs: list = None, max_recs_per: int = 2) -> dict:
    """Live LLM-as-judge over sample applicants. Skips cleanly without a key."""
    llm = get_langchain_llm()
    if llm is None:
        return {
            "skipped": True,
            "reason": "No ANTHROPIC_API_KEY; the LLM judge needs a model.",
        }

    slugs = slugs or DEFAULT_SLUGS
    rec_scores, rec_cases = [], []
    elig_scores, elig_consistent, elig_cases = [], [], []
    total_violations = 0
    deterministic_violations = 0

    for slug in slugs:
        pdf = APPLICATIONS_DIR / f"{slug}.pdf"
        if not pdf.exists():
            continue
        text = pdf_ingest.extract_text(str(pdf))
        profile = rag_llamaindex._heuristic_profile(text)
        pdict = profile.model_dump()

        result = graphrag.recommend(profile)
        for rec in result.get("recommendations", [])[:max_recs_per]:
            rd = rec.model_dump()
            verdict = judge_recommendation(pdict, rd, llm)
            det = faithfulness_violations(rd["match_reason"], rd["amenities"])
            deterministic_violations += len(det)
            total_violations += len(verdict["violations"])
            if verdict["score"] is not None:
                rec_scores.append(verdict["score"])
            rec_cases.append(
                {
                    "slug": slug,
                    "property": rd["name"],
                    "score": verdict["score"],
                    "reason": verdict["reason"],
                    "violations": verdict["violations"],
                    "deterministic_violations": det,
                }
            )

        elig = eligibility.evaluate(profile, explain=True)
        verdict = judge_eligibility(
            pdict, elig.verdict, elig.explanation, elig.reasons, llm
        )
        if verdict["score"] is not None:
            elig_scores.append(verdict["score"])
        elig_consistent.append(verdict["consistent"])
        elig_cases.append(
            {
                "slug": slug,
                "verdict": elig.verdict,
                "consistent": verdict["consistent"],
                "score": verdict["score"],
                "reason": verdict["reason"],
            }
        )

    consistency_rate = (
        round(sum(elig_consistent) / len(elig_consistent), 4)
        if elig_consistent
        else None
    )
    return {
        "skipped": False,
        "model": getattr(llm, "model", ""),
        "recommendation": {
            "mean_groundedness": _mean(rec_scores),
            "mean_groundedness_pct": (
                round(_mean(rec_scores) / 5, 4) if rec_scores else None
            ),
            "judge_violations": total_violations,
            "deterministic_violations": deterministic_violations,
            "n": len(rec_cases),
            "per_case": rec_cases,
        },
        "eligibility": {
            "mean_score": _mean(elig_scores),
            "consistency_rate": consistency_rate,
            "n": len(elig_cases),
            "per_case": elig_cases,
        },
    }


if __name__ == "__main__":
    import pprint

    pprint.pp(run_judge_suite())
