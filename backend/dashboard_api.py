"""API endpoints for the dashboard home page.

One aggregate endpoint that answers "how is the system doing?" in a single
call: applicant counts, eligibility verdict breakdown, property inventory,
request traffic, and feedback sentiment. Everything is computed from data we
already store (SQLite + properties.json) -- no LLM calls, so it is fast and
deterministic. The eligibility verdict per applicant comes from the rules
engine with explain=False, which skips the Claude explanation step.
"""

from fastapi import APIRouter

import eligibility
import graph
import store

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


def _applicant_stats() -> tuple[int, dict, list]:
    """Totals, verdict counts, and the newest 5 applicants with verdicts.

    Applicants whose profile can't be loaded (e.g. deleted mid-request) are
    skipped rather than failing the whole endpoint.
    """
    applicants = store.list_applicants()  # already newest-first
    verdicts = {"qualified": 0, "needs_review": 0, "not_qualified": 0}
    recent = []
    total = 0
    for row in applicants:
        profile = store.get_profile(row["id"])
        if profile is None:
            continue
        try:
            verdict = eligibility.evaluate(profile, explain=False).verdict
        except Exception:  # noqa: BLE001 -- one bad profile shouldn't 500
            continue
        total += 1
        verdicts[verdict] = verdicts.get(verdict, 0) + 1
        if len(recent) < 5:
            recent.append(
                {
                    "id": row["id"],
                    "name": row["name"],
                    "created_at": row["created_at"],
                    "verdict": verdict,
                }
            )
    return total, verdicts, recent


def _property_stats() -> dict:
    props = graph.load_properties()
    rents = [p.get("monthly_rent") or 0 for p in props]
    by_area: dict = {}
    for p in props:
        name = (p.get("neighborhood") or {}).get("name")
        if name:
            by_area[name] = by_area.get(name, 0) + 1
    return {
        "total": len(props),
        "rent_min": min(rents) if rents else 0,
        "rent_max": max(rents) if rents else 0,
        "pets_allowed": sum(1 for p in props if p.get("pets_allowed")),
        "areas": len(by_area),
        "by_area": dict(sorted(by_area.items(), key=lambda kv: -kv[1])),
    }



# A streamed SSE response's true "latency" is measured to its natural
# completion — a request genuinely stuck this long isn't realistic, so a
# reading past this ceiling is teardown/idle time (e.g. an abandoned
# connection), not backend work. Excluded from the average, never from
# events_total, so one bad row can't silently corrupt the whole chart.
_MAX_SANE_LATENCY_MS = 60_000  # 60s


def _traffic_stats() -> dict:
    events = store.recent_events(500)
    by_endpoint: dict = {}
    requests_by_endpoint: dict = {}
    violations = 0
    outliers_excluded = 0
    for e in events:
        violations += int(e.get("faithfulness_violations") or 0)
        requests_by_endpoint[e["endpoint"]] = requests_by_endpoint.get(e["endpoint"], 0) + 1
        latency = e.get("latency_ms")
        if latency is None:
            continue
        if float(latency) > _MAX_SANE_LATENCY_MS:
            outliers_excluded += 1
            continue
        bucket = by_endpoint.setdefault(e["endpoint"], [0.0, 0])
        bucket[0] += float(latency)
        bucket[1] += 1
    return {
        "events_total": len(events),
        "avg_latency_ms_by_endpoint": {
            ep: round(total / count) for ep, (total, count) in by_endpoint.items()
        },
        "requests_by_endpoint": dict(
            sorted(requests_by_endpoint.items(), key=lambda kv: -kv[1])
        ),
        "outliers_excluded": outliers_excluded,
        "faithfulness_violations": violations,
    }


def _feedback_stats() -> dict:
    rows = store.recent_feedback(500)
    return {
        "up": sum(1 for r in rows if r.get("rating") == "up"),
        "down": sum(1 for r in rows if r.get("rating") == "down"),
    }


@router.get("/stats")
def stats() -> dict:
    """At-a-glance system stats for the dashboard home page."""
    total, verdicts, recent = _applicant_stats()
    return {
        "applicants_total": total,
        "verdicts": verdicts,
        "recent_applicants": recent,
        "properties": _property_stats(),
        "traffic": _traffic_stats(),
        "feedback": _feedback_stats(),
    }
