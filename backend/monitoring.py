"""Online (production) monitoring — distinct from offline evaluation.

Eval asks "how good is the system on a fixed golden set?" Monitoring asks
"what is happening to LIVE traffic right now, and is it getting worse?"

We log one telemetry row per served request (see store.log_event) and compute:
  * live ops metrics  -- volume, latency p50/p95, error-ish signals;
  * a quality signal  -- the deterministic faithfulness-tripwire violations;
  * user feedback     -- thumbs up/down rate;
  * drift             -- recent window vs the previous window;
  * alerts            -- threshold rules that fire on the above.

An optional Datadog sink forwards these metrics when a key is configured; it's
a graceful no-op otherwise, so nothing here requires external services.
"""

from settings import settings
import store

# Alert thresholds (kept here so they're easy to find and tune).
THRESHOLDS = {
    "latency_p95_ms": 18000.0,   # explanations call the LLM; warn if slow
    "mean_violations": 0.5,      # avg hallucination-tripwire hits per request
    "down_rate": 0.4,            # fraction of feedback that is thumbs-down
    "drift_latency_pct": 0.5,    # 50% jump in mean latency window-over-window
    "drift_violations_abs": 0.5,  # absolute jump in mean violations
}


def _percentile(values: list, p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return round(s[0], 2)
    k = (len(s) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return round(s[lo] + (s[hi] - s[lo]) * frac, 2)


def _latency_stats(events: list) -> dict:
    lat = [e["latency_ms"] for e in events if e.get("latency_ms") is not None]
    return {
        "p50_ms": _percentile(lat, 0.50),
        "p95_ms": _percentile(lat, 0.95),
        "mean_ms": round(sum(lat) / len(lat), 2) if lat else 0.0,
        "n": len(lat),
    }


def _feedback_stats(feedback: list) -> dict:
    up = sum(1 for f in feedback if f["rating"] == "up")
    down = sum(1 for f in feedback if f["rating"] == "down")
    total = up + down
    return {
        "up": up,
        "down": down,
        "total": total,
        "down_rate": round(down / total, 4) if total else 0.0,
        "satisfaction": round(up / total, 4) if total else None,
    }


def overview(window: int = 200) -> dict:
    events = store.recent_events(window)
    feedback = store.recent_feedback(window)

    by_endpoint: dict = {}
    for e in events:
        by_endpoint[e["endpoint"]] = by_endpoint.get(e["endpoint"], 0) + 1

    rec_events = [e for e in events if e["endpoint"] == "recommend"]
    viol = [e.get("faithfulness_violations", 0) or 0 for e in rec_events]
    mean_viol = round(sum(viol) / len(viol), 4) if viol else 0.0

    return {
        "total_requests": len(events),
        "by_endpoint": by_endpoint,
        "latency": _latency_stats(events),
        "mean_violations": mean_viol,
        "feedback": _feedback_stats(feedback),
    }


def drift() -> dict:
    """Compare the most recent half of events to the previous half."""
    events = store.recent_events(200)
    if len(events) < 8:
        return {"enough_data": False, "n": len(events)}

    # recent_events is newest-first; split into recent vs previous windows.
    half = len(events) // 2
    recent, previous = events[:half], events[half: half * 2]

    def mean_lat(rows):
        lat = [r["latency_ms"] for r in rows if r.get("latency_ms") is not None]
        return sum(lat) / len(lat) if lat else 0.0

    def mean_viol(rows):
        rec = [r for r in rows if r["endpoint"] == "recommend"]
        v = [r.get("faithfulness_violations", 0) or 0 for r in rec]
        return sum(v) / len(v) if v else 0.0

    r_lat, p_lat = mean_lat(recent), mean_lat(previous)
    r_v, p_v = mean_viol(recent), mean_viol(previous)
    lat_change = (r_lat - p_lat) / p_lat if p_lat else 0.0

    return {
        "enough_data": True,
        "window_size": half,
        "latency": {
            "recent_mean_ms": round(r_lat, 2),
            "previous_mean_ms": round(p_lat, 2),
            "pct_change": round(lat_change, 4),
            "drifted": abs(lat_change) >= THRESHOLDS["drift_latency_pct"],
        },
        "violations": {
            "recent_mean": round(r_v, 4),
            "previous_mean": round(p_v, 4),
            "abs_change": round(r_v - p_v, 4),
            "drifted": (r_v - p_v) >= THRESHOLDS["drift_violations_abs"],
        },
    }


def alerts(ov: dict, dr: dict) -> list:
    """Threshold rules over the live metrics + drift. Returns fired alerts."""
    fired = []

    p95 = ov["latency"]["p95_ms"]
    if p95 > THRESHOLDS["latency_p95_ms"]:
        fired.append(
            {
                "level": "warning",
                "metric": "latency_p95_ms",
                "message": f"p95 latency {p95:.0f}ms exceeds "
                f"{THRESHOLDS['latency_p95_ms']:.0f}ms.",
            }
        )

    if ov["mean_violations"] > THRESHOLDS["mean_violations"]:
        fired.append(
            {
                "level": "critical",
                "metric": "mean_violations",
                "message": f"Avg faithfulness violations {ov['mean_violations']} "
                f"exceeds {THRESHOLDS['mean_violations']} — explanations may be "
                "hallucinating.",
            }
        )

    fb = ov["feedback"]
    if fb["total"] >= 3 and fb["down_rate"] > THRESHOLDS["down_rate"]:
        fired.append(
            {
                "level": "warning",
                "metric": "down_rate",
                "message": f"Thumbs-down rate {fb['down_rate']:.0%} over "
                f"{fb['total']} ratings exceeds "
                f"{THRESHOLDS['down_rate']:.0%}.",
            }
        )

    if dr.get("enough_data"):
        if dr["latency"]["drifted"]:
            fired.append(
                {
                    "level": "warning",
                    "metric": "drift_latency",
                    "message": f"Latency drifted "
                    f"{dr['latency']['pct_change']:+.0%} vs the previous window.",
                }
            )
        if dr["violations"]["drifted"]:
            fired.append(
                {
                    "level": "critical",
                    "metric": "drift_violations",
                    "message": "Faithfulness violations rose "
                    f"{dr['violations']['abs_change']:+.2f} vs the previous "
                    "window.",
                }
            )

    return fired


def snapshot() -> dict:
    ov = overview()
    dr = drift()
    return {
        "overview": ov,
        "drift": dr,
        "alerts": alerts(ov, dr),
        "datadog_configured": settings.has_datadog,
        "thresholds": THRESHOLDS,
    }


def push_to_datadog() -> dict:
    """Forward the current live metrics to Datadog (optional, graceful)."""
    if not settings.has_datadog:
        return {
            "skipped": True,
            "reason": "No DATADOG_API_KEY; set it to forward metrics.",
        }
    try:
        import time

        import requests

        ov = overview()
        now = int(time.time())
        series = [
            _series("rentready.requests.total", ov["total_requests"], now),
            _series("rentready.latency.p95_ms", ov["latency"]["p95_ms"], now),
            _series("rentready.latency.p50_ms", ov["latency"]["p50_ms"], now),
            _series("rentready.faithfulness.mean_violations", ov["mean_violations"], now),
            _series("rentready.feedback.down_rate", ov["feedback"]["down_rate"], now),
        ]
        resp = requests.post(
            f"https://api.{settings.datadog_site}/api/v2/series",
            headers={
                "DD-API-KEY": settings.datadog_api_key,
                "Content-Type": "application/json",
            },
            json={"series": series},
            timeout=10,
        )
        return {"skipped": False, "status": resp.status_code, "metrics": len(series)}
    except Exception as exc:  # noqa: BLE001
        return {"skipped": True, "reason": f"{type(exc).__name__}: {exc}"}


def _series(metric: str, value, ts: int) -> dict:
    return {
        "metric": metric,
        "type": 3,  # gauge
        "points": [{"timestamp": ts, "value": float(value or 0)}],
        "tags": ["app:rentready", "env:dev"],
    }
