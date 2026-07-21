"""Offline tests for the online-monitoring layer.

We point the store at a temp SQLite DB, seed synthetic production events and
feedback, and assert the aggregation, drift detection, and alert rules behave.
"""

import pytest

import monitoring
import store


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "test.db")
    store.init_db()
    return store


def test_percentile_basic():
    assert monitoring._percentile([10, 20, 30, 40], 0.5) == 25.0
    assert monitoring._percentile([], 0.95) == 0.0
    assert monitoring._percentile([7], 0.95) == 7


def test_overview_aggregates_events_and_feedback(temp_db):
    for ms in (100, 200, 300):
        temp_db.log_event(endpoint="recommend", latency_ms=ms, faithfulness_violations=0)
    temp_db.log_event(endpoint="eligibility", latency_ms=50)
    temp_db.save_feedback("a1", "recommendation", "up")
    temp_db.save_feedback("a1", "recommendation", "down")

    ov = monitoring.overview()
    assert ov["total_requests"] == 4
    assert ov["by_endpoint"]["recommend"] == 3
    assert ov["by_endpoint"]["eligibility"] == 1
    assert ov["latency"]["p50_ms"] > 0
    assert ov["feedback"]["up"] == 1
    assert ov["feedback"]["down"] == 1
    assert ov["feedback"]["down_rate"] == 0.5


def test_alert_fires_on_high_violations(temp_db):
    for _ in range(5):
        temp_db.log_event(endpoint="recommend", latency_ms=100, faithfulness_violations=2)
    ov = monitoring.overview()
    fired = monitoring.alerts(ov, {"enough_data": False})
    metrics = {a["metric"] for a in fired}
    assert "mean_violations" in metrics
    assert any(a["level"] == "critical" for a in fired)


def test_alert_fires_on_high_down_rate(temp_db):
    for _ in range(4):
        temp_db.save_feedback("a1", "recommendation", "down")
    temp_db.save_feedback("a1", "recommendation", "up")
    ov = monitoring.overview()
    fired = monitoring.alerts(ov, {"enough_data": False})
    assert any(a["metric"] == "down_rate" for a in fired)


def test_no_alerts_when_healthy(temp_db):
    for _ in range(5):
        temp_db.log_event(endpoint="recommend", latency_ms=200, faithfulness_violations=0)
    ov = monitoring.overview()
    assert monitoring.alerts(ov, {"enough_data": False}) == []


def test_drift_needs_enough_data(temp_db):
    temp_db.log_event(endpoint="recommend", latency_ms=100)
    assert monitoring.drift()["enough_data"] is False


def test_drift_detects_violation_jump(temp_db):
    # Older window (clean), then recent window (hallucinating). recent_events is
    # newest-first, so insert the clean batch first, the bad batch last.
    for _ in range(6):
        temp_db.log_event(endpoint="recommend", latency_ms=100, faithfulness_violations=0)
    for _ in range(6):
        temp_db.log_event(endpoint="recommend", latency_ms=100, faithfulness_violations=2)
    dr = monitoring.drift()
    assert dr["enough_data"] is True
    assert dr["violations"]["drifted"] is True


def test_push_to_datadog_skips_without_key(temp_db, monkeypatch):
    monkeypatch.setattr(monitoring.settings, "datadog_api_key", "")
    out = monitoring.push_to_datadog()
    assert out["skipped"] is True
