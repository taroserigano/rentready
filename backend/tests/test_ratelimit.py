"""Regression tests for ratelimit.py and the admin gate.

These exist because the original limiter had three defects that no test caught
(the module had zero test coverage):

  1. It keyed on ``request.client.host``, which behind nginx is always
     127.0.0.1 -- putting every client on Earth in ONE bucket, so a single
     caller could 429 everybody.
  2. uvicorn trusts ``X-Forwarded-For`` from 127.0.0.1 and nginx did not
     overwrite it, so a spoofed header minted a fresh bucket -- the limiter
     was bypassable with one header. Verified live against the deployed box
     before the fix: rotating X-Forwarded-For sailed straight through.
  3. ``_counters`` was never pruned, so spoofed IPs grew it without bound.
"""

import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import ratelimit
from admin_auth import require_admin


@pytest.fixture(autouse=True)
def _clear_counters():
    """The limiter's state is a module global -- isolate every test."""
    ratelimit._counters.clear()
    yield
    ratelimit._counters.clear()


@pytest.fixture
def app():
    a = FastAPI()
    a.middleware("http")(ratelimit.rate_limit_middleware)

    @a.post("/concierge/ask")
    def ask() -> dict:
        return {"ok": True}

    @a.get("/unlimited")
    def unlimited() -> dict:
        return {"ok": True}

    return a


def _limit_for(prefix: str) -> int:
    return next(mx for p, mx, _ in ratelimit._LIMITS if p == prefix)


def test_limit_fires_after_the_configured_budget(app):
    limit = _limit_for("/concierge/ask")
    with TestClient(app) as c:
        codes = [c.post("/concierge/ask").status_code for _ in range(limit + 3)]
    assert codes[:limit] == [200] * limit
    assert codes[limit:] == [429, 429, 429]


def test_429_carries_retry_after(app):
    limit = _limit_for("/concierge/ask")
    with TestClient(app) as c:
        for _ in range(limit):
            c.post("/concierge/ask")
        resp = c.post("/concierge/ask")
    assert resp.status_code == 429
    assert int(resp.headers["Retry-After"]) >= 1


def test_unlisted_paths_are_never_limited(app):
    with TestClient(app) as c:
        codes = {c.get("/unlimited").status_code for _ in range(60)}
    assert codes == {200}


def test_buckets_are_per_ip_not_shared_globally(app):
    """THE bug that made this protection an availability DoS.

    Two different callers must not share a budget: exhausting one must leave
    the other completely unaffected."""
    limit = _limit_for("/concierge/ask")
    with TestClient(app) as c:
        for _ in range(limit + 2):
            c.post("/concierge/ask", headers={"X-Real-IP": "203.0.113.9"})
        assert c.post("/concierge/ask", headers={"X-Real-IP": "203.0.113.9"}).status_code == 429
        # A different client still has its full budget.
        other = c.post("/concierge/ask", headers={"X-Real-IP": "198.51.100.4"})
    assert other.status_code == 200


def test_spoofed_x_forwarded_for_cannot_mint_a_fresh_bucket(app):
    """THE bypass. X-Real-IP is set by nginx and always overwritten, so it
    identifies the caller even when the client sends its own X-Forwarded-For."""
    limit = _limit_for("/concierge/ask")
    hdr = {"X-Real-IP": "203.0.113.9"}
    with TestClient(app) as c:
        for _ in range(limit):
            c.post("/concierge/ask", headers=hdr)
        # Same real caller, rotating a forged X-Forwarded-For each time.
        codes = [
            c.post(
                "/concierge/ask",
                headers={**hdr, "X-Forwarded-For": f"9.9.9.{i}"},
            ).status_code
            for i in range(5)
        ]
    assert codes == [429] * 5, "forged X-Forwarded-For bypassed the limiter"


def test_expensive_llm_and_admin_routes_are_all_covered():
    """Every route that costs money or CPU must match a limit prefix."""
    must_be_limited = [
        "/concierge/ask", "/risk/chat", "/residents/chat", "/tours/chat",
        "/graph-ask", "/ask", "/recommend", "/eligibility/abc", "/simulate",
        "/evals/run", "/apply", "/upload",
        "/risk", "/residents", "/properties", "/dashboard/stats",
    ]
    for path in must_be_limited:
        assert any(path.startswith(p) for p, _, _ in ratelimit._LIMITS), path


def test_eval_routes_get_a_much_tighter_budget_than_chat():
    """An /evals run walks a whole golden set through Claude (and /ragas forks
    a subprocess), so it must not share the ordinary chat budget."""
    assert _limit_for("/evals") < _limit_for("/concierge/ask")


def test_counters_are_pruned_and_bounded(app):
    """Expired windows are dropped, so rotating IPs can't grow the dict
    without bound on a 1GB box."""
    now = time.monotonic()
    longest = max(w for _, _, w in ratelimit._LIMITS)
    # Seed well past the cap with windows that have already expired.
    for i in range(ratelimit._MAX_COUNTERS + 50):
        ratelimit._counters[(f"10.0.{i // 256}.{i % 256}", "/ask")] = (now - longest - 1, 1)
    ratelimit._prune(now)
    assert len(ratelimit._counters) <= ratelimit._MAX_COUNTERS


def test_window_rolls_over_and_restores_the_budget(app, monkeypatch):
    limit = _limit_for("/concierge/ask")
    with TestClient(app) as c:
        for _ in range(limit):
            c.post("/concierge/ask")
        assert c.post("/concierge/ask").status_code == 429
        # Jump past the window instead of sleeping through it.
        real = time.monotonic
        window = next(w for p, _, w in ratelimit._LIMITS if p == "/concierge/ask")
        monkeypatch.setattr(time, "monotonic", lambda: real() + window + 1)
        assert c.post("/concierge/ask").status_code == 200


# ---------------------------------------------------------------------------
# Admin gate (admin_auth.require_admin)
# ---------------------------------------------------------------------------
@pytest.fixture
def admin_app():
    from fastapi import Depends

    a = FastAPI()

    @a.post("/evals/run", dependencies=[Depends(require_admin)])
    def run() -> dict:
        return {"ok": True}

    return a


def test_admin_route_allows_local_caller_when_no_token_set(admin_app, monkeypatch):
    from settings import settings

    monkeypatch.setattr(settings, "admin_token", "")
    with TestClient(admin_app) as c:
        # TestClient presents as a local peer.
        assert c.post("/evals/run").status_code == 200


def test_admin_route_404s_a_public_caller_when_no_token_set(admin_app, monkeypatch):
    """A real internet request always arrives with X-Real-IP (nginx sets it),
    which marks it non-local. 404 rather than 403 so an unconfigured deploy
    doesn't advertise the route."""
    from settings import settings

    monkeypatch.setattr(settings, "admin_token", "")
    with TestClient(admin_app) as c:
        resp = c.post("/evals/run", headers={"X-Real-IP": "203.0.113.9"})
    assert resp.status_code == 404


def test_admin_route_requires_the_token_when_configured(admin_app, monkeypatch):
    from settings import settings

    monkeypatch.setattr(settings, "admin_token", "s3cret")
    with TestClient(admin_app) as c:
        pub = {"X-Real-IP": "203.0.113.9"}
        assert c.post("/evals/run", headers=pub).status_code == 404
        assert c.post("/evals/run", headers={**pub, "X-Admin-Token": "wrong"}).status_code == 404
        assert c.post("/evals/run", headers={**pub, "X-Admin-Token": "s3cret"}).status_code == 200


def test_configured_token_also_gates_local_callers(admin_app, monkeypatch):
    """Once a token is set it is the ONLY way in -- being local is not a
    second, weaker credential."""
    from settings import settings

    monkeypatch.setattr(settings, "admin_token", "s3cret")
    with TestClient(admin_app) as c:
        assert c.post("/evals/run").status_code == 404
