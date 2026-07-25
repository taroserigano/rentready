"""A minimal in-process rate limiter for the expensive endpoints.

Every LLM-backed route (chat/ask/recommend) invokes the Anthropic API once
per request, and every model route runs an XGBoost inference pass. Reachable
from the public internet (see infra/), that's a straightforward cost-DoS:
anyone can hammer these routes and run up the bill.

Deliberately simple: a fixed-window counter per client IP, in a plain dict.
No Redis/external store -- this app's only deployed shape (infra/) is a
single EC2 instance running one uvicorn worker, so in-process state is the
real state. This would need a shared store (Redis, DynamoDB) behind a
load balancer or multiple workers; not needed for the current deploy target.

CLIENT IDENTITY -- the subtle part
---------------------------------
``request.client.host`` is NOT the caller here. nginx terminates the
connection and proxies to 127.0.0.1, so the app sees 127.0.0.1 for every
client on Earth; keying on it puts the whole internet in one bucket, which
turns this protection into an availability DoS (one caller 429s everybody).

Nor can a client-supplied ``X-Forwarded-For`` be trusted: uvicorn enables
``ProxyHeadersMiddleware`` by default and trusts that header from
127.0.0.1, so a spoofed ``X-Forwarded-For`` both rewrites ``request.client``
AND mints a fresh bucket -- i.e. the limiter is bypassable with one header.

So we read ``X-Real-IP``, which nginx sets from ``$remote_addr`` and, being
a single-value header it always overwrites, a client cannot forge. nginx is
also configured to overwrite ``X-Forwarded-For`` (infra/user_data.sh) so the
uvicorn-rewritten ``request.client`` is trustworthy too; ``X-Real-IP`` is
preferred because it is unambiguous (no client-controlled list to parse).
"""

import time

from fastapi import Request
from starlette.responses import JSONResponse

# path prefix -> (max requests, window seconds). Checked in order; a path
# matching none of these is never rate-limited by this middleware.
#
# LLM-backed (one Anthropic call per request -- real money):
_LIMITS: list[tuple[str, int, int]] = [
    ("/concierge/ask", 20, 60),
    ("/risk/chat", 20, 60),
    ("/residents/chat", 20, 60),
    ("/tours/chat", 20, 60),
    ("/graph-ask", 20, 60),
    ("/ask", 20, 60),
    ("/recommend", 20, 60),
    ("/eligibility", 20, 60),      # explain=True -> an Anthropic call
    ("/simulate", 20, 60),         # runs eligibility + recommend
    # Eval suites: each walks a golden set through Claude, and /evals/ragas
    # forks a subprocess. Far more expensive than a normal request, so a
    # much tighter budget on top of the admin-token gate in eval_api.
    ("/evals", 3, 60),
    # Write amplification: each POST persists a row + a generated PDF +
    # embeddings, permanently slowing every O(N) endpoint after it.
    ("/apply", 10, 60),
    ("/upload", 10, 60),
    # Model-inference routes (XGBoost passes; no LLM, so a looser budget).
    ("/risk", 60, 60),
    ("/residents", 60, 60),
    ("/properties", 60, 60),
    ("/dashboard", 60, 60),
]

# client_key -> (window_start_monotonic, count), one entry per (ip, prefix).
_counters: dict[tuple[str, str], tuple[float, int]] = {}

# Bound the dict so a caller rotating spoofed IPs can't grow it without
# limit on a 1GB box. When we cross this, drop every entry whose window has
# already expired (they carry no information); if that isn't enough, clear
# outright -- losing counters fails OPEN for one window, which is the same
# exposure as a restart and far better than an OOM kill.
_MAX_COUNTERS = 10_000


def _client_key(request: Request) -> str:
    """The real caller's IP, from the trusted proxy hop.

    ``X-Real-IP`` is set by nginx from ``$remote_addr``; because it is a
    single-value header that nginx always overwrites, a client cannot forge
    it. Falls back to ``request.client`` for direct (no-proxy) access, e.g.
    local dev and the test suite.
    """
    real_ip = request.headers.get("x-real-ip")
    if real_ip:
        return real_ip.strip()
    return request.client.host if request.client else "unknown"


def _prune(now: float) -> None:
    """Drop expired windows; clear everything if still over the cap."""
    longest_window = max(w for _, _, w in _LIMITS)
    for key in [k for k, (start, _) in _counters.items() if now - start >= longest_window]:
        _counters.pop(key, None)
    if len(_counters) > _MAX_COUNTERS:
        _counters.clear()


async def rate_limit_middleware(request: Request, call_next):
    path = request.url.path
    limit = next((l for l in _LIMITS if path.startswith(l[0])), None)
    if limit is not None:
        prefix, max_requests, window_s = limit
        key = (_client_key(request), prefix)
        now = time.monotonic()
        if len(_counters) > _MAX_COUNTERS:
            _prune(now)
        window_start, count = _counters.get(key, (now, 0))
        if now - window_start >= window_s:
            window_start, count = now, 0
        count += 1
        _counters[key] = (window_start, count)
        if count > max_requests:
            retry_after = max(1, int(window_s - (now - window_start)))
            return JSONResponse(
                {"detail": "Too many requests -- please slow down and try again shortly."},
                status_code=429,
                headers={"Retry-After": str(retry_after)},
            )
    return await call_next(request)
