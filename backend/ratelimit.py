"""A minimal in-process rate limiter for the LLM-backed endpoints.

Every LLM-backed route (chat/ask/recommend) invokes the Anthropic API once
per request with no other throttle in front of it (only /upload has a size
cap). Reachable from the public internet (see infra/), that's a
straightforward cost-DoS: anyone can hammer these routes and run up the bill.

Deliberately simple: a fixed-window counter per client IP, in a plain dict.
No Redis/external store -- this app's only deployed shape (infra/) is a
single EC2 instance running one uvicorn worker, so in-process state is the
real state. This would need a shared store (Redis, DynamoDB) behind a
load balancer or multiple workers; not needed for the current deploy target.
"""

import time

from fastapi import Request
from starlette.responses import JSONResponse

# path prefix -> (max requests, window seconds). Checked in order; a path
# matching none of these is never rate-limited by this middleware.
_LIMITS: list[tuple[str, int, int]] = [
    ("/concierge/ask", 20, 60),
    ("/risk/chat", 20, 60),
    ("/residents/chat", 20, 60),
    ("/tours/chat", 20, 60),
    ("/ask", 20, 60),
    ("/recommend", 20, 60),
    ("/graph-ask", 20, 60),
]

# client_key -> (window_start_epoch, count), one entry per (ip, path prefix).
_counters: dict[tuple[str, str], tuple[float, int]] = {}


def _client_key(request: Request) -> str:
    # Behind nginx (see infra/user_data.sh) there's no X-Forwarded-For hop
    # added, so request.client is the real peer for this deploy target.
    return request.client.host if request.client else "unknown"


async def rate_limit_middleware(request: Request, call_next):
    path = request.url.path
    limit = next((l for l in _LIMITS if path.startswith(l[0])), None)
    if limit is not None:
        prefix, max_requests, window_s = limit
        key = (_client_key(request), prefix)
        now = time.monotonic()
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
