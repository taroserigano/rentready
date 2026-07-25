"""Gate for administrative / expensive-operation routes.

Some routes are developer tooling that must never be driven by an anonymous
internet caller: the ``/evals/*`` suites walk golden sets through Claude (and
``/evals/ragas`` forks a subprocess), and ``/seed-graph`` runs
``MATCH (n) DETACH DELETE n`` before reseeding.

Policy (fail CLOSED for public callers):

  * ``ADMIN_TOKEN`` set   -> require a matching ``X-Admin-Token`` header.
  * ``ADMIN_TOKEN`` unset -> allow LOCAL callers only; public callers get 404.

404 rather than 401/403 is deliberate: an unconfigured deployment should not
advertise that the route exists at all.

"Local" means no trusted-proxy hop identified the caller AND the direct peer
is loopback/private -- true for local dev and the test suite, false for
anything arriving through nginx from the internet (nginx always sets
``X-Real-IP``; see infra/user_data.sh). On the deployed box an operator with
SSH can still reach these over 127.0.0.1, which is the intended escape hatch.
"""

from __future__ import annotations

import ipaddress

from fastapi import HTTPException, Request

from settings import settings


def _is_local_caller(request: Request) -> bool:
    # A trusted proxy identified a real remote client -> not local.
    if request.headers.get("x-real-ip"):
        return False
    host = request.client.host if request.client else ""
    if not host:
        return False
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        # Starlette's TestClient uses the non-IP host "testclient".
        return host == "testclient"
    return ip.is_loopback or ip.is_private


def require_admin(request: Request) -> None:
    """FastAPI dependency. Raises 404 unless the caller is authorized."""
    token = settings.admin_token
    if token:
        supplied = request.headers.get("x-admin-token", "")
        # Compare with a constant-time check so the token can't be recovered
        # byte-by-byte from response timing.
        import hmac

        if not supplied or not hmac.compare_digest(supplied, token):
            raise HTTPException(404, "Not Found")
        return
    if _is_local_caller(request):
        return
    raise HTTPException(404, "Not Found")
