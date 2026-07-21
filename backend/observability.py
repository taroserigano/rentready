"""Tracing setup: LangSmith (LangChain side) + Phoenix (LlamaIndex side).

Two lenses on the same app:
  - LangSmith   -> the graph / recommendation / eligibility code (LangChain).
  - Arize Phoenix (local) -> the PDF RAG code (LlamaIndex).

Both are best-effort: if a tracer can't start, the app keeps running.
"""

import os
import socket
from urllib.parse import urlparse

from settings import settings

_initialized = False


def _collector_reachable(endpoint: str, timeout: float = 0.5) -> bool:
    """Fast TCP probe of the Phoenix OTLP collector.

    Phoenix's exporter otherwise blocks on connection retries for every span
    when the collector is down — which serially stalls each traced
    ``knowledge.search`` (e.g. a 12-item eval run took ~250s over HTTP). If the
    host:port isn't accepting connections we skip instrumentation entirely, so
    a missing collector costs nothing. When Phoenix is actually running the
    probe succeeds and tracing turns on as before.
    """
    try:
        u = urlparse(endpoint)
        host = u.hostname or "localhost"
        port = u.port or (443 if u.scheme == "https" else 6006)
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:  # noqa: BLE001
        return False


def init_observability() -> dict:
    """Turn on tracing once. Returns a small status dict for /health."""
    global _initialized
    status = {"langsmith": False, "phoenix": False}
    if _initialized:
        return status

    # --- LangSmith: configured purely through env vars that LangChain reads.
    if settings.has_langsmith:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
        os.environ["LANGCHAIN_ENDPOINT"] = settings.langsmith_endpoint
        # Newer SDK names too, for good measure.
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
        status["langsmith"] = True

    # --- Phoenix: local OpenTelemetry collector for LlamaIndex spans.
    if settings.phoenix_enabled and not _collector_reachable(
        settings.phoenix_collector_endpoint
    ):
        print(
            "Phoenix tracing skipped: collector unreachable at "
            f"{settings.phoenix_collector_endpoint} (start it to enable tracing)."
        )
    elif settings.phoenix_enabled:
        try:
            from phoenix.otel import register
            from openinference.instrumentation.llama_index import (
                LlamaIndexInstrumentor,
            )

            tracer_provider = register(
                project_name="rentready",
                endpoint=f"{settings.phoenix_collector_endpoint}/v1/traces",
            )
            LlamaIndexInstrumentor().instrument(
                tracer_provider=tracer_provider
            )
            status["phoenix"] = True
        except Exception as exc:  # noqa: BLE001
            print(f"Phoenix tracing not started: {type(exc).__name__}: {exc}")

    _initialized = True
    return status
