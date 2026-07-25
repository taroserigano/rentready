"""Tests for the shared LLM client helpers (llm.py)."""

import llm


def test_anthropic_chat_kwargs_sets_a_bounded_timeout():
    """Regression: ChatAnthropic previously had no explicit timeout, so a
    stalled Anthropic call (network hang, corporate TLS proxy hiccup) could
    block a synchronous FastAPI request thread for the SDK's own default
    (several minutes) instead of degrading like the rest of the app."""
    kwargs = llm.anthropic_chat_kwargs("claude-sonnet-5")
    assert kwargs.get("timeout") is not None
    assert 0 < kwargs["timeout"] <= 60


def test_anthropic_chat_kwargs_omits_temperature_for_deprecated_models():
    kwargs = llm.anthropic_chat_kwargs("claude-sonnet-5")
    assert "temperature" not in kwargs

    kwargs_other = llm.anthropic_chat_kwargs("claude-haiku-4-5")
    assert kwargs_other["temperature"] == 0.0


def test_anthropic_chat_kwargs_omits_temperature_for_dated_snapshot_ids():
    """Regression: an exact-string check against "claude-sonnet-5" alone
    misses a dated snapshot id like "claude-sonnet-5-20250929" -- temperature
    would silently get added back and the real Anthropic API would 400."""
    kwargs = llm.anthropic_chat_kwargs("claude-sonnet-5-20250929")
    assert "temperature" not in kwargs

    # Must not over-match an unrelated model that merely shares a prefix.
    kwargs_unrelated = llm.anthropic_chat_kwargs("claude-sonnet-5x")
    assert kwargs_unrelated["temperature"] == 0.0
