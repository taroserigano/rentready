"""Shared LLM helpers.

Anthropic Claude is used both via LlamaIndex (PDF RAG) and via LangChain
(graph + eligibility). When no ANTHROPIC_API_KEY is set, callers fall back
to mock/heuristic logic so the whole app still runs.
"""

from functools import lru_cache

from settings import settings

# claude-sonnet-5 rejects any explicit `temperature` (400: "`temperature` is
# deprecated for this model") -- only the implicit default is accepted. Older
# models (e.g. Haiku 4.5) still take temperature=0 fine, so this is an
# opt-out per model rather than a blanket removal.
_NO_TEMPERATURE_MODELS = {"claude-sonnet-5"}


def anthropic_chat_kwargs(model: str) -> dict:
    """Extra ChatAnthropic kwargs for `model`, minus any it has deprecated."""
    kwargs: dict = {"max_tokens": 1024}
    if model not in _NO_TEMPERATURE_MODELS:
        kwargs["temperature"] = 0.0
    return kwargs


@lru_cache(maxsize=1)
def get_llamaindex_llm():
    """LLM for LlamaIndex, or None if no key.

    We wrap the LangChain Claude client with LlamaIndex's LangChainLLM
    adapter. LangChain passes the model name straight to the API, so this
    works with any Claude model the key supports (unlike LlamaIndex's
    native Anthropic class, which validates against a hard-coded list).
    """
    chat = get_langchain_llm()
    if chat is None:
        return None
    from llama_index.llms.langchain import LangChainLLM

    return LangChainLLM(llm=chat)


@lru_cache(maxsize=1)
def get_langchain_llm():
    """Anthropic chat model for LangChain, or None if no key."""
    if not settings.has_anthropic:
        return None
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(
        model=settings.anthropic_model,
        api_key=settings.anthropic_api_key,
        **anthropic_chat_kwargs(settings.anthropic_model),
    )
