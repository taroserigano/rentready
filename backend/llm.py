"""Shared LLM helpers.

Anthropic Claude is used both via LlamaIndex (PDF RAG) and via LangChain
(graph + eligibility). When no ANTHROPIC_API_KEY is set, callers fall back
to mock/heuristic logic so the whole app still runs.
"""

from functools import lru_cache

from settings import settings


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
        temperature=0.0,
        max_tokens=1024,
    )
