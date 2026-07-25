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


def _rejects_temperature(model: str) -> bool:
    """True if `model` is (or is a dated snapshot of) a model in
    _NO_TEMPERATURE_MODELS. An exact-string check alone misses dated ids like
    "claude-sonnet-5-20250929", which would otherwise silently get
    temperature=0.0 added back and 400 against the real API."""
    return any(
        model == base or model.startswith(base + "-") for base in _NO_TEMPERATURE_MODELS
    )


def anthropic_chat_kwargs(model: str) -> dict:
    """Extra ChatAnthropic kwargs for `model`, minus any it has deprecated."""
    kwargs: dict = {
        "max_tokens": 1024,
        # Every LLM-backed route runs synchronously in a FastAPI request
        # thread; ChatAnthropic's own default timeout is otherwise the
        # Anthropic SDK's (several minutes), so a stalled call — a network
        # hang, a corporate TLS proxy hiccup — ties up a worker thread far
        # longer than any caller will wait. 30s is generous for one
        # completion but keeps a hang bounded and turned into a normal
        # "unavailable" fallback instead.
        "timeout": 30.0,
    }
    if not _rejects_temperature(model):
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
    """Anthropic chat model for LangChain, or None if no key.

    For models in _NO_TEMPERATURE_MODELS, the returned instance also refuses
    any *later* attempt to set `.temperature` -- omitting the kwarg at
    construction only stops OUR code from sending it; third-party wrappers
    (e.g. ragas.llms.LangchainLLMWrapper, which unconditionally does
    `self.langchain_llm.temperature = value` before every call) can still
    reintroduce the deprecated param, get a 400 back, and silently swallow it
    as a NaN score.
    """
    if not settings.has_anthropic:
        return None
    from langchain_anthropic import ChatAnthropic as _ChatAnthropic

    model = settings.anthropic_model
    cls = _ChatAnthropic
    if _rejects_temperature(model):
        class _NoTemperatureChatAnthropic(_ChatAnthropic):
            def __setattr__(self, name, value):
                if name == "temperature":
                    return
                super().__setattr__(name, value)

        cls = _NoTemperatureChatAnthropic

    return cls(
        model=model,
        api_key=settings.anthropic_api_key,
        **anthropic_chat_kwargs(model),
    )
