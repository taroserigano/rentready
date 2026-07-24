"""A minimal, real LangGraph wiring for the Tour Scheduler's two-stage flow:
resolve the property, then run the booking state machine.

Kept as its own module so ``tours_chat.py``'s deterministic logic stays
framework-free and independently testable -- this module just assembles two
existing functions into an actual compiled ``StateGraph`` and invokes it.
Every ``/tours/chat`` request runs through this graph (see
``tours_chat._handle``); on any error the caller's own try/except degrades to
a templated reply, same as before this existed.
"""

from __future__ import annotations

from typing import Callable, TypedDict

from langgraph.graph import END, StateGraph

from models import ChatState, TourChatRequest, TourChatResponse

ResolveFn = Callable[[TourChatRequest, ChatState], "tuple[dict | None, TourChatResponse | None]"]
RunFlowFn = Callable[[TourChatRequest, ChatState, object, dict], TourChatResponse]


class _FlowState(TypedDict):
    req: TourChatRequest
    chat_state: ChatState
    now: object  # datetime; kept loose so the graph schema doesn't need to model it
    prop: dict | None
    response: TourChatResponse | None


_GRAPH_CACHE: dict[tuple[ResolveFn, RunFlowFn], object] = {}


def _build_graph(resolve_property: ResolveFn, run_flow: RunFlowFn):
    def resolve_node(state: _FlowState) -> dict:
        prop, early = resolve_property(state["req"], state["chat_state"])
        return {"prop": prop, "response": early}

    def run_flow_node(state: _FlowState) -> dict:
        resp = run_flow(state["req"], state["chat_state"], state["now"], state["prop"])
        return {"response": resp}

    def route(state: _FlowState) -> str:
        return END if state["response"] is not None else "run_flow"

    g = StateGraph(_FlowState)
    g.add_node("resolve_property", resolve_node)
    g.add_node("run_flow", run_flow_node)
    g.set_entry_point("resolve_property")
    g.add_conditional_edges("resolve_property", route, {"run_flow": "run_flow", END: END})
    g.add_edge("run_flow", END)
    return g.compile()


def run(
    resolve_property: ResolveFn,
    run_flow: RunFlowFn,
    req: TourChatRequest,
    chat_state: ChatState,
    now,
) -> TourChatResponse:
    """Build (once per function pair, then cached) and invoke the graph."""
    key = (resolve_property, run_flow)
    app = _GRAPH_CACHE.get(key)
    if app is None:
        app = _build_graph(resolve_property, run_flow)
        _GRAPH_CACHE[key] = app
    result = app.invoke(
        {"req": req, "chat_state": chat_state, "now": now, "prop": None, "response": None}
    )
    return result["response"]
