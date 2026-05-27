"""StateGraph 조립.

흐름:
    START → decompose → agent ⇄ exec → format → coerce → END

agent ↔ exec 는 conditional edge 로 묶여, <code> 블록이 있으면 exec, 아니면 format.
agent 노드 안에서 step_count 가드 + final_answer 설정.
"""
from __future__ import annotations

from langgraph.graph import END, StateGraph

from .nodes import (
    agent_node,
    coerce_node,
    decompose_node,
    exec_node,
    format_pass_node,
    route_after_agent,
)
from .state import GAIAState


def build_graph():
    g = StateGraph(GAIAState)
    g.add_node("decompose", decompose_node)
    g.add_node("agent", agent_node)
    g.add_node("exec", exec_node)
    g.add_node("format", format_pass_node)
    g.add_node("coerce", coerce_node)

    g.set_entry_point("decompose")
    g.add_edge("decompose", "agent")
    g.add_conditional_edges(
        "agent",
        route_after_agent,
        {"exec": "exec", "format": "format"},
    )
    g.add_edge("exec", "agent")
    g.add_edge("format", "coerce")
    g.add_edge("coerce", END)

    return g.compile()
