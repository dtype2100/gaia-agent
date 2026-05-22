"""tool_executor 노드 + agent 이후 라우팅 함수.

agent 노드가 만든 마지막 AIMessage 의 tool_calls 를 모두 실행해 ToolMessage 로
누적한다. route_after_agent 는 agent → (tools | format) 분기 결정.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, ToolMessage

from ..state import GAIAState
from ..tools import ALL_TOOLS


_TOOL_MAP = {t.name: t for t in ALL_TOOLS}
# tool 결과는 보통 길어서 truncate 안 하면 다음 LLM 컨텍스트가 폭주.
_MAX_TOOL_RESULT_CHARS = 14000


def tool_executor_node(state: GAIAState) -> dict:
    messages = state.get("messages") or []
    if not messages:
        return {}
    last = messages[-1]
    if not isinstance(last, AIMessage):
        return {}
    tool_calls = getattr(last, "tool_calls", None) or []
    if not tool_calls:
        return {}

    new_msgs = []
    for tc in tool_calls:
        name = tc.get("name", "")
        args = tc.get("args") or {}
        tid = tc.get("id") or name
        if name not in _TOOL_MAP:
            content = (
                f"Tool not found: {name!r}. "
                f"Available: {sorted(_TOOL_MAP.keys())}"
            )
        else:
            tool = _TOOL_MAP[name]
            try:
                # LangChain BaseTool.invoke 는 dict 또는 단일 인자를 받는다.
                result = tool.invoke(args)
                content = str(result)
            except Exception as e:
                content = f"Tool {name} error: {type(e).__name__}: {e}"
        if len(content) > _MAX_TOOL_RESULT_CHARS:
            content = content[:_MAX_TOOL_RESULT_CHARS] + "\n...[truncated]"
        new_msgs.append(ToolMessage(content=content, name=name, tool_call_id=tid))

    return {"messages": new_msgs}


def route_after_agent(state: GAIAState) -> str:
    """agent → tools (tool 호출 있으면) 또는 format (final_answer / 예산 소진)."""
    if state.get("raw_answer") is not None:
        return "format"
    messages = state.get("messages") or []
    if messages and isinstance(messages[-1], AIMessage):
        if getattr(messages[-1], "tool_calls", None):
            return "tools"
    # tool_call 도 final_answer 도 없으면 모델이 형식을 어긴 것 — 한 번 더 돌게 둘
    # 수 있지만 무한루프 방지 위해 step_count 가드는 agent_node 가 책임. 여기선
    # tool 호출이 없으면 format 으로 빠지게 두어 그래프가 멈춘다.
    return "format"
