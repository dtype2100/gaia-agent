"""Code execution node + agent post-routing.

CodeAct 방식: agent 노드가 <code>...python...</code> 블록을 내면 여기서 in-process
exec 한다. 도구는 plain Python function 으로 namespace 에 주입되어 있어
`web_search(query="...")` 처럼 호출 가능. 변수는 state.metadata['py_scope'] 에 살아
다음 turn 까지 보존됨.

설계 메모:
- in-process exec 라서 격리는 없음 (HF Space 컨테이너 안에서 동작). 별도 subprocess
  격리가 필요해지면 후속 step 에서 AST whitelist sandbox 도입.
- stdout/stderr 만 캡쳐해서 ToolMessage(content=...) 로 메시지 로그에 누적.
- 결과가 비면 "[no output]" 힌트를 줘서 모델이 print() 를 빼먹는 패턴 교정.
"""
from __future__ import annotations

import contextlib
import io
import traceback

from langchain_core.messages import AIMessage, ToolMessage

from ..state import GAIAState
from ..tools import ALL_TOOLS


# tool 결과는 보통 길어서 truncate 안 하면 다음 LLM 컨텍스트가 폭주.
_MAX_OUTPUT_CHARS = 14000


def _build_tool_namespace() -> dict:
    """Return a dict mapping tool name → plain Python function wrapping the LangChain tool.

    `exec_python_code` 는 의도적으로 제외 (이미 <code> 블록 자체가 Python 실행).
    """
    ns: dict = {}
    for t in ALL_TOOLS:
        if t.name == "exec_python_code":
            continue

        def _make(tool):
            def _call(**kwargs):
                return tool.invoke(kwargs)

            _call.__name__ = tool.name
            _call.__doc__ = tool.description
            return _call

        ns[t.name] = _make(t)
    return ns


def exec_node(state: GAIAState) -> dict:
    messages = state.get("messages") or []
    if not messages:
        return {}
    last = messages[-1]
    if not isinstance(last, AIMessage):
        return {}
    code = None
    if hasattr(last, "additional_kwargs") and last.additional_kwargs:
        code = last.additional_kwargs.get("code")
    if not code:
        return {}

    metadata = state.get("metadata") or {}
    scope = metadata.get("py_scope")
    if scope is None:
        scope = _build_tool_namespace()

    buf = io.StringIO()
    error_msg = None
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            exec(code, scope)
    except Exception as e:
        tb = traceback.format_exc()
        error_msg = f"{type(e).__name__}: {e}\n{tb}"

    output = buf.getvalue()
    if error_msg:
        output = (output + f"\n[ERROR] {error_msg}").strip()
    if not output.strip():
        output = "[no output — remember to print() what you need to see]"
    if len(output) > _MAX_OUTPUT_CHARS:
        output = output[:_MAX_OUTPUT_CHARS] + "\n...[truncated]"

    new_msg = ToolMessage(content=output, name="exec", tool_call_id="exec_call")
    return {"messages": [new_msg], "metadata": {**metadata, "py_scope": scope}}


def route_after_agent(state: GAIAState) -> str:
    """agent → exec (if <code> emitted) | format (final_answer or formatting failure)."""
    if state.get("raw_answer") is not None:
        return "format"
    messages = state.get("messages") or []
    if messages and isinstance(messages[-1], AIMessage):
        last = messages[-1]
        if (
            hasattr(last, "additional_kwargs")
            and last.additional_kwargs
            and last.additional_kwargs.get("code")
        ):
            return "exec"
    # <code> 도 <final_answer> 도 없으면 모델이 형식을 어긴 것 — format 으로 빠져
    # raw_answer=None → "UNKNOWN" 으로 종결.
    return "format"
