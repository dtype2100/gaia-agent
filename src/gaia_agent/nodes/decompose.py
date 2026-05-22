"""decompose 노드.

질문을 SINGLE-HOP 또는 numbered plan 으로 분류. 출력은 state['plan'] 갱신만.
실패 시 plan=None 으로 안전 진행.
"""
import re

from langchain_core.messages import HumanMessage, SystemMessage

from ..llm import get_llm
from ..prompts import DECOMPOSITION_PROMPT
from ..state import GAIAState


def _normalize(raw: str) -> str:
    """마크다운 펜스/머리말 정리. SINGLE-HOP 한 줄이거나 번호 plan 본문만 남김."""
    t = (raw or "").strip()
    if not t:
        return t
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
        if "```" in t:
            t = t.split("```", 1)[0]
        t = t.strip()
    if re.match(r"^\s*SINGLE[\s\-]*HOP\s*$", t, re.IGNORECASE):
        return t
    m = re.search(r"(?m)^\s*\d+[\.\)]\s+", t)
    if m:
        return t[m.start():].strip()
    return t


def decompose_node(state: GAIAState) -> dict:
    question = state["question"]
    llm = get_llm()
    try:
        out = llm.invoke_with(
            [SystemMessage(content=DECOMPOSITION_PROMPT), HumanMessage(content=question)],
            max_new_tokens=256,
            temperature=0.0,
        )
        text = _normalize(out.content)
        if re.match(r"^\s*SINGLE[\s\-]*HOP", text, re.IGNORECASE):
            return {"plan": None}
        return {"plan": text or None}
    except Exception as e:
        print(f"decompose_node failed (proceeding without plan): {e}")
        return {"plan": None}
