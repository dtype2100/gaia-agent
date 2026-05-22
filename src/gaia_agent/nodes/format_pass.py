"""format_pass 노드. raw_answer 를 GAIA 채점 포맷으로 LLM 재변환.

호출 실패 또는 UNKNOWN 입력 시 raw 유지.
"""
import unicodedata

from langchain_core.messages import HumanMessage, SystemMessage

from ..llm import get_llm
from ..prompts import FORMAT_PASS_PROMPT
from ..state import GAIAState


def format_pass_node(state: GAIAState) -> dict:
    raw = (state.get("raw_answer") or "").strip()
    # FINAL ANSWER 접두/양옆 따옴표 1차 정리 (agent_course/app.py:__call__ 의 1·2단계 이식)
    import re

    raw = re.sub(r"^\s*FINAL\s*ANSWER\s*[:\-]?\s*", "", raw, flags=re.IGNORECASE).strip()
    if len(raw) >= 2 and (
        (raw[0] == '"' and raw[-1] == '"')
        or (raw[0] == "'" and raw[-1] == "'")
    ):
        raw = raw[1:-1].strip()

    if not raw or raw.upper() == "UNKNOWN":
        return {"formatted_answer": raw or "UNKNOWN"}

    llm = get_llm()
    try:
        out = llm.invoke_with(
            [
                SystemMessage(content=FORMAT_PASS_PROMPT),
                HumanMessage(
                    content=(
                        f"Question: {state['question']}\n\n"
                        f"Draft answer: {raw}\n\nFinal answer:"
                    )
                ),
            ],
            max_new_tokens=128,
            temperature=0.0,
        )
        formatted = (out.content or "").strip()
        if not formatted:
            return {"formatted_answer": raw}
        if len(formatted) >= 2 and (
            (formatted[0] == '"' and formatted[-1] == '"')
            or (formatted[0] == "'" and formatted[-1] == "'")
        ):
            formatted = formatted[1:-1].strip()
        formatted = unicodedata.normalize("NFC", formatted)
        return {"formatted_answer": formatted}
    except Exception as e:
        print(f"format_pass_node failed (using raw): {e}")
        return {"formatted_answer": raw}
