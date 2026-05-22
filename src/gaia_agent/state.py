"""LangGraph state 정의.

GAIAState는 노드 간 전달되는 누적 컨텍스트. TypedDict 라서 LangGraph가 partial
update를 자동 머지한다(각 노드는 변경된 키만 반환하면 됨).

설계 메모:
- `messages`는 LangChain 메시지 리스트(Human/AI/Tool). agent ↔ tool_executor 루프
  에서 추가만 되고 truncate는 별도 정책으로(컨텍스트 보호용).
- `step_count`는 agent 노드 진입마다 +1. max_steps 도달 시 강제로 format으로 분기.
- `final_answer`가 세팅되면 agent 루프 종료 → format/coerce 진행.
- `raw_answer`는 final_format_pass 전 원본을 보존(graceful degrade 시 사용).
"""
from __future__ import annotations

from typing import Annotated, Any, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class GAIAState(TypedDict, total=False):
    # --- 입력 ---
    question: str
    task_id: Optional[str]          # tools/attachments 인덱스에서 매칭. 없으면 첨부 없는 문제.

    # --- decompose 노드가 채움 ---
    plan: Optional[str]             # 멀티홉 plan 또는 None(단일 lookup).

    # --- agent ↔ tool_executor 루프 ---
    messages: Annotated[list[BaseMessage], add_messages]  # add_messages = LangGraph 표준 reducer
    step_count: int
    max_steps: int

    # --- 종료 ---
    final_answer: Optional[str]     # agent가 final_answer 호출 시 세팅.
    raw_answer: Optional[str]       # format_pass 전 원본.
    formatted_answer: Optional[str] # format_pass 후, coerce 전.
    answer: Optional[str]           # coerce 후 최종.

    # --- 디버그 ---
    error: Optional[str]            # 예외 발생 시 메시지.
    metadata: dict[str, Any]
