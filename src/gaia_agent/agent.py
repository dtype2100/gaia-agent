"""GaiaAgent — 외부 노출 진입점.

기존 agent_course/app.py:BasicAgent 와 동일한 시그니처 `__call__(question) -> str`
을 유지해서 평가 루프(run_and_submit_all)에 그대로 꽂힌다.
"""
from __future__ import annotations

from .graph import build_graph
from .tools import prefetch_question_index, set_current_task, set_question_index


class GaiaAgent:
    """LangGraph StateGraph 를 래핑한 GAIA 에이전트."""

    def __init__(self, max_steps: int = 12) -> None:
        print("GaiaAgent initializing (LangGraph + Gemma 3n E4B)")
        self.graph = build_graph()
        self.max_steps = max_steps

        # /questions 한 번 prefetch — get_attached_file 가 task_id를 자동 해석할 수
        # 있게 한다(시그니처 제약 우회).
        idx = prefetch_question_index()
        set_question_index(idx)
        print(f"Prefetched question index: {len(idx)} entries")

    def __call__(self, question: str) -> str:
        print(f"Agent received question (first 50 chars): {question[:50]}...")
        tid = set_current_task(question)
        if tid:
            print(f"  → matched task_id: {tid}")
        else:
            print("  → no matched task_id (question not in cache)")

        initial: dict = {
            "question": question,
            "task_id": tid,
            "messages": [],
            "step_count": 0,
            "max_steps": self.max_steps,
            "metadata": {},
        }
        try:
            # recursion_limit 은 노드 방문 횟수 상한. step_count 가드와 별개로 안전망.
            final_state = self.graph.invoke(
                initial, config={"recursion_limit": self.max_steps * 4}
            )
            answer = (
                final_state.get("answer")
                or final_state.get("formatted_answer")
                or final_state.get("raw_answer")
                or "UNKNOWN"
            )
            print(f"Agent returning answer: {answer}")
            return answer
        except Exception as e:
            import traceback

            err_type = type(e).__name__
            print(f"Agent error ({err_type}): {e}")
            print(traceback.format_exc()[-600:])
            return f"AGENT_ERROR: {err_type}"
