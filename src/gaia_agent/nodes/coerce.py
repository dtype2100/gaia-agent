"""coerce 노드. formatter.coerce_answer 호출해 최종 answer 세팅."""
from ..formatter import coerce_answer
from ..state import GAIAState


def coerce_node(state: GAIAState) -> dict:
    candidate = (
        state.get("formatted_answer")
        or state.get("raw_answer")
        or "UNKNOWN"
    )
    final = coerce_answer(state["question"], candidate)
    return {"answer": final}
