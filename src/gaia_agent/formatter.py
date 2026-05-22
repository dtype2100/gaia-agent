"""결정적 답변 후처리 (yes/no, 숫자, 통화). agent_course/formatter.py 의
coerce_answer 부분만 분리해서 포팅. LLM 기반 final_format_pass 는 노드로 옮겨감
(nodes/format_pass.py).
"""
import re


_YES_NO_STARTS = (
    "is ", "are ", "was ", "were ", "do ", "does ", "did ",
    "has ", "have ", "had ", "can ", "could ", "should ",
    "will ", "would ", "may ", "might ",
)


def _looks_yes_no(question: str) -> bool:
    q = question.strip().lower()
    if "yes or no" in q or "yes/no" in q:
        return True
    if not q.endswith("?"):
        return False
    return any(q.startswith(s) for s in _YES_NO_STARTS)


def _looks_numeric(question: str) -> bool:
    q = question.lower()
    return (
        "how many" in q
        or "what number" in q
        or "what is the number of" in q
    )


def coerce_answer(question: str, answer: str) -> str:
    """질문 형식 힌트에 맞춰 LLM 답을 보정. 힌트 없거나 매칭 실패 시 원본 반환."""
    a = answer.strip()
    if not a:
        return a

    if _looks_yes_no(question):
        first = a.split(None, 1)[0].rstrip(",.").lower() if a.split() else ""
        if first == "yes":
            return "Yes"
        if first == "no":
            return "No"
        return a

    if _looks_numeric(question):
        m = re.search(r"-?\d+(?:\.\d+)?", a.replace(",", ""))
        if m:
            num = m.group(0)
            try:
                f = float(num)
                if f.is_integer():
                    return str(int(f))
                return num
            except ValueError:
                pass
        return a

    if re.fullmatch(r"\s*[\$€£¥]?\s*-?[\d,]+(?:\.\d+)?\s*", a):
        cleaned = re.sub(r"[\$€£¥,\s]", "", a)
        if cleaned:
            return cleaned

    return a
