"""Per-task 답변 캐시. agent_course/answer_cache.py에서 거의 그대로 포팅.

저장 위치만 gaia-agent-lee/.cache/answers.json. 형식과 원자적 저장 로직은 동일.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Iterable, Optional

# Space 배포 시 작업 디렉토리가 / 또는 /home/user 가 될 수 있어 상대경로로 안전하게.
_CACHE_PATH = Path(".cache") / "answers.json"


def load_cache() -> dict:
    if not _CACHE_PATH.exists():
        return {}
    try:
        return json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Warning: cache load failed ({e}); starting empty.")
        return {}


def save_answer(task_id: str, question: str, answer: str) -> None:
    if not task_id or is_retryable_answer(answer):
        return
    cache = load_cache()
    cache[task_id] = {"question": question, "answer": answer}
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="answers.", suffix=".tmp", dir=str(_CACHE_PATH.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
        os.replace(tmp, _CACHE_PATH)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def get_cached_answer(task_id: str, cache: Optional[dict] = None) -> Optional[str]:
    if cache is None:
        cache = load_cache()
    entry = cache.get(task_id)
    if entry and isinstance(entry, dict):
        return entry.get("answer")
    return None


def is_retryable_answer(answer: Optional[str]) -> bool:
    if answer is None:
        return True
    a = str(answer).strip()
    if not a:
        return True
    upper = a.upper()
    return (
        upper == "UNKNOWN"
        or upper == "UNK"
        or upper.startswith("AGENT_ERROR:")
        or upper.startswith("AGENT ERROR:")
        or "CANNOT ANSWER" in upper
        or "NO FINAL ANSWER" in upper
    )


def clear_cache() -> None:
    if _CACHE_PATH.exists():
        _CACHE_PATH.unlink()


def invalidate_tasks(task_ids: Iterable[str]) -> int:
    cache = load_cache()
    removed = 0
    for tid in task_ids:
        if tid in cache:
            del cache[tid]
            removed += 1
    if removed == 0:
        return 0
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="answers.", suffix=".tmp", dir=str(_CACHE_PATH.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
        os.replace(tmp, _CACHE_PATH)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return removed
