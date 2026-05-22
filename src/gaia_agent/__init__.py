"""gaia_agent — LangGraph 기반 GAIA Level 1 에이전트.

기존 agent_course/(smolagents CodeAgent + Qwen2.5-72B)를 다음으로 리팩토링:
- LangGraph 커스텀 StateGraph (decompose → agent ↔ tools → format → coerce)
- Gemma 3n E4B SLM (멀티모달, transformers 백엔드)
- HF ZeroGPU / RunPod GPU 호스팅

엔트리포인트는 `GaiaAgent` 클래스. 모듈 레벨 __getattr__ 로 lazy import 하여
의존성(langgraph/transformers/torch)이 깔리지 않은 환경에서도 가벼운 서브모듈
(state/prompts/cache/formatter)은 단독 import 가능하다.
"""
from __future__ import annotations

__all__ = ["GaiaAgent"]


def __getattr__(name: str):
    if name == "GaiaAgent":
        from .agent import GaiaAgent

        return GaiaAgent
    raise AttributeError(f"module 'gaia_agent' has no attribute {name!r}")
