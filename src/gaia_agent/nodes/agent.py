"""agent 노드. ReAct 루프의 LLM 호출 + 출력 파싱.

설계:
- Gemma 3n은 네이티브 function-calling 토큰이 없으므로 프롬프트 컨벤션을 강제하고
  XML-스타일 마커를 regex로 파싱한다:
    <thought>...</thought>
    <tool_call>{"name":"...","args":{...}}</tool_call>   # 호출하려면
    <final_answer>...</final_answer>                     # 끝내려면
- 첫 진입 시 system + user 메시지를 부트스트랩하고 state.messages 에 누적.
- step_count 가 max_steps 도달하면 final_answer 미발화여도 raw_answer 를 강제 세팅
  하고 다음 conditional edge 에서 format 으로 빠진다.
"""
from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ..llm import get_llm
from ..prompts import AGENT_SYSTEM_PROMPT
from ..state import GAIAState
from ..tools import ALL_TOOLS


_THOUGHT_RE = re.compile(r"<thought>\s*(.*?)\s*</thought>", re.DOTALL | re.IGNORECASE)
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL | re.IGNORECASE)
_FINAL_ANSWER_RE = re.compile(r"<final_answer>\s*(.*?)\s*</final_answer>", re.DOTALL | re.IGNORECASE)


def _render_tool_catalog(tools) -> str:
    """LangChain BaseTool 리스트 → 시스템 프롬프트용 텍스트 카탈로그."""
    parts = []
    for t in tools:
        args_desc = ""
        try:
            schema = t.args_schema.model_json_schema() if t.args_schema else {}
            props = schema.get("properties", {})
            if props:
                args_desc = " args: " + json.dumps(
                    {k: v.get("type", "string") for k, v in props.items()}
                )
            else:
                args_desc = " args: {}"
        except Exception:
            args_desc = ""
        # 첫 문장만 추출(LLM-facing description 은 짧게 유지).
        desc = (t.description or "").split("\n")[0].strip()
        parts.append(f"- {t.name}: {desc}{args_desc}")
    return "\n".join(parts)


def _parse_agent_output(text: str) -> dict:
    """모델 출력에서 thought/tool_call/final_answer 추출."""
    out: dict[str, Any] = {"thought": None, "tool_calls": [], "final_answer": None}
    m = _THOUGHT_RE.search(text)
    if m:
        out["thought"] = m.group(1).strip()
    for i, m in enumerate(_TOOL_CALL_RE.finditer(text)):
        raw = m.group(1)
        try:
            tc = json.loads(raw)
        except json.JSONDecodeError:
            # 작은 JSON 위반 회복 시도: 양옆에 백틱 등이 묻은 경우.
            cleaned = raw.strip().strip("`").strip()
            try:
                tc = json.loads(cleaned)
            except json.JSONDecodeError:
                continue
        if isinstance(tc, dict) and "name" in tc:
            out["tool_calls"].append(
                {
                    "name": tc["name"],
                    "args": tc.get("args") or tc.get("arguments") or {},
                    "id": f"call_{i}",
                }
            )
    m = _FINAL_ANSWER_RE.search(text)
    if m:
        out["final_answer"] = m.group(1).strip()
    return out


def _load_image(path: str):
    from PIL import Image
    return Image.open(path).convert("RGB")


def _load_audio(path: str) -> dict | None:
    try:
        import soundfile as sf
        import numpy as np
        data, samplerate = sf.read(path)
        if len(data.shape) > 1:
            data = data.mean(axis=1)
        if samplerate != 16000:
            try:
                import librosa
                data = librosa.resample(data, orig_sr=samplerate, target_sr=16000)
            except ImportError:
                print("Warning: librosa not installed, using raw sample rate for audio. Downsampling might fail.")
            samplerate = 16000
        return {"raw": data.astype(np.float32), "sampling_rate": samplerate}
    except Exception as e:
        print(f"Error loading audio file {path}: {e}")
        return None


def _bootstrap_messages(state: GAIAState) -> list:
    """첫 agent 진입 시 system + user 메시지를 구성."""
    max_steps = state.get("max_steps", 12)
    commit_by = max(1, max_steps - 4)
    sys = AGENT_SYSTEM_PROMPT.format(
        tool_catalog=_render_tool_catalog(ALL_TOOLS),
        max_steps=max_steps,
        commit_by=commit_by,
    )
    question = state["question"]
    plan = state.get("plan")
    if plan:
        user_text = (
            f"{question}\n\n"
            f"--- Suggested decomposition plan (guidance — deviate as tool results show) ---\n"
            f"{plan}\n"
            f"--- end plan ---\n"
            f"The final answer must address the ORIGINAL question above, not the plan."
        )
    else:
        user_text = question

    # --- Task 4: Prefetch/Bootstrap Multimodal Media Natively ---
    from ..tools.attachments import download_attachment
    task_id = state.get("task_id")
    multimodal_content = []

    if task_id:
        meta = download_attachment(task_id)
        if meta:
            ext = meta.get("ext", "")
            content_type = meta.get("content_type", "")
            abs_path = meta.get("abs_path")

            is_img = ext in ("png", "jpg", "jpeg", "webp", "gif", "bmp") or "image" in content_type
            is_aud = ext in ("mp3", "wav", "m4a", "ogg", "flac") or "audio" in content_type

            if is_img and abs_path:
                try:
                    print(f"[Bootstrap] Loading image native payload from {abs_path}")
                    img = _load_image(abs_path)
                    multimodal_content.append({"type": "image", "image": img})
                    user_text = "[IMAGE ATTACHED] " + user_text
                except Exception as e:
                    print(f"Failed to load image native payload: {e}")
            elif is_aud and abs_path:
                try:
                    print(f"[Bootstrap] Loading audio native payload from {abs_path}")
                    audio_payload = _load_audio(abs_path)
                    if audio_payload is not None:
                        multimodal_content.append({"type": "audio", "audio": audio_payload})
                        user_text = "[AUDIO ATTACHED] " + user_text
                except Exception as e:
                    print(f"Failed to load audio native payload: {e}")

    if multimodal_content:
        multimodal_content.append({"type": "text", "text": user_text})
        user_message_content = multimodal_content
    else:
        user_message_content = user_text

    return [SystemMessage(content=sys), HumanMessage(content=user_message_content)]


def agent_node(state: GAIAState) -> dict:
    step = state.get("step_count", 0) + 1
    max_steps = state.get("max_steps", 12)

    existing = state.get("messages") or []
    if not existing:
        bootstrap = _bootstrap_messages(state)
        messages_for_llm = bootstrap
        # bootstrap 메시지도 그래프 메시지 리스트에 누적되도록 함께 반환.
        msgs_to_append: list = list(bootstrap)
    else:
        messages_for_llm = list(existing)
        msgs_to_append = []

    # --- Task 6: Anti-Loop & Dynamic Self-Correction ---
    is_loop = False
    loop_tool_name = ""
    ai_tool_calls = []
    
    # Gather tool calls in reversed order to inspect loops
    for msg in reversed(messages_for_llm):
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            ai_tool_calls.append(msg.tool_calls)
            if len(ai_tool_calls) >= 2:
                break
                
    if len(ai_tool_calls) >= 2:
        tc1 = ai_tool_calls[0][0] if ai_tool_calls[0] else None
        tc2 = ai_tool_calls[1][0] if ai_tool_calls[1] else None
        if tc1 and tc2 and tc1.get("name") == tc2.get("name") and tc1.get("args") == tc2.get("args"):
            is_loop = True
            loop_tool_name = tc1.get("name")

    # Inject Anti-Loop warning if loop detected
    if is_loop:
        warning_msg = HumanMessage(
            content=f"WARNING: You are stuck in a loop calling `{loop_tool_name}` with the exact same arguments! "
                    f"You MUST change your strategy immediately. If a Python script failed or gave the same result, "
                    f"rewrite it to approach the problem differently. If web search failed, use different search terms. "
                    f"DO NOT repeat the same tool call with the same arguments."
        )
        print(f"[Anti-Loop] Stuck detected for tool `{loop_tool_name}`. Injecting warning to LLM.")
        messages_for_llm = list(messages_for_llm) + [warning_msg]

    # Inject Self-Correction warning if previous tool execution errored out
    has_error = False
    if messages_for_llm:
        last_msg = messages_for_llm[-1]
        last_content = str(last_msg.content)
        last_lower = last_content.lower()
        if (
            "error" in last_lower
            or "failed" in last_lower
            or "exception" in last_lower
            or "exited with status" in last_lower
            or "traceback" in last_lower
        ):
            has_error = True
            
    if has_error and not is_loop:  # Avoid warning overload if already in a loop
        correction_msg = HumanMessage(
            content="WARNING: Your previous tool execution returned an ERROR or FAILED. "
                    "Please analyze the error carefully, fix your logic, and correct the error in your next tool call. "
                    "Do not repeat the exact same failing script or arguments."
        )
        print("[Self-Correction] Error detected in last tool output. Injecting warning to LLM.")
        messages_for_llm = list(messages_for_llm) + [correction_msg]

    llm = get_llm()
    try:
        ai = llm.invoke(messages_for_llm)
    except Exception as e:
        print(f"agent_node LLM call failed: {e}")
        return {
            "messages": msgs_to_append,
            "step_count": step,
            "raw_answer": "UNKNOWN",
            "error": f"llm: {type(e).__name__}: {e}",
        }

    parsed = _parse_agent_output(ai.content)

    # AIMessage 에 tool_calls 를 부착해 tool_executor 가 인식하게 한다.
    if parsed["tool_calls"]:
        ai.tool_calls = parsed["tool_calls"]
    msgs_to_append.append(ai)

    update: dict = {"messages": msgs_to_append, "step_count": step}

    if parsed["final_answer"] is not None:
        update["raw_answer"] = parsed["final_answer"]
    elif step >= max_steps:
        # 예산 소진 — final_answer 못 받았으면 thought 또는 UNKNOWN 으로 강제 종결.
        update["raw_answer"] = (parsed["thought"] or "UNKNOWN").strip() or "UNKNOWN"

    return update
