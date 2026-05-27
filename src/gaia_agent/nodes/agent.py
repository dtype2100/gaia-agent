"""agent 노드. ReAct 루프의 LLM 호출 + 출력 파싱.

설계 (CodeAct 방식, Step 1):
- 액션 포맷은 Python 코드:
    <thought>...</thought>
    <code>...python...</code>           # 행동 = 짧은 파이썬 스니펫
    <final_answer>...</final_answer>    # 끝내려면
- 도구는 exec_node 의 namespace 에 함수로 주입되어 있다. agent 는 `web_search(query=...)`
  같은 함수 호출을 코드 안에 쓰면 됨.
- 첫 진입 시 system + user 메시지를 부트스트랩하고 state.messages 에 누적.
- step_count 가 max_steps 도달하면 final_answer 미발화여도 raw_answer 를 강제 세팅
  하고 다음 conditional edge 에서 format 으로 빠진다.
"""
from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ..llm import get_llm
from ..prompts import AGENT_SYSTEM_PROMPT
from ..state import GAIAState
from ..tools import ALL_TOOLS


_THOUGHT_RE = re.compile(r"<thought>\s*(.*?)\s*</thought>", re.DOTALL | re.IGNORECASE)
_CODE_RE = re.compile(r"<code>\s*(.*?)\s*</code>", re.DOTALL | re.IGNORECASE)
_FINAL_ANSWER_RE = re.compile(r"<final_answer>\s*(.*?)\s*</final_answer>", re.DOTALL | re.IGNORECASE)


def _render_tool_catalog(tools) -> str:
    """LangChain BaseTool 리스트 → Python 함수 시그니처 카탈로그.

    `exec_python_code` 는 <code> 블록 자체가 Python 이라 노출하지 않음.
    """
    parts = []
    for t in tools:
        if t.name == "exec_python_code":
            continue
        try:
            schema = t.args_schema.model_json_schema() if t.args_schema else {}
            props = schema.get("properties", {})
            sig_args = ", ".join(
                f"{k}: {v.get('type', 'string')}" for k, v in props.items()
            )
        except Exception:
            sig_args = ""
        desc = (t.description or "").split("\n")[0].strip()
        parts.append(f"- {t.name}({sig_args}) -> str: {desc}")
    return "\n".join(parts)


def _parse_agent_output(text: str) -> dict:
    """모델 출력에서 thought / code / final_answer 추출."""
    out: dict[str, Any] = {"thought": None, "code": None, "final_answer": None}
    m = _THOUGHT_RE.search(text)
    if m:
        out["thought"] = m.group(1).strip()
    m = _CODE_RE.search(text)
    if m:
        code = m.group(1).strip()
        # 모델이 ```python ... ``` 마크다운 펜스를 추가로 두른 경우 정리.
        if code.startswith("```"):
            code = re.sub(r"^```[a-zA-Z]*\n?", "", code)
            if "```" in code:
                code = code.rsplit("```", 1)[0]
            code = code.strip()
        out["code"] = code
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

    # Append routing directive if a route hint is present in metadata
    metadata = state.get("metadata") or {}
    route_hint = metadata.get("route_hint")
    if route_hint:
        if route_hint == "get_attached_file":
            user_text += (
                "\n\n[System Directive] An attached file is detected. For your very first step, "
                "you MUST call `get_attached_file()` inside a <code> block to download and inspect "
                "the file."
            )
        elif route_hint == "youtube_info":
            user_text += (
                "\n\n[System Directive] A YouTube link/video is referenced. For your very first step, "
                "you MUST call `youtube_info(url=...)` inside a <code> block with the video URL."
            )

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

    # --- Anti-Loop & Dynamic Self-Correction ---
    # 동일한 <code> 블록이 연속 2회 나오면 루프로 판단.
    is_loop = False
    recent_codes: list[str] = []
    for msg in reversed(messages_for_llm):
        if isinstance(msg, AIMessage):
            kw = getattr(msg, "additional_kwargs", None) or {}
            code = kw.get("code")
            if code:
                recent_codes.append(code)
                if len(recent_codes) >= 2:
                    break

    if len(recent_codes) >= 2 and recent_codes[0] == recent_codes[1]:
        is_loop = True

    if is_loop:
        warning_msg = HumanMessage(
            content=(
                "WARNING: You are stuck in a loop — your last two <code> blocks were "
                "identical! You MUST change your strategy immediately. Rewrite the snippet "
                "with different queries, different tools, or a different approach. "
                "DO NOT submit the same code again."
            )
        )
        print("[Anti-Loop] Identical consecutive <code> blocks detected. Injecting warning to LLM.")
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

    # AIMessage.additional_kwargs 에 code 를 부착해 exec_node 가 인식하게 한다.
    if parsed["code"]:
        ai.additional_kwargs["code"] = parsed["code"]
    msgs_to_append.append(ai)

    update: dict = {"messages": msgs_to_append, "step_count": step}

    if parsed["final_answer"] is not None:
        update["raw_answer"] = parsed["final_answer"]
    elif step >= max_steps:
        # 예산 소진 — final_answer 못 받았으면 thought 또는 UNKNOWN 으로 강제 종결.
        update["raw_answer"] = (parsed["thought"] or "UNKNOWN").strip() or "UNKNOWN"

    return update
