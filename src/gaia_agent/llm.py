"""Gemma 3n E4B LLM wrapper.

기존 hf-inference InferenceClient(외부 API) → 직접 transformers 로 모델 로드로 전환.
HF ZeroGPU 또는 RunPod 등 GPU 환경에서 동작하며, 멀티모달(이미지/오디오) 처리를 네이티브 지원.

설계 메모:
- LangChain의 `BaseChatModel`을 완전히 구현하지 않고 최소 인터페이스(`invoke`)만 노출.
- ZeroGPU: `@spaces.GPU(duration=120)` 데코레이터를 통해 GPU 임대 시간에만 VRAM 점유.
- Lazy Loading: 전역 CPU 모델 로딩을 강제하지 않고, 첫 GPU 세션 내에서 `device_map="auto"`로
  초기화하여 ZeroGPU가 PyTorch 디바이스 포인터를 투명하게 스왑하도록 유도함.
- AutoProcessor: 텍스트뿐 아니라 이미지/오디오 멀티모달 입력을 함께 처리.
"""
from __future__ import annotations

import os
import threading
from typing import Any, Optional

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)


# --- ZeroGPU 데코레이터 ---
try:
    import spaces  # type: ignore

    _IN_ZEROGPU = True
except Exception:
    spaces = None  # type: ignore
    _IN_ZEROGPU = False


def _gpu(fn):
    """ZeroGPU 가용 시 @spaces.GPU 로 감싸고, 아니면 그대로 반환."""
    if _IN_ZEROGPU:
        return spaces.GPU(duration=120)(fn)
    return fn


# --- 모델 캐시 (싱글톤) ---
_MODEL_CACHE: dict[str, Any] = {}
_PROCESSOR_CACHE: dict[str, Any] = {}
_LOAD_LOCK = threading.Lock()


def _load_model(model_id: str):
    """ZeroGPU 세션 내 첫 호출 시 모델 및 프로세서 초기화."""
    if model_id in _MODEL_CACHE:
        return _MODEL_CACHE[model_id], _PROCESSOR_CACHE[model_id]
    with _LOAD_LOCK:
        if model_id in _MODEL_CACHE:
            return _MODEL_CACHE[model_id], _PROCESSOR_CACHE[model_id]
        
        import torch
        import transformers
        print(f"Hugging Face Spaces transformers version: {transformers.__version__}")
        
        from transformers import AutoProcessor, AutoModelForCausalLM
        
        print(f"Loading multimodal processor and model {model_id}...")
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        
        processor = AutoProcessor.from_pretrained(model_id)
        
        # device_map="auto" allows transparent mapping to GPU during the @spaces.GPU hook.
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        model.eval()
        
        _MODEL_CACHE[model_id] = model
        _PROCESSOR_CACHE[model_id] = processor
        return model, processor


# --- 메시지 변환 ---

def _messages_to_chat_and_media(messages: list[BaseMessage]) -> tuple[list[dict], list[Any], list[Any]]:
    """LangChain 메시지를 Gemma 3 Multimodal chat template 형식으로 변환.
    이미지와 오디오 객체는 별도 리스트로 수집하여 프로세서에 전달한다.
    """
    out: list[dict] = []
    images = []
    audios = []
    pending_system: Optional[str] = None
    
    for msg in messages:
        if isinstance(msg, SystemMessage):
            pending_system = (pending_system + "\n\n" if pending_system else "") + str(msg.content)
        
        elif isinstance(msg, HumanMessage):
            if isinstance(msg.content, list):
                # 멀티모달 메시지 처리 (agent.py에서 조립)
                content_parts = []
                # System prompt가 남아있다면 첫 파트 최상단에 주입
                if pending_system:
                    content_parts.append({"type": "text", "text": pending_system + "\n\n"})
                    pending_system = None
                    
                for part in msg.content:
                    if isinstance(part, dict) and part.get("type") == "image":
                        images.append(part["image"])
                        content_parts.append({"type": "image"})
                    elif isinstance(part, dict) and part.get("type") == "audio":
                        audios.append(part["audio"])
                        content_parts.append({"type": "audio"})
                    elif isinstance(part, dict) and part.get("type") == "text":
                        content_parts.append({"type": "text", "text": part["text"]})
                    elif isinstance(part, str):
                        content_parts.append({"type": "text", "text": part})
                
                out.append({"role": "user", "content": content_parts})
            else:
                content = str(msg.content)
                if pending_system:
                    content = pending_system + "\n\n" + content
                    pending_system = None
                out.append({"role": "user", "content": [{"type": "text", "text": content}]})
        
        elif isinstance(msg, AIMessage):
            out.append({"role": "assistant", "content": [{"type": "text", "text": str(msg.content)}]})
            
        elif isinstance(msg, ToolMessage):
            tool_text = f"[Tool result for `{getattr(msg, 'name', '?')}`]\n{msg.content}"
            out.append({"role": "user", "content": [{"type": "text", "text": tool_text}]})
            
        else:
            out.append({"role": "user", "content": [{"type": "text", "text": str(msg.content)}]})
            
    if pending_system:
        out.append({"role": "user", "content": [{"type": "text", "text": pending_system}]})
        
    return out, images, audios


@_gpu
def _generate(model_id: str, chat: list[dict], images: list[Any], audios: list[Any], max_new_tokens: int, temperature: float) -> str:
    """실제 GPU 추론. ZeroGPU에서 이 함수만 GPU slot을 점유한다."""
    import torch
    model, processor = _load_model(model_id)
    
    # 템플릿 적용 시 이미지/오디오 토큰 마커가 자동으로 삽입됨.
    prompt = processor.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )
    
    # 프로세서 kwargs 조립
    kwargs = {"text": prompt, "return_tensors": "pt"}
    if images:
        kwargs["images"] = images
    if audios:
        kwargs["audio"] = audios
        
    inputs = processor(**kwargs).to(model.device)
    
    # PAD 토큰 설정 (Gemma는 eos_token_id를 주로 사용)
    pad_token_id = getattr(processor.tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = processor.tokenizer.eos_token_id
        
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-3),
            top_p=0.95,
            pad_token_id=pad_token_id,
        )
        
    # 입력부를 잘라내고 생성된 토큰만 디코딩
    input_len = inputs["input_ids"].shape[1]
    new_tokens = out[0][input_len:]
    return processor.decode(new_tokens, skip_special_tokens=True)


# --- 공개 wrapper ---

class GemmaChat:
    """LangGraph 노드가 호출하는 최소 LLM 인터페이스."""

    def __init__(
        self,
        model_id: str = "google/gemma-4-E4B-it",
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> None:
        self.model_id = os.getenv("GAIA_MODEL_ID", model_id)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def invoke(self, messages: list[BaseMessage]) -> AIMessage:
        """동기 호출. 메시지 리스트 → AIMessage(content=...)."""
        chat, images, audios = _messages_to_chat_and_media(messages)
        text = _generate(
            self.model_id, chat, images, audios, self.max_new_tokens, self.temperature
        )
        return AIMessage(content=text.strip())

    def invoke_with(
        self,
        messages: list[BaseMessage],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> AIMessage:
        chat, images, audios = _messages_to_chat_and_media(messages)
        text = _generate(
            self.model_id,
            chat,
            images,
            audios,
            max_new_tokens if max_new_tokens is not None else self.max_new_tokens,
            temperature if temperature is not None else self.temperature,
        )
        return AIMessage(content=text.strip())


# --- 공유 싱글톤 ---
_SHARED: Optional[GemmaChat] = None


def get_llm() -> GemmaChat:
    global _SHARED
    if _SHARED is None:
        _SHARED = GemmaChat()
    return _SHARED
