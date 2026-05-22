---
title: GAIA Agent Lee
emoji: 🕵🏻‍♂️
colorFrom: indigo
colorTo: indigo
sdk: gradio
sdk_version: 5.25.2
app_file: app.py
pinned: false
hf_oauth: true
hf_oauth_expiration_minutes: 480
short_description: GAIA Level 1 agent — LangGraph + Gemma 3n E4B
---

# GAIA Agent Lee

GAIA Level 1 벤치마크용 LangGraph 에이전트. Gemma 3n E4B SLM 을 HF ZeroGPU 에서
구동한다.

## 아키텍처

```
question
  → decompose       SINGLE-HOP vs 멀티홉 plan
  → agent ⇄ tools   ReAct 루프 (XML tool-call 컨벤션)
  → format          LLM 한 번 더 호출해 GAIA 채점 포맷 정규화
  → coerce          yes/no, 숫자, 통화 결정적 후처리
  → final answer
```

도구 6종: `web_search`, `visit_webpage`, `wikipedia_search`, `youtube_info`,
`exec_python_code`, `get_attached_file`.

## 환경 변수 (Space secrets)

| 이름 | 필수? | 용도 |
|---|---|---|
| `HF_TOKEN` | 필수 | Gemma 3n 모델 다운로드 (gated). https://huggingface.co/google/gemma-3n-E4B-it 에서 라이선스 수락 후 발급한 read token. |
| `GAIA_MODEL_ID` | 선택 | 기본 `google/gemma-3n-E4B-it`. 다른 chat-template 호환 모델로 override 가능. |
| `TAVILY_API_KEY` | 선택 | Tavily 검색 백엔드 (SearXNG 폴백). |
| `BRAVE_API_KEY` | 선택 | Brave 검색 백엔드 (SearXNG 폴백). |

## Hardware

Space settings → Hardware 에서 **ZeroGPU (Nvidia A10G/H200)** 선택.
`@spaces.GPU(duration=120)` 데코레이터가 자동 감지됨 (`src/gaia_agent/llm.py`).
