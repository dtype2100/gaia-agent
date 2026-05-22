"""Gradio app — GAIA Level 1 평가 + 제출.

agent_course/app.py 의 run_and_submit_all 흐름을 그대로 유지하되, BasicAgent
대신 src/gaia_agent/GaiaAgent (LangGraph + Gemma 3n E4B) 를 호출한다.

HF Space 배포 시 README.md 프론트매터를 Gradio Space 용으로 교체해야 한다
(현재 README는 GAIA 데이터셋 카드라 dataset 형식). 데이터셋 사용/Space 배포 분리
시점에 사용자가 결정.
"""
from __future__ import annotations

import os
import sys

# pip install -r requirements.txt 가 `-e .` 로 패키지를 깔지 않은 환경(예: 직접
# python app.py 만 실행)에서도 src/ 를 import 경로에 넣어준다.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import gradio as gr
import pandas as pd
import requests

from gaia_agent import GaiaAgent
from gaia_agent.cache import is_retryable_answer, load_cache, save_answer

DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"


def run_and_submit_all(profile: "gr.OAuthProfile | None"):
    """Fetches all questions, runs the agent, submits answers, returns status + table."""
    space_id = os.getenv("SPACE_ID")

    if profile:
        username = f"{profile.username}"
        print(f"User logged in: {username}")
    else:
        print("User not logged in.")
        return "Please Login to Hugging Face with the button.", None

    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    try:
        agent = GaiaAgent()
    except Exception as e:
        print(f"Error instantiating agent: {e}")
        return f"Error initializing agent: {e}", None

    if space_id:
        agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"
    else:
        agent_code = "https://huggingface.co/docs/hub/spaces"
        print("SPACE_ID unset — using docs URL for agent_code.")
    print(agent_code)

    print(f"Fetching questions from: {questions_url}")
    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        if not questions_data:
            return "Fetched questions list is empty or invalid format.", None
        print(f"Fetched {len(questions_data)} questions.")
    except Exception as e:
        print(f"Error fetching questions: {e}")
        return f"Error fetching questions: {e}", None

    results_log: list[dict] = []
    answers_payload: list[dict] = []
    cache = load_cache()
    print(f"Running agent on {len(questions_data)} questions... (cache: {len(cache)} entries)")

    for item in questions_data:
        task_id = item.get("task_id")
        question_text = item.get("question")
        if not task_id or question_text is None:
            print(f"Skipping item with missing task_id or question: {item}")
            continue

        cached = cache.get(task_id)
        if cached and isinstance(cached, dict) and "answer" in cached:
            submitted_answer = cached["answer"]
            if is_retryable_answer(submitted_answer):
                print(
                    f"  [cache stale] task_id={task_id}: retrying "
                    f"instead of reusing {submitted_answer!r}"
                )
            else:
                print(f"  [cache hit] task_id={task_id}: {submitted_answer[:80]}")
                answers_payload.append(
                    {"task_id": task_id, "submitted_answer": submitted_answer}
                )
                results_log.append(
                    {
                        "Task ID": task_id,
                        "Question": question_text,
                        "Submitted Answer": submitted_answer,
                    }
                )
                continue
        try:
            submitted_answer = agent(question_text)
            save_answer(task_id, question_text, submitted_answer)
            if is_retryable_answer(submitted_answer):
                print(f"  [skip retryable answer] task_id={task_id}: {submitted_answer!r}")
                results_log.append(
                    {
                        "Task ID": task_id,
                        "Question": question_text,
                        "Submitted Answer": submitted_answer,
                    }
                )
                continue
            answers_payload.append(
                {"task_id": task_id, "submitted_answer": submitted_answer}
            )
            results_log.append(
                {
                    "Task ID": task_id,
                    "Question": question_text,
                    "Submitted Answer": submitted_answer,
                }
            )
        except Exception as e:
            err_type = type(e).__name__
            print(f"Error running agent on task {task_id} ({err_type}): {e}")
            results_log.append(
                {
                    "Task ID": task_id,
                    "Question": question_text,
                    "Submitted Answer": f"AGENT_ERROR: {err_type}",
                }
            )

    if not answers_payload:
        print("Agent did not produce any answers to submit.")
        return "Agent did not produce any answers to submit.", pd.DataFrame(results_log)

    submission_data = {
        "username": username.strip(),
        "agent_code": agent_code,
        "answers": answers_payload,
    }
    print(f"Submitting {len(answers_payload)} answers to: {submit_url}")
    try:
        response = requests.post(submit_url, json=submission_data, timeout=60)
        response.raise_for_status()
        result_data = response.json()
        final_status = (
            f"Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Overall Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)\n"
            f"Message: {result_data.get('message', 'No message received.')}"
        )
        print("Submission successful.")
        return final_status, pd.DataFrame(results_log)
    except requests.exceptions.HTTPError as e:
        detail = f"Server responded with status {e.response.status_code}."
        try:
            detail += f" Detail: {e.response.json().get('detail', e.response.text)}"
        except Exception:
            detail += f" Response: {e.response.text[:500]}"
        return f"Submission Failed: {detail}", pd.DataFrame(results_log)
    except Exception as e:
        return f"Submission failed: {e}", pd.DataFrame(results_log)


with gr.Blocks() as demo:
    gr.Markdown("# GAIA Agent — LangGraph + Gemma 3n E4B")
    gr.Markdown(
        """
        1. Log in to your Hugging Face account using the button below.
        2. Click 'Run Evaluation & Submit All Answers' to fetch GAIA questions,
           run the agent, submit the answers, and see the score.

        Model: `google/gemma-3n-E4B-it` (override with `GAIA_MODEL_ID` env var).
        Runtime: HF ZeroGPU (`@spaces.GPU` auto-detected) or any CUDA GPU (e.g., RunPod).
        """
    )

    gr.LoginButton()

    run_button = gr.Button("Run Evaluation & Submit All Answers")

    status_output = gr.Textbox(label="Run Status / Submission Result", lines=5, interactive=False)
    results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True)

    run_button.click(fn=run_and_submit_all, outputs=[status_output, results_table])


if __name__ == "__main__":
    print("\n" + "-" * 30 + " App Starting " + "-" * 30)
    space_host = os.getenv("SPACE_HOST")
    space_id = os.getenv("SPACE_ID")
    if space_host:
        print(f"SPACE_HOST: {space_host}")
    if space_id:
        print(f"SPACE_ID: {space_id}")
    print("-" * 74 + "\n")
    demo.launch(debug=True, share=False)
