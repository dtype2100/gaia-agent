"""파이썬 코드 실행. agent_course/tools/exec_code.py 에서 포팅."""
import os
import subprocess
import sys
import tempfile
from langchain_core.tools import tool


@tool
def exec_python_code(code: str) -> str:
    """Execute Python source code and return its captured stdout.
    Use this when the question asks for the output of a piece of attached or referenced
    Python code (e.g., "What is the final numeric output of the code?"). Pass the code
    body verbatim. Captures both stdout and stderr; returns up to ~12k characters.

    Args:
        code: The Python source code to execute.
    """
    if len(code) > 50000:
        return "exec_python_code error: code is too large to execute safely"

    try:
        with tempfile.TemporaryDirectory(prefix="gaia_exec_") as tmpdir:
            script_path = os.path.join(tmpdir, "snippet.py")
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(code)
            result = subprocess.run(
                [sys.executable, "-I", script_path],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=20,
            )
        out = (result.stdout or "") + (result.stderr or "")
        if result.returncode != 0:
            out = f"exec_python_code exited with status {result.returncode}\n{out}"
    except subprocess.TimeoutExpired as e:
        partial = ((e.stdout or "") + (e.stderr or ""))[:4000]
        return f"exec_python_code error: TimeoutExpired after 20s\n--- partial output ---\n{partial}"
    except Exception as e:
        import traceback
        return (
            f"exec_python_code error: {type(e).__name__}: {e}\n"
            f"--- traceback ---\n{traceback.format_exc()[-1000:]}\n"
        )

    if len(out) > 12000:
        out = out[:12000] + "\n...[truncated]"
    return out or "(no stdout output)"
