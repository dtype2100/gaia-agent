"""Agent가 호출하는 도구 묶음.

agent_course/tools/ 에서 포팅. smolagents @tool → LangChain @tool 로 데코레이터만 교체.
대부분 본문은 동일.

ALL_TOOLS 리스트는 그래프 build 시 tool_executor에 일괄 등록된다.
"""
from .search import web_search
from .webpage import visit_webpage
from .wikipedia import wikipedia_search
from .youtube import youtube_info
from .exec_code import exec_python_code
from .attachments import (
    get_attached_file,
    prefetch_question_index,
    set_question_index,
    set_current_task,
)

# tool_executor 에 등록할 LangChain BaseTool 객체들.
ALL_TOOLS = [
    web_search,
    visit_webpage,
    wikipedia_search,
    youtube_info,
    exec_python_code,
    get_attached_file,
]

__all__ = [
    "ALL_TOOLS",
    "web_search",
    "visit_webpage",
    "wikipedia_search",
    "youtube_info",
    "exec_python_code",
    "get_attached_file",
    "prefetch_question_index",
    "set_question_index",
    "set_current_task",
]
