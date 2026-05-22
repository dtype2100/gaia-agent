"""웹 검색 툴. agent_course/tools/search.py 에서 포팅.

변경점: @smolagents.tool → @langchain_core.tools.tool. 본문은 동일.
백엔드 우선순위: SearXNG → Tavily → Brave → DuckDuckGo.
"""
import os
import random
import requests
from langchain_core.tools import tool

_TAVILY_URL = "https://api.tavily.com/search"
_BRAVE_URL = "https://api.search.brave.com/res/v1/web/search"

_SEARXNG_INSTANCES = (
    "https://searx.be",
    "https://searx.tiekoetter.com",
    "https://search.inetol.net",
    "https://searxng.online",
    "https://priv.au",
)
_SEARXNG_TRY_COUNT = 3
_SEARXNG_TIMEOUT = 5


def _format_results(items) -> str:
    lines = [f"- {t}\n  {u}\n  {b}" for t, u, b in items if (t or u or b)]
    return "\n".join(lines) if lines else ""


def _search_searxng(query: str):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json",
    }
    candidates = random.sample(_SEARXNG_INSTANCES, _SEARXNG_TRY_COUNT)
    for base in candidates:
        try:
            r = requests.get(
                f"{base}/search",
                params={"q": query, "format": "json", "language": "en"},
                headers=headers,
                timeout=_SEARXNG_TIMEOUT,
            )
            if r.status_code != 200:
                continue
            results = r.json().get("results", [])
            if not results:
                continue
            items = [
                (x.get("title", ""), x.get("url", ""), x.get("content", ""))
                for x in results[:8]
            ]
            formatted = _format_results(items)
            if formatted:
                return formatted
        except Exception as e:
            print(f"SearXNG ({base}) failed: {e}")
            continue
    return None


def _search_tavily(query: str):
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return None
    try:
        r = requests.post(
            _TAVILY_URL,
            json={"api_key": api_key, "query": query, "max_results": 8},
            timeout=15,
        )
        r.raise_for_status()
        results = r.json().get("results", [])
        if not results:
            return None
        items = [
            (x.get("title", ""), x.get("url", ""), x.get("content", ""))
            for x in results
        ]
        return _format_results(items) or None
    except Exception as e:
        print(f"Tavily search failed (falling back): {e}")
        return None


def _search_brave(query: str):
    api_key = os.getenv("BRAVE_API_KEY")
    if not api_key:
        return None
    try:
        r = requests.get(
            _BRAVE_URL,
            params={"q": query, "count": 8},
            headers={"X-Subscription-Token": api_key, "Accept": "application/json"},
            timeout=15,
        )
        r.raise_for_status()
        results = r.json().get("web", {}).get("results", [])
        if not results:
            return None
        items = [
            (x.get("title", ""), x.get("url", ""), x.get("description", ""))
            for x in results
        ]
        return _format_results(items) or None
    except Exception as e:
        print(f"Brave search failed (falling back): {e}")
        return None


def _search_ddg(query: str) -> str:
    last_err = None
    for module_name in ("ddgs", "duckduckgo_search"):
        try:
            mod = __import__(module_name, fromlist=["DDGS"])
            DDGS = getattr(mod, "DDGS")
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=8))
            if not results:
                continue
            items = [
                (
                    r.get("title", ""),
                    r.get("href", "") or r.get("url", ""),
                    r.get("body", "") or r.get("snippet", ""),
                )
                for r in results
            ]
            formatted = _format_results(items)
            if formatted:
                return formatted
        except Exception as e:
            last_err = e
            continue
    if last_err:
        return f"web_search error: {last_err}"
    return "No results found."


@tool
def web_search(query: str) -> str:
    """Search the web and return a list of titles, URLs, and snippets.
    Backend priority: Tavily/Brave (only if their API keys are set) -> DuckDuckGo (highly stable & fast) -> SearXNG fallback.

    Args:
        query: The search query string.
    """
    # 1. Try Tavily or Brave first if API keys are set
    for backend in (_search_tavily, _search_brave):
        out = backend(query)
        if out:
            return out

    # 2. Try DuckDuckGo (highly reliable and rate-limit resistant for free tiers)
    ddg_out = _search_ddg(query)
    if ddg_out and "No results found." not in ddg_out and "web_search error" not in ddg_out:
        return ddg_out

    # 3. Fallback to SearXNG only if DuckDuckGo failed
    searx_out = _search_searxng(query)
    if searx_out:
        return searx_out

    return ddg_out or "No results found."
