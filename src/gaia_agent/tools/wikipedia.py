"""위키피디아 본문(표 포함) 추출. agent_course/tools/wikipedia.py 에서 포팅."""
import re
import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool

_HEADERS = {
    "User-Agent": "GAIA-Agent/1.0 (HF agents course unit 4; https://huggingface.co/spaces)"
}


@tool
def wikipedia_search(query: str) -> str:
    """Search English Wikipedia and return the FULL article body (text + tables) of the top matching
    article, rendered as readable text and truncated to ~14k chars. Includes section headers and
    table rows so factual lookups (winners lists, rosters, dates, etc.) are answerable.

    Args:
        query: The search term or article title.
    """
    try:
        s = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "format": "json",
                "list": "search",
                "srsearch": query,
                "srlimit": 3,
            },
            headers=_HEADERS,
            timeout=15,
        )
        s.raise_for_status()
        hits = s.json().get("query", {}).get("search", [])
        if not hits:
            return "No Wikipedia results."
        title = hits[0]["title"]
        candidates = [h.get("title", "") for h in hits[:3]]

        page = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "parse",
                "format": "json",
                "page": title,
                "prop": "text",
                "redirects": True,
            },
            headers=_HEADERS,
            timeout=20,
        )
        page.raise_for_status()
        html = page.json().get("parse", {}).get("text", {}).get("*", "")
        if not html:
            return f"Top hit: {title} (no body available)."

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup.select(
            "sup.reference, .mw-editsection, .reference, .navbox, "
            ".infobox.metadata, .hatnote, .printfooter, script, style"
        ):
            tag.decompose()

        for tbl in soup.find_all("table"):
            rows_text = []
            header_emitted = False
            for tr in tbl.find_all("tr"):
                ths = tr.find_all("th")
                tds = tr.find_all("td")
                cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
                if not cells:
                    continue
                if not header_emitted and ths and not tds and len(ths) >= 2:
                    rows_text.append("| " + " | ".join(cells) + " |")
                    rows_text.append("| " + " | ".join(["---"] * len(cells)) + " |")
                    header_emitted = True
                else:
                    rows_text.append("| " + " | ".join(cells) + " |")
            if rows_text:
                tbl.replace_with("\n[TABLE]\n" + "\n".join(rows_text) + "\n[/TABLE]\n")
            else:
                tbl.replace_with("")

        text = soup.get_text("\n", strip=True)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if len(text) > 14000:
            text = text[:14000] + "\n...[truncated]"
        url = f"https://en.wikipedia.org/wiki/{requests.utils.quote(title.replace(' ', '_'))}"
        candidates_str = ", ".join(candidates) if len(candidates) > 1 else title
        return (
            f"Title: {title}\n"
            f"URL: {url}\n"
            f"Candidates (top 3): {candidates_str}\n\n"
            f"{text}"
        )
    except Exception as e:
        return f"wikipedia_search error: {e}"
