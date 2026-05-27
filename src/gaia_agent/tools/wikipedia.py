"""위키피디아 본문(표 포함) 추출. agent_course/tools/wikipedia.py 에서 포팅."""
import re
import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool

_HEADERS = {
    "User-Agent": "GAIA-Agent/1.0 (HF agents course unit 4; https://huggingface.co/spaces)"
}


@tool
def wikipedia_search(query: str, page_title: str = "") -> str:
    """Search English Wikipedia and return the FULL article body (text + tables) of the matching
    article, rendered as readable text and truncated to ~14k chars. Includes section headers and
    table rows so factual lookups (winners lists, rosters, dates, etc.) are answerable.

    Args:
        query: The search term to find articles (used if page_title is not provided).
        page_title: The exact title of a specific Wikipedia article to fetch directly, bypassing search.
                    Use this when you know the precise title or to resolve a disambiguation warning.
    """
    try:
        title = page_title.strip()
        candidates = []
        
        if not title:
            # 1. Search Wikipedia first to find the best page matching the query
            s = requests.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "format": "json",
                    "list": "search",
                    "srsearch": query,
                    "srlimit": 5,
                },
                headers=_HEADERS,
                timeout=15,
            )
            s.raise_for_status()
            hits = s.json().get("query", {}).get("search", [])
            if not hits:
                return "No Wikipedia results."
            title = hits[0]["title"]
            candidates = [h.get("title", "") for h in hits[:5]]
        else:
            candidates = [title]

        # 2. Fetch the actual parsed article content
        page = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "parse",
                "format": "json",
                "page": title,
                "prop": "text|templates",
                "redirects": True,
            },
            headers=_HEADERS,
            timeout=20,
        )
        page.raise_for_status()
        parse_json = page.json()
        if "error" in parse_json:
            err = parse_json["error"].get("info", "Unknown parse error")
            return f"Wikipedia parse error for page '{title}': {err}. Try using a different search query."

        parsed_data = parse_json.get("parse", {})
        html = parsed_data.get("text", {}).get("*", "")
        templates = [t.get("*", "") for t in parsed_data.get("templates", [])]
        
        # Check if this is a disambiguation page
        is_disambig = any("disambig" in t.lower() for t in templates)
        
        if not html:
            return f"Top hit: {title} (no body available)."

        soup = BeautifulSoup(html, "html.parser")
        
        # Clean unwanted elements
        for tag in soup.select(
            "sup.reference, .mw-editsection, .reference, .navbox, "
            ".infobox.metadata, .hatnote, .printfooter, script, style"
        ):
            tag.decompose()

        if is_disambig:
            # Extract bullet points from disambiguation list
            options = []
            for li in soup.find_all("li"):
                li_text = li.get_text(" ", strip=True)
                if li_text:
                    options.append(f"- {li_text}")
            options_str = "\n".join(options[:15])
            
            return (
                f"WARNING: The page '{title}' is a DISAMBIGUATION PAGE. It may refer to multiple topics.\n"
                f"Please search again by calling wikipedia_search with the exact title in the `page_title` argument.\n\n"
                f"Possible topics found:\n{options_str}"
            )

        # Parse tables as markdown
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
            f"Candidates (top 5): {candidates_str}\n\n"
            f"{text}"
        )
    except Exception as e:
        return f"wikipedia_search error: {e}"
