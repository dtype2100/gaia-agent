"""웹페이지 fetch + 마크다운 변환. agent_course/tools/webpage.py 에서 포팅."""
import io
import re
import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from langchain_core.tools import tool


def _handle_pdf_url(content: bytes) -> str:
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(content))
        parts = []
        for i, page in enumerate(reader.pages):
            try:
                txt = page.extract_text() or ""
            except Exception as pe:
                txt = f"(extraction failed: {pe})"
            parts.append(f"--- Page {i+1} ---\n{txt}")
        combined = "\n\n".join(parts)
        if len(combined) > 12000:
            combined = combined[:12000] + "\n...[truncated]"
        return f"[PDF, {len(reader.pages)} pages]\n{combined}"
    except Exception as e:
        return f"PDF parse error: {e}"


@tool
def visit_webpage(url: str) -> str:
    """Fetch a web page (HTML or PDF) and return its readable text (truncated to ~12k chars).

    HTML pages are converted to markdown. PDF URLs are parsed page-by-page via pypdf —
    useful for arxiv papers, NASA technical reports, and other linked PDF documents.

    Args:
        url: The full URL of the webpage or PDF to fetch.
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; GAIA-Agent/1.0)"}
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "").lower()
        if "application/pdf" in content_type or url.lower().endswith(".pdf"):
            return _handle_pdf_url(resp.content)
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        markdown = md(str(soup))
        markdown = re.sub(r"\n{3,}", "\n\n", markdown).strip()
        if len(markdown) > 12000:
            markdown = markdown[:12000] + "\n...[truncated]"
        return markdown
    except Exception as e:
        return f"visit_webpage error: {e}"
