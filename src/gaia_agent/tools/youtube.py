"""YouTube 메타데이터 + 자막. agent_course/tools/youtube.py 에서 포팅."""
import re
import requests
from langchain_core.tools import tool


def _extract_video_id(url_or_id: str):
    s = url_or_id.strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", s):
        return s
    m = re.search(
        r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/|youtube\.com/shorts/)([A-Za-z0-9_-]{11})",
        s,
    )
    if m:
        return m.group(1)
    m = re.search(r"[?&]v=([A-Za-z0-9_-]{11})", s)
    if m:
        return m.group(1)
    return None


def _fetch_metadata(video_id: str) -> str:
    try:
        r = requests.get(
            "https://www.youtube.com/oembed",
            params={
                "url": f"https://www.youtube.com/watch?v={video_id}",
                "format": "json",
            },
            timeout=15,
        )
        r.raise_for_status()
        meta = r.json()
        title = meta.get("title", "")
        author = meta.get("author_name", "")
        return f"Title: {title}\nChannel: {author}"
    except Exception as e:
        return f"Metadata fetch failed: {e}"


def _fetch_transcript(video_id: str) -> str:
    try:
        from youtube_transcript_api import YouTubeTranscriptApi

        try:
            segments = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        except Exception:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            chosen = None
            for t in transcript_list:
                try:
                    if getattr(t, "is_translatable", False):
                        chosen = t.translate("en").fetch()
                        break
                except Exception:
                    continue
            if chosen is None:
                first = next(iter(transcript_list))
                chosen = first.fetch()
            segments = chosen

        texts = []
        for seg in segments:
            if isinstance(seg, dict):
                texts.append(seg.get("text", ""))
            else:
                texts.append(getattr(seg, "text", ""))
        text = " ".join(t for t in texts if t).strip()
        if not text:
            return "Transcript unavailable (empty)."
        if len(text) > 14000:
            text = text[:14000] + "\n...[truncated]"
        return text
    except Exception as e:
        return f"Transcript unavailable: {e}"


@tool
def youtube_info(url: str) -> str:
    """Fetch a YouTube video's title, channel, and transcript text.
    Use this whenever a question references a YouTube link, video, or asks about its contents.
    The transcript covers spoken content (auto-generated if no manual captions exist) and is
    truncated to ~14k chars. If the video has no captions, only metadata is returned and you
    should fall back to web/wikipedia searches for the question's specific facts.

    Args:
        url: A full YouTube URL (watch, youtu.be, embed, shorts) or a bare 11-character video ID.
    """
    video_id = _extract_video_id(url)
    if not video_id:
        return f"Could not parse YouTube video ID from: {url}"
    metadata = _fetch_metadata(video_id)
    transcript = _fetch_transcript(video_id)
    return (
        f"Video ID: {video_id}\n"
        f"URL: https://www.youtube.com/watch?v={video_id}\n"
        f"{metadata}\n\n"
        f"--- Transcript ---\n"
        f"{transcript}"
    )
