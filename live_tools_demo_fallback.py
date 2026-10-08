"""
UNPLLM - lightweight live tools used by main.py.

No API key is required for the demo paths below.
"""

import re
from datetime import datetime, timezone
import requests

WIKI_API_URL = "https://en.wikipedia.org/w/api.php"
HEADERS = {
    "User-Agent": "UNPLLM-Demo/1.0 (local student project)"
}


def _wiki_get(params, timeout=8):
    try:
        r = requests.get(
            WIKI_API_URL,
            params=params,
            headers=HEADERS,
            timeout=timeout,
        )
        r.raise_for_status()
        return r.json()
    except (requests.RequestException, ValueError):
        return None


def _wiki_search(term):
    data = _wiki_get({
        "action": "query",
        "list": "search",
        "srsearch": term,
        "srlimit": 1,
        "format": "json",
    })
    if not data:
        return None
    hits = data.get("query", {}).get("search", [])
    return hits[0]["title"] if hits else None


def _wiki_extract(title):
    data = _wiki_get({
        "action": "query",
        "prop": "extracts|info",
        "explaintext": 1,
        "exintro": 1,
        "inprop": "url",
        "titles": title,
        "format": "json",
    })
    if not data:
        return None
    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        text = (page.get("extract") or "").strip()
        url = page.get("fullurl") or f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
        if text:
            return text, url
    return None


def _is_current_query(q):
    patterns = [
        r"\b(current|currently|right now|today|latest|newest|recent|recently)\b",
        r"\b20\d{2}\b",
        r"\bwho is (the )?(current )?(pm|prime minister|president|chief minister)\b",
        r"\b(pm|prime minister) of (india|canada|uk|united states|usa)\b",
        r"\b(current|latest) (price|rate|score|weather|temperature|news|exchange rate)\b",
    ]
    return any(re.search(p, q, re.IGNORECASE) for p in patterns)


def _handle_pm_query(q):
    lower = q.lower()
    if not ("pm" in lower or "prime minister" in lower):
        return None
    countries = {
        "india": "Prime Minister of India",
        "canada": "Prime Minister of Canada",
        "united states": "President of the United States",
        "usa": "President of the United States",
        "uk": "Prime Minister of the United Kingdom",
    }
    for country, topic in countries.items():
        if country in lower:
            title = _wiki_search(topic)
            if not title:
                return {
                    "text": "Live lookup is currently unavailable. I will not guess.",
                    "tool": "wikipedia",
                    "status": "unavailable",
                }
            extracted = _wiki_extract(title)
            if not extracted:
                return {
                    "text": "Live lookup is currently unavailable. I will not guess.",
                    "tool": "wikipedia",
                    "status": "unavailable",
                }
            text, url = extracted
            return {
                "text": (
                    f"Live Wikipedia lookup ({datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')})\\n\\n"
                    f"{text}\\n\\nSource: {url}"
                ),
                "tool": "wikipedia",
                "status": "live",
            }
    return None


def try_live_tools(prompt):
    """Return a live-tool result dict, or None when this is not a supported live query."""
    q = prompt.strip()
    if not _is_current_query(q):
        return None

    # First-class deterministic handler for the demo's current PM question.
    pm = _handle_pm_query(q)
    if pm is not None:
        return pm

    # Generic current-information query: do not silently fall back to RAG.
    title = _wiki_search(q)
    if title:
        extracted = _wiki_extract(title)
        if extracted:
            text, url = extracted
            return {
                "text": f"Live Wikipedia lookup\\n\\n{text}\\n\\nSource: {url}",
                "tool": "wikipedia",
                "status": "live",
            }

    return {
        "text": "Live lookup is currently unavailable for this current-information question. I will not guess.",
        "tool": "wikipedia",
        "status": "unavailable",
    }
