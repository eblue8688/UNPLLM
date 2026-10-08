"""UNPLLM Stage 3: connectivity watcher and opportunistic Wikipedia sync."""

import json
import re
import time
from datetime import datetime, timezone

import ollama
import requests

from gap_engine import load_unresolved_gaps, cluster_gaps, cosine_similarity
from query_logger import mark_topic_resolved
from rag_utils import chunk_text, embed_texts, get_collection, enforce_storage_cap

CHECK_INTERVAL_SECONDS = 15
CONNECTIVITY_CHECK_URL = "https://www.google.com"
TOP_N_TOPICS_PER_SYNC = None  # None = sync ALL unresolved topics
MAX_EXTRACT_WORDS = 2500
MAX_SYNCED_CHUNKS = 100
MODEL_NAME = "phi4-mini"
STATUS_FILE = "./sync_status.json"
RELEVANCE_SANITY_FLOOR = 0.15

WIKI_API_URL = "https://en.wikipedia.org/w/api.php"
WIKI_HEADERS = {"User-Agent": "UNPLLM-FinalYearProject/1.0 (student project; contact: n/a)"}
WIKI_MAX_RETRIES = 2
WIKI_SEARCH_TIMEOUT = 10
WIKI_EXTRACT_TIMEOUT = 15

# Deterministic first-line protection for questions that should never be
# cached as static Wikipedia knowledge.
REALTIME_PATTERNS = [
    r"\bcurrent\b", r"\bcurrently\b", r"\bright now\b", r"\btoday\b",
    r"\btonight\b", r"\byesterday\b", r"\btomorrow\b", r"\blatest\b",
    r"\brecent\b", r"\bthis week\b", r"\bthis month\b", r"\bthis year\b",
    r"\bwho is the (?:current|new)\b", r"\bwhat is the current\b",
    r"\bwhat's the current\b", r"\bprice\b", r"\bweather\b",
    r"\btemperature\b", r"\blive score\b", r"\blive scores\b",
    r"\bexchange rate\b", r"\bstock price\b", r"\btoday'?s news\b",
]


def is_online():
    try:
        requests.head(CONNECTIVITY_CHECK_URL, timeout=3, allow_redirects=True)
        return True
    except requests.RequestException:
        return False


def write_status(is_online_now, last_sync_time=None, last_synced_topics=None):
    status = {
        "is_online": is_online_now,
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }
    try:
        with open(STATUS_FILE, "r", encoding="utf-8") as f:
            previous = json.load(f)
        status["last_sync_time"] = previous.get("last_sync_time")
        status["last_synced_topics"] = previous.get("last_synced_topics", [])
    except Exception:
        status["last_sync_time"] = None
        status["last_synced_topics"] = []
    if last_sync_time is not None:
        status["last_sync_time"] = last_sync_time
    if last_synced_topics is not None:
        status["last_synced_topics"] = last_synced_topics
    try:
        with open(STATUS_FILE, "w", encoding="utf-8") as f:
            json.dump(status, f, indent=2)
    except Exception:
        pass


def has_realtime_language(query):
    text = query.lower().strip()
    return any(re.search(pattern, text) for pattern in REALTIME_PATTERNS)


def is_realtime_query(query):
    """Deterministic live/current detection first, local-model classifier second."""
    if has_realtime_language(query):
        return True
    try:
        result = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": (
                    "Classify the user's question. Reply with exactly YES or NO. "
                    "Reply YES only when answering requires information that can "
                    "change over time, such as current weather, today's news, "
                    "live sports, current prices, current political office holders, "
                    "or present-day status. Reply NO for stable historical, "
                    "scientific, educational, or general-knowledge questions."
                )},
                {"role": "user", "content": query},
            ],
            options={"temperature": 0.0, "num_predict": 5},
        )
        return result["message"]["content"].strip().upper().startswith("YES")
    except Exception:
        return False


def extract_search_terms(user_query):
    """Generate concise Wikipedia keywords; relevance is checked later."""
    try:
        result = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": (
                    "Extract exactly 3 to 5 concise search keywords that would "
                    "find the Wikipedia article answering the user's question. "
                    "Reply ONLY with keywords separated by spaces. No punctuation. "
                    "No explanation. Preserve important names, entities and places. "
                    "Do NOT add dates, countries, organizations, events, or facts "
                    "that are not explicitly present or clearly implied.\n"
                    "Example: how tall is the Eiffel Tower -> Eiffel Tower height"
                )},
                {"role": "user", "content": user_query},
            ],
            options={"temperature": 0.0, "num_predict": 30},
        )
        terms = result["message"]["content"].strip().strip('"')
        terms = " ".join(terms.split()[:5])
        return terms if terms else user_query
    except Exception:
        return user_query


def _wiki_get(params, timeout, operation_name):
    """Retry Wikipedia requests and contain all network failures."""
    last_error = None
    for attempt in range(1, WIKI_MAX_RETRIES + 1):
        try:
            response = requests.get(
                WIKI_API_URL,
                params=params,
                headers=WIKI_HEADERS,
                timeout=timeout,
            )
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt < WIKI_MAX_RETRIES:
                wait_seconds = attempt * 2
                print(
                    f"    Wikipedia {operation_name} failed "
                    f"(attempt {attempt}/{WIKI_MAX_RETRIES}): "
                    f"{type(exc).__name__}. Retrying in {wait_seconds}s..."
                )
                time.sleep(wait_seconds)
    print(
        f"    Wikipedia {operation_name} failed after {WIKI_MAX_RETRIES} attempt(s): "
        f"{type(last_error).__name__}: {last_error}"
    )
    print("    Topic remains unresolved and can be retried later.")
    return None


def find_wikipedia_title(query):
    params = {"action": "query", "list": "search", "srsearch": query,
              "srlimit": 1, "format": "json"}
    data = _wiki_get(params, WIKI_SEARCH_TIMEOUT, "search")
    if not data:
        return None
    results = data.get("query", {}).get("search", [])
    return results[0]["title"] if results else None


def fetch_wikipedia_extract(title):
    params = {"action": "query", "prop": "extracts", "explaintext": True,
              "titles": title, "format": "json"}
    data = _wiki_get(params, WIKI_EXTRACT_TIMEOUT, "article fetch")
    if not data:
        return ""
    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        return page.get("extract", "") or ""
    return ""


def sync_topic(representative_query, query_embedding):
    """Sync one topic. False means keep unresolved for a later retry."""
    if is_realtime_query(representative_query):
        print(
            "  Skipping static Wikipedia sync (question requires current/live "
            f"information): {representative_query}"
        )
        # Static Wikipedia cannot satisfy this gap; do not keep retrying it.
        return True

    search_terms = extract_search_terms(representative_query)
    print(f"  Searching Wikipedia for: {representative_query}")
    print(f"    (search terms used: {search_terms})")

    try:
        title = find_wikipedia_title(search_terms)
        if not title:
            print("    No Wikipedia match found — keeping topic unresolved.")
            return False

        print(f"    Found article: {title}")
        extract = fetch_wikipedia_extract(title)
        if not extract:
            print("    Article could not be fetched — keeping topic unresolved.")
            return False

        # Compare fetched content to the ORIGINAL question, not generated keywords.
        sample_embedding = embed_texts([extract[:2000]])[0]
        relevance = cosine_similarity(sample_embedding, query_embedding)
        print(
            f"    Relevance check: similarity {relevance:.3f} "
            f"(floor: {RELEVANCE_SANITY_FLOOR})"
        )
        if relevance < RELEVANCE_SANITY_FLOOR:
            print(f"    '{title}' doesn't look related enough — discarding, not stored.")
            return False

        words = extract.split()
        if len(words) > MAX_EXTRACT_WORDS:
            extract = " ".join(words[:MAX_EXTRACT_WORDS])

        chunks = chunk_text(extract)
        if not chunks:
            print("    Article produced no usable chunks.")
            return False

        ids = [f"sync-{title}-{i}" for i in range(len(chunks))]
        synced_at = datetime.now(timezone.utc).isoformat()
        metadatas = [
            {"source": f"wikipedia:{title}", "chunk_index": i,
             "synced_at": synced_at, "is_synced": True}
            for i in range(len(chunks))
        ]

        embeddings = embed_texts(chunks)
        collection = get_collection()
        collection.upsert(ids=ids, embeddings=embeddings,
                          documents=chunks, metadatas=metadatas)
        print(f"    Added {len(chunks)} chunk(s) from '{title}' to the knowledge base.")
        return True

    except Exception as exc:
        # Final per-topic guard: one bad topic can never kill the watcher.
        print(f"    Unexpected error while syncing this topic: {type(exc).__name__}: {exc}")
        print("    Topic remains unresolved for a future retry.")
        return False


def run_sync_cycle():
    print(
        f"\n[{datetime.now().strftime('%H:%M:%S')}] "
        "Back online — checking for knowledge gaps..."
    )

    try:
        gap_queries = load_unresolved_gaps()
    except Exception as exc:
        print(f"  Could not load knowledge gaps: {type(exc).__name__}: {exc}")
        print("  Sync cycle aborted safely; watcher will continue.\n")
        return

    if not gap_queries:
        print("  No unresolved gaps right now. Nothing to sync.\n")
        return

    try:
        clusters = cluster_gaps(gap_queries)
    except Exception as exc:
        print(f"  Could not cluster knowledge gaps: {type(exc).__name__}: {exc}")
        print("  Sync cycle aborted safely; watcher will continue.\n")
        return

    # Sync every unresolved topic, not just the top three.
    # Clustering is still used to group duplicate/similar questions so
    # each topic is fetched once per cycle.
    top_clusters = clusters if TOP_N_TOPICS_PER_SYNC is None else clusters[:TOP_N_TOPICS_PER_SYNC]
    print(f"  {len(clusters)} unresolved topic(s) found. Syncing all {len(top_clusters)} topic(s).")

    synced_this_cycle = []
    for index, cluster in enumerate(top_clusters, start=1):
        representative = cluster["queries"][0]
        print(f"\n  Topic {index}/{len(top_clusters)} ({len(cluster['queries'])} related query/queries)")
        try:
            success = sync_topic(representative["text"], representative["embedding"])
            if not success:
                continue

            all_texts = [q["text"] for q in cluster["queries"]]
            try:
                mark_topic_resolved(all_texts)
            except Exception as exc:
                print(f"    Could not mark topic resolved: {type(exc).__name__}: {exc}")
                continue
            synced_this_cycle.append(representative["text"])
        except Exception as exc:
            print(f"    Topic failed safely: {type(exc).__name__}: {exc}")
            print("    Continuing with the next topic...")

    print("\nSync cycle complete.")

    try:
        evicted = enforce_storage_cap(max_synced_chunks=MAX_SYNCED_CHUNKS)
        if evicted:
            print(
                f"Storage cap reached — evicted {evicted} oldest synced chunk(s) "
                f"to stay under {MAX_SYNCED_CHUNKS}."
            )
    except Exception as exc:
        print(f"Storage-cap enforcement failed safely: {type(exc).__name__}: {exc}")

    if synced_this_cycle:
        write_status(
            is_online_now=True,
            last_sync_time=datetime.now(timezone.utc).isoformat(),
            last_synced_topics=synced_this_cycle,
        )

    print("Resuming offline monitoring.\n")


def main():
    print("UNPLLM sync watcher started.")
    print(f"Checking connectivity every {CHECK_INTERVAL_SECONDS}s. Ctrl+C to stop.\n")

    was_online = is_online()
    print(f"Initial status: {'ONLINE' if was_online else 'OFFLINE'}")
    write_status(is_online_now=was_online)

    while True:
        time.sleep(CHECK_INTERVAL_SECONDS)
        now_online = is_online()

        if now_online and not was_online:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Connectivity restored.")
            try:
                run_sync_cycle()
            except Exception as exc:
                print(
                    "Sync cycle crashed unexpectedly, but the watcher will continue: "
                    f"{type(exc).__name__}: {exc}\n"
                )
        elif (not now_online) and was_online:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Connection lost — offline mode.")

        write_status(is_online_now=now_online)
        was_online = now_online


if __name__ == "__main__":
    main()
