"""
UNPLLM - Stage 3: Connectivity watcher & opportunistic sync

Run in its own terminal, alongside main.py:

    python3 sync_watcher.py

When the machine comes back online, this looks at ALL unresolved knowledge
gaps from query_logger's log, fetches relevant Wikipedia content for each,
and adds it to the same vector store main.py retrieves from.

Robustness rules in this version:
  * No network call can crash the watcher. Timeouts are retried, and a topic
    that still fails is skipped and retried later.
  * One bad topic never stops the others.
  * Topics that keep failing on CONTENT (irrelevant article, no match) are
    given up on after MAX_FAILED_ATTEMPTS so they stop slowing every sync.
    They are NOT marked resolved, so your closure-rate metric stays honest.
"""

import re
import html
import time
import json
import requests
import ollama
from datetime import datetime, timezone

from gap_engine import load_unresolved_gaps, cluster_gaps, cosine_similarity
from query_logger import mark_topic_resolved
from rag_utils import chunk_text, embed_texts, get_collection, enforce_storage_cap

CHECK_INTERVAL_SECONDS = 15
RETRY_DELAY_SECONDS = 60          # wait this long before retrying after a network failure
CONNECTIVITY_CHECK_URL = "https://www.google.com"
MAX_EXTRACT_WORDS = 2500
MAX_SYNCED_CHUNKS = 100
MAX_FAILED_ATTEMPTS = 3           # content failures per topic before giving up
SEARCH_CANDIDATES = 5             # Wikipedia results to compare, instead of trusting result #1
MODEL_NAME = "phi4-mini"
STATUS_FILE = "./sync_status.json"
ATTEMPTS_FILE = "./sync_attempts.json"

RELEVANCE_SANITY_FLOOR = 0.15

WIKI_API_URL = "https://en.wikipedia.org/w/api.php"
WIKI_HEADERS = {"User-Agent": "UNPLLM-FinalYearProject/1.0 (student project; contact: n/a)"}

# Cheap, deterministic gate that runs BEFORE the (slow, unreliable) LLM check.
REALTIME_PATTERNS = [
    r"\btoday\b", r"\bright now\b", r"\btonight\b", r"\blive score",
    r"\bprice of\b", r"\bexchange rate\b", r"\bweather\b",
    r"\btemperature (in|at)\b", r"\bforecast\b", r"\blatest news\b",
    r"\b(bit ?coin|btc|ethereum|crypto)\b.*\bprice\b",
    r"\bprice\b.*\b(bit ?coin|btc|ethereum|crypto)\b",
]
_REALTIME_RE = re.compile("|".join(REALTIME_PATTERNS), re.IGNORECASE)


# ---------------------------------------------------------------- network

def is_online():
    try:
        requests.head(CONNECTIVITY_CHECK_URL, timeout=3)
        return True
    except requests.RequestException:
        return False


def safe_get(url, params=None, timeout=(5, 20), tries=3):
    """GET with retries. Returns the response, or None if every try failed.
    Never raises a network error to the caller."""
    for attempt in range(tries):
        try:
            resp = requests.get(url, params=params, headers=WIKI_HEADERS, timeout=timeout)
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            print(f"    (network problem: {type(e).__name__}, try {attempt + 1}/{tries})")
            if attempt < tries - 1:
                time.sleep(2 * (attempt + 1))
    return None


# ---------------------------------------------------------------- status + attempts files

def write_status(is_online_now, last_sync_time=None, last_synced_topics=None):
    status = {
        "is_online": is_online_now,
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }
    try:
        with open(STATUS_FILE, "r") as f:
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
        with open(STATUS_FILE, "w") as f:
            json.dump(status, f)
    except Exception:
        pass


def load_attempts():
    try:
        with open(ATTEMPTS_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def save_attempts(attempts):
    try:
        with open(ATTEMPTS_FILE, "w") as f:
            json.dump(attempts, f)
    except Exception:
        pass


# ---------------------------------------------------------------- query understanding

def extract_search_terms(user_query):
    """Turns a natural-language question into concise Wikipedia keywords."""
    try:
        result = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": (
                    "Extract 3 to 5 concise keywords that would find the "
                    "Wikipedia article answering the user's question. Use "
                    "only words from the question, or obvious synonyms. "
                    "NEVER add people's names, dates or facts that are not "
                    "in the question. Do not write the words 'search' or "
                    "'keywords'. Reply with ONLY the keywords separated by "
                    "spaces. Example: for 'how tall is the Eiffel Tower', "
                    "reply 'Eiffel Tower height'."
                )},
                {"role": "user", "content": user_query},
            ],
            options={"temperature": 0.0, "num_predict": 30},
        )
        terms = result["message"]["content"].strip().strip('"')
        terms = re.sub(r"\b(search|keywords?)\b", "", terms, flags=re.IGNORECASE)
        terms = re.sub(r"[^\w\s'-]", " ", terms)
        terms = " ".join(terms.split()[:5])
        return terms if terms else user_query
    except Exception:
        return user_query


def is_realtime_query(query):
    """True if no static source could answer correctly (prices, weather, live scores...)."""
    if _REALTIME_RE.search(query):
        return True
    try:
        result = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": (
                    "Does answering this question require real-time or "
                    "live information (current weather, live scores, "
                    "today's news, current prices) that a static, "
                    "cached knowledge source could never correctly "
                    "provide? Reply with only YES or NO."
                )},
                {"role": "user", "content": query},
            ],
            options={"temperature": 0.0, "num_predict": 5},
        )
        return result["message"]["content"].strip().upper().startswith("YES")
    except Exception:
        return False


# ---------------------------------------------------------------- wikipedia

def find_wikipedia_candidates(search_terms, limit=SEARCH_CANDIDATES):
    """Returns a list of {title, snippet}, [] if no match, None if the network failed."""
    params = {
        "action": "query", "list": "search", "srsearch": search_terms,
        "srlimit": limit, "format": "json",
    }
    resp = safe_get(WIKI_API_URL, params, timeout=(5, 15))
    if resp is None:
        return None
    out = []
    for r in resp.json().get("query", {}).get("search", []):
        snippet = html.unescape(re.sub(r"<[^>]+>", "", r.get("snippet", "")))
        out.append({"title": r["title"], "snippet": snippet})
    return out


def pick_best_candidate(candidates, query_embedding):
    """Compares each result (title + snippet) with the ORIGINAL question and
    picks the closest, instead of blindly trusting Wikipedia's result #1."""
    texts = [f"{c['title']}. {c['snippet']}" for c in candidates]
    embs = embed_texts(texts)
    scored = [(cosine_similarity(e, query_embedding), c) for e, c in zip(embs, candidates)]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1], scored[0][0]


def fetch_wikipedia_extract(title):
    """Plain-text article. Returns None if the network failed, '' if no text."""
    params = {
        "action": "query", "prop": "extracts", "explaintext": True,
        "titles": title, "format": "json",
    }
    resp = safe_get(WIKI_API_URL, params, timeout=(5, 30))
    if resp is None:
        return None
    pages = resp.json().get("query", {}).get("pages", {})
    for page in pages.values():
        return page.get("extract", "")
    return ""


# ---------------------------------------------------------------- sync

def sync_topic(representative_query, query_embedding):
    """Returns one of:
        "resolved" - synced, or correctly recognised as unsyncable (real-time)
        "failed"   - content problem (no match / irrelevant); counts toward giving up
        "network"  - network problem; retry later, does not count toward giving up
    """
    if is_realtime_query(representative_query):
        print(f"  Skipping static Wikipedia sync (question requires current/live "
              f"information): {representative_query}")
        return "resolved"

    search_terms = extract_search_terms(representative_query)
    print(f"  Searching Wikipedia for: {representative_query}")
    print(f"    (search terms used: {search_terms})")

    candidates = find_wikipedia_candidates(search_terms)
    if candidates is None:
        print("    Wikipedia unreachable - will retry later.")
        return "network"
    if not candidates:
        print("    No Wikipedia match found - skipping this topic.")
        return "failed"

    best, title_score = pick_best_candidate(candidates, query_embedding)
    title = best["title"]
    print(f"    Candidates: {', '.join(c['title'] for c in candidates)}")
    print(f"    Chose article: {title} (title similarity {title_score:.3f})")

    extract = fetch_wikipedia_extract(title)
    if extract is None:
        print("    Article download failed - will retry later.")
        return "network"
    if not extract:
        print("    Article had no usable text - skipping.")
        return "failed"

    sample_embedding = embed_texts([extract[:1000]])[0]
    relevance = cosine_similarity(sample_embedding, query_embedding)
    print(f"    Relevance check: similarity {relevance:.3f} (floor: {RELEVANCE_SANITY_FLOOR})")
    if relevance < RELEVANCE_SANITY_FLOOR:
        print(f"    '{title}' doesn't look related enough - discarding, not stored.")
        return "failed"

    words = extract.split()
    if len(words) > MAX_EXTRACT_WORDS:
        extract = " ".join(words[:MAX_EXTRACT_WORDS])

    chunks = chunk_text(extract)
    ids = [f"sync-{title}-{i}" for i in range(len(chunks))]
    now = datetime.now(timezone.utc).isoformat()
    metadatas = [
        {"source": f"wikipedia:{title}", "chunk_index": i,
         "synced_at": now, "is_synced": True}
        for i in range(len(chunks))
    ]
    embeddings = embed_texts(chunks)

    get_collection().upsert(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)
    print(f"    Added {len(chunks)} chunk(s) from '{title}' to the knowledge base.")
    return "resolved"


def run_sync_cycle():
    """Syncs ALL unresolved topics. Returns True if some topics hit network
    trouble and a retry should be scheduled."""
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Checking for knowledge gaps...")
    gap_queries = load_unresolved_gaps()

    if not gap_queries:
        print("  No unresolved gaps right now. Nothing to sync.\n")
        return False

    clusters = cluster_gaps(gap_queries)
    total = len(clusters)
    print(f"  {total} unresolved topic(s) found. Syncing all of them.")

    attempts = load_attempts()
    synced_this_cycle = []
    needs_retry = False

    for i, cluster in enumerate(clusters, 1):
        if not is_online():
            print("\n  Connection dropped mid-sync - pausing; will resume when back online.")
            needs_retry = True
            break

        representative = cluster["queries"][0]
        key = representative["text"]
        print(f"\n  Topic {i}/{total} ({len(cluster['queries'])} related query/queries)")

        if attempts.get(key, 0) >= MAX_FAILED_ATTEMPTS:
            print(f"    Skipping - no good article found after {MAX_FAILED_ATTEMPTS} attempts: {key}")
            continue

        try:
            status = sync_topic(key, representative["embedding"])
        except requests.RequestException as e:
            print(f"    Network error ({type(e).__name__}) - will retry later.")
            status = "network"
        except Exception as e:
            print(f"    Unexpected error ({type(e).__name__}: {e}) - skipping this topic.")
            status = "failed"

        if status == "resolved":
            mark_topic_resolved([q["text"] for q in cluster["queries"]])
            synced_this_cycle.append(key)
            attempts.pop(key, None)
        elif status == "failed":
            attempts[key] = attempts.get(key, 0) + 1
        else:  # network
            needs_retry = True

    save_attempts(attempts)
    print("\nSync cycle complete.")

    try:
        evicted = enforce_storage_cap(max_synced_chunks=MAX_SYNCED_CHUNKS)
        if evicted:
            print(f"Storage cap reached - evicted {evicted} oldest synced chunk(s) "
                  f"to stay under {MAX_SYNCED_CHUNKS}.")
    except Exception as e:
        print(f"(storage cap check failed: {type(e).__name__}: {e})")

    if synced_this_cycle:
        write_status(
            is_online_now=True,
            last_sync_time=datetime.now(timezone.utc).isoformat(),
            last_synced_topics=synced_this_cycle,
        )

    if needs_retry:
        print(f"Some topics hit network problems - retrying in {RETRY_DELAY_SECONDS}s.")
    print("Resuming offline monitoring.\n")
    return needs_retry


def main():
    print("UNPLLM sync watcher started.")
    print(f"Checking connectivity every {CHECK_INTERVAL_SECONDS}s. Ctrl+C to stop.\n")

    was_online = is_online()
    print(f"Initial status: {'ONLINE' if was_online else 'OFFLINE'}")
    write_status(is_online_now=was_online)

    retry_at = 0.0
    while True:
        time.sleep(CHECK_INTERVAL_SECONDS)
        now_online = is_online()
        should_sync = False

        if now_online and not was_online:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Connectivity restored.")
            should_sync = True
        elif (not now_online) and was_online:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Connection lost - offline mode.")
        elif now_online and retry_at and time.time() >= retry_at:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Retrying topics that failed earlier...")
            should_sync = True

        if should_sync:
            try:
                needs_retry = run_sync_cycle()
            except Exception as e:  # the watcher must never die
                print(f"Sync cycle failed ({type(e).__name__}: {e}) - will retry.")
                needs_retry = True
            retry_at = time.time() + RETRY_DELAY_SECONDS if needs_retry else 0.0

        write_status(is_online_now=now_online)
        was_online = now_online


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nSync watcher stopped.")
