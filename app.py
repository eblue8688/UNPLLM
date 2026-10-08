"""
UNPLLM - Stage 4: Chat UI
Simple Streamlit interface so the sync engine's behavior is visible
without watching a terminal — this is what makes a demo legible to
someone who isn't reading your code.

Run (with main.py AND sync_watcher.py both already running):
    streamlit run app.py
"""

import json
import os
import time
import requests
import streamlit as st
from datetime import datetime, timezone

API_URL = "http://127.0.0.1:8001"
STATUS_FILE = "./sync_status.json"

st.set_page_config(page_title="UNPLLM", page_icon="🔌", layout="centered")


def load_sync_status():
    if not os.path.exists(STATUS_FILE):
        return {"is_online": None, "last_sync_time": None, "last_synced_topics": []}
    try:
        with open(STATUS_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {"is_online": None, "last_sync_time": None, "last_synced_topics": []}


def format_time_ago(iso_timestamp):
    if not iso_timestamp:
        return "never"
    try:
        then = datetime.fromisoformat(iso_timestamp)
        now = datetime.now(timezone.utc)
        seconds = int((now - then).total_seconds())
        if seconds < 60:
            return f"{seconds}s ago"
        minutes = seconds // 60
        if minutes < 60:
            return f"{minutes}m ago"
        return f"{minutes // 60}h ago"
    except Exception:
        return "unknown"


# --- Sidebar: live status, read from sync_watcher.py's status file ---
st.sidebar.title("UNPLLM Status")
status = load_sync_status()

if status.get("is_online") is True:
    st.sidebar.success("🟢 Online")
elif status.get("is_online") is False:
    st.sidebar.error("🔴 Offline")
else:
    st.sidebar.warning("⚪ Unknown — is sync_watcher.py running?")

st.sidebar.caption(f"Last synced: {format_time_ago(status.get('last_sync_time'))}")

if status.get("last_synced_topics"):
    st.sidebar.caption("Recently added:")
    for topic in status["last_synced_topics"]:
        st.sidebar.caption(f"  • {topic}")

if st.sidebar.button("Refresh status"):
    st.rerun()

st.sidebar.divider()
st.sidebar.caption("Answer source key:")
st.sidebar.caption("🟩 Original documents")
st.sidebar.caption("🟦 Synced content")
st.sidebar.caption("🟨 Weak match (gap logged)")
st.sidebar.caption("⬜ No relevant context")
st.sidebar.caption("🟧 Live API data (weather, air quality, currency)")
st.sidebar.caption("🟫 Last saved live reading (offline)")

# --- Main chat interface ---
st.title("🔌 UNPLLM")
st.caption("Offline-first local assistant with adaptive knowledge sync")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("badge"):
            st.caption(msg["badge"])

BADGES = {
    "original_documents": "🟩 Answered from original documents",
    "synced_content": "🟦 Answered using synced knowledge",
    "weak_match": "🟨 Weak match — knowledge gap logged for future sync",
    "no_context": "⬜ No relevant local context found",
    "live_api": "🟧 Live data from a free public API, fetched just now",
    "live_cached": "🟫 Offline — last reading saved earlier (see time in answer)",
    "live_unavailable": "⬛ Live data unavailable while offline",
}

prompt = st.chat_input("Ask something...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking... (CPU-only, can take a bit)"):
            try:
                start = time.perf_counter()
                resp = requests.post(
                    f"{API_URL}/generate_rag",
                    json={"prompt": prompt},
                    timeout=300,
                )
                total_ms = int((time.perf_counter() - start) * 1000)
                resp.raise_for_status()
                data = resp.json()

                answer = data["response"]
                badge = BADGES.get(data.get("answer_source"), "")

                # Total round trip, plus model-only time when an LLM was involved
                timing = f"⏱ {total_ms:,} ms total"
                gen_ns = data.get("generation_time_ns")
                if gen_ns:
                    timing += f" ({gen_ns / 1e6:,.0f} ms model)"
                badge_line = f"{badge} · {timing}" if badge else timing

                st.write(answer)
                st.caption(badge_line)

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "badge": badge_line}
                )

            except requests.RequestException as e:
                error_msg = f"Couldn't reach the API server. Is main.py running? ({e})"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
