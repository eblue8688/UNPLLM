"""
UNPLLM - Stage 5: Evaluation harness
Runs a fixed set of test questions through three conditions and
records latency, RAM, and retrieval metadata for each:

  A_no_rag        - plain model, no retrieval at all
  B_rag_baseline  - RAG restricted to originally-ingested documents
                    only (the "Stage 2 static baseline" condition)
  C_rag_full      - RAG against the full knowledge base, including
                    everything the sync engine has fetched

Correctness scoring is deliberately left as a manual column — that
judgment can't be honestly automated. Latency, RAM, retrieval count,
and top distance are all measured directly, not estimated.

Run (main.py doesn't need to be running — this calls Ollama and the
vector store directly):
    python3 evaluate.py
Produces: evaluation_results.csv
"""

import csv
import time
import sqlite3
import psutil
import ollama

from rag_utils import get_collection, embed_texts, retrieve_chunks
from query_logger import DB_PATH

MODEL = "phi4-mini"
SYSTEM_PROMPT = "You are a helpful, concise assistant running fully offline."
TOP_K = 4  # chunks fed to the model in conditions B and C. Latency on CPU scales
           # heavily with this — re-run with 2 to measure the tradeoff.

# Replace with ~15-20 real questions representative of your project.
# Mix: some your original documents should answer well, some that
# were gaps you've since synced (e.g. "What is the Turing machine?"
# if you ran that demo), and a couple of genuinely hard/edge cases.
TEST_QUESTIONS = [
    "What is UNPLLM?",
    "What is the Turing machine?",
    "What is anoxygenic photosynthesis?",
    "What is the latest iPhone model?",
    # add your own here
]


def get_ram_mb():
    return psutil.virtual_memory().used / (1024 * 1024)


def retrieve_baseline_only(query, top_k=TOP_K):
    """Same retrieval as rag_utils.retrieve_chunks, but restricted to
    originally-ingested documents (is_synced=False) — this is the
    Stage 2 'static baseline' condition for comparison."""
    collection = get_collection()
    query_embedding = embed_texts([query])[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where={"is_synced": False},
    )
    docs = results["documents"][0] if results["documents"] else []
    dists = results["distances"][0] if results["distances"] else []
    return list(zip(docs, dists))


def build_rag_prompt(question, retrieved):
    """Same template used in main.py's /generate_rag — keeping this
    identical matters, otherwise you'd be evaluating a different
    system than the one you're shipping."""
    if not retrieved:
        return question
    context = "\n\n".join(f"- {text}" for text, _ in retrieved)
    return (
        f"Use the following context if it's relevant to answer the "
        f"question. If the context doesn't help, answer normally.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )


def run_condition(question, mode):
    """mode: 'no_rag', 'baseline_rag', or 'full_rag'"""
    start = time.time()

    if mode == "no_rag":
        prompt = question
        retrieved = []
    elif mode == "baseline_rag":
        retrieved = retrieve_baseline_only(question)
        prompt = build_rag_prompt(question, retrieved)
    else:  # full_rag
        full_retrieved, _ = retrieve_chunks(question, top_k=TOP_K)
        retrieved = [(r["text"], r["distance"]) for r in full_retrieved]
        prompt = build_rag_prompt(question, retrieved)

    result = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        options={"num_predict": 200},
    )

    elapsed = time.time() - start
    ram_after = get_ram_mb()

    return {
        "answer": result["message"]["content"],
        "latency_sec": round(elapsed, 2),
        "system_ram_used_mb": round(ram_after),
        "chunks_retrieved": len(retrieved),
        "top_distance": round(retrieved[0][1], 4) if retrieved else None,
    }


def compute_gap_closure_rate():
    """Your signature metric: of all gaps ever logged, what percentage
    are now resolved? Computed directly from the real query log."""
    conn = sqlite3.connect(DB_PATH)
    total = conn.execute("SELECT COUNT(*) FROM query_log WHERE is_gap = 1").fetchone()[0]
    resolved = conn.execute(
        "SELECT COUNT(*) FROM query_log WHERE is_gap = 1 AND resolved = 1"
    ).fetchone()[0]
    conn.close()
    rate = (resolved / total * 100) if total > 0 else 0
    return total, resolved, rate


def main():
    print(f"Running {len(TEST_QUESTIONS)} questions across 3 conditions "
          f"({len(TEST_QUESTIONS) * 3} total generations — this will take a while on CPU)...\n")
    rows = []

    # Warm-up: the first generation includes loading the model into RAM
    # (~3 GB, tens of seconds on CPU). Without this, the very first
    # row's latency and RAM figures are a cold-start artifact.
    print("Warming up the model (not recorded)...")
    ollama.chat(model=MODEL, messages=[{"role": "user", "content": "hi"}], options={"num_predict": 5})

    for i, question in enumerate(TEST_QUESTIONS, start=1):
        print(f"[{i}/{len(TEST_QUESTIONS)}] {question}")

        print("  A: plain model, no RAG...")
        a = run_condition(question, "no_rag")

        print("  B: RAG, baseline only (original documents)...")
        b = run_condition(question, "baseline_rag")

        print("  C: RAG, full knowledge base (with synced content)...")
        c = run_condition(question, "full_rag")

        for condition_name, result in [("A_no_rag", a), ("B_rag_baseline", b), ("C_rag_full", c)]:
            rows.append({
                "question": question,
                "condition": condition_name,
                "answer": result["answer"],
                "latency_sec": result["latency_sec"],
                "system_ram_used_mb": result["system_ram_used_mb"],
                "chunks_retrieved": result["chunks_retrieved"],
                "top_distance": result["top_distance"],
                "correctness_score_1to5": "",  # fill in by hand after reading each answer
                "unsupported_claims_y_n": "",  # y if the answer states facts not in the retrieved context
                "notes": "",
            })

    with open("evaluation_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to evaluation_results.csv")
    print("Open it in a spreadsheet, read each answer, and fill in "
          "correctness_score_1to5 (1-5, your own judgment) — that's "
          "the one column that has to stay human.")

    total, resolved, rate = compute_gap_closure_rate()
    print(f"\nKnowledge-gap closure rate: {resolved}/{total} gaps resolved ({rate:.1f}%)")
    print("This is your signature metric — cite it directly in your results chapter.")


if __name__ == "__main__":
    main()
