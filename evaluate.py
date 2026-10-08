"""
UNPLLM - Stage 5: Evaluation harness (v3)
Runs every question in eval_questions.txt through three conditions:

  A_no_rag        - plain model, no retrieval
  B_rag_baseline  - RAG over originally-ingested documents only
                    (the "Stage 2 static baseline")
  C_rag_full      - RAG over the full knowledge base, incl. synced content

Both RAG conditions follow the SAME rules as main.py: the shared prompt
template from prompts.py, and context is only injected when the top match is
under GAP_THRESHOLD (otherwise the question is asked plainly).

Deterministic by design: temperature 0 and a fixed seed, so reruns give the
same answers. Each run writes its OWN timestamped CSV (no more overwriting).
Correctness stays a manual column — that judgment can't be honestly automated.

Run (main.py/app.py/sync_watcher.py should NOT be running — they steal CPU and
distort latency):
    python3 evaluate.py
"""

import csv
import time
import sqlite3
import ollama

from rag_utils import get_collection, embed_texts, retrieve_chunks
from query_logger import DB_PATH, GAP_THRESHOLD
from prompts import build_rag_prompt

MODEL = "phi4-mini"
SYSTEM_PROMPT = "You are a helpful, concise assistant running fully offline."
TOP_K = 4            # chunks retrieved for conditions B and C
NUM_PREDICT = 200    # output cap; answers can end mid-sentence — say so in the report
QUESTIONS_FILE = "eval_questions.txt"


def load_questions():
    with open(QUESTIONS_FILE, encoding="utf-8") as f:
        questions = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]
    if not questions:
        raise SystemExit(f"No questions found in {QUESTIONS_FILE}.")
    return questions


def retrieve_baseline_only(query, top_k=TOP_K):
    """Same retrieval as rag_utils.retrieve_chunks, restricted to originally
    ingested documents (is_synced=False): the Stage 2 static baseline."""
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


def make_prompt(question, retrieved):
    """Mirrors main.py: inject context only when the top match is good enough.
    Returns (prompt, context_injected)."""
    if not retrieved or retrieved[0][1] > GAP_THRESHOLD:
        return question, False
    return build_rag_prompt(question, [text for text, _ in retrieved]), True


def run_condition(question, mode):
    """mode: 'no_rag', 'baseline_rag' or 'full_rag'"""
    start = time.time()

    if mode == "no_rag":
        retrieved, prompt, injected = [], question, False
    else:
        if mode == "baseline_rag":
            retrieved = retrieve_baseline_only(question)
        else:
            full, _ = retrieve_chunks(question, top_k=TOP_K)
            retrieved = [(r["text"], r["distance"]) for r in full]
        prompt, injected = make_prompt(question, retrieved)

    result = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        options={"num_predict": NUM_PREDICT, "temperature": 0, "seed": 42},
    )

    return {
        "answer": result["message"]["content"],
        "latency_sec": round(time.time() - start, 2),
        # Ollama's own timing split — separates reading the prompt from writing the answer
        "prefill_sec": round((result.get("prompt_eval_duration") or 0) / 1e9, 2),
        "generate_sec": round((result.get("eval_duration") or 0) / 1e9, 2),
        "prompt_tokens": result.get("prompt_eval_count"),
        "output_tokens": result.get("eval_count"),
        "chunks_retrieved": len(retrieved),
        "context_injected": injected,
        "top_distance": round(retrieved[0][1], 4) if retrieved else None,
    }


def compute_gap_closure_rate():
    """Signature metric: of all gaps ever logged, what % are now resolved?
    Read from the real query log."""
    conn = sqlite3.connect(DB_PATH)
    total = conn.execute("SELECT COUNT(*) FROM query_log WHERE is_gap = 1").fetchone()[0]
    resolved = conn.execute(
        "SELECT COUNT(*) FROM query_log WHERE is_gap = 1 AND resolved = 1").fetchone()[0]
    conn.close()
    return total, resolved, (resolved / total * 100) if total else 0


def main():
    questions = load_questions()
    print(f"{len(questions)} questions x 3 conditions = {len(questions) * 3} generations "
          f"(CPU: expect minutes per question).\n")

    print("Warming up the model (not recorded)...")
    ollama.chat(model=MODEL, messages=[{"role": "user", "content": "hi"}], options={"num_predict": 5})

    rows = []
    for i, question in enumerate(questions, start=1):
        print(f"[{i}/{len(questions)}] {question}")
        for label, mode, name in (("A: no RAG", "no_rag", "A_no_rag"),
                                  ("B: RAG, original docs only", "baseline_rag", "B_rag_baseline"),
                                  ("C: RAG, full knowledge base", "full_rag", "C_rag_full")):
            print(f"  {label}...")
            r = run_condition(question, mode)
            rows.append({
                "question": question, "condition": name, "answer": r["answer"],
                "latency_sec": r["latency_sec"], "prefill_sec": r["prefill_sec"],
                "generate_sec": r["generate_sec"], "prompt_tokens": r["prompt_tokens"],
                "output_tokens": r["output_tokens"], "chunks_retrieved": r["chunks_retrieved"],
                "context_injected": r["context_injected"], "top_distance": r["top_distance"],
                "correctness_score_1to5": "",   # fill in by hand: 5 correct+complete, 3 partial/hedged, 1 wrong/invented
                "unsupported_claims_y_n": "",   # y if it states facts not in its context / invents details
                "notes": "",
            })

    out = f"evaluation_results_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {out}")

    print(f"\nKnowledge-gap rate by condition (top distance > {GAP_THRESHOLD}):")
    for name in ("B_rag_baseline", "C_rag_full"):
        ds = [r["top_distance"] for r in rows if r["condition"] == name]
        gaps = sum(1 for d in ds if d is None or d > GAP_THRESHOLD)
        print(f"  {name}: {gaps}/{len(ds)} questions ({gaps / len(ds) * 100:.0f}%)")

    total, resolved, rate = compute_gap_closure_rate()
    print(f"\nKnowledge-gap closure rate (from the query log): {resolved}/{total} resolved ({rate:.1f}%)")


if __name__ == "__main__":
    main()
