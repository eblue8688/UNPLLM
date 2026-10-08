"""
UNPLLM - does the model actually USE its retrieved context?
Runs the OLD and NEW RAG prompts on the same retrieved chunks and prints the
answers side by side, so you can see the difference with your own model.
Takes a few minutes on CPU (3 variants x each question). Stop main.py,
app.py and sync_watcher.py first so they don't slow it down.

    python3 prompt_compare.py
Edit QUESTIONS below to test others.
"""

import time
import ollama

from rag_utils import retrieve_chunks
from prompts import build_rag_prompt

MODEL = "phi4-mini"
SYSTEM_PROMPT = "You are a helpful, concise assistant running fully offline."
QUESTIONS = [
    "which macbook was the latest from apple",
    "What is the latest iPhone model?",
]


def old_prompt(question, chunks):
    context = "\n\n".join(f"- {c}" for c in chunks)
    return ("Use the following context if it's relevant to answer the question. "
            "If the context doesn't help, answer normally.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}")


def ask(prompt):
    start = time.time()
    r = ollama.chat(
        model=MODEL,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        options={"num_predict": 200, "temperature": 0, "seed": 42},
    )
    return r["message"]["content"], time.time() - start


def main():
    ollama.chat(model=MODEL, messages=[{"role": "user", "content": "hi"}], options={"num_predict": 5})
    for q in QUESTIONS:
        retrieved, _ = retrieve_chunks(q, top_k=4)
        chunks = [r["text"] for r in retrieved]
        print("=" * 90)
        print("QUESTION:", q)
        print("top distances:", [round(r["distance"], 3) for r in retrieved])
        for label, prompt in (
            ("OLD prompt, 4 chunks", old_prompt(q, chunks)),
            ("NEW prompt, 4 chunks", build_rag_prompt(q, chunks)),
            ("NEW prompt, top 2 chunks", build_rag_prompt(q, chunks[:2])),
        ):
            answer, secs = ask(prompt)
            print(f"\n--- {label} ({secs:.0f}s) ---\n{answer}")


if __name__ == "__main__":
    main()
