"""
UNPLLM - Stage 1: Hardware benchmark script
Measures tokens/sec, latency, and RAM usage for your chosen local model.
This produces the numbers for your report's "Results" Table 1.

Run (model must already be pulled via `ollama pull <model>` and `ollama serve` running):
    python benchmark.py
"""

import time
import psutil
import ollama

MODEL = "phi4-mini"  # change to match the model you're benchmarking

TEST_PROMPTS = [
    "Explain the difference between supervised and unsupervised learning.",
    "What is a hash table and why is it fast?",
    "Summarize the plot of a story about a robot learning empathy.",
    "Write a two-sentence explanation of recursion.",
    "What causes inflation in an economy?",
    # Add ~10 more prompts representative of your project's actual use case
    # (e.g. questions similar to what UNPLLM will be asked in your demo)
]


def get_ram_usage_mb():
    return psutil.virtual_memory().used / (1024 * 1024)


def run_benchmark():
    results = []
    for i, prompt in enumerate(TEST_PROMPTS, start=1):
        print(f"  [{i}/{len(TEST_PROMPTS)}] Running: {prompt[:50]}...", flush=True)

        ram_before = get_ram_usage_mb()
        start = time.time()

        response = ollama.generate(
            model=MODEL,
            prompt=prompt,
            options={"num_predict": 200},  # caps output so each prompt finishes in seconds
        )

        elapsed = time.time() - start
        ram_after = get_ram_usage_mb()

        tokens = response.get("eval_count", 0)
        tokens_per_sec = tokens / elapsed if elapsed > 0 else 0

        print(f"      done in {elapsed:.1f}s — {tokens_per_sec:.2f} tok/s", flush=True)

        results.append({
            "prompt": prompt[:40] + "...",
            "latency_sec": round(elapsed, 2),
            "tokens_generated": tokens,
            "tokens_per_sec": round(tokens_per_sec, 2),
            "ram_delta_mb": round(ram_after - ram_before, 1),
        })

    return results


if __name__ == "__main__":
    print(f"Benchmarking model: {MODEL}\n")
    results = run_benchmark()

    for r in results:
        print(r)

    avg_tps = sum(r["tokens_per_sec"] for r in results) / len(results)
    avg_latency = sum(r["latency_sec"] for r in results) / len(results)
    print(f"\nAverage tokens/sec: {avg_tps:.2f}")
    print(f"Average latency: {avg_latency:.2f}s")
