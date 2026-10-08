"""
UNPLLM - Stage 1: Core local inference service
Wraps a local Ollama model behind a simple FastAPI endpoint.

Setup:
    1. ollama serve                  (run in a separate terminal, keep it open)
    2. ollama pull phi4-mini         (or qwen3:4b / gemma3:4b depending on your RAM)
    3. Run it EITHER of these two ways:
         python3 main.py
         uvicorn main:app --reload --port 8001
       Both do the same thing. Either way, the terminal will show
       "Uvicorn running on http://127.0.0.1:8001" and then sit there
       waiting for requests — that's correct, it's a server, not a
       one-shot script. It won't return to your prompt; open a
       second terminal to test it or stop it with Ctrl+C.

       Port is set once below (PORT constant) rather than hardcoded
       in multiple places — change it there if 8001 is ever
       contested too, and update app.py's API_URL to match.

Test:
    curl -X POST http://localhost:8001/generate \
      -H "Content-Type: application/json" \
      -d '{"prompt": "Explain recursion in two sentences."}'
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import ollama
from rag_utils import retrieve_chunks
from query_logger import log_query
from live_tools import try_live_tools
from prompts import build_rag_prompt

PORT = 8001  # moved off 8000 due to a port conflict with an unrelated process

app = FastAPI(title="UNPLLM Core Inference Service")

# Change this to match the model you pulled in Week 2
DEFAULT_MODEL = "phi4-mini"
DEFAULT_SYSTEM_PROMPT = "You are a helpful, concise assistant running fully offline."


class GenerateRequest(BaseModel):
    prompt: str
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    temperature: float = 0.7
    max_tokens: int = 512
    model: str = DEFAULT_MODEL


@app.get("/health")
def health():
    """Quick check that the service is up and which model is default."""
    return {"status": "ok", "default_model": DEFAULT_MODEL}


@app.post("/generate")
def generate(req: GenerateRequest):
    """Send a prompt to the local model and return its response."""
    try:
        result = ollama.chat(
            model=req.model,
            messages=[
                {"role": "system", "content": req.system_prompt},
                {"role": "user", "content": req.prompt},
            ],
            options={
                "temperature": req.temperature,
                "num_predict": req.max_tokens,
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ollama call failed: {e}")

    return {
        "response": result["message"]["content"],
        "model": req.model,
        "tokens_generated": result.get("eval_count"),
        "generation_time_ns": result.get("eval_duration"),
    }


class RagRequest(BaseModel):
    prompt: str
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    temperature: float = 0.7
    max_tokens: int = 512
    model: str = DEFAULT_MODEL
    top_k: int = 4


@app.post("/generate_rag")
def generate_rag(req: RagRequest):
    """Retrieve relevant chunks from the local knowledge base, then
    generate an answer grounded in that context. Falls back to a
    plain (non-augmented) prompt if the knowledge base is empty."""
    # Real-time questions (weather, air quality, exchange rates) are answered by
    # live tools, not the knowledge base. No LLM call, no logging as a "gap".
    live = try_live_tools(req.prompt)
    if live is not None:
        return {
            "response": live["text"],
            "model": f"live_tool:{live['tool']}",
            "retrieved_chunks": [],
            "is_knowledge_gap": False,
            "answer_source": {"live": "live_api", "cached": "live_cached",
                              "unavailable": "live_unavailable"}[live["status"]],
            "tokens_generated": 0,
            "generation_time_ns": 0,
        }

    retrieved, query_embedding = retrieve_chunks(req.prompt, top_k=req.top_k)

    top_distance = retrieved[0]["distance"] if retrieved else None
    top_chunk_preview = retrieved[0]["text"][:150] if retrieved else ""
    is_gap = log_query(
        query_text=req.prompt,
        query_embedding=query_embedding,
        top_distance=top_distance,
        retrieved_chunk_preview=top_chunk_preview,
    )

    # Only inject context when retrieval was a good match. For gap queries the
    # retrieved chunks are irrelevant noise the model ends up talking about
    # ("the notes provided do not contain...").
    if retrieved and not is_gap:
        augmented_prompt = build_rag_prompt(req.prompt, [r["text"] for r in retrieved])
    else:
        augmented_prompt = req.prompt

    try:
        result = ollama.chat(
            model=req.model,
            messages=[
                {"role": "system", "content": req.system_prompt},
                {"role": "user", "content": augmented_prompt},
            ],
            options={
                "temperature": req.temperature,
                "num_predict": req.max_tokens,
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ollama call failed: {e}")

    # For the UI: was this grounded in originally-ingested documents,
    # opportunistically-synced content, a weak/gap match, or nothing?
    if not retrieved:
        answer_source = "no_context"
    elif is_gap:
        answer_source = "weak_match"
    elif any(r["metadata"].get("is_synced") for r in retrieved):
        answer_source = "synced_content"
    else:
        answer_source = "original_documents"

    return {
        "response": result["message"]["content"],
        "model": req.model,
        "retrieved_chunks": [
            {"text": r["text"][:150] + "...", "distance": round(r["distance"], 4)}
            for r in retrieved
        ],
        "is_knowledge_gap": is_gap,
        "answer_source": answer_source,
        "tokens_generated": result.get("eval_count"),
        "generation_time_ns": result.get("eval_duration"),
    }


if __name__ == "__main__":
    # Lets `python3 main.py` start the server directly, in addition
    # to `uvicorn main:app --reload --port 8001`. Same effect either way.
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=PORT, reload=True)
