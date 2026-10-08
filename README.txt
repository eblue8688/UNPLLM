UNPLLM DEMO HOTFIX
==================

RECOMMENDED QUICK FIX
1. Stop the current main.py / uvicorn process with Ctrl+C.
2. Replace your project's main.py with this main.py.
3. Replace your project's query_logger.py with this query_logger.py.
4. Keep your existing live_tools.py for now.
5. Start the server exactly as before:
      python3 main.py
   or
      uvicorn main:app --reload --port 8001

WHAT THIS FIXES
- Current/live-looking questions such as "PM of india in 2026" no longer fall through into ordinary RAG.
- Weak retrieval matches (distance > 1.25) are logged as gaps and are NOT passed to the model as context.
- The local model is explicitly instructed not to claim it is Microsoft/OpenAI/etc.
- The UI can still show weak_match / synced_content / original_documents.

DEMO TESTS
A) Type: PM of india in 2026
   Expected: a safe live-tool/unavailable response, NOT a hallucinated Microsoft answer.

B) Type: What is UNPLLM?
   Expected: normal RAG answer when a strong local chunk exists.

C) Type: capital of France
   Expected: weak/no-context handling rather than unrelated retrieved context.

NOTE
The included live_tools_demo_fallback.py is a reference implementation for a Wikipedia-backed live path.
Do not overwrite your existing live_tools.py unless you intentionally want to replace its current APIs.
