"""
UNPLLM - shared RAG prompt template.

Used by main.py AND evaluate.py, so the system you evaluate is exactly
the system you ship. Change the wording here, nowhere else.

Why this wording: with the old "use the context if it's relevant, otherwise
answer normally" prompt, phi4-mini often ignored perfectly good retrieved
context on "latest/newest" questions and fell back to "as of my last update
in 2023...". The prompt now (1) gives today's date, (2) says the notes may be
newer than the model's training data and should win on conflicts, (3) tells
it to name the newest item the notes describe, and (4) forbids invented facts.
"""

from datetime import date


def build_rag_prompt(question, context_chunks):
    notes = "\n\n".join(f"[{i}] {chunk}" for i, chunk in enumerate(context_chunks, start=1))
    return (
        f"Today's date is {date.today():%B %d, %Y}.\n"
        "Below are notes retrieved from your local knowledge base. They may be newer "
        "than your training data. Answer the question using the notes. If the notes "
        "conflict with what you remember, trust the notes. For 'latest', 'newest' or "
        "'current' questions, name the newest item the notes describe. Do not invent "
        "names, places or numbers. If the notes do not contain the answer, say so "
        "briefly, then answer from general knowledge.\n\n"
        f"NOTES:\n{notes}\n\n"
        f"QUESTION: {question}\n"
        "ANSWER:"
    )
