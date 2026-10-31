"""
src/prompt.py
─────────────────────────────────────────────────────────────────────────────
Task 7 – RAG Prompt Construction

Returns (system_prompt, user_prompt) tuple consumed by generate.py.
"""
from __future__ import annotations

SYSTEM_PROMPT = """You are the SPIT Assistant — an AI for Sardar Patel Institute of Technology (SPIT), Mumbai.

Rules (follow strictly):
1. Answer ONLY using the numbered context sources provided below.
2. If the context does not contain sufficient information to answer the question, reply EXACTLY:
   "I don't have this information in the SPIT knowledge base."
3. Do NOT use any external knowledge or make assumptions beyond the provided context.
4. After each factual claim, cite the source using the format [S1], [S2], etc.
5. If multiple sources support a claim, cite all relevant ones (e.g., [S1][S3]).
6. Keep answers concise, accurate and structured. Use bullet points where appropriate.
7. If sources conflict, state the conflict and cite both.
"""


def build_prompt(
    question: str,
    results: list[tuple[dict, float]],
) -> tuple[str, str]:
    """
    Build the (system_prompt, user_prompt) tuple for the generator.

    Args:
        question: the user's question string
        results:  list of (chunk_dict, similarity_score) from a retriever

    Returns:
        (system_prompt, user_prompt)
    """
    context_lines = []
    for i, (chunk, score) in enumerate(results, start=1):
        source   = chunk.get("source", "Unknown")
        section  = chunk.get("section", "")
        doc_id   = chunk.get("doc_id", "")
        text     = chunk.get("text", "").strip()
        sec_tag  = f", section: {section}" if section else ""
        context_lines.append(
            f"[S{i}] (source: {source}{sec_tag}, doc_id: {doc_id}, "
            f"score: {score:.3f})\n{text}"
        )

    context_block = "\n\n".join(context_lines)

    user_prompt = (
        f"CONTEXT:\n{context_block}\n\n"
        f"QUESTION: {question}\n\n"
        f"ANSWER (cite sources inline as [S1], [S2], …):"
    )
    return SYSTEM_PROMPT, user_prompt


# ── Unanswerable / threshold-rejected prompt ──────────────────────────────────
NO_CONTEXT_RESPONSE = "I don't have this information in the SPIT knowledge base."


# ── ICL prompt (Experiment D baseline) ───────────────────────────────────────
ICL_SYSTEM_PROMPT = """You are the SPIT Assistant. You have been given the following
documents about Sardar Patel Institute of Technology (SPIT) as your knowledge source.
Answer the question using ONLY the provided documents.
If the answer cannot be found in the documents, say exactly:
"I don't have this information in the SPIT knowledge base."
Do not use any outside knowledge.
"""

def build_icl_prompt(question: str, documents: list[str]) -> tuple[str, str]:
    """Build ICL prompt by stuffing full documents into context."""
    doc_block = "\n\n---\n\n".join(
        f"[DOC {i+1}]\n{d}" for i, d in enumerate(documents)
    )
    user_prompt = (
        f"DOCUMENTS:\n{doc_block}\n\n"
        f"QUESTION: {question}\n\n"
        f"ANSWER:"
    )
    return ICL_SYSTEM_PROMPT, user_prompt
