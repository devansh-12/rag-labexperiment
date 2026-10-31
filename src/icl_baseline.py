"""
src/icl_baseline.py
─────────────────────────────────────────────────────────────────────────────
Experiment D – In-Context Learning baseline

Stuffs as many full documents as fit in the context window into one prompt,
then generates an answer without any retrieval step.

Usage:
    from icl_baseline import run_icl
    result = run_icl(question, all_docs, gen_obj, max_context_tokens=6000)
"""
from __future__ import annotations
import time
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import config
from prompt import build_icl_prompt
from generate import generate_answer
from chunkers import count_tokens


def run_icl(
    question:          str,
    all_docs:          list[dict],   # list of {doc_id, text, source_name, …}
    gen_obj:           dict,
    max_context_tokens: int = 6000,
    temperature:       float | None = None,
) -> dict:
    """
    Pack as many documents as possible into the context window, generate answer.

    Returns:
        {
            answer, cited_ids,
            n_docs_included, n_docs_total, truncated,
            t_generate, tokens_used
        }
    """
    docs_included: list[str] = []
    tokens_used = 0

    for doc in all_docs:
        text = doc.get("text") or doc.get("full_text", "")
        header = doc.get("source_name") or doc.get("title", "")
        doc_text = f"[{header}]\n{text}" if header else text
        doc_tokens = count_tokens(doc_text)
        if tokens_used + doc_tokens > max_context_tokens:
            break
        docs_included.append(doc_text)
        tokens_used += doc_tokens

    truncated = len(docs_included) < len(all_docs)

    system_p, user_p = build_icl_prompt(question, docs_included)

    t0 = time.perf_counter()
    answer, cited_ids = generate_answer(
        (system_p, user_p), gen_obj, temperature=temperature
    )
    t_generate = time.perf_counter() - t0

    return {
        "answer":           answer,
        "cited_ids":        cited_ids,
        "n_docs_included":  len(docs_included),
        "n_docs_total":     len(all_docs),
        "truncated":        truncated,
        "t_generate":       round(t_generate, 3),
        "tokens_used":      tokens_used,
        "t_retrieve":       0.0,   # ICL has no retrieval step
        "t_rerank":         0.0,
    }
