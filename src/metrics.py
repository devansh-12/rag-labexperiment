"""
src/metrics.py
─────────────────────────────────────────────────────────────────────────────
Custom retrieval + generation metrics for CE413 Experiment 8.

Metrics implemented:
  • hit_at_k         – was a gold doc/chunk in the top-k?
  • precision_at_k   – fraction of top-k that are relevant
  • mrr              – Mean Reciprocal Rank
  • ndcg_at_k        – nDCG (optional, graded relevance)
  • latency_breakdown – structured latency dict
  • answer_exact_match, answer_f1 – text-overlap metrics

Aggregation:
  • run_metrics(qa_set, results_list) → pandas DataFrame + summary dict
"""
from __future__ import annotations
import re
import time
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# 1. Relevance judgement helper
# ─────────────────────────────────────────────────────────────────────────────

def is_relevant(
    chunk: dict,
    gold_doc_ids: list[str],
    gold_evidence: list[str],
    soft: bool = False,
) -> bool:
    """
    A chunk is relevant if:
      • hard: its doc_id is in gold_doc_ids AND its text contains ≥1 gold evidence substring
      • soft: its doc_id is in gold_doc_ids (ignores evidence substring)

    Use substring matching after normalising whitespace and case.
    Never use chunk_id-level labels (they change per chunking strategy).
    """
    doc_match = chunk.get("doc_id", "") in gold_doc_ids
    if not doc_match:
        return False
    if soft:
        return True
    chunk_text = _normalize(chunk.get("text", ""))
    return any(_normalize(ev) in chunk_text for ev in gold_evidence)


def _normalize(text: str) -> str:
    return re.sub(r'\s+', ' ', text.lower()).strip()


# ─────────────────────────────────────────────────────────────────────────────
# 2. Retrieval metrics
# ─────────────────────────────────────────────────────────────────────────────

def hit_at_k(
    retrieved: list[dict],
    gold_doc_ids: list[str],
    gold_evidence: list[str],
    k: int,
    soft: bool = False,
) -> float:
    """1 if ≥1 relevant chunk in top-k, else 0."""
    for chunk in retrieved[:k]:
        if is_relevant(chunk, gold_doc_ids, gold_evidence, soft=soft):
            return 1.0
    return 0.0


def precision_at_k(
    retrieved: list[dict],
    gold_doc_ids: list[str],
    gold_evidence: list[str],
    k: int,
    soft: bool = False,
) -> float:
    """Fraction of top-k chunks that are relevant."""
    if not retrieved:
        return 0.0
    topk = retrieved[:k]
    n_rel = sum(
        1 for c in topk
        if is_relevant(c, gold_doc_ids, gold_evidence, soft=soft)
    )
    return n_rel / len(topk)


def mrr(
    retrieved: list[dict],
    gold_doc_ids: list[str],
    gold_evidence: list[str],
    soft: bool = False,
) -> float:
    """Reciprocal rank of the first relevant chunk."""
    for rank, chunk in enumerate(retrieved, start=1):
        if is_relevant(chunk, gold_doc_ids, gold_evidence, soft=soft):
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    retrieved: list[dict],
    gold_doc_ids: list[str],
    gold_evidence: list[str],
    k: int,
    soft: bool = False,
) -> float:
    """nDCG@k with binary relevance (1 = relevant, 0 = not)."""
    import math
    gains = [
        1.0 if is_relevant(c, gold_doc_ids, gold_evidence, soft=soft) else 0.0
        for c in retrieved[:k]
    ]
    dcg  = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    # ideal: all relevant at top
    n_ideal = min(k, sum(gains))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(int(n_ideal)))
    return dcg / idcg if idcg > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 3. Generation metrics
# ─────────────────────────────────────────────────────────────────────────────

def answer_exact_match(pred: str, gold: str) -> float:
    """1 if normalised strings match exactly."""
    return float(_normalize(pred) == _normalize(gold))


def answer_f1(pred: str, gold: str) -> float:
    """Token-level F1 between predicted and gold answer."""
    pred_tokens = _normalize(pred).split()
    gold_tokens = _normalize(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    prec = len(common) / len(pred_tokens)
    rec  = len(common) / len(gold_tokens)
    return 2 * prec * rec / (prec + rec)


def is_refusal(answer: str) -> bool:
    """True if the model produced the canonical refusal string."""
    return "don't have this information" in answer.lower()


# ─────────────────────────────────────────────────────────────────────────────
# 4. Latency
# ─────────────────────────────────────────────────────────────────────────────

def latency_breakdown(
    t_retrieve: float,
    t_rerank:   float,
    t_generate: float,
) -> dict:
    return {
        "t_retrieve": round(t_retrieve, 4),
        "t_rerank":   round(t_rerank, 4),
        "t_generate": round(t_generate, 4),
        "t_total":    round(t_retrieve + t_rerank + t_generate, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. Failure tagging
# ─────────────────────────────────────────────────────────────────────────────

def tag_failure(
    retrieved:    list[dict],
    answer:       str,
    gold_doc_ids: list[str],
    gold_evidence: list[str],
    gold_answer:  Optional[str],
) -> str:
    """
    Classify a wrong answer into one of four failure types:
      retrieval_failure | generation_failure | chunking_failure | abstention_failure
    Returns 'correct' if none apply.
    """
    if gold_answer is None:
        # Unanswerable question
        if is_refusal(answer):
            return "correct"
        return "abstention_failure"

    hit = hit_at_k(retrieved, gold_doc_ids, gold_evidence, k=len(retrieved))
    soft_hit = hit_at_k(retrieved, gold_doc_ids, gold_evidence,
                        k=len(retrieved), soft=True)
    f1 = answer_f1(answer, gold_answer)

    if f1 >= 0.5:
        return "correct"

    if not soft_hit:
        return "retrieval_failure"

    if soft_hit and not hit:
        # Doc retrieved but evidence substring not in any chunk
        return "chunking_failure"

    # Doc + evidence retrieved but answer still wrong
    return "generation_failure"


# ─────────────────────────────────────────────────────────────────────────────
# 6. Aggregate over a run
# ─────────────────────────────────────────────────────────────────────────────

def run_metrics(
    qa_rows:  list[dict],
    run_rows: list[dict],
    k:        int = 3,
    soft:     bool = False,
) -> dict:
    """
    Compute aggregate metrics over a full evaluation run.

    Args:
        qa_rows:  list of QA dicts from qa_set.jsonl
        run_rows: list of result dicts from experiments.py
                  Each must have: retrieved_chunks, answer, t_retrieve, t_rerank, t_generate

    Returns:
        dict with macro-averaged metrics
    """
    hits, precs, rrs, f1s = [], [], [], []
    t_ret, t_gen = [], []

    for qa, res in zip(qa_rows, run_rows):
        gold_doc_ids  = qa.get("gold_doc_ids", [])
        gold_evidence = qa.get("gold_evidence", [])
        gold_answer   = qa.get("gold_answer")
        retrieved     = res.get("retrieved_chunks", [])
        answer        = res.get("answer", "")

        hits.append(hit_at_k(retrieved, gold_doc_ids, gold_evidence, k, soft))
        precs.append(precision_at_k(retrieved, gold_doc_ids, gold_evidence, k, soft))
        rrs.append(mrr(retrieved, gold_doc_ids, gold_evidence, soft))
        if gold_answer:
            f1s.append(answer_f1(answer, gold_answer))
        t_ret.append(res.get("t_retrieve", 0))
        t_gen.append(res.get("t_generate", 0))

    n = len(hits) or 1
    return {
        f"hit@{k}":       round(sum(hits)  / n, 4),
        f"precision@{k}": round(sum(precs) / n, 4),
        "mrr":            round(sum(rrs)   / n, 4),
        "answer_f1":      round(sum(f1s)   / max(len(f1s), 1), 4),
        "avg_t_retrieve": round(sum(t_ret) / n, 4),
        "avg_t_generate": round(sum(t_gen) / n, 4),
        "n":              n,
    }


# ── CLI quick-test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Sanity check
    chunk = {"doc_id": "D001", "text": "SPIT is located in Mumbai Andheri West."}
    print(hit_at_k([chunk], ["D001"], ["located in Mumbai"], k=3))   # 1.0
    print(mrr([chunk],       ["D001"], ["located in Mumbai"]))        # 1.0
    print(answer_f1("SPIT is in Mumbai", "SPIT Mumbai Andheri"))      # ~0.5
    print(tag_failure([chunk], "I don't know", ["D001"],
                      ["located in Mumbai"], "SPIT is in Mumbai"))    # generation_failure
