"""
src/retrievers.py
─────────────────────────────────────────────────────────────────────────────
Task 6 + Advanced Challenge – All Retrieval Strategies

Every retriever has the same signature:
    retrieve_<name>(query, ..., k=3) -> list[tuple[chunk_dict, float]]

Strategies:
  • Dense          – FAISS inner product
  • BM25           – sparse keyword search
  • Hybrid (RRF)   – Reciprocal Rank Fusion of dense + BM25
  • MMR            – Maximum Marginal Relevance (diversity)
  • Rerank         – cross-encoder reranking on top of any retriever
  • Threshold gate – reject low-confidence results

Also exports:
  build_bm25(chunks) -> BM25Okapi
  cosine_sim(a, b)   -> float
"""
from __future__ import annotations
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import config

import re

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he",
    "in", "is", "it", "its", "of", "on", "that", "the", "to", "was", "were",
    "will", "with", "what", "where", "when", "who", "how", "which", "tell", "me", "about"
}

def _tokenize(text: str) -> list[str]:
    words = re.findall(r"\b\w+\b", text.lower())
    filtered = [w for w in words if w not in _STOPWORDS]
    return filtered if filtered else words

# ── BM25 builder ──────────────────────────────────────────────────────────────
def build_bm25(chunks: list[dict]):
    """
    Build a BM25Okapi index over chunk texts.
    Returns the BM25 object; keep a reference alongside the chunk list.
    """
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        raise ImportError("pip install rank-bm25")

    tokenised = [_tokenize(c["text"]) for c in chunks]
    return BM25Okapi(tokenised)


# ── Dense retrieval (Task 6 baseline) ────────────────────────────────────────
def retrieve_dense(
    query:  str,
    model,           # SentenceTransformer instance
    index,           # faiss.IndexFlatIP
    chunks: list[dict],
    k:      int = 3,
) -> list[tuple[dict, float]]:
    """
    Embed the query with the BGE prefix and run FAISS inner-product search.
    """
    from embed_index import encode_query
    q_vec   = encode_query(query)
    scores, ids = index.search(q_vec, k)
    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0 or idx >= len(chunks):
            continue
        c = dict(chunks[idx])   # shallow copy so we can annotate freely
        results.append((c, float(score)))
    return results


# ── BM25 retrieval ────────────────────────────────────────────────────────────
def retrieve_bm25(
    query:  str,
    bm25,
    chunks: list[dict],
    k:      int = 3,
) -> list[tuple[dict, float]]:
    """
    Sparse keyword retrieval. Particularly strong on codes (CE413), names, etc.
    """
    tokens = _tokenize(query)
    scores = bm25.get_scores(tokens)
    top_ids = np.argsort(scores)[::-1][:k]
    return [(dict(chunks[i]), float(scores[i])) for i in top_ids if scores[i] > 0]


# ── Hybrid retrieval via Reciprocal Rank Fusion ───────────────────────────────
def retrieve_hybrid_rrf(
    query:  str,
    model,
    index,
    bm25,
    chunks: list[dict],
    k:      int  = 3,
    rrf_k:  int  = 60,
    fetch:  int  = 30,
) -> list[tuple[dict, float]]:
    """
    RRF score = Σ  1 / (rrf_k + rank_i)  over dense and BM25 ranks.
    Fetch top `fetch` from each leg, fuse, then cut to k.
    """
    dense_results  = retrieve_dense(query, model, index, chunks, k=fetch)
    sparse_results = retrieve_bm25(query, bm25, chunks, k=fetch)

    scores: dict[str, float]     = {}
    chunk_map: dict[str, dict]   = {}

    for rank, (c, _) in enumerate(dense_results):
        cid = c["chunk_id"]
        scores[cid]    = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)
        chunk_map[cid] = c

    for rank, (c, _) in enumerate(sparse_results):
        cid = c["chunk_id"]
        scores[cid]    = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)
        chunk_map[cid] = c

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    return [(chunk_map[cid], rrf_score) for cid, rrf_score in ranked]


# ── MMR – Maximum Marginal Relevance ─────────────────────────────────────────
def retrieve_mmr(
    query:  str,
    model,
    index,
    chunks: list[dict],
    k:      int   = 3,
    lam:    float = 0.6,   # λ: 1 = pure relevance, 0 = pure diversity
    fetch:  int   = 20,
) -> list[tuple[dict, float]]:
    """
    Fetch `fetch` dense candidates, then greedily select k with MMR.
    λ controls the relevance-diversity trade-off.
    """
    candidates = retrieve_dense(query, model, index, chunks, k=fetch)
    if not candidates:
        return []

    selected:  list[tuple[dict, float]] = []
    remaining: list[tuple[dict, float]] = list(candidates)

    while len(selected) < k and remaining:
        if not selected:
            best = max(remaining, key=lambda x: x[1])
        else:
            def mmr_score(item: tuple[dict, float]) -> float:
                rel      = item[1]
                max_red  = max(
                    _cosine_text(item[0]["text"], s[0]["text"])
                    for s in selected
                )
                return lam * rel - (1 - lam) * max_red
            best = max(remaining, key=mmr_score)

        selected.append(best)
        remaining.remove(best)

    return selected


def _cosine_text(a: str, b: str) -> float:
    """Quick word-overlap cosine for MMR diversity (avoids re-embedding)."""
    sa, sb = set(a.lower().split()), set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / (len(sa | sb) + 1e-9)


# ── Cross-encoder Reranker ─────────────────────────────────────────────────────
_CE_CACHE: dict[str, object] = {}

def _get_cross_encoder(model_name: str):
    if model_name not in _CE_CACHE:
        from sentence_transformers import CrossEncoder
        print(f"[Reranker] Loading: {model_name}")
        _CE_CACHE[model_name] = CrossEncoder(model_name)
    return _CE_CACHE[model_name]


def rerank(
    query:      str,
    candidates: list[tuple[dict, float]],
    model_name: str | None = None,
    k:          int = 3,
) -> list[tuple[dict, float]]:
    """
    Score every (query, chunk) pair with a cross-encoder and keep top k.
    Stores pre-rerank rank in chunk["_pre_rank"] for logging.
    """
    if not candidates:
        return []

    model_name = model_name or config.RERANKER_MODEL
    ce         = _get_cross_encoder(model_name)

    pairs  = [[query, c["text"]] for c, _ in candidates]
    scores = ce.predict(pairs, show_progress_bar=False)

    annotated = []
    for pre_rank, ((c, _), ce_score) in enumerate(zip(candidates, scores)):
        c = dict(c)
        c["_pre_rank"] = pre_rank
        annotated.append((c, float(ce_score)))

    annotated.sort(key=lambda x: x[1], reverse=True)
    return annotated[:k]


# ── Relevance threshold gate ──────────────────────────────────────────────────
def apply_threshold(
    results: list[tuple[dict, float]],
    tau:     float,
) -> list[tuple[dict, float]]:
    """
    If the top-scoring result is below τ, return an empty list.
    An empty list triggers the "information not available" response.
    """
    if not results:
        return []
    if results[0][1] < tau:
        return []
    return results


# ── Convenience: full retrieve pipeline ──────────────────────────────────────
def retrieve(
    query:     str,
    model,
    index,
    chunks:    list[dict],
    bm25,
    strategy:  str   = "hybrid",
    k:         int   = 3,
    tau:       float = 0.0,
    do_rerank: bool  = False,
) -> list[tuple[dict, float]]:
    """
    Single entry point used by app.py and experiments.py.
    strategy: dense | bm25 | hybrid | mmr | rerank
    """
    if strategy == "dense":
        results = retrieve_dense(query, model, index, chunks, k=k)
    elif strategy == "bm25":
        results = retrieve_bm25(query, bm25, chunks, k=k)
    elif strategy == "mmr":
        results = retrieve_mmr(query, model, index, chunks, k=k)
    elif strategy in ("rerank", "hybrid"):
        results = retrieve_hybrid_rrf(query, model, index, bm25, chunks, k=k)
        if strategy == "rerank" or do_rerank:
            results = rerank(query, results, k=k)
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")

    if tau > 0:
        results = apply_threshold(results, tau)

    return results
