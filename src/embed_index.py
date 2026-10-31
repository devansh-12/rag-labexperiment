"""
src/embed_index.py
─────────────────────────────────────────────────────────────────────────────
Tasks 4 & 5 – Embedding Generation + FAISS Index

build_index(chunks, model_name, chunker_id)
    → encodes all chunks, builds IndexFlatIP, saves to disk

load_index(chunker_id, model_key)
    → loads index + chunk list from disk
"""
from __future__ import annotations
import json
import sys, os
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
import config


# ── Model cache (avoid re-loading the same model in one process) ──────────────
_MODEL_CACHE: dict[str, object] = {}

def _get_model(model_name: str):
    if model_name not in _MODEL_CACHE:
        from sentence_transformers import SentenceTransformer
        print(f"[Embed] Loading model: {model_name}")
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]


# ── Helpers ───────────────────────────────────────────────────────────────────
def _index_tag(chunker_id: str, model_key: str) -> str:
    return f"{chunker_id}_{model_key}"

def _index_path(tag: str) -> Path:
    return config.INDEX_DIR / f"index_{tag}.faiss"

def _chunks_path(tag: str) -> Path:
    return config.INDEX_DIR / f"chunks_{tag}.jsonl"


# ── Build ─────────────────────────────────────────────────────────────────────
def build_index(
    chunks: list[dict],
    model_key: str = "A",
    chunker_id: str = "C1",
    batch_size: int = 64,
    show_progress: bool = True,
) -> tuple:
    """
    Encode all chunks and build a FAISS IndexFlatIP.

    Uses the BGE query prefix for consistent embedding.
    Vectors are L2-normalised so inner product == cosine similarity.

    Returns (index, chunks) tuple.
    """
    import numpy as np
    import faiss

    model_name = config._EMBED_MODELS.get(model_key, config.EMBED_MODEL_NAME)
    model      = _get_model(model_name)
    tag        = _index_tag(chunker_id, model_key)

    # For C6 parent-child: only embed children (is_child=True) for indexing
    # Parents are stored but not directly embedded
    index_chunks = [c for c in chunks if not c.get("is_parent", False)]
    if not index_chunks:
        index_chunks = chunks   # fallback if no child markers

    # Prepend BGE passage prefix
    texts = [
        config.BGE_QUERY_PREFIX + c["text"]
        for c in index_chunks
    ]

    print(f"[Embed] Encoding {len(texts)} chunks with {model_name}…")
    vecs = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    )
    vecs = vecs.astype("float32")

    dim   = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)

    # Persist to disk
    faiss.write_index(index, str(_index_path(tag)))
    with open(_chunks_path(tag), "w", encoding="utf-8") as f:
        for c in index_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # Also persist full chunks (including parents for C6)
    if len(index_chunks) != len(chunks):
        with open(config.INDEX_DIR / f"chunks_full_{tag}.jsonl", "w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"[Embed] Index saved → {_index_path(tag)} ({index.ntotal} vectors, dim={dim})")
    return index, index_chunks


# ── Load ──────────────────────────────────────────────────────────────────────
def load_index(
    chunker_id: str | None = None,
    model_key:  str | None = None,
) -> tuple:
    """
    Load the most recent (or specified) FAISS index + chunk list from disk.
    Falls back to the first available index if none specified.
    """
    import faiss

    chunker_id = chunker_id or config.DEFAULT_CHUNKER
    model_key  = model_key  or config.EMBED_MODEL_KEY
    tag        = _index_tag(chunker_id, model_key)

    idx_path = _index_path(tag)
    chk_path = _chunks_path(tag)

    # Auto-discover if exact tag not found
    if not idx_path.exists():
        candidates = list(config.INDEX_DIR.glob("index_*.faiss"))
        if not candidates:
            raise FileNotFoundError(
                "No FAISS index found. Run:\n"
                "  python src/embed_index.py"
            )
        idx_path = candidates[0]
        tag      = idx_path.stem.replace("index_", "")
        chk_path = config.INDEX_DIR / f"chunks_{tag}.jsonl"
        print(f"[Embed] Auto-discovered index: {idx_path.name}")

    index  = faiss.read_index(str(idx_path))
    chunks = [json.loads(l) for l in open(chk_path, encoding="utf-8")]
    print(f"[Embed] Loaded index ({index.ntotal} vectors) + {len(chunks)} chunks")
    return index, chunks


# ── Encode a single query ─────────────────────────────────────────────────────
def encode_query(query: str, model_key: str = "A") -> "np.ndarray":
    """Encode one query with the BGE query prefix. Returns float32 (1, dim)."""
    import numpy as np
    model_name = config._EMBED_MODELS.get(model_key, config.EMBED_MODEL_NAME)
    model      = _get_model(model_name)
    vec = model.encode(
        [config.BGE_QUERY_PREFIX + query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")
    return vec


# ── CLI: build all (chunker × model) pairs ────────────────────────────────────
if __name__ == "__main__":
    import sys, json as _json
    from chunkers import chunk
    from ingest   import ingest_kb

    strategies = ["C1", "C3", "C4", "C6"]
    model_keys = ["A", "B"]

    units = ingest_kb()
    for strat in strategies:
        chunks_all = chunk(units, strategy=strat)
        for mk in model_keys:
            print(f"\n{'='*60}")
            print(f" Building index: {strat} × Model {mk}")
            print('='*60)
            build_index(chunks_all, model_key=mk, chunker_id=strat)
