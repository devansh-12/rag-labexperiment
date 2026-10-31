"""
src/chunkers.py
─────────────────────────────────────────────────────────────────────────────
Task 3 + Advanced Challenge – Six Chunking Strategies

All strategies expose the same interface:
    chunk_<strategy>(units: list[dict], **kwargs) -> list[dict]

Each returned chunk dict:
    chunk_id | doc_id | source | source_name | source_url | title
    section | text | n_tokens | start_char | parent_id (C6 only)

Unified entry point:
    chunk(units, strategy="C1", **kwargs) -> list[dict]
"""
from __future__ import annotations
import json
import uuid
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import config

# ── Token counter ─────────────────────────────────────────────────────────────
try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text: str) -> int:
        return len(_enc.encode(text))
except ImportError:
    # Fallback: rough word-based approximation
    def count_tokens(text: str) -> int:  # type: ignore[misc]
        return int(len(text.split()) * 1.35)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _make_chunk(unit: dict, text: str, section: str | None = None,
                start_char: int = 0, parent_id: str | None = None) -> dict:
    """Create a canonical chunk dict from a raw extraction unit."""
    cid = str(uuid.uuid4())[:8]
    return {
        "chunk_id":   cid,
        "doc_id":     unit.get("doc_id", ""),
        "source":     unit.get("source_name", unit.get("source", "")),
        "source_url": unit.get("source_url", ""),
        "title":      unit.get("title", ""),
        "section":    section if section is not None else unit.get("section", ""),
        "text":       text.strip(),
        "n_tokens":   count_tokens(text),
        "start_char": start_char,
        **({"parent_id": parent_id} if parent_id else {}),
    }


def _prepend_header(unit: dict, text: str) -> str:
    """Prepend title + section for contextual embedding (always do this)."""
    header_parts = []
    title = unit.get("title") or unit.get("source_name") or unit.get("source", "")
    section = unit.get("section", "")
    if title:
        header_parts.append(title)
    if section and section not in ("main", "erp_row", title):
        header_parts.append(section)
    header = " | ".join(header_parts)
    return f"{header}\n\n{text}" if header else text


def _split_text_into_token_windows(text: str, size: int, overlap: int) -> list[tuple[str, int]]:
    """
    Slide a token window over `text`. Returns list of (chunk_text, start_char).
    Works character-by-character for start_char tracking.
    """
    words  = text.split()
    chunks = []
    i      = 0
    while i < len(words):
        window = words[i : i + size]
        chunk_text = " ".join(window)
        # approximate start_char
        start_char = len(" ".join(words[:i]))
        chunks.append((chunk_text, start_char))
        i += max(1, size - overlap)
    return chunks


# ═════════════════════════════════════════════════════════════════════════════
# C1 – Fixed-size token windows (500 tok, 50 overlap) — Lab baseline
# ═════════════════════════════════════════════════════════════════════════════
def chunk_fixed(units: list[dict], size: int = 500, overlap: int = 50) -> list[dict]:
    chunks = []
    for unit in units:
        text = _prepend_header(unit, unit["text"])
        for chunk_text, start in _split_text_into_token_windows(text, size, overlap):
            if len(chunk_text.strip()) < 20:
                continue
            chunks.append(_make_chunk(unit, chunk_text, start_char=start))
    return chunks


# ═════════════════════════════════════════════════════════════════════════════
# C2 – Small fixed windows (200 tok, 30 overlap) — higher precision
# ═════════════════════════════════════════════════════════════════════════════
def chunk_small_fixed(units: list[dict], size: int = 200, overlap: int = 30) -> list[dict]:
    return chunk_fixed(units, size=size, overlap=overlap)


# ═════════════════════════════════════════════════════════════════════════════
# C3 – Recursive character splitter
#       Splits on \n\n → \n → ". " → " "  to respect natural boundaries
# ═════════════════════════════════════════════════════════════════════════════
def chunk_recursive(units: list[dict], target: int = 500, min_size: int = 50) -> list[dict]:
    separators = ["\n\n", "\n", ". ", " "]

    def _split(text: str, sep_list: list[str]) -> list[str]:
        if not sep_list or count_tokens(text) <= target:
            return [text]
        sep = sep_list[0]
        parts = text.split(sep)
        result, buffer = [], ""
        for part in parts:
            candidate = (buffer + sep + part).strip() if buffer else part
            if count_tokens(candidate) <= target:
                buffer = candidate
            else:
                if buffer:
                    result.append(buffer)
                # Try splitting the overflowing part with next separator
                sub = _split(part, sep_list[1:])
                result.extend(sub[:-1])
                buffer = sub[-1] if sub else ""
        if buffer:
            result.append(buffer)
        return [r for r in result if r.strip()]

    chunks = []
    for unit in units:
        text = _prepend_header(unit, unit["text"])
        parts = _split(text, separators)
        for part in parts:
            if len(part.strip()) < min_size:
                continue
            chunks.append(_make_chunk(unit, part))
    return chunks


# ═════════════════════════════════════════════════════════════════════════════
# C4 – Heading-aware (structure) chunker
#       One chunk per section from the _sections.json sidecar.
#       Splits sections > max_tokens using C3.
# ═════════════════════════════════════════════════════════════════════════════
def chunk_heading_aware(units: list[dict], max_tokens: int = 600) -> list[dict]:
    chunks = []
    for unit in units:
        section  = unit.get("section", "main")
        heading  = unit.get("title", "") or unit.get("source_name", "")
        raw_text = unit["text"]
        # Prepend heading to make each chunk self-contained
        full_text = f"{heading} | {section}\n\n{raw_text}" if heading else raw_text

        if count_tokens(full_text) <= max_tokens:
            chunks.append(_make_chunk(unit, full_text, section=section))
        else:
            # Split overlong section with recursive splitter
            sub_units = [dict(unit, text=full_text)]
            sub_chunks = chunk_recursive(sub_units, target=max_tokens)
            for c in sub_chunks:
                c["section"] = section  # keep original section label
            chunks.extend(sub_chunks)
    return chunks


# ═════════════════════════════════════════════════════════════════════════════
# C5 – Semantic chunking
#       Embeds sentences, breaks where cosine similarity to the next sentence
#       drops below a percentile threshold.
# ═════════════════════════════════════════════════════════════════════════════
def chunk_semantic(units: list[dict], threshold_pct: int = 80,
                   max_tokens: int = 600) -> list[dict]:
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except ImportError:
        raise ImportError("pip install sentence-transformers numpy")

    model = SentenceTransformer(config.EMBED_MODEL_NAME)
    chunks = []

    for unit in units:
        text = unit["text"]
        # Split into sentences (simple period-split; good enough for semantic boundary)
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        if len(sentences) <= 2:
            chunks.append(_make_chunk(unit, _prepend_header(unit, text)))
            continue

        vecs = model.encode(sentences, normalize_embeddings=True, batch_size=32)
        # cosine similarity between consecutive sentences (vectors are normalised)
        sims = [float(vecs[i] @ vecs[i+1]) for i in range(len(vecs)-1)]
        cutoff = float(import_numpy_percentile(sims, threshold_pct))

        current, groups = [], []
        for i, sent in enumerate(sentences):
            current.append(sent)
            if i < len(sims) and sims[i] < cutoff:
                groups.append(" ".join(current))
                current = []
        if current:
            groups.append(" ".join(current))

        for group in groups:
            full = _prepend_header(unit, group)
            if count_tokens(full) > max_tokens:
                sub = chunk_recursive([dict(unit, text=group)], target=max_tokens)
                chunks.extend(sub)
            else:
                chunks.append(_make_chunk(unit, full))
    return chunks

import re as _re
def import_numpy_percentile(values, pct):
    import numpy as np
    return np.percentile(values, pct)


# ═════════════════════════════════════════════════════════════════════════════
# C6 – Parent-child chunker
#       Index small children (150 tok); return larger parent (700 tok) as context.
#       child["parent_id"] points to parent["chunk_id"].
# ═════════════════════════════════════════════════════════════════════════════
def chunk_parent_child(units: list[dict], child_size: int = 150,
                       parent_size: int = 700) -> list[dict]:
    all_chunks: list[dict] = []

    # First pass: build parents (700-tok windows)
    parent_windows = chunk_fixed(units, size=parent_size, overlap=0)

    for parent in parent_windows:
        parent["is_parent"] = True
        pid = parent["chunk_id"]
        all_chunks.append(parent)

        # Second pass: split each parent into 150-tok children
        pseudo_unit = {
            **{k: parent.get(k, "") for k in
               ("doc_id", "source", "source_url", "title", "section")},
            "text": parent["text"],
        }
        children = chunk_fixed([pseudo_unit], size=child_size, overlap=0)
        for child in children:
            child["parent_id"] = pid
            child["is_child"]  = True
            all_chunks.append(child)

    return all_chunks


# ═════════════════════════════════════════════════════════════════════════════
# Unified entry point
# ═════════════════════════════════════════════════════════════════════════════
_STRATEGIES = {
    "C1": chunk_fixed,
    "C2": chunk_small_fixed,
    "C3": chunk_recursive,
    "C4": chunk_heading_aware,
    "C5": chunk_semantic,
    "C6": chunk_parent_child,
}

def chunk(units: list[dict], strategy: str = "C1", **kwargs) -> list[dict]:
    """
    Main entry point. strategy ∈ {C1, C2, C3, C4, C5, C6}.
    Returns list of chunk dicts ready for embedding.
    """
    if strategy not in _STRATEGIES:
        raise ValueError(f"Unknown chunking strategy {strategy!r}. "
                         f"Choose from {list(_STRATEGIES)}")
    fn      = _STRATEGIES[strategy]
    chunks  = fn(units, **kwargs)
    # Filter: remove very short or whitespace-only chunks
    chunks  = [c for c in chunks if len(c["text"].strip()) >= 20]
    print(f"[Chunker {strategy}] {len(units)} units → {len(chunks)} chunks")
    return chunks


if __name__ == "__main__":
    import sys, json
    from pathlib import Path
    units_path = config.KB_DIR / "units.jsonl"
    if not units_path.exists():
        print("Run ingest.py first.")
        sys.exit(1)
    units = [json.loads(l) for l in open(units_path)]
    for strat in ["C1", "C3", "C4", "C6"]:
        chunks = chunk(units, strategy=strat)
        out = config.KB_DIR / f"chunks_{strat}.jsonl"
        with open(out, "w") as f:
            for c in chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        print(f"  → {out}")
