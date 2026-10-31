"""
src/ingest.py
─────────────────────────────────────────────────────────────────────────────
Task 2 – Document Extraction & Cleaning

Reads from kb/ and produces a list of raw extraction units:
  {doc_id, source, section, text}

Supported sources:
  • PDFs         → pymupdf (fitz)
  • HTML / .txt  → trafilatura + section JSON sidecar
  • CSV (ERP)    → pandas → natural-language sentences
  • JSONL        → passthrough
"""
from __future__ import annotations
import csv
import json
import re
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
import config

# ── privacy gate ──────────────────────────────────────────────────────────────
_PII = re.compile(
    r'\b[A-Z]{2}\d{2}[A-Z0-9]{6}\b'   # roll number
    r'|\b[6-9]\d{9}\b'                 # Indian mobile
    r'|\b[\w.+-]+@[\w-]+\.[a-z]{2,}\b' # email
    r'|\b\d{12}\b'                      # Aadhaar
)

def _pii_check(text: str, doc_id: str) -> None:
    m = _PII.search(text)
    if m:
        raise ValueError(
            f"[PRIVACY] PII detected in {doc_id}: {m.group()!r}. "
            "Remove before indexing."
        )


# ── text normalisers ──────────────────────────────────────────────────────────
def _normalize(text: str) -> str:
    text = re.sub(r'-\n', '', text)           # fix PDF hyphenation across lines
    text = re.sub(r'\n{3,}', '\n\n', text)    # collapse excess blank lines
    text = re.sub(r'[ \t]+', ' ', text)       # collapse inline spaces
    return text.strip()


# ── PDF extraction ─────────────────────────────────────────────────────────────
def extract_pdf(path: str | Path, doc_id: str) -> list[dict]:
    """
    Extract text page-by-page. Strip lines that appear on >50% of pages
    (repeated headers / footers).
    """
    try:
        import fitz  # pymupdf
    except ImportError:
        raise ImportError("pip install pymupdf")

    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = config._ROOT / path
    doc  = fitz.open(str(path))
    pages: list[str] = [page.get_text() for page in doc]
    doc.close()

    if not pages:
        return []

    # Detect repeated header/footer lines
    from collections import Counter
    line_counts = Counter(
        ln.strip()
        for p in pages
        for ln in p.splitlines()
        if ln.strip()
    )
    if len(pages) > 2:
        threshold = len(pages) * 0.5
        repeated = {ln for ln, c in line_counts.items() if c > threshold and ln}
    elif len(pages) == 2:
        repeated = {ln for ln, c in line_counts.items() if c > 1 and ln}
    else:
        repeated = set()

    units = []
    for i, page_text in enumerate(pages):
        cleaned = "\n".join(
            ln for ln in page_text.splitlines()
            if ln.strip() not in repeated
        )
        cleaned = _normalize(cleaned)
        if len(cleaned) < 50:
            continue
        _pii_check(cleaned, doc_id)
        units.append({
            "doc_id":  doc_id,
            "source":  path.name,
            "section": f"page_{i + 1}",
            "text":    cleaned,
        })
    return units


# ── HTML / plain-text extraction ──────────────────────────────────────────────
def extract_html(path: str | Path, doc_id: str) -> list[dict]:
    """
    Extract from a saved HTML or .txt file.
    If a _sections.json sidecar exists (from the scraper), use it for
    heading-aware units; otherwise fall back to full-text.
    """
    path = Path(path)
    sidecar = path.parent / (path.stem + "_sections.json")

    if sidecar.exists():
        sections = json.loads(sidecar.read_text(encoding="utf-8"))
        units = []
        for s in sections:
            txt = _normalize(s.get("content", ""))
            if len(txt) < 30:
                continue
            _pii_check(txt, doc_id)
            units.append({
                "doc_id":  doc_id,
                "source":  path.name,
                "section": s.get("heading", "main"),
                "text":    txt,
            })
        return units

    # Fallback: read plain text
    text = _normalize(path.read_text(encoding="utf-8", errors="replace"))
    _pii_check(text, doc_id)
    return [{"doc_id": doc_id, "source": path.name,
             "section": "main", "text": text}]


# ── ERP CSV extraction ────────────────────────────────────────────────────────
def extract_csv(path: str | Path, doc_id: str) -> list[dict]:
    """
    Convert each ERP CSV row into a natural-language sentence.
    Raw CSV rows embed poorly; NL sentences embed much better.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pip install pandas")

    path = Path(path)
    df   = pd.read_csv(str(path))
    units = []
    for _, row in df.iterrows():
        sentence = _row_to_sentence(row.to_dict())
        _pii_check(sentence, doc_id)
        units.append({
            "doc_id":  doc_id,
            "source":  path.name,
            "section": "erp_row",
            "text":    sentence,
        })
    return units


def _row_to_sentence(row: dict) -> str:
    parts = [
        f"{str(k).replace('_', ' ')} is {v}"
        for k, v in row.items()
        if v is not None and str(v).strip() not in ("", "nan", "NaN")
    ]
    return "Course record: " + "; ".join(parts) + "."


# ── JSONL passthrough ─────────────────────────────────────────────────────────
def extract_jsonl(path: str | Path, doc_id: str) -> list[dict]:
    path  = Path(path)
    units = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            txt = _normalize(obj.get("text", ""))
            if len(txt) < 30:
                continue
            _pii_check(txt, doc_id)
            units.append({
                "doc_id":  doc_id,
                "source":  obj.get("source", path.name),
                "section": obj.get("section", "main"),
                "text":    txt,
            })
    return units


# ── Master ingest function ────────────────────────────────────────────────────
def ingest_kb(metadata_csv: str | Path | None = None) -> list[dict]:
    """
    Read metadata.csv, iterate all docs, dispatch to the right extractor.
    Returns a flat list of extraction units.
    """
    if metadata_csv is None:
        metadata_csv = config.KB_DIR / "metadata.csv"

    metadata_csv = Path(metadata_csv)
    if not metadata_csv.exists():
        raise FileNotFoundError(
            f"metadata.csv not found at {metadata_csv}. "
            "Run the scraper first: python src/scraper/crawler.py"
        )

    all_units: list[dict] = []

    with open(metadata_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            doc_id     = row.get("doc_id", "")
            local_path = row.get("local_path", "")
            doc_type   = row.get("type", "")

            p = Path(local_path)
            if not p.is_absolute():
                if not p.exists():
                    p = config._ROOT / local_path
                if not p.exists():
                    p = config.KB_DIR / Path(local_path).name

            if not p.exists():
                print(f"[SKIP] {doc_id}: file not found → {local_path}")
                continue
            local_path = str(p)

            try:
                if "pdf" in doc_type.lower() or local_path.endswith(".pdf"):
                    units = extract_pdf(local_path, doc_id)
                elif local_path.endswith(".csv"):
                    units = extract_csv(local_path, doc_id)
                elif local_path.endswith(".jsonl"):
                    units = extract_jsonl(local_path, doc_id)
                else:
                    units = extract_html(local_path, doc_id)

                # Attach metadata from CSV to every unit
                for u in units:
                    u["source_url"]  = row.get("source_url", "")
                    u["title"]       = row.get("title", "")
                    u["source_name"] = row.get("source_name", u["source"])

                all_units.extend(units)
                print(f"[OK] {doc_id}: {len(units)} units ← {Path(local_path).name}")

            except Exception as e:
                print(f"[ERROR] {doc_id}: {e}")

    print(f"\n✅ Ingested {len(all_units)} units from {metadata_csv.parent}")
    return all_units


if __name__ == "__main__":
    units = ingest_kb()
    out = config.KB_DIR / "units.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for u in units:
            f.write(json.dumps(u, ensure_ascii=False) + "\n")
    print(f"Saved → {out}")
