"""
eval/experiments.py
─────────────────────────────────────────────────────────────────────────────
Runs all pipeline configurations R0–R6 + ICL over qa_set.jsonl,
writes results to results/runs.jsonl and results/ablation_table.csv.

Usage:
    python eval/experiments.py --exp all
    python eval/experiments.py --exp A          # k sweep
    python eval/experiments.py --exp B          # retrieval failure
    python eval/experiments.py --exp chunker    # chunker sweep
    python eval/experiments.py --config R5      # single config
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

# Make src importable
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import config
from embed_index   import load_index, encode_query, _get_model
from retrievers    import build_bm25, retrieve, rerank
from prompt        import build_prompt
from generate      import load_generator, generate_answer
from metrics       import (hit_at_k, precision_at_k, mrr, answer_f1,
                            latency_breakdown, tag_failure, run_metrics)
from icl_baseline  import run_icl
from chunkers      import chunk
from ingest        import ingest_kb


# ── Load QA set ───────────────────────────────────────────────────────────────
def load_qa(path: str | Path | None = None) -> list[dict]:
    path = path or ROOT / "eval" / "qa_set.jsonl"
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


# ── Pipeline configurations ───────────────────────────────────────────────────
#   key → (chunker, retriever, do_rerank, tau)
CONFIGS = {
    "R0": ("C1",   "dense",  False, 0.0),
    "R1": ("best", "dense",  False, 0.0),
    "R2": ("best", "bm25",   False, 0.0),
    "R3": ("best", "hybrid", False, 0.0),
    "R4": ("best", "mmr",    False, 0.0),
    "R5": ("best", "hybrid", True,  0.0),
    "R6": ("best", "hybrid", True,  0.4),   # tau tuned on dev set
}
# "best" resolved after chunker sweep; default to C3 until then
BEST_CHUNKER = os.environ.get("BEST_CHUNKER", "C3")


def resolve_chunker(c: str) -> str:
    return BEST_CHUNKER if c == "best" else c


# ── Single question through pipeline ─────────────────────────────────────────
def run_question(
    question: str,
    model,
    index,
    chunks:   list[dict],
    bm25,
    gen_obj:  dict,
    retriever: str  = "hybrid",
    k:         int  = 3,
    tau:       float = 0.0,
    do_rerank: bool  = False,
) -> dict:
    t0 = time.perf_counter()
    results = retrieve(question, model, index, chunks, bm25,
                       strategy=retriever, k=k, tau=tau, do_rerank=do_rerank)
    t_retrieve = time.perf_counter() - t0

    t1 = time.perf_counter()
    # If rerank was already done inside retrieve(), t_rerank is captured above.
    # For separate logging we just measure the total retrieval stage here.
    t_rerank = time.perf_counter() - t1

    if not results:
        return {
            "answer": "I don't have this information in the SPIT knowledge base.",
            "retrieved_chunks": [],
            "cited_ids": [],
            "t_retrieve": round(t_retrieve, 4),
            "t_rerank":   round(t_rerank, 4),
            "t_generate": 0.0,
        }

    system_p, user_p = build_prompt(question, results)

    t2 = time.perf_counter()
    answer, cited_ids = generate_answer((system_p, user_p), gen_obj)
    t_generate = time.perf_counter() - t2

    return {
        "answer":           answer,
        "retrieved_chunks": [c for c, _ in results],
        "scores":           [s for _, s in results],
        "cited_ids":        cited_ids,
        **latency_breakdown(t_retrieve, t_rerank, t_generate),
    }


# ── Run one config over full QA set ──────────────────────────────────────────
def run_config(
    config_name: str,
    qa_rows:     list[dict],
    gen_obj:     dict,
    k:           int = 3,
    split:       str = "all",   # all | dev | test
) -> list[dict]:
    chunker_id, retriever, do_rerank, tau = CONFIGS[config_name]
    chunker_id = resolve_chunker(chunker_id)

    index, chunks = load_index(chunker_id=chunker_id)
    bm25 = build_bm25(chunks)
    model_name = config.EMBED_MODEL_NAME
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)

    rows = []
    for qa in qa_rows:
        if split == "dev" and not qa.get("split") == "dev":
            continue
        if split == "test" and not qa.get("split") == "test":
            continue

        res = run_question(
            qa["question"], model, index, chunks, bm25, gen_obj,
            retriever=retriever, k=k, tau=tau, do_rerank=do_rerank,
        )
        row = {
            "config":   config_name,
            "chunker":  chunker_id,
            "retriever":retriever,
            "k":        k,
            "qid":      qa["qid"],
            "type":     qa.get("type", ""),
            "question": qa["question"],
            **res,
            "gold_doc_ids":  qa.get("gold_doc_ids", []),
            "gold_evidence": qa.get("gold_evidence", []),
            "gold_answer":   qa.get("gold_answer"),
        }
        # Add per-question metrics
        row["hit_k"]    = hit_at_k(res["retrieved_chunks"],
                                   qa.get("gold_doc_ids", []),
                                   qa.get("gold_evidence", []), k)
        row["prec_k"]   = precision_at_k(res["retrieved_chunks"],
                                         qa.get("gold_doc_ids", []),
                                         qa.get("gold_evidence", []), k)
        row["mrr"]      = mrr(res["retrieved_chunks"],
                              qa.get("gold_doc_ids", []),
                              qa.get("gold_evidence", []))
        row["ans_f1"]   = answer_f1(res["answer"], qa.get("gold_answer") or "")
        row["failure"]  = tag_failure(
            res["retrieved_chunks"], res["answer"],
            qa.get("gold_doc_ids", []), qa.get("gold_evidence", []),
            qa.get("gold_answer"),
        )
        rows.append(row)
        print(f"  [{config_name}] Q{qa['qid']}: hit={row['hit_k']} "
              f"f1={row['ans_f1']:.2f}  {row['failure']}")

    return rows


# ── Experiment A: k sweep ─────────────────────────────────────────────────────
def experiment_A(qa_rows: list[dict], gen_obj: dict):
    print("\n" + "="*60)
    print(" Experiment A – Effect of Top-k")
    all_rows = []
    for k in [1, 3, 5, 10]:
        rows = run_config("R1", qa_rows, gen_obj, k=k)
        for r in rows:
            r["k"] = k
        all_rows.extend(rows)
        agg = run_metrics(qa_rows, rows, k=k)
        print(f"  k={k}: hit@k={agg[f'hit@{k}']}, mrr={agg['mrr']}, "
              f"f1={agg['answer_f1']}, t_gen={agg['avg_t_generate']}s")
    _save_rows(all_rows, "experiment_A.jsonl")
    return all_rows


# ── Experiment B: retrieval failure ──────────────────────────────────────────
def experiment_B(qa_rows: list[dict], gen_obj: dict):
    print("\n" + "="*60)
    print(" Experiment B – Retrieval Failure (gold docs removed)")
    # This experiment requires manually rebuilding index without gold docs.
    # We simulate by running with low tau so the system is forced to answer
    # even without evidence, then compare R1 vs R6.
    rows_r1 = run_config("R1", qa_rows, gen_obj, k=3)
    rows_r6 = run_config("R6", qa_rows, gen_obj, k=3)
    all_rows = rows_r1 + rows_r6
    _save_rows(all_rows, "experiment_B.jsonl")
    return all_rows


# ── Experiment C: irrelevant context ─────────────────────────────────────────
def experiment_C(qa_rows: list[dict], gen_obj: dict):
    print("\n" + "="*60)
    print(" Experiment C – Irrelevant / Misleading Context")
    print("  NOTE: Run with distractor/conflict corpus variants.")
    print("  Rebuild index (python src/embed_index.py) after swapping kb/.")
    rows = run_config("R3", qa_rows, gen_obj, k=5)
    _save_rows(rows, "experiment_C.jsonl")
    return rows


# ── Experiment D: RAG vs ICL ──────────────────────────────────────────────────
def experiment_D(qa_rows: list[dict], gen_obj: dict, all_docs: list[dict]):
    print("\n" + "="*60)
    print(" Experiment D – RAG vs ICL")
    rag_rows = run_config("R5", qa_rows, gen_obj, k=3)

    icl_rows = []
    for qa in qa_rows:
        res = run_icl(qa["question"], all_docs, gen_obj)
        row = {
            "config":           "ICL",
            "qid":              qa["qid"],
            "type":             qa.get("type", ""),
            "question":         qa["question"],
            **res,
            "retrieved_chunks": [],
            "gold_doc_ids":     qa.get("gold_doc_ids", []),
            "gold_answer":      qa.get("gold_answer"),
            "ans_f1":           answer_f1(res["answer"], qa.get("gold_answer") or ""),
        }
        icl_rows.append(row)
        print(f"  [ICL] Q{qa['qid']}: docs_used={res['n_docs_included']}/{res['n_docs_total']}"
              f"  f1={row['ans_f1']:.2f}")

    _save_rows(rag_rows + icl_rows, "experiment_D.jsonl")
    return rag_rows, icl_rows


# ── Chunker sweep ─────────────────────────────────────────────────────────────
def chunker_sweep(qa_rows: list[dict], gen_obj: dict):
    print("\n" + "="*60)
    print(" Chunker Sweep – C1 vs C3 vs C4 vs C6")
    all_rows = []
    for strat in ["C1", "C3", "C4", "C6"]:
        # Temporarily override CONFIGS R1 chunker
        CONFIGS["R1"] = (strat, "dense", False, 0.0)
        rows = run_config("R1", qa_rows, gen_obj, k=3)
        for r in rows:
            r["chunker"] = strat
        all_rows.extend(rows)
        agg = run_metrics(qa_rows, rows, k=3)
        print(f"  {strat}: hit@3={agg['hit@3']}, mrr={agg['mrr']}, f1={agg['answer_f1']}")
    _save_rows(all_rows, "chunker_sweep.jsonl")


# ── Ablation table ────────────────────────────────────────────────────────────
def build_ablation_table(qa_rows: list[dict], gen_obj: dict):
    print("\n" + "="*60)
    print(" Building Ablation Table – R0 through R6")
    all_agg = []
    for cfg_name in CONFIGS:
        rows = run_config(cfg_name, qa_rows, gen_obj, k=3)
        agg  = run_metrics(qa_rows, rows, k=3)
        agg["config"] = cfg_name
        all_agg.append(agg)

    out = config.RESULTS_DIR / "ablation_table.csv"
    fields = ["config", "hit@3", "precision@3", "mrr", "answer_f1",
              "avg_t_retrieve", "avg_t_generate", "n"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_agg)
    print(f"  Ablation table → {out}")


# ── Helpers ───────────────────────────────────────────────────────────────────
def _save_rows(rows: list[dict], filename: str):
    out = config.RESULTS_DIR / filename
    with open(out, "w", encoding="utf-8") as f:
        for row in rows:
            # make row JSON-serialisable
            safe = {k: (v if not hasattr(v, 'tolist') else v.tolist())
                    for k, v in row.items()}
            f.write(json.dumps(safe, ensure_ascii=False) + "\n")
    print(f"  Saved → {out}")


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="SPIT RAG Experiments")
    parser.add_argument("--exp", default="ablation",
                        choices=["all","A","B","C","D","chunker","ablation"] +
                                list(CONFIGS.keys()),
                        help="Which experiment to run")
    parser.add_argument("--qa", default=None, help="Path to QA JSONL")
    parser.add_argument("--split", default="all", choices=["all","dev","test"])
    args = parser.parse_args()

    qa_rows = load_qa(args.qa)
    gen_obj = load_generator()
    all_docs = ingest_kb()   # needed for ICL

    print(f"[Experiments] Loaded {len(qa_rows)} questions. Provider: {config.LLM_PROVIDER}")

    exp = args.exp
    if exp in CONFIGS:
        rows = run_config(exp, qa_rows, gen_obj, split=args.split)
        _save_rows(rows, f"{exp}.jsonl")
    elif exp == "A":
        experiment_A(qa_rows, gen_obj)
    elif exp == "B":
        experiment_B(qa_rows, gen_obj)
    elif exp == "C":
        experiment_C(qa_rows, gen_obj)
    elif exp == "D":
        experiment_D(qa_rows, gen_obj, all_docs)
    elif exp == "chunker":
        chunker_sweep(qa_rows, gen_obj)
    elif exp == "ablation":
        build_ablation_table(qa_rows, gen_obj)
    elif exp == "all":
        chunker_sweep(qa_rows, gen_obj)
        experiment_A(qa_rows, gen_obj)
        experiment_B(qa_rows, gen_obj)
        experiment_C(qa_rows, gen_obj)
        experiment_D(qa_rows, gen_obj, all_docs)
        build_ablation_table(qa_rows, gen_obj)


if __name__ == "__main__":
    main()
