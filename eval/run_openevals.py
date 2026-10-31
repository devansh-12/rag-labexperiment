"""
eval/run_openevals.py
─────────────────────────────────────────────────────────────────────────────
openevals LLM-as-judge evaluation.

Judges:
  • correctness      – answer vs gold_answer   → Answer Accuracy
  • groundedness     – answer vs retrieved ctx → Faithfulness
  • hallucination    – detects unsupported claims
  • retrieval_rel    – each chunk vs question  → Precision@k (LLM-judged)
  • refusal_correct  – for unanswerable questions

Usage:
    python eval/run_openevals.py --results results/R5.jsonl --qa eval/qa_set.jsonl
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))
import config
from run_ragas import get_ragas_llm


def _make_evaluator(prompt: str, key: str, judge) -> Callable:
    from openevals.llm import create_llm_as_judge

    return create_llm_as_judge(
        prompt=prompt,
        feedback_key=key,
        judge=judge,
        use_reasoning=True,
    )


def _score(evaluator: Callable, *, question: str, answer: str,
           context: str = "", reference: str = "") -> int:
    result = evaluator(
        inputs={"question": question, "context": context},
        outputs={"answer": answer},
        reference_outputs={"answer": reference},
    )
    return int(bool(result["score"]))


# ── Per-question evaluation ───────────────────────────────────────────────────

def evaluate_row(qa: dict, res: dict, evaluators: dict[str, Callable]) -> dict:
    question   = res.get("question", qa.get("question", ""))
    answer     = res.get("answer", "")
    gold       = qa.get("gold_answer")
    q_type     = qa.get("type", "factual")
    retrieved  = res.get("retrieved_chunks", [])
    context    = "\n---\n".join(c.get("text", "") for c in retrieved) or "(none)"

    scores = {}

    # Correctness (skip unanswerable)
    if gold:
        scores["correctness"] = _score(
            evaluators["correctness"], question=question, answer=answer,
            reference=gold,
        )
    else:
        scores["correctness"] = None

    scores["groundedness"] = _score(
        evaluators["groundedness"], question=question, answer=answer,
        context=context,
    )
    scores["no_hallucination"] = _score(
        evaluators["no_hallucination"], question=question, answer=answer,
        context=context,
    )

    # Retrieval relevance (per-chunk, then average)
    if retrieved:
        rel_scores = []
        for chunk in retrieved:
            rel_scores.append(_score(
                evaluators["retrieval_relevance"],
                question=question,
                answer=chunk.get("text", ""),
            )
        )
        scores["retrieval_precision_llm"] = sum(rel_scores) / len(rel_scores)
    else:
        scores["retrieval_precision_llm"] = 0.0

    # Refusal correctness (only for unanswerable)
    if q_type == "unanswerable" or gold is None:
        scores["refusal_correct"] = _score(
            evaluators["refusal"], question=question, answer=answer,
        )
    else:
        scores["refusal_correct"] = None

    return {**res, **scores, "qid": qa["qid"], "type": q_type}


# ── Run full evaluation ───────────────────────────────────────────────────────

def run_openevals(
    qa_path:      str | Path,
    results_path: str | Path,
) -> dict:
    def load_jsonl(p):
        with open(p) as f:
            return [json.loads(l) for l in f if l.strip()]

    qa_rows  = load_jsonl(qa_path)
    run_rows = load_jsonl(results_path)

    qa_by_qid = {q["qid"]: q for q in qa_rows}
    evaluated  = []
    judge = get_ragas_llm()
    evaluators = {
        "correctness": _make_evaluator(
            """Judge whether the answer correctly answers the question and agrees with the reference.
Question: {inputs}
Answer: {outputs}
Reference answer: {reference_outputs}""",
            "correctness", judge,
        ),
        "groundedness": _make_evaluator(
            """Judge whether every factual claim in the answer is supported by the retrieved context.
Question and context: {inputs}
Answer: {outputs}""",
            "groundedness", judge,
        ),
        "no_hallucination": _make_evaluator(
            """Judge whether the answer contains any factual claim unsupported by the retrieved context.
Return true only when no unsupported factual claims are present.
Question and context: {inputs}
Answer: {outputs}""",
            "no_hallucination", judge,
        ),
        "retrieval_relevance": _make_evaluator(
            """Judge whether the retrieved text is relevant to answering the question.
Question and context: {inputs}
Retrieved text: {outputs}""",
            "retrieval_relevance", judge,
        ),
        "refusal": _make_evaluator(
            """Judge whether the assistant appropriately declined to answer a question that cannot be answered from the knowledge base, without inventing facts.
Question and context: {inputs}
Answer: {outputs}""",
            "refusal_correct", judge,
        ),
    }

    for i, row in enumerate(run_rows, 1):
        qid = row.get("qid")
        qa  = qa_by_qid.get(qid)
        if not qa:
            continue
        print(f"  [openevals] Q{qid} ({i}/{len(run_rows)})…", end=" ", flush=True)
        scored = evaluate_row(qa, row, evaluators)
        evaluated.append(scored)
        print(f"correct={scored['correctness']} "
              f"grounded={scored['groundedness']} "
              f"no_halluc={scored['no_hallucination']}")

    # Aggregate
    keys    = ["correctness", "groundedness", "no_hallucination",
               "retrieval_precision_llm", "refusal_correct"]
    summary = {}
    for k in keys:
        vals = [r[k] for r in evaluated if r.get(k) is not None]
        summary[k] = round(sum(vals) / len(vals), 4) if vals else None

    summary["hallucination_rate"] = round(
        1 - (summary["no_hallucination"] or 0), 4
    )
    summary["answer_accuracy"]    = summary["correctness"]
    summary["faithfulness"]       = summary["groundedness"]
    summary["n"] = len(evaluated)

    # Save
    out_jsonl = config.RESULTS_DIR / (Path(results_path).stem + "_openevals.jsonl")
    with open(out_jsonl, "w") as f:
        for row in evaluated:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    print("\n── openevals Summary ──────────────────────────")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"Saved → {out_jsonl}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--qa",      default="eval/qa_set.jsonl")
    args = parser.parse_args()
    run_openevals(args.qa, args.results)
