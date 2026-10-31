"""
eval/run_ragas.py
─────────────────────────────────────────────────────────────────────────────
RAGAS evaluation over the run results.

Metrics used:
  • Faithfulness
  • ResponseRelevancy
  • LLMContextPrecisionWithReference
  • LLMContextRecall
  • FactualCorrectness
  • NoiseSensitivity (Experiment C)

Usage:
    python eval/run_ragas.py --results results/R5.jsonl --qa eval/qa_set.jsonl
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
import config

def load_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def build_ragas_dataset(qa_rows: list[dict], run_rows: list[dict]):
    """Convert QA + run results into RAGAS Dataset format."""
    try:
        from ragas import EvaluationDataset, SingleTurnSample
    except ImportError:
        raise ImportError("pip install ragas")

    samples = []
    qa_by_qid = {q["qid"]: q for q in qa_rows}

    for row in run_rows:
        qa = qa_by_qid.get(row.get("qid"))
        if not qa:
            continue

        retrieved_contexts = [c.get("text", "") for c in row.get("retrieved_chunks", [])]
        if not retrieved_contexts:
            retrieved_contexts = [""]

        sample = SingleTurnSample(
            user_input        = row["question"],
            response          = row.get("answer", ""),
            reference         = qa.get("gold_answer") or "",
            retrieved_contexts= retrieved_contexts,
        )
        samples.append(sample)

    return EvaluationDataset(samples=samples)


def get_ragas_llm():
    """Return a RAGAS-compatible LLM wrapper for the configured provider."""
    provider = config.LLM_PROVIDER

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=config.OPENAI_JUDGE_MODEL,
            temperature=0,
            api_key=config.OPENAI_API_KEY,
            base_url=config.OPENAI_BASE_URL,
            timeout=60,
            max_retries=1,
        )
    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=config.GEMINI_JUDGE_MODEL,
            temperature=0,
            google_api_key=config.GOOGLE_API_KEY,
        )
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=config.CLAUDE_JUDGE_MODEL,
            temperature=0,
            api_key=config.ANTHROPIC_API_KEY,
        )
    elif provider == "ollama":
        from langchain_community.llms import Ollama
        return Ollama(model=config.OLLAMA_JUDGE_MODEL,
                      base_url=config.OLLAMA_BASE_URL)
    elif provider == "groq":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=config.GROQ_JUDGE_MODEL,
            temperature=0,
            api_key=config.GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1",
            timeout=60,
            max_retries=1,
        )
    raise ValueError(f"Unknown provider: {provider}")


def get_ragas_embeddings():
    """Return a RAGAS-compatible embeddings wrapper."""
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=config.EMBED_MODEL_NAME)


def run_ragas(
    qa_path:     str | Path,
    results_path: str | Path,
    include_noise_sensitivity: bool = False,
) -> dict:
    from ragas import evaluate
    from ragas.run_config import RunConfig
    from ragas.metrics import (
        Faithfulness,
        ResponseRelevancy,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        FactualCorrectness,
    )

    metrics = [
        Faithfulness(),
        ResponseRelevancy(strictness=1),
        LLMContextPrecisionWithReference(),
        LLMContextRecall(),
        FactualCorrectness(),
    ]

    if include_noise_sensitivity:
        try:
            from ragas.metrics import NoiseSensitivity
            metrics.append(NoiseSensitivity())
        except ImportError:
            print("[WARN] NoiseSensitivity not available in this RAGAS version.")

    qa_rows  = load_jsonl(qa_path)
    run_rows = load_jsonl(results_path)
    dataset  = build_ragas_dataset(qa_rows, run_rows)

    llm        = get_ragas_llm()
    embeddings = get_ragas_embeddings()

    print(f"[RAGAS] Evaluating {len(dataset.samples)} samples…")
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        run_config=RunConfig(
            timeout=60,
            max_retries=5,
            max_wait=30,
            max_workers=1,
        ),
        raise_exceptions=True,
        show_progress=False,
        batch_size=1,
    )
    df = result.to_pandas()
    metric_columns = [
        column for column in df.select_dtypes(include="number").columns
    ]
    scored_cells = df[metric_columns].notna().sum().sum()
    if not scored_cells:
        raise RuntimeError("RAGAS completed without producing any metric scores.")

    out = Path(results_path).stem + "_ragas.csv"
    df.to_csv(config.RESULTS_DIR / out, index=False)
    print(f"[RAGAS] Results saved → {config.RESULTS_DIR / out}")

    summary = df.mean(numeric_only=True).to_dict()
    print("\n── RAGAS Summary ──────────────────────────")
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--qa",      default="eval/qa_set.jsonl")
    parser.add_argument("--noise",   action="store_true",
                        help="Include NoiseSensitivity (Experiment C)")
    args = parser.parse_args()
    run_ragas(args.qa, args.results, include_noise_sensitivity=args.noise)
