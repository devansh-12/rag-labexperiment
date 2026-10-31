"""
SPIT RAG Chatbot – FastAPI Backend
CE413 Generative AI Lab | Experiment 8
"""
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Literal, Optional
import csv
import json
import math
import os
import sys
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor

from fastapi.middleware.cors import CORSMiddleware

# ── make src importable ───────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

app = FastAPI(title="SPIT RAG Chatbot", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Mount static files ────────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory="static"), name="static")

# ── Lazy pipeline state ───────────────────────────────────────────────────────
_pipelines = {}
_pipeline_lock = threading.Lock()
_evaluation_jobs: dict[str, dict] = {}
_evaluation_lock = threading.Lock()
_evaluation_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rag-eval")

def _json_safe(value):
    if hasattr(value, "item"):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value

def _persist_evaluation_job(job_id: str, updates: dict) -> None:
    import config
    with _evaluation_lock:
        job = _evaluation_jobs[job_id]
        job.update(_json_safe(updates))
        status_path = config.RESULTS_DIR / f"{job_id}_status.json"
        status_path.write_text(
            json.dumps(job, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )

def available_indexes() -> list[dict[str, str]]:
    import config
    indexes = []
    for index_path in config.INDEX_DIR.glob("index_*_*.faiss"):
        tag = index_path.stem.removeprefix("index_")
        chunk_path = config.INDEX_DIR / f"chunks_{tag}.jsonl"
        if not chunk_path.exists():
            continue
        chunker, separator, model_key = tag.rpartition("_")
        if separator and chunker in {"C1", "C2", "C3", "C4", "C5", "C6"} and model_key in {"A", "B"}:
            indexes.append({"chunker": chunker, "embed": model_key})
    return sorted(indexes, key=lambda item: (item["chunker"], item["embed"]))

def get_pipeline(chunker: str = "C1", embed: str = "A"):
    key = (chunker, embed)
    if key not in _pipelines:
        with _pipeline_lock:
            if key not in _pipelines:
                if not any(
                    item["chunker"] == chunker and item["embed"] == embed
                    for item in available_indexes()
                ):
                    raise FileNotFoundError(
                        f"No {chunker}/{embed} index is built. Build it before selecting it."
                    )
                from embed_index import load_index, _get_model
                from retrievers import build_bm25
                from generate import load_generator
                import config
                idx, chunks = load_index(chunker_id=chunker, model_key=embed)
                bm25 = build_bm25(chunks)
                gen = load_generator()
                model_name = config._EMBED_MODELS[embed]
                model = _get_model(model_name)
                _pipelines[key] = {
                    "index": idx, "chunks": chunks, "bm25": bm25,
                    "gen": gen, "model": model,
                }
    return _pipelines[key]

def build_pipeline():
    """Load the configured default pipeline and keep startup failures visible."""
    import config
    try:
        return get_pipeline(config.DEFAULT_CHUNKER, config.EMBED_MODEL_KEY)
    except Exception as e:
        print(f"[WARN] Pipeline not ready, using DEMO mode: {e}")
        return None

# ── Request / Response schemas ─────────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str
    k: int = 3
    retriever: Literal["dense", "bm25", "hybrid", "mmr", "rerank"] = "hybrid"
    chunker: Literal["C1", "C2", "C3", "C4", "C5", "C6"] = "C1"
    embed: Literal["A", "B"] = "A"
    threshold: float = 0.0
    temperature: float = 0.0

class EvaluationRequest(BaseModel):
    evaluator: Literal["openevals", "ragas"]
    sample_size: Literal[1, 5, 10, 30] = 5
    k: int = 3
    retriever: Literal["dense", "bm25", "hybrid", "mmr", "rerank"] = "hybrid"
    chunker: Literal["C1", "C2", "C3", "C4", "C5", "C6"] = "C1"
    embed: Literal["A", "B"] = "A"
    threshold: float = 0.0

class ChunkInfo(BaseModel):
    chunk_id: str
    source: str
    section: str
    text: str
    score: float
    rank_before_rerank: Optional[int] = None

class ChatResponse(BaseModel):
    answer: str
    chunks: list[ChunkInfo]
    cited_ids: list[str]
    t_retrieve: float
    t_rerank: float
    t_generate: float
    hit_at_k: Optional[float] = None
    mode: str = "live"

# ── Startup: eagerly load pipeline so first request is fast ───────────────────
@app.on_event("startup")
def startup_load_pipeline():
    print("[Startup] Pre-loading pipeline…")
    get_pipeline()
    print("[Startup] Pipeline ready.")

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return FileResponse(
        "templates/index.html",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )

@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """Sync handler — FastAPI runs this in a thread pool automatically,
    so blocking model inference won't freeze the event loop."""
    try:
        import config
        if req.threshold < 0 or req.threshold > 1:
            raise HTTPException(status_code=422, detail="threshold must be between 0 and 1")
        if req.threshold and req.retriever in {"hybrid", "bm25"}:
            raise HTTPException(
                status_code=422,
                detail="The threshold slider applies to dense, MMR, and rerank scores; "
                       "hybrid and BM25 scores use different scales.",
            )
        pipeline = get_pipeline(req.chunker, req.embed)
        return _run_live_chat(req, pipeline)
    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

def _run_live_chat(req: ChatRequest, pipeline: dict) -> ChatResponse:
    import retrievers as R
    from generate import generate_answer
    from prompt import build_prompt

    model = pipeline["model"]
    chunks = pipeline["chunks"]
    index = pipeline["index"]
    bm25 = pipeline["bm25"]

    t0 = time.perf_counter()
    if req.retriever == "dense":
        results = R.retrieve_dense(req.question, model, index, chunks, req.k)
    elif req.retriever == "bm25":
        results = R.retrieve_bm25(req.question, bm25, chunks, req.k)
    elif req.retriever == "mmr":
        results = R.retrieve_mmr(req.question, model, index, chunks, req.k)
    else:
        candidate_k = min(len(chunks), max(req.k * 5, 20)) if req.retriever == "rerank" else req.k
        results = R.retrieve_hybrid_rrf(
            req.question, model, index, bm25, chunks, k=candidate_k,
        )
    t_retrieve = time.perf_counter() - t0

    t1 = time.perf_counter()
    if req.retriever == "rerank" and results:
        results = R.rerank(req.question, results, k=req.k)
    t_rerank = time.perf_counter() - t1

    if req.threshold > 0:
        results = R.apply_threshold(results, req.threshold)

    if not results:
        reason = "threshold_rejected" if req.threshold > 0 else "no_results"
        return ChatResponse(
            answer="I don't have this information in the SPIT knowledge base.",
            chunks=[], cited_ids=[], t_retrieve=round(t_retrieve, 3),
            t_rerank=round(t_rerank, 3), t_generate=0.0, mode=reason,
        )

    t2 = time.perf_counter()
    prompt = build_prompt(req.question, results)
    answer, cited_ids = generate_answer(
        prompt, pipeline["gen"], temperature=req.temperature,
    )
    t_generate = time.perf_counter() - t2

    chunk_infos = [
        ChunkInfo(
            chunk_id=c.get("chunk_id", "?"),
            source=c.get("source", "?"),
            section=c.get("section", "?"),
            text=c.get("text", ""),
            score=round(float(score), 4),
            rank_before_rerank=c.get("_pre_rank"),
        )
        for c, score in results
    ]
    return ChatResponse(
        answer=answer, chunks=chunk_infos, cited_ids=cited_ids,
        t_retrieve=round(t_retrieve, 3), t_rerank=round(t_rerank, 3),
        t_generate=round(t_generate, 3), mode="live",
    )

@app.get("/api/status")
async def status():
    import config
    indexes = available_indexes()
    return {
        "pipeline_ready": (config.DEFAULT_CHUNKER, config.EMBED_MODEL_KEY) in _pipelines,
        "mode": "live" if indexes else "demo",
        "available_indexes": indexes,
        "qa_count": sum(
            1 for line in (config._ROOT / "eval" / "qa_set.jsonl").open(encoding="utf-8")
            if line.strip()
        ),
    }

@app.post("/api/evaluation/run", status_code=202)
def start_evaluation(req: EvaluationRequest):
    import config
    if req.threshold < 0 or req.threshold > 1:
        raise HTTPException(status_code=422, detail="threshold must be between 0 and 1")
    if req.threshold and req.retriever in {"hybrid", "bm25"}:
        raise HTTPException(
            status_code=422,
            detail="The threshold slider applies to dense, MMR, and rerank scores; "
                   "hybrid and BM25 scores use different scales.",
        )
    try:
        get_pipeline(req.chunker, req.embed)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    if req.evaluator == "ragas":
        provider_keys = {
            "openai": config.OPENAI_API_KEY,
            "gemini": config.GOOGLE_API_KEY,
            "anthropic": config.ANTHROPIC_API_KEY,
            "groq": config.GROQ_API_KEY,
        }
        if config.LLM_PROVIDER != "ollama" and not provider_keys.get(config.LLM_PROVIDER):
            raise HTTPException(
                status_code=400,
                detail=f"No judge API key is configured for {config.LLM_PROVIDER}.",
            )

    with _evaluation_lock:
        if any(job["status"] == "running" for job in _evaluation_jobs.values()):
            raise HTTPException(status_code=409, detail="An evaluation is already running.")
        job_id = uuid.uuid4().hex[:12]
        _evaluation_jobs[job_id] = {
            "job_id": job_id,
            "status": "running",
            "phase": "Generating answers",
            "evaluator": req.evaluator,
            "completed": 0,
            "total": req.sample_size,
            "summary": None,
            "error": None,
            "output_file": None,
        }
    _persist_evaluation_job(job_id, {})
    _evaluation_executor.submit(_run_evaluation, job_id, req)
    return {"job_id": job_id, "status": "running"}

@app.get("/api/evaluation/{job_id}")
def evaluation_status(job_id: str):
    import config
    with _evaluation_lock:
        job = _evaluation_jobs.get(job_id)
        if job is not None:
            job = _json_safe(job)
            if job.get("status") == "failed" and job.get("error"):
                job["error"] = _evaluation_error_message(Exception(job["error"]))
            if job.get("status") == "completed" and not _has_finite_evaluation_score(job):
                job.update({
                    "status": "failed",
                    "phase": "Failed",
                    "error": "The evaluator returned no valid metric scores.",
                })
            return job
    status_path = config.RESULTS_DIR / f"{job_id}_status.json"
    if status_path.exists():
        job = _json_safe(json.loads(status_path.read_text(encoding="utf-8")))
        if job.get("status") == "failed" and job.get("error"):
            job["error"] = _evaluation_error_message(Exception(job["error"]))
        if job.get("status") == "running":
            job.update({
                "status": "failed",
                "phase": "Failed",
                "error": "The server restarted before this evaluation finished.",
            })
            status_path.write_text(
                json.dumps(job, ensure_ascii=False, allow_nan=False),
                encoding="utf-8",
            )
        if job.get("status") == "completed" and not _has_finite_evaluation_score(job):
            job.update({
                "status": "failed",
                "phase": "Failed",
                "error": "The evaluator returned no valid metric scores.",
            })
            status_path.write_text(
                json.dumps(job, ensure_ascii=False, allow_nan=False),
                encoding="utf-8",
            )
        return job
    raise HTTPException(status_code=404, detail="Evaluation job not found.")

def _has_finite_evaluation_score(job: dict) -> bool:
    summary = job.get("summary") or {}
    scores = [value for key, value in summary.items() if key != "n"]
    return any(
        isinstance(value, (int, float)) and math.isfinite(value)
        for value in scores
    )

def _evaluation_error_message(error: Exception) -> str:
    message = str(error)
    lowered = message.lower()
    if any(marker in lowered for marker in (
        "rate_limit_exceeded",
        "input tokens per minute",
        "error code: 429",
        "error code: 413",
        "rate limit",
    )):
        return (
            "The configured judge provider rate-limited this evaluation. "
            "Wait for the quota window to reset, then retry with a smaller sample."
        )
    return message

def _run_evaluation(job_id: str, req: EvaluationRequest) -> None:
    import config
    qa_path = config._ROOT / "eval" / "qa_set.jsonl"
    results_path = config.RESULTS_DIR / f"{job_id}_run.jsonl"
    try:
        with qa_path.open(encoding="utf-8") as f:
            qa_rows = [json.loads(line) for line in f if line.strip()]
        qa_rows = qa_rows[:req.sample_size]
        pipeline = get_pipeline(req.chunker, req.embed)
        run_rows = []
        for qa in qa_rows:
            chat_req = ChatRequest(
                question=qa["question"],
                k=req.k,
                retriever=req.retriever,
                chunker=req.chunker,
                embed=req.embed,
                threshold=req.threshold,
            )
            response = _run_live_chat(chat_req, pipeline)
            run_rows.append({
                "qid": qa["qid"],
                "question": qa["question"],
                "answer": response.answer,
                "retrieved_chunks": [
                    chunk.model_dump() for chunk in response.chunks
                ],
                "t_retrieve": response.t_retrieve,
                "t_rerank": response.t_rerank,
                "t_generate": response.t_generate,
            })
            _persist_evaluation_job(job_id, {"completed": len(run_rows)})

        with results_path.open("w", encoding="utf-8") as f:
            for row in run_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        sys.path.insert(0, str(config._ROOT / "eval"))
        _persist_evaluation_job(job_id, {
            "phase": f"Scoring with {req.evaluator}",
        })
        if req.evaluator == "openevals":
            from run_openevals import run_openevals
            summary = run_openevals(qa_path, results_path)
            output_file = str(config.RESULTS_DIR / f"{results_path.stem}_openevals.jsonl")
        else:
            from run_ragas import run_ragas
            summary = run_ragas(qa_path, results_path)
            output_file = str(config.RESULTS_DIR / f"{results_path.stem}_ragas.csv")

        _persist_evaluation_job(job_id, {
            "status": "completed",
            "phase": "Complete",
            "summary": summary,
            "output_file": os.path.relpath(output_file, config._ROOT),
        })
    except Exception as e:
        traceback.print_exc()
        _persist_evaluation_job(job_id, {
            "status": "failed",
            "phase": "Failed",
            "error": _evaluation_error_message(e),
        })

@app.get("/api/metrics/history")
async def metrics_history():
    """Return last N run metrics from results/runs.jsonl if it exists."""
    path = "results/runs.jsonl"
    if not os.path.exists(path):
        return {"runs": []}
    runs = []
    with open(path) as f:
        for line in f:
            try:
                runs.append(json.loads(line))
            except Exception:
                pass
    return {"runs": runs[-50:]}   # last 50

# ── Demo mode (no real pipeline) ──────────────────────────────────────────────
def _demo_response(req: ChatRequest) -> ChatResponse:
    """Return plausible-looking mock data while the pipeline loads."""
    import random, time
    time.sleep(0.6)
    demo_chunks = [
        ChunkInfo(chunk_id="a1b2c3", source="About SPIT",
                  section="Introduction", score=round(random.uniform(0.78,0.96),3),
                  text="Sardar Patel Institute of Technology (SPIT) is an autonomous institute "
                       "affiliated to the University of Mumbai, located at Bhavan's Campus, "
                       "Munshi Nagar, Andheri (West), Mumbai-400058."),
        ChunkInfo(chunk_id="d4e5f6", source="Admissions",
                  section="Eligibility", score=round(random.uniform(0.60,0.77),3),
                  text="Admissions to SPIT are conducted through the MHT-CET / JEE Main "
                       "counselling process conducted by the state authority. Candidates must "
                       "have passed 10+2 with Physics, Chemistry and Mathematics."),
        ChunkInfo(chunk_id="g7h8i9", source="CE Dept",
                  section="About", score=round(random.uniform(0.45,0.59),3),
                  text="The Computer Engineering department at SPIT offers B.Tech, M.Tech and "
                       "Ph.D. programmes. It is equipped with state-of-the-art labs including "
                       "an AI/ML lab, a cybersecurity lab and a cloud computing centre."),
    ][:req.k]

    answer = (
        f"**[DEMO MODE — connect real pipeline]**\n\n"
        f"Based on the retrieved context, SPIT (Sardar Patel Institute of Technology) "
        f"is an autonomous institute affiliated to the University of Mumbai. "
        f"Your question was: *\"{req.question}\"* — retrieved using **{req.retriever}** "
        f"with k={req.k}. [S1][S2]"
    )
    return ChatResponse(
        answer=answer, chunks=demo_chunks, cited_ids=["a1b2c3","d4e5f6"],
        t_retrieve=round(random.uniform(0.05,0.25),3),
        t_rerank=round(random.uniform(0.0,0.15),3),
        t_generate=round(random.uniform(0.3,1.2),3),
        mode="demo"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
