"""
src/config.py
─────────────────────────────────────────────────────────────────────────────
Central config loader — reads .env and exposes a typed Settings object.
All other modules import from here instead of os.getenv directly.
"""
from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root (two levels up from this file)
_ROOT = Path(__file__).parent.parent
load_dotenv(_ROOT / ".env", override=True)

def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)

# ── LLM ──────────────────────────────────────────────────────────────────────
LLM_PROVIDER        = _env("LLM_PROVIDER", "openai").lower()
LLM_TEMPERATURE     = float(_env("LLM_TEMPERATURE", "0.0"))
LLM_MAX_TOKENS      = int(_env("LLM_MAX_TOKENS", "512"))

OPENAI_API_KEY      = _env("OPENAI_API_KEY")
OPENAI_BASE_URL     = _env("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL        = _env("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_JUDGE_MODEL  = _env("OPENAI_JUDGE_MODEL", "gpt-4o")

GOOGLE_API_KEY      = _env("GOOGLE_API_KEY")
GEMINI_MODEL        = _env("GEMINI_MODEL", "gemini-1.5-flash")
GEMINI_JUDGE_MODEL  = _env("GEMINI_JUDGE_MODEL", "gemini-1.5-pro")

ANTHROPIC_API_KEY   = _env("ANTHROPIC_API_KEY")
CLAUDE_MODEL        = _env("CLAUDE_MODEL", "claude-3-haiku-20240307")
CLAUDE_JUDGE_MODEL  = _env("CLAUDE_JUDGE_MODEL", "claude-3-5-sonnet-20241022")

OLLAMA_BASE_URL     = _env("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL        = _env("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_JUDGE_MODEL  = _env("OLLAMA_JUDGE_MODEL", "llama3.1:70b")

GROQ_API_KEY        = _env("GROQ_API_KEY")
GROQ_MODEL          = _env("GROQ_MODEL", "qwen/qwen3.8-27b")
GROQ_JUDGE_MODEL    = _env("GROQ_JUDGE_MODEL", "qwen/qwen3.8-27b")

# ── Hugging Face Hub ──────────────────────────────────────────────────────────
HF_TOKEN = _env("HF_TOKEN")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN

# ── Embedding ─────────────────────────────────────────────────────────────────
_EMBED_MODELS = {
    "A": _env("EMBED_MODEL_A", "BAAI/bge-small-en-v1.5"),
    "B": _env("EMBED_MODEL_B", "BAAI/bge-base-en-v1.5"),
}
EMBED_MODEL_KEY  = _env("EMBED_MODEL", "A").upper()
EMBED_MODEL_NAME = _EMBED_MODELS.get(EMBED_MODEL_KEY, _EMBED_MODELS["A"])
RERANKER_MODEL   = _env("RERANKER_MODEL", "BAAI/bge-reranker-base")

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# ── Pipeline defaults ─────────────────────────────────────────────────────────
DEFAULT_K             = int(_env("DEFAULT_K", "3"))
DEFAULT_RETRIEVER     = _env("DEFAULT_RETRIEVER", "hybrid")
DEFAULT_CHUNKER       = _env("DEFAULT_CHUNKER", "C1")
RELEVANCE_THRESHOLD   = float(_env("RELEVANCE_THRESHOLD", "0.0"))

# ── Paths ─────────────────────────────────────────────────────────────────────
KB_DIR       = _ROOT / _env("KB_DIR", "kb")
INDEX_DIR    = _ROOT / _env("INDEX_DIR", "indexes")
RESULTS_DIR  = _ROOT / _env("RESULTS_DIR", "results")
INDEX_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Scraper ───────────────────────────────────────────────────────────────────
CRAWL_DELAY_SECONDS = float(_env("CRAWL_DELAY_SECONDS", "2.5"))
CRAWL_MAX_PAGES     = int(_env("CRAWL_MAX_PAGES", "80"))
SCRAPER_USER_AGENT  = _env("SCRAPER_USER_AGENT",
                            "SPITRagBot/1.0 (academic-research; CE413)")

def active_llm_model() -> str:
    return {
        "openai":    OPENAI_MODEL,
        "gemini":    GEMINI_MODEL,
        "anthropic": CLAUDE_MODEL,
        "ollama":    OLLAMA_MODEL,
        "groq":      GROQ_MODEL,
    }.get(LLM_PROVIDER, OPENAI_MODEL)

def active_judge_model() -> str:
    return {
        "openai":    OPENAI_JUDGE_MODEL,
        "gemini":    GEMINI_JUDGE_MODEL,
        "anthropic": CLAUDE_JUDGE_MODEL,
        "ollama":    OLLAMA_JUDGE_MODEL,
        "groq":      GROQ_JUDGE_MODEL,
    }.get(LLM_PROVIDER, OPENAI_JUDGE_MODEL)
