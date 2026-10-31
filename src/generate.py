"""
src/generate.py
─────────────────────────────────────────────────────────────────────────────
Task 8 – Answer Generation
Supports: OpenAI, Gemini, Anthropic Claude, Ollama (local).
Provider is chosen from config.LLM_PROVIDER at runtime.

Returns (answer: str, cited_ids: list[str])
"""
from __future__ import annotations
import re
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
import config

# ─────────────────────────────────────────────────────────────────────────────
# Provider loader – lazy-imported so unused SDKs are never required
# ─────────────────────────────────────────────────────────────────────────────

def load_generator():
    """
    Return a callable generator object.
    Each provider returns a different object; generate_answer() handles dispatch.
    """
    provider = config.LLM_PROVIDER
    if provider == "openai":
        if not config.OPENAI_API_KEY:
            print("[INFO] No OPENAI_API_KEY set; using grounded extractive generator.")
            return {"provider": "grounded"}
        return _load_openai()
    elif provider == "gemini":
        if not config.GOOGLE_API_KEY:
            print("[INFO] No GOOGLE_API_KEY set; using grounded extractive generator.")
            return {"provider": "grounded"}
        return _load_gemini()
    elif provider == "anthropic":
        if not config.ANTHROPIC_API_KEY:
            print("[INFO] No ANTHROPIC_API_KEY set; using grounded extractive generator.")
            return {"provider": "grounded"}
        return _load_anthropic()
    elif provider == "ollama":
        return _load_ollama()
    elif provider == "groq":
        return _load_groq()
    elif provider in ("grounded", "local", "mock"):
        return {"provider": "grounded"}
    else:
        return {"provider": "grounded"}


# ── OpenAI ────────────────────────────────────────────────────────────────────
def _load_openai():
    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=config.OPENAI_API_KEY,
            base_url=config.OPENAI_BASE_URL,
        )
        return {"provider": "openai", "client": client}
    except ImportError:
        raise ImportError("pip install openai")


def _gen_openai(gen_obj, system_prompt: str, user_prompt: str,
                temperature: float, max_tokens: int) -> str:
    client = gen_obj["client"]
    resp = client.chat.completions.create(
        model=config.OPENAI_MODEL,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system",  "content": system_prompt},
            {"role": "user",    "content": user_prompt},
        ],
    )
    return resp.choices[0].message.content.strip()


# ── Gemini ────────────────────────────────────────────────────────────────────
def _load_gemini():
    try:
        import google.generativeai as genai
        genai.configure(api_key=config.GOOGLE_API_KEY)
        model = genai.GenerativeModel(
            model_name=config.GEMINI_MODEL,
            system_instruction="You are the SPIT assistant.",
        )
        return {"provider": "gemini", "model": model}
    except ImportError:
        raise ImportError("pip install google-generativeai")


def _gen_gemini(gen_obj, system_prompt: str, user_prompt: str,
                temperature: float, max_tokens: int) -> str:
    import google.generativeai as genai
    cfg = genai.types.GenerationConfig(
        temperature=temperature, max_output_tokens=max_tokens
    )
    full = f"{system_prompt}\n\n{user_prompt}"
    resp = gen_obj["model"].generate_content(full, generation_config=cfg)
    return resp.text.strip()


# ── Anthropic ─────────────────────────────────────────────────────────────────
def _load_anthropic():
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        return {"provider": "anthropic", "client": client}
    except ImportError:
        raise ImportError("pip install anthropic")


def _gen_anthropic(gen_obj, system_prompt: str, user_prompt: str,
                   temperature: float, max_tokens: int) -> str:
    client = gen_obj["client"]
    resp = client.messages.create(
        model=config.CLAUDE_MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return resp.content[0].text.strip()


# ── Ollama ────────────────────────────────────────────────────────────────────
def _load_ollama():
    """Ollama uses the OpenAI-compatible /v1 endpoint."""
    try:
        from openai import OpenAI
        client = OpenAI(
            api_key="ollama",   # any non-empty string
            base_url=f"{config.OLLAMA_BASE_URL}/v1",
        )
        return {"provider": "ollama", "client": client}
    except ImportError:
        raise ImportError("pip install openai  (used for Ollama compat endpoint)")


def _load_groq():
    """Load Groq client for the Groq provider."""
    try:
        from groq import Groq
        client = Groq(api_key=config.GROQ_API_KEY)
        return {"provider": "groq", "client": client}
    except ImportError:
        raise ImportError("pip install groq")


def _gen_ollama(gen_obj, system_prompt: str, user_prompt: str,
                temperature: float, max_tokens: int) -> str:
    client = gen_obj["client"]
    resp = client.chat.completions.create(
        model=config.OLLAMA_MODEL,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
    )
    return resp.choices[0].message.content.strip()


def _gen_groq(gen_obj, system_prompt: str, user_prompt: str,
                temperature: float, max_tokens: int) -> str:
    client = gen_obj["client"]
    resp = client.chat.completions.create(
        model=config.GROQ_MODEL,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
    )
    return resp.choices[0].message.content.strip()


def _gen_grounded(gen_obj, system_prompt: str, user_prompt: str,
                  temperature: float, max_tokens: int) -> str:
    """
    Grounded synthesizer for fast prototype execution without external API keys.
    Extracts relevant facts directly from retrieved context and attaches citations.
    Strictly refuses if context does not match question.
    """
    # Extract QUESTION
    q_match = re.search(r'QUESTION:\s*(.*?)(?:\n\nANSWER|$)', user_prompt, re.DOTALL)
    question = q_match.group(1).strip() if q_match else ""

    # Extract [Si] sources
    source_blocks = re.findall(r'\[S(\d+)\]\s*\(([^)]*)\)\s*(.*?)(?=(?:\[S\d+\])|$)', user_prompt, re.DOTALL)
    if not source_blocks:
        return "I don't have this information in the SPIT knowledge base."

    # Stopwords
    stopwords = {"what", "is", "the", "at", "of", "and", "in", "to", "for", "are", "spit", "how", "much", "many", "does", "do", "a", "an", "on", "with", "tell", "me", "about"}
    q_words = [w.lower().strip("?,.!") for w in question.split() if w.lower().strip("?,.!") not in stopwords and len(w) > 2]

    # Find matching sentences
    scored_sentences = []
    for s_idx, meta, text in source_blocks:
        # Split text into sentences
        raw_sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n+', text) if len(s.strip()) > 15]
        for sent in raw_sents:
            sent_lower = sent.lower()
            matches = sum(1 for w in q_words if w in sent_lower)
            if matches > 0:
                scored_sentences.append((matches, s_idx, sent))

    if not scored_sentences:
        return "I don't have this information in the SPIT knowledge base."

    scored_sentences.sort(key=lambda x: x[0], reverse=True)
    # Take top unique sentences
    seen = set()
    selected = []
    for score, s_idx, sent in scored_sentences:
        clean = sent.strip()
        if clean not in seen and len(selected) < 4:
            seen.add(clean)
            selected.append(f"{clean} [S{s_idx}]")

    if not selected:
        return "I don't have this information in the SPIT knowledge base."

    return "Based on the SPIT knowledge base:\n\n" + "\n\n".join(selected)

_DISPATCH = {
    "openai":    _gen_openai,
    "gemini":    _gen_gemini,
    "anthropic": _gen_anthropic,
    "ollama":    _gen_ollama,
    "groq":      _gen_groq,
    "grounded":  _gen_grounded,
}


def generate_answer(
    prompt_tuple: tuple[str, str],   # (system_prompt, user_prompt)
    gen_obj: dict,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> tuple[str, list[str]]:
    """
    Call the LLM and return (answer_text, [cited_chunk_ids]).

    cited_chunk_ids are parsed from [S1], [S2] … tags in the answer.
    """
    system_prompt, user_prompt = prompt_tuple
    temperature = temperature if temperature is not None else config.LLM_TEMPERATURE
    max_tokens  = max_tokens  if max_tokens  is not None else config.LLM_MAX_TOKENS

    provider = gen_obj.get("provider", config.LLM_PROVIDER)
    fn = _DISPATCH.get(provider)
    if fn is None:
        raise ValueError(f"No generator for provider: {provider!r}")

    raw = fn(gen_obj, system_prompt, user_prompt, temperature, max_tokens)
    cited_ids = _extract_citations(raw)
    return raw, cited_ids


def _extract_citations(text: str) -> list[str]:
    """Return list of cited source numbers, e.g. ['1','2'] from [S1][S2]."""
    return re.findall(r'\[S(\d+)\]', text)


# ─────────────────────────────────────────────────────────────────────────────
# Judge (for evaluation, uses a DIFFERENT model than the generator)
# ─────────────────────────────────────────────────────────────────────────────

def call_judge(prompt: str) -> str:
    """
    Call the judge model (always different from the generator) for evaluation.
    Returns raw judge response text.
    """
    provider = config.LLM_PROVIDER
    judge_model_name = config.active_judge_model()

    if provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=config.OPENAI_API_KEY,
                        base_url=config.OPENAI_BASE_URL)
        resp = client.chat.completions.create(
            model=judge_model_name, temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content.strip()

    elif provider == "gemini":
        import google.generativeai as genai
        genai.configure(api_key=config.GOOGLE_API_KEY)
        m = genai.GenerativeModel(model_name=judge_model_name)
        return m.generate_content(prompt).text.strip()

    elif provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        resp = client.messages.create(
            model=judge_model_name, max_tokens=256, temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()

    elif provider == "ollama":
        from openai import OpenAI
        client = OpenAI(api_key="ollama",
                        base_url=f"{config.OLLAMA_BASE_URL}/v1")
        resp = client.chat.completions.create(
            model=judge_model_name, temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content.strip()

    elif provider == "groq":
        from groq import Groq
        client = Groq(api_key=config.GROQ_API_KEY)
        resp = client.chat.completions.create(
            model=judge_model_name,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content.strip()

    raise ValueError(f"Unknown provider for judge: {provider!r}")
