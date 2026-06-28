"""
pipeline.py — Phase 2 RAG pipeline.
Hybrid retrieval (BM25 + vector) → RRF fusion → cross-encoder re-ranking → generation.
"""

import os
import yaml
from dotenv import load_dotenv
from openai import OpenAI
from app.hybrid_retriever import hybrid_retrieve
from app.reranker import rerank

load_dotenv()

GENERATION_MODEL    = "openrouter/auto"
PROMPTS_PATH        = "prompts/prompts.yaml"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
RERANK_TOP_K        = 5
HYBRID_TOP_K        = 10


def _load_prompts() -> dict:
    with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _format_chunks(chunks: list[dict]) -> str:
    formatted = []
    for i, chunk in enumerate(chunks, start=1):
        formatted.append(
            f"[Chunk {i}]\n"
            f"[Source: {chunk['source']}, page {chunk['page']}]\n"
            f"{chunk['text']}"
        )
    return "\n\n---\n\n".join(formatted)


def _get_client() -> OpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in .env file.")
    return OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)


def _enforce_citation(answer: str) -> bool:
    """
    Basic citation enforcement check.
    Returns True if the answer appears grounded (contains citations or a refusal).
    """
    refusal_phrases = [
        "i cannot answer",
        "not found in the provided",
        "the documents do not",
        "no information",
    ]
    answer_lower = answer.lower()

    if any(phrase in answer_lower for phrase in refusal_phrases):
        return True

    if "[source:" in answer_lower:
        return True

    return False


def run_rag_pipeline(question: str) -> dict:
    """
    Phase 2 RAG pipeline:
    1. Hybrid retrieval  — BM25 + vector search → RRF fusion (top 10)
    2. Re-ranking        — cross-encoder rescores to top 5
    3. Generation        — LLM answers with strict citation prompt
    4. Citation check    — enforce grounded output
    """
    prompts = _load_prompts()

    # ── Step 1: Hybrid Retrieval ─────────────────────────────────────────────
    print("\n[Hybrid Retriever]")
    candidates = hybrid_retrieve(question, top_k=HYBRID_TOP_K)

    if not candidates:
        return {
            "answer":  "I cannot answer this from the provided documents.",
            "sources": [],
            "phase":   "hybrid_retrieval_empty",
        }

    # ── Step 2: Re-ranking ───────────────────────────────────────────────────
    print(f"\n[Re-ranker] Scoring {len(candidates)} candidates...")
    chunks = rerank(question, candidates, top_k=RERANK_TOP_K)
    print(f"  ✅ Top {len(chunks)} chunks selected after re-ranking.")

    # ── Step 3: Generation ───────────────────────────────────────────────────
    context      = _format_chunks(chunks)
    final_prompt = prompts["rag_prompt"].format(
        chunks=context,
        question=question,
    )

    print(f"\n[Generator] Calling OpenRouter ({GENERATION_MODEL})...")
    client   = _get_client()
    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[{"role": "user", "content": final_prompt}],
    )
    answer = response.choices[0].message.content.strip()

    # ── Step 4: Citation Enforcement ─────────────────────────────────────────
    if not _enforce_citation(answer):
        print("  ⚠ Citation enforcement triggered — answer lacks citations.")
        answer = (
            "I cannot answer this from the provided documents. "
            "(The generated answer did not cite its sources.)"
        )

    # ── Collect unique sources ────────────────────────────────────────────────
    seen    = set()
    sources = []
    for chunk in chunks:
        key = (chunk["source"], chunk["page"])
        if key not in seen:
            seen.add(key)
            sources.append({"source": chunk["source"], "page": chunk["page"]})

    return {
        "answer":  answer,
        "sources": sources,
        "phase":   "phase2_hybrid_rerank",
    }