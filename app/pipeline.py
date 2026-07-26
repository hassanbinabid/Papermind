"""
pipeline.py — Phase 2 RAG pipeline.
Hybrid retrieval (BM25 + vector) → RRF fusion → cross-encoder re-ranking → generation.
"""

import os
import yaml
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APIError
from langfuse import observe
from app.observability import langfuse
from app.hybrid_retriever import hybrid_retrieve
from app.reranker import rerank

load_dotenv()

GENERATION_MODEL  = "llama-3.3-70b-versatile"
GROQ_BASE_URL     = "https://api.groq.com/openai/v1"
PROMPTS_PATH      = "prompts/prompts.yaml"
RERANK_TOP_K      = 5
HYBRID_TOP_K      = 10
MAX_OUTPUT_TOKENS = 256  # cap response length so it fits low/free-tier token budgets

# Phases that count as a fully successful pipeline run for the failure-rate metric.
SUCCESS_PHASES = {"phase2_hybrid_rerank"}


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
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in .env file.")
    return OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)


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


def _score_and_return(result: dict, grounded: bool | None = None) -> dict:
    """
    Log failure-rate and (when known) citation-coverage scores to the
    current trace, then return the result dict unchanged.

    Centralizing this in one place ensures every exit path from
    run_rag_pipeline — not just the happy path — is reflected in the
    failure-rate metric. Citation grounding is only meaningful once an
    answer has actually been generated, so `grounded` is left as None
    on earlier exit paths (e.g. empty retrieval, API errors) and no
    citation_grounded score is logged for those.
    """
    phase = result.get("phase")

    langfuse.score_current_trace(
        name="pipeline_success",
        value=1 if phase in SUCCESS_PHASES else 0,
        data_type="BOOLEAN",
        comment=phase,
    )

    if grounded is not None:
        langfuse.score_current_trace(
            name="citation_grounded",
            value=1 if grounded else 0,
            data_type="BOOLEAN",
        )

    return result


@observe(name="rag_pipeline")
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
        return _score_and_return({
            "answer":  "I cannot answer this from the provided documents.",
            "sources": [],
            "phase":   "hybrid_retrieval_empty",
        })

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

    print(f"\n[Generator] Calling Groq ({GENERATION_MODEL})...")
    client = _get_client()

    with langfuse.start_as_current_observation(
        name="groq_generation",
        as_type="generation",
        model=GENERATION_MODEL,
        input=final_prompt,
    ) as generation:
        try:
            response = client.chat.completions.create(
                model=GENERATION_MODEL,
                messages=[{"role": "user", "content": final_prompt}],
                max_tokens=MAX_OUTPUT_TOKENS,
            )
        except RateLimitError as e:
            print(f"  ⚠ Groq rate limit hit: {e}")
            generation.update(level="ERROR", status_message=str(e))
            return _score_and_return({
                "answer": (
                    "⚠ Rate limit reached on the free tier. "
                    "Please wait a minute and try again."
                ),
                "sources": [],
                "phase": "generation_rate_limited",
            })
        except APIError as e:
            print(f"  ⚠ Groq API error: {e}")
            generation.update(level="ERROR", status_message=str(e))
            return _score_and_return({
                "answer": f"⚠ Groq API error — real error: {e}",
                "sources": [],
                "phase": "generation_api_error",
            })
        except Exception as e:
            print(f"  ⚠ Unexpected error calling Groq: {e}")
            generation.update(level="ERROR", status_message=str(e))
            return _score_and_return({
                "answer": f"⚠ Unexpected error — real error: {e}",
                "sources": [],
                "phase": "generation_unexpected_error",
            })

        choice        = response.choices[0]
        raw_content   = choice.message.content
        finish_reason = getattr(choice, "finish_reason", None)

        usage = getattr(response, "usage", None)
        usage_details = None
        if usage:
            usage_details = {
                "input":  getattr(usage, "prompt_tokens", None),
                "output": getattr(usage, "completion_tokens", None),
                "total":  getattr(usage, "total_tokens", None),
            }

        generation.update(
            output=raw_content,
            usage_details=usage_details,
            metadata={"finish_reason": finish_reason},
        )

        if not raw_content:
            print(f"  ⚠ Empty response from model. finish_reason={finish_reason}")
            print(f"  ⚠ Raw response: {response}")
            return _score_and_return({
                "answer": f"⚠ Model returned empty content. finish_reason={finish_reason}",
                "sources": [],
                "phase": "generation_empty_response",
            })

        answer = raw_content.strip()

    # ── Step 4: Citation Enforcement ─────────────────────────────────────────
    grounded = _enforce_citation(answer)
    if not grounded:
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

    return _score_and_return({
        "answer":  answer,
        "sources": sources,
        "phase":   "phase2_hybrid_rerank",
    }, grounded=grounded)