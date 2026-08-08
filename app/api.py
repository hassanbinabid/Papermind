"""
api.py — PaperMind FastAPI application (entrypoint).

Run with:
    uvicorn app.api:app --reload --port 8000

Then visit:
    http://localhost:8000/docs   ← interactive API docs
    http://localhost:8000/health ← system health check

This file only wires things together — route logic lives in
app/routers/*.py, settings in app/config.py, DI in app/dependencies.py.
Keeping assembly separate from route logic is what makes this
readable as the project grows, instead of one large file.
"""

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.config import get_settings
from app.middleware import RequestLoggingMiddleware
from app.rate_limit import limiter
from app.routers import documents, health, query

load_dotenv()
settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    description=(
        "Production-grade RAG system using hybrid retrieval (BM25 + vector), "
        "cross-encoder re-ranking, and citation enforcement. "
        "Built on the AI Index Report 2026 and research papers."
    ),
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── Rate limiting (slowapi) ─────────────────────────────────────────────────
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# ── CORS ─────────────────────────────────────────────────────────────────
# Was allow_origins=["*"] — now driven by settings.frontend_origin.
# Set FRONTEND_ORIGIN in your .env before deploying publicly.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request logging / timing ────────────────────────────────────────────────
app.add_middleware(RequestLoggingMiddleware)

# ── Routers ──────────────────────────────────────────────────────────────
app.include_router(health.router)
app.include_router(query.router)
app.include_router(documents.router)


# ── Startup event ─────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    """Pre-load models on startup so first request isn't slow."""
    print("[PaperMind API] Starting up...")
    try:
        from app.embeddings import embed_query
        from app.reranker import _reranker
        print("[PaperMind API] Models loaded successfully.")
    except Exception as e:
        print(f"[PaperMind API] Warning: Could not pre-load models: {e}")
