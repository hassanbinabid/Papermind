"""
routers/health.py — System health check.
"""

from fastapi import APIRouter, Depends

from app.config import Settings, get_settings
from app.dependencies import get_vectorstore_module
from app.models import HealthResponse

router = APIRouter(tags=["System"])


@router.get("/health", response_model=HealthResponse)
async def health(
    settings: Settings = Depends(get_settings),
    vectorstore=Depends(get_vectorstore_module),
):
    """
    Verifies Pinecone is reachable and the LLM API key is configured.
    Used by deployment systems / uptime checks to confirm the server is live.
    """
    try:
        count = vectorstore.collection_count()
        db_status = "ok"
    except Exception as e:
        count = 0
        db_status = f"error: {str(e)}"

    llm_status = "ok" if settings.groq_api_key else "missing API key"
    overall = "ok" if db_status == "ok" and llm_status == "ok" else "degraded"

    return HealthResponse(
        status=overall,
        pinecone=db_status,
        llm=llm_status,
        total_chunks=count,
        message=f"PaperMind RAG is running. {count} chunks indexed across all documents.",
    )
