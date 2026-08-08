"""
routers/query.py — Core RAG query endpoint.
"""

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.dependencies import get_pipeline_fn
from app.models import QueryRequest, QueryResponse, Source
from app.rate_limit import limiter

router = APIRouter(tags=["RAG"])


@router.post("/query", response_model=QueryResponse)
@limiter.limit("60/minute")
async def query(
    request: Request,  # required by slowapi's limiter decorator
    req: QueryRequest,
    run_rag_pipeline=Depends(get_pipeline_fn),
):
    """
    Hybrid retrieval -> re-ranking -> generation -> citation enforcement.
    Returns the answer with source citations.
    """
    if not req.question.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Question cannot be empty.",
        )

    try:
        result = run_rag_pipeline(req.question)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline error: {str(e)}",
        )

    return QueryResponse(
        question=req.question,
        answer=result["answer"],
        sources=[Source(**s) for s in result["sources"]],
        pipeline=result.get("phase", "phase2_hybrid_rerank"),
    )
