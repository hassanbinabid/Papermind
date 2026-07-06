---
title: PaperMind RAG
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# PaperMind RAG

Production-grade RAG system using hybrid retrieval (BM25 + vector search), cross-encoder re-ranking, and citation enforcement.

## Features
- Hybrid retrieval (BM25 + Pinecone vector search)
- Cross-encoder re-ranking
- Citation enforcement
- FastAPI REST API
- Evaluated with LLM-as-judge

## API Endpoints
- GET /health
- POST /query
- GET /documents
- DELETE /documents/{source}
