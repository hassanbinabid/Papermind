"""
main.py — Entry point for PaperMind RAG (Phase 2).
Ingests documents → builds vector + BM25 index → runs a test query.
"""

import os
import sys

# ── Dependency check ──────────────────────────────────────────────────────────
missing = []
try:
    from google import genai
except ImportError:
    pass  # google-genai is optional now
try:
    import sentence_transformers
except ImportError:
    missing.append("sentence-transformers")
try:
    import openai
except ImportError:
    missing.append("openai")
try:
    import pinecone
except ImportError:
    missing.append("pinecone")
try:
    import pypdf
except ImportError:
    missing.append("pypdf")
try:
    import dotenv
except ImportError:
    missing.append("python-dotenv")
try:
    import yaml
except ImportError:
    missing.append("pyyaml")
try:
    import rank_bm25
except ImportError:
    missing.append("rank-bm25")

if missing:
    print("❌ Missing dependencies. Run:")
    print(f"   pip install {' '.join(missing)}")
    sys.exit(1)

# ── Environment check ─────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

if not os.getenv("OPENROUTER_API_KEY"):
    print("❌ OPENROUTER_API_KEY is missing or empty in your .env file.")
    print("   Get one at openrouter.ai/keys")
    sys.exit(1)

if not os.getenv("PINECONE_API_KEY"):
    print("❌ PINECONE_API_KEY is missing or empty in your .env file.")
    print("   Get one at pinecone.io")
    sys.exit(1)

# ── Imports ───────────────────────────────────────────────────────────────────
from app.ingest import load_documents, load_pdf
from app.embeddings import embed_texts
from app.vectorstore import add_documents, collection_count
from app.bm25_retriever import build_bm25_index
from app.pipeline import run_rag_pipeline

DOCS_DIR  = r"C:\Users\hassa\OneDrive\Desktop"
PDF_FILE  = "ai_index_report_2026 (1).pdf"
DOCS_PATH = os.path.join(DOCS_DIR, PDF_FILE)

TEST_QUESTION = "What are the key AI trends highlighted in this report?"


def ingest():
    print("=" * 60)
    print("PHASE 2 — INGESTION (Vector + BM25)")
    print("=" * 60)

    local_docs  = "data/docs"
    local_files = [
        f for f in os.listdir(local_docs)
        if f.endswith(".pdf") or f.endswith(".md")
    ] if os.path.isdir(local_docs) else []

    if local_files:
        print(f"\nLoading documents from: {local_docs}/")
        chunks = load_documents(local_docs)
    elif os.path.exists(DOCS_PATH):
        print(f"\nLoading document from: {DOCS_PATH}")
        chunks = load_pdf(DOCS_PATH)
        print(f"  → {len(chunks)} chunks from {PDF_FILE}")
    else:
        print(f"\n❌ Could not find documents.")
        print(f"   Put your PDF in: data/docs/")
        sys.exit(1)

    print(f"\n✅ Total chunks created: {len(chunks)}")

    # ── Vector embeddings ────────────────────────────────────────────────────
    print("\n[Vector Store] Generating embeddings...")
    texts = [c["text"] for c in chunks]
    embeddings, valid_indices = embed_texts(texts)

    # Keep only chunks that were successfully embedded (skip bad/empty ones)
    valid_chunks = [chunks[i] for i in valid_indices]

    stored = add_documents(valid_chunks, embeddings)
    print(f"✅ Chunks stored in Pinecone: {stored}")
    print(f"✅ Total in collection: {collection_count()}")

    # ── BM25 index ───────────────────────────────────────────────────────────
    print("\n[BM25 Index] Building keyword search index...")
    build_bm25_index(valid_chunks)


def query():
    print("\n" + "=" * 60)
    print("PHASE 2 — QUERY (Hybrid + Re-ranked)")
    print("=" * 60)
    print(f"\nQuestion: {TEST_QUESTION}\n")

    result = run_rag_pipeline(TEST_QUESTION)

    print("\n" + "─" * 60)
    print("ANSWER:")
    print("─" * 60)
    print(result["answer"])

    print("\n" + "─" * 60)
    print("SOURCES USED:")
    print("─" * 60)
    if result["sources"]:
        for s in result["sources"]:
            print(f"  • {s['source']}, page {s['page']}")
    else:
        print("  No sources cited.")

    print(f"\n  Pipeline: {result.get('phase', 'unknown')}")


if __name__ == "__main__":
    ingest()
    query()
