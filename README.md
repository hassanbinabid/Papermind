# PaperMind RAG

Production-grade retrieval-augmented generation (RAG) system for question-answering over PDF and Markdown documents. PaperMind combines hybrid retrieval (keyword + semantic), cross-encoder re-ranking, and strict citation enforcement so answers are always grounded in the source documents rather than model guesswork.

Built on top of the AI Index Report 2026 and other research papers, PaperMind exposes a FastAPI REST API (and a lightweight Streamlit UI) for uploading documents and asking questions against them, with every answer traceable back to a specific file and page.

## Features

- **Hybrid retrieval** — combines BM25 keyword search with Pinecone vector search, fused via Reciprocal Rank Fusion (RRF)
- **Cross-encoder re-ranking** — rescores the fused candidates by reading the query and each chunk together for higher precision
- **Citation enforcement** — every answer must cite `[Source: filename, page N]` for each claim, or the pipeline falls back to a refusal rather than an ungrounded answer
- **FastAPI REST API** — with interactive Swagger docs at `/docs`
- **Automated evaluation** — a golden dataset of 50 question/answer pairs scored for pass rate and faithfulness, wired into CI

## Architecture

The pipeline runs in two phases: an **ingestion** phase that chunks and indexes documents, and a **query** phase that retrieves, re-ranks, and generates a grounded answer.

```
Document ingestion (PDF / Markdown, chunked and embedded)
        │
        ▼
Pinecone + BM25 index (vector store and keyword index)
        │
        ▼
Hybrid retrieval (RRF fusion of top 10 from each)
        │
        ▼
Cross-encoder reranker (rescores fused results to top 5)
        │
        ▼
LLM generation (OpenRouter, citation enforced)
        │
        ▼
FastAPI response (answer with source citations)
```

**Ingestion**
1. PDFs are parsed page-by-page (`pypdf`) and Markdown files are read directly; text is split into ~600-token chunks with 100-token overlap.
2. Chunks are embedded locally with `sentence-transformers` (`all-MiniLM-L6-v2`, 384-dim, no API key required).
3. Embeddings are upserted into a Pinecone serverless index; the same chunks are tokenized and written to a local BM25 index (`rank-bm25`, persisted as JSON).

**Query**
1. **Hybrid retrieval** — the query is embedded and matched against Pinecone (semantic top 10) while BM25 independently scores keyword matches (top 10). Reciprocal Rank Fusion (`k=60`) merges both ranked lists into one.
2. **Re-ranking** — a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) reads `(query, chunk)` pairs directly and rescores the top 10 fused candidates down to the 5 most relevant.
3. **Generation** — the top 5 chunks are formatted into a strict prompt (`prompts/prompts.yaml`) and sent to an LLM via OpenRouter (`openrouter/auto`).
4. **Citation enforcement** — the raw answer is checked for `[Source: ...]` citations or an explicit refusal phrase; if neither is present, the pipeline overrides the answer with a "cannot answer from the provided documents" response rather than returning an ungrounded claim.

## Tech stack

| Layer | Technology | Purpose |
|---|---|---|
| API framework | FastAPI + Uvicorn | REST API, request validation, auto-generated `/docs` |
| UI | Streamlit | Lightweight chat interface for manual testing |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) | Local, no-API-key text embeddings (384-dim) |
| Vector store | Pinecone (serverless, AWS `us-east-1`) | Persistent semantic search, survives restarts |
| Keyword search | rank-bm25 (BM25Okapi) | Exact-term and keyword matching, complements vector search |
| Fusion | Reciprocal Rank Fusion (custom) | Merges vector + BM25 rankings into one list |
| Re-ranking | sentence-transformers CrossEncoder (`ms-marco-MiniLM-L-6-v2`) | Precision rescoring of fused candidates |
| Generation | OpenAI SDK → OpenRouter (`openrouter/auto`) | LLM answer generation with citation prompting |
| Document parsing | pypdf | PDF text extraction, page-level chunking |
| Config | python-dotenv, PyYAML | Environment variables and externalized prompts |
| Deployment | Docker, Hugging Face Spaces, Render (Procfile) | Containerized deployment targets |
| CI / evaluation | GitHub Actions (`eval.yml`) | Runs the golden-dataset evaluation on push |

## How to run locally

### Prerequisites

- Python 3.11+
- An [OpenRouter](https://openrouter.ai/keys) API key
- A [Pinecone](https://pinecone.io) API key

### 1. Clone and install

```bash
git clone <your-repo-url>
cd Papermind
pip install -r requirements.txt
```

### 2. Configure environment variables

Create a `.env` file in the project root:

```env
OPENROUTER_API_KEY=your_openrouter_key_here
PINECONE_API_KEY=your_pinecone_key_here
```

### 3. Ingest documents

Place PDF or Markdown files in `data/docs/`, then run:

```bash
python main.py
```

This chunks and embeds the documents, upserts them into Pinecone, builds the local BM25 index, and runs a sample query to confirm everything works end to end.

### 4. Start the API

```bash
uvicorn app.api:app --reload --port 8000
```

- Interactive docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

### 5. (Optional) Start the Streamlit UI

```bash
streamlit run streamlit_app.py
```

### Running with Docker

```bash
docker-compose up --build
```

This starts the API on port 8000 and mounts a volume so the BM25 index persists across container restarts (Pinecone handles vector persistence in the cloud). To run ingestion as a one-off task instead:

```bash
docker-compose run --rm ingest
```

## API endpoints

Base URL when running locally: `http://localhost:8000`

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Reports overall system status, Pinecone connectivity, LLM key presence, and total indexed chunks |
| `POST` | `/query` | Core RAG endpoint — accepts `{"question": str, "top_k": int}`, runs hybrid retrieval → re-ranking → generation → citation check, returns the answer and its sources |
| `POST` | `/ingest` | Upload a `.pdf` or `.md` file (multipart form) — chunks, embeds, and stores it, then rebuilds the BM25 index |
| `GET` | `/documents` | Lists all documents currently indexed in Pinecone with their chunk counts |
| `DELETE` | `/documents/{source}` | Deletes all chunks belonging to a given source filename from Pinecone |

### Example: `POST /query`

Request:
```json
{
  "question": "What are the key AI trends highlighted in this report?",
  "top_k": 5
}
```

Response:
```json
{
  "question": "What are the key AI trends highlighted in this report?",
  "answer": "... [Source: ai_index_report_2026.pdf, page 12] ...",
  "sources": [
    { "source": "ai_index_report_2026.pdf", "page": 12 }
  ],
  "pipeline": "phase2_hybrid_rerank"
}
```


## Evaluation results

PaperMind is evaluated against a golden dataset of 50 question/answer pairs (25 factual, 15 conceptual, 10 intentionally unanswerable) using an LLM-as-judge faithfulness score, wired into CI via `.github/workflows/eval.yml`.

| Metric | Result |
|---|---|
| Pass rate | **74%** (37/50) |
| Average faithfulness | **0.65** |
| Faithfulness threshold | ≥ 0.45 |
| Judge model | `openrouter/auto` |
| Overall CI status | ✅ Pass |

Faithfulness is scored on a 0.0–1.0 scale by fact-checking each generated answer against its retrieved context; refusals on unanswerable questions are automatically scored 1.0 since declining to answer is the correct behavior for that category.

Run the evaluation yourself:

```bash
python eval/evaluate.py
```

Optional flags: `--category factual|conceptual|unanswerable` to filter, `--sample N` to evaluate a random subset.
