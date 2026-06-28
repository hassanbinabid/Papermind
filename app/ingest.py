"""
ingest.py — Load and chunk PDF and Markdown documents.
"""

import os
from pathlib import Path
from pypdf import PdfReader


# 1 token ≈ 4 characters
CHARS_PER_TOKEN = 4
TARGET_CHUNK_TOKENS = 600          # aim for middle of 500–800 range
TARGET_CHUNK_CHARS = TARGET_CHUNK_TOKENS * CHARS_PER_TOKEN   # 2400 chars
OVERLAP_TOKENS = 100
OVERLAP_CHARS = OVERLAP_TOKENS * CHARS_PER_TOKEN             # 400 chars


def _chunk_text(text: str, source: str, page: int) -> list[dict]:
    """Split a block of text into overlapping chunks with metadata."""
    chunks = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = start + TARGET_CHUNK_CHARS
        chunk_text = text[start:end].strip()

        if chunk_text:
            chunks.append({
                "text": chunk_text,
                "source": source,
                "page": page,
                "chunk_index": chunk_index,
            })
            chunk_index += 1

        # Move forward by chunk size minus overlap
        start += TARGET_CHUNK_CHARS - OVERLAP_CHARS

    return chunks


def load_pdf(file_path: str) -> list[dict]:
    """Load a PDF and return chunks with page-level metadata."""
    chunks = []
    filename = Path(file_path).name
    reader = PdfReader(file_path)

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text and text.strip():
            page_chunks = _chunk_text(text, source=filename, page=page_num)
            chunks.extend(page_chunks)

    return chunks


def load_markdown(file_path: str) -> list[dict]:
    """Load a markdown file and return chunks."""
    filename = Path(file_path).name
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    return _chunk_text(text, source=filename, page=1)


def load_documents(docs_dir: str) -> list[dict]:
    """Load all PDFs and markdown files from a directory."""
    docs_path = Path(docs_dir)
    all_chunks = []

    pdf_files = list(docs_path.glob("*.pdf"))
    md_files  = list(docs_path.glob("*.md"))
    all_files = pdf_files + md_files

    if not all_files:
        raise FileNotFoundError(f"No PDF or markdown files found in: {docs_dir}")

    for file_path in all_files:
        print(f"  Loading: {file_path.name}")
        if file_path.suffix.lower() == ".pdf":
            chunks = load_pdf(str(file_path))
        else:
            chunks = load_markdown(str(file_path))
        print(f"    → {len(chunks)} chunks")
        all_chunks.extend(chunks)

    return all_chunks
