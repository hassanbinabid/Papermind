"""
chat.py — Terminal chat for PaperMind RAG (Phase 2).
Run with: python chat.py
"""

import sys
from dotenv import load_dotenv
load_dotenv()

from app.pipeline import run_rag_pipeline

print("=" * 60)
print("PaperMind RAG — Phase 2 Terminal Chat")
print("Hybrid Retrieval + Cross-Encoder Re-ranking")
print("Type your question and press Enter. Type 'exit' to quit.")
print("=" * 60)

while True:
    try:
        print()
        question = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nGoodbye!")
        sys.exit(0)

    if not question:
        continue

    if question.lower() in ["exit", "quit", "q"]:
        print("Goodbye!")
        break

    result = run_rag_pipeline(question)

    print()
    print("Answer:")
    print("-" * 40)
    print(result["answer"])

    if result["sources"]:
        print()
        print("Sources:")
        for s in result["sources"]:
            print(f"  • {s['source']} — page {s['page']}")
    print("-" * 40)
