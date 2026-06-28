"""
generate_golden_dataset.py — Auto-generate Q&A pairs from the AI Index Report.
Run this ONCE to create eval/golden_dataset.json.

Usage: python generate_golden_dataset.py
"""

import json
import os
import random
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
GENERATION_MODEL    = "openrouter/auto"
OUTPUT_PATH         = "eval/golden_dataset.json"

UNANSWERABLE = [
    {"question": "What is the capital of France?",                    "answer": "I cannot answer this from the provided documents.", "category": "unanswerable"},
    {"question": "Who won the FIFA World Cup in 2022?",               "answer": "I cannot answer this from the provided documents.", "category": "unanswerable"},
    {"question": "What is the recipe for chocolate cake?",            "answer": "I cannot answer this from the provided documents.", "category": "unanswerable"},
    {"question": "How many planets are in the solar system?",         "answer": "I cannot answer this from the provided documents.", "category": "unanswerable"},
    {"question": "What is the boiling point of water?",               "answer": "I cannot answer this from the provided documents.", "category": "unanswerable"},
    {"question": "Who wrote Romeo and Juliet?",                       "answer": "I cannot answer this from the provided documents.", "category": "unanswerable"},
    {"question": "What is the population of Tokyo?",                  "answer": "I cannot answer this from the provided documents.", "category": "unanswerable"},
    {"question": "How do you make a cup of tea?",                     "answer": "I cannot answer this from the provided documents.", "category": "unanswerable"},
    {"question": "What programming language was Python named after?", "answer": "I cannot answer this from the provided documents.", "category": "unanswerable"},
    {"question": "What is the speed of light?",                       "answer": "I cannot answer this from the provided documents.", "category": "unanswerable"},
]


def get_client():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in .env file.")
    return OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)


def load_chunks():
    from app.vectorstore import _get_collection
    collection = _get_collection()
    results    = collection.get(include=["documents", "metadatas"])
    chunks = []
    for text, meta in zip(results["documents"], results["metadatas"]):
        chunks.append({"text": text, "source": meta["source"], "page": meta["page"]})
    print(f"  Loaded {len(chunks)} chunks from ChromaDB.")
    return chunks


def generate_qa_from_chunk(client, chunk: dict, category: str) -> dict | None:
    if category == "factual":
        instruction = (
            "Generate ONE specific factual question that can be answered using ONLY "
            "the text below. The question should ask about a specific fact, number, "
            "statistic, or finding. Then provide the exact answer from the text.\n\n"
        )
    else:
        instruction = (
            "Generate ONE conceptual question about the topic covered in the text below. "
            "The question should ask about a concept, trend, or insight. "
            "Then provide a clear answer based only on the text.\n\n"
        )

    prompt = (
        f"{instruction}"
        f"Text:\n{chunk['text'][:1000]}\n\n"
        f"Respond in this exact JSON format only, no other text:\n"
        f'{{"question": "...", "answer": "..."}}'
    )

    try:
        response = client.chat.completions.create(
            model=GENERATION_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.choices[0].message.content.strip()

        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        qa = json.loads(raw)
        if "question" not in qa or "answer" not in qa:
            return None

        return {
            "question": qa["question"].strip(),
            "answer":   qa["answer"].strip(),
            "source":   chunk["source"],
            "page":     chunk["page"],
            "category": category,
        }
    except Exception as e:
        print(f"    ⚠ Skipped chunk (page {chunk['page']}): {e}")
        return None


def generate_dataset():
    print("=" * 60)
    print("GENERATING GOLDEN DATASET")
    print("=" * 60)

    client = get_client()
    chunks = load_chunks()

    random.seed(42)
    sampled           = random.sample(chunks, min(60, len(chunks)))
    factual_chunks    = sampled[:30]
    conceptual_chunks = sampled[30:50]

    dataset = []

    print(f"\n[1/3] Generating factual questions (target: 25)...")
    factual_count = 0
    for i, chunk in enumerate(factual_chunks):
        if factual_count >= 25:
            break
        print(f"  Chunk {i+1}/{len(factual_chunks)} (page {chunk['page']})...", end=" ")
        qa = generate_qa_from_chunk(client, chunk, "factual")
        if qa:
            dataset.append(qa)
            factual_count += 1
            print("✅")
        else:
            print("⚠ skipped")

    print(f"\n[2/3] Generating conceptual questions (target: 15)...")
    conceptual_count = 0
    for i, chunk in enumerate(conceptual_chunks):
        if conceptual_count >= 15:
            break
        print(f"  Chunk {i+1}/{len(conceptual_chunks)} (page {chunk['page']})...", end=" ")
        qa = generate_qa_from_chunk(client, chunk, "conceptual")
        if qa:
            dataset.append(qa)
            conceptual_count += 1
            print("✅")
        else:
            print("⚠ skipped")

    print(f"\n[3/3] Adding 10 unanswerable questions...")
    dataset.extend(UNANSWERABLE)

    os.makedirs("eval", exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=True)

    print(f"\n{'=' * 60}")
    print(f"✅ Golden dataset saved: {OUTPUT_PATH}")
    print(f"   Total pairs:    {len(dataset)}")
    print(f"   Factual:        {factual_count}")
    print(f"   Conceptual:     {conceptual_count}")
    print(f"   Unanswerable:   {len(UNANSWERABLE)}")
    print(f"{'=' * 60}")
    print(f"\n⚠  IMPORTANT: Review eval/golden_dataset.json and fix any wrong answers.")


if __name__ == "__main__":
    generate_dataset()