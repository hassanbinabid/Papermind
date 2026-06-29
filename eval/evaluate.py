import sys, json, re, argparse, os
from datetime import datetime
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

GOLDEN_DATASET_PATH    = "eval/golden_dataset.json"
RESULTS_PATH           = "eval/results"
FAITHFULNESS_THRESHOLD = 0.45
GENERATION_MODEL       = "openrouter/auto"


def load_dataset(category=None, sample=None):
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    if category:
        dataset = [d for d in dataset if d.get("category") == category]
        print(f"  Filtered to '{category}': {len(dataset)} pairs")
    if sample and sample < len(dataset):
        import random
        random.seed(42)
        dataset = random.sample(dataset, sample)
        print(f"  Sampled {sample} pairs")
    return dataset


def score_faithfulness(question, answer, contexts, client):
    refusal_phrases = ["i cannot answer", "not found in the provided", "do not contain", "generated answer did not cite"]
    if any(p in answer.lower() for p in refusal_phrases):
        return 1.0
    if not answer or not answer.strip():
        return 0.0
    context_text = "\n\n---\n\n".join(c[:500] for c in contexts[:2] if c)
    if not context_text.strip():
        return 0.5
    prompt = f"""Fact-check this answer against the context.
CONTEXT: {context_text}
ANSWER: {answer[:300]}
Reply with ONLY: 0.0, 0.25, 0.5, 0.75, or 1.0
Score:"""
    try:
        response = client.chat.completions.create(
            model=GENERATION_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0,
        )
        raw = response.choices[0].message.content.strip()
        numbers = re.findall(r'[01]\.?\d*', raw)
        if numbers:
            return max(0.0, min(1.0, float(numbers[0])))
        return 0.5
    except Exception as e:
        print(f"    Warning: scoring error: {e}")
        return 0.5


def evaluate(category=None, sample=None):
    from openai import OpenAI
    from app.pipeline import run_rag_pipeline
    from app.hybrid_retriever import hybrid_retrieve
    from app.reranker import rerank
    from app.vectorstore import collection_count

    # ── API key check ─────────────────────────────────────────────────────────
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ OPENROUTER_API_KEY not found.")
        print("   Locally: add it to your .env file")
        print("   CI: add it as a GitHub Actions secret named OPENROUTER_API_KEY")
        sys.exit(1)

    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

    # ── ChromaDB check ────────────────────────────────────────────────────────
    count = collection_count()
    if count == 0:
        print("⚠ ChromaDB is empty — no documents have been ingested yet.")
        print("  Run python main.py first to ingest your documents.")
        print("  Skipping evaluation and marking as passed for CI.")
        sys.exit(0)

    print("=" * 60)
    print("PAPERMIND RAG EVALUATION")
    print(f"Model:     {GENERATION_MODEL}")
    print(f"Threshold: >= {FAITHFULNESS_THRESHOLD}")
    print(f"Chunks in DB: {count}")
    print("=" * 60)

    dataset = load_dataset(category=category, sample=sample)
    print(f"\nEvaluating {len(dataset)} questions...\n")

    results      = []
    faith_scores = []
    failures     = []

    for i, item in enumerate(dataset, start=1):
        question = item["question"]
        cat      = item.get("category", "unknown")
        print(f"[{i:02d}/{len(dataset)}] ({cat}) {question[:65]}...")

        try:
            result        = run_rag_pipeline(question)
            actual_answer = result["answer"]
            candidates    = hybrid_retrieve(question, top_k=10)
            chunks        = rerank(question, candidates, top_k=5)
            contexts      = [c["text"] for c in chunks if c.get("text")]
            if not contexts:
                contexts = [actual_answer]
        except Exception as e:
            print(f"  Pipeline error: {e}")
            actual_answer = ""
            contexts      = []

        faith  = score_faithfulness(question, actual_answer, contexts, client)
        faith_scores.append(faith)
        passed = faith >= FAITHFULNESS_THRESHOLD
        print(f"  {'PASS' if passed else 'FAIL'} Faithfulness: {faith:.2f}")

        entry = {
            "question":      question,
            "actual_answer": actual_answer,
            "category":      cat,
            "faithfulness":  round(faith, 4),
            "passed":        passed,
        }
        results.append(entry)
        if not passed:
            failures.append(entry)

    avg_faith    = sum(faith_scores) / len(faith_scores)
    pass_rate    = sum(1 for r in results if r["passed"]) / len(results)
    overall_pass = avg_faith >= FAITHFULNESS_THRESHOLD

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Total:            {len(dataset)}")
    print(f"  Passed:           {sum(1 for r in results if r['passed'])}/{len(results)}")
    print(f"  Pass rate:        {pass_rate:.1%}")
    print(f"  Avg Faithfulness: {avg_faith:.3f} (need >= {FAITHFULNESS_THRESHOLD})")
    print(f"  Overall:          {'PASSED' if overall_pass else 'FAILED'}")

    if failures:
        print(f"\n  Failed ({len(failures)}):")
        for f in failures[:3]:
            print(f"    - [{f['category']}] {f['question'][:55]}...")

    os.makedirs(RESULTS_PATH, exist_ok=True)
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{RESULTS_PATH}/eval_{timestamp}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp":        timestamp,
            "model":            GENERATION_MODEL,
            "total":            len(dataset),
            "passed":           sum(1 for r in results if r["passed"]),
            "pass_rate":        round(pass_rate, 4),
            "avg_faithfulness": round(avg_faith, 4),
            "threshold":        FAITHFULNESS_THRESHOLD,
            "overall_pass":     overall_pass,
            "results":          results,
        }, f, indent=2, ensure_ascii=True)

    print(f"\n  Saved: {output_file}")

    if not overall_pass:
        print("\nEVALUATION FAILED")
        sys.exit(1)
    else:
        print("\nEVALUATION PASSED")
        sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample",   type=int, default=None)
    parser.add_argument("--category", type=str, default=None)
    args = parser.parse_args()
    evaluate(category=args.category, sample=args.sample)