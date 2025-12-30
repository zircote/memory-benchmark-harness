#!/usr/bin/env python3
"""Quick validation script for SubcogAdapter with LoCoMo benchmark.

This script runs a small subset of LoCoMo questions against the subcog
memory system to validate the adapter is working correctly.

Usage:
    python scripts/validate_subcog.py [--questions N] [--search-limit N]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.adapters import SubcogAdapter
from src.benchmarks.locomo.dataset import load_locomo
from src.clients.openai_client import OpenAIClient
from src.evaluation.judge import JudgmentResult, LLMJudge

logger = logging.getLogger(__name__)


def run_validation(
    num_questions: int = 10,
    memory_search_limit: int = 20,
    verbose: bool = False,  # noqa: ARG001 - used for future enhancements
) -> dict:
    """Run validation against subcog adapter.

    Args:
        num_questions: Number of questions to test
        memory_search_limit: Number of memories to retrieve per search

    Returns:
        Dictionary with results
    """
    print("\n" + "=" * 60)
    print("SUBCOG ADAPTER VALIDATION")
    print("=" * 60)

    # Initialize adapter
    print("\n[1/5] Initializing SubcogAdapter...")
    try:
        adapter = SubcogAdapter()
        version = adapter.get_version()
        print(f"  ✓ SubcogAdapter initialized (version: {version})")

        # Skip clear() to test with existing memories
        # The clear() creates a new temp data dir which causes issues
        stats = adapter.get_stats()
        print(f"  ✓ Existing memories: {stats['memory_count']}")
    except Exception as e:
        print(f"  ✗ Failed to initialize: {e}")
        return {"error": str(e)}

    # Load dataset
    print("\n[2/5] Loading LoCoMo dataset...")
    try:
        dataset = load_locomo()
        total_conversations = len(dataset.conversations)
        print(f"  ✓ Loaded {total_conversations} conversations")

        # Take first conversation
        conv = dataset.conversations[0]
        print(f"  ✓ Using conversation: {conv.sample_id}")
        print(f"    Sessions: {len(conv.sessions)}, Questions: {len(conv.questions)}")
    except Exception as e:
        print(f"  ✗ Failed to load dataset: {e}")
        return {"error": str(e)}

    # Ingest memories
    print("\n[3/5] Ingesting session memories...")
    try:
        for i, session in enumerate(conv.sessions):
            # Format session as memory content
            turns_text = []
            for turn in session.turns:
                turns_text.append(f"{turn.speaker}: {turn.text}")
            content = f"Session {session.session_num} ({session.timestamp}):\n" + "\n".join(
                turns_text
            )

            result = adapter.add(
                content,
                metadata={
                    "namespace": "sessions",
                    "tags": ["locomo", f"session_{session.session_num}"],
                },
            )

            if result.success:
                print(f"  ✓ Ingested session {i + 1}/{len(conv.sessions)}: #{session.session_num}")
            else:
                print(f"  ✗ Failed session {i + 1}: {result.error}")

        stats = adapter.get_stats()
        print(f"  ✓ Total memories: {stats['memory_count']}")
    except Exception as e:
        print(f"  ✗ Failed during ingestion: {e}")
        import traceback

        traceback.print_exc()
        return {"error": str(e)}

    # Run questions
    print(f"\n[4/5] Running {num_questions} questions...")
    llm = OpenAIClient()
    judge = LLMJudge()

    questions = conv.questions[:num_questions]
    results = []

    for i, qa in enumerate(questions):
        print(f"\n  Q{i + 1}: {qa.question[:60]}...")

        # Search for relevant memories
        memories = adapter.search(qa.question, limit=memory_search_limit)
        print(f"    → Found {len(memories)} memories")

        if memories:
            # Build context from memories
            context = "\n\n".join([f"Memory {j + 1}:\n{m.content}" for j, m in enumerate(memories)])

            # Generate answer using LLM
            system_prompt = (
                "You are a helpful assistant answering questions based on conversation memories. "
                "Answer concisely based only on the memories provided. "
                "If the answer cannot be determined from the memories, say so."
            )
            user_message = f"""Based on the following conversation memories, answer the question.

Memories:
{context}

Question: {qa.question}

Answer:"""

            response = llm.complete(
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
            answer = response.content
        else:
            answer = "I don't have any relevant memories to answer this question."

        # Judge the answer
        judgment = judge.judge(
            question=qa.question,
            reference_answer=qa.answer,
            model_answer=answer,
        )

        is_correct = judgment.result == JudgmentResult.CORRECT
        results.append({
            "question": qa.question,
            "predicted": answer,
            "reference": qa.answer,
            "correct": is_correct,
            "score": judgment.score,
        })

        status = "✓" if is_correct else "✗"
        print(f"    {status} Score: {judgment.score:.2f}")
        expected_str = str(qa.answer)
        print(f"    Expected: {expected_str[:50]}...")
        print(f"    Got: {answer[:50]}...")

    # Summary
    print("\n" + "=" * 60)
    print("[5/5] SUMMARY")
    print("=" * 60)

    correct = sum(1 for r in results if r["correct"])
    total = len(results)
    accuracy = correct / total * 100 if total > 0 else 0

    print(f"\nSubcog Results:")
    print(f"  Correct: {correct}/{total} ({accuracy:.1f}%)")

    # Show failures
    failures = [r for r in results if not r["correct"]]
    if failures:
        print(f"\nFailures ({len(failures)}):")
        for f in failures:
            print(f"  - Q: {f['question'][:50]}...")
            print(f"    Expected: {str(f['reference'])[:40]}...")
            print(f"    Got: {f['predicted'][:40]}...")

    # Cleanup
    adapter.cleanup()

    return {
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
        "results": results,
    }


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Validate SubcogAdapter")
    parser.add_argument(
        "--questions",
        type=int,
        default=10,
        help="Number of questions to test (default: 10)",
    )
    parser.add_argument(
        "--search-limit",
        type=int,
        default=20,
        help="Number of memories to retrieve per search (default: 20)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging output",
    )

    args = parser.parse_args()

    # Configure logging based on verbosity
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    results = run_validation(
        num_questions=args.questions,
        memory_search_limit=args.search_limit,
        verbose=args.verbose,
    )

    if "error" in results:
        sys.exit(1)


if __name__ == "__main__":
    main()
