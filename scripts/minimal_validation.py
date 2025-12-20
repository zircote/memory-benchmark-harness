#!/usr/bin/env python3
"""Minimal end-to-end validation of the benchmark pipeline.

Tests the critical OpenAI integration by running a simple LLM judge evaluation.

Usage:
    uv run python scripts/minimal_validation.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypedDict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestCase(TypedDict):
    """Test case structure."""

    question: str
    reference: str
    model_answer: str
    expected: bool


def run_minimal_validation() -> dict[str, Any]:
    """Run minimal validation and return results."""
    from src.evaluation.judge import LLMJudge

    print("=" * 60)
    print("Minimal End-to-End Validation")
    print("=" * 60)

    # Check OpenAI API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        return {"success": False, "error": "OPENAI_API_KEY not set"}

    print(f"\n1. API Key: ...{api_key[-8:]}")

    print("\n2. Creating LLM Judge...")
    judge = LLMJudge()
    print("   LLMJudge created (using GPT-4o)")

    print("\n3. Testing judgment (calling OpenAI API)...")

    # Test case: Simple factual question
    test_cases: list[TestCase] = [
        {
            "question": "What is the capital of France?",
            "reference": "Paris",
            "model_answer": "The capital of France is Paris.",
            "expected": True,
        },
        {
            "question": "What is 2 + 2?",
            "reference": "4",
            "model_answer": "The answer is 4.",
            "expected": True,
        },
        {
            "question": "Who wrote Romeo and Juliet?",
            "reference": "William Shakespeare",
            "model_answer": "I'm not sure who wrote it.",
            "expected": False,
        },
    ]

    results: list[dict[str, Any]] = []
    for i, tc in enumerate(test_cases, 1):
        print(f"\n   Test {i}: {tc['question'][:40]}...")

        try:
            judgment = judge.judge(
                question=tc["question"],
                reference_answer=tc["reference"],
                model_answer=tc["model_answer"],
            )

            # Check if result matches expected (CORRECT = True, INCORRECT = False)
            is_correct = judgment.result.value == "correct"
            match = "✓" if is_correct == tc["expected"] else "✗"
            print(f"      Model answer: {tc['model_answer'][:50]}...")
            print(f"      Judgment: {judgment.result.value} {match}")
            print(f"      Score: {judgment.score:.2f}")
            print(f"      Reasoning: {judgment.reasoning[:80]}...")

            results.append(
                {
                    "question": tc["question"],
                    "model_answer": tc["model_answer"],
                    "judgment_result": judgment.result.value,
                    "is_correct": is_correct,
                    "expected": tc["expected"],
                    "match": is_correct == tc["expected"],
                    "score": judgment.score,
                }
            )

        except Exception as e:
            print(f"      ERROR: {e}")
            results.append(
                {
                    "question": tc["question"],
                    "error": str(e),
                    "match": False,
                }
            )

    # Summary
    matches = sum(1 for r in results if r.get("match", False))
    total = len(results)

    print("\n" + "=" * 60)
    print(f"VALIDATION RESULTS: {matches}/{total} tests passed")
    print("=" * 60)

    if matches == total:
        print("\n✓ LLM Judge is working correctly")
        print("✓ OpenAI API integration verified")
        print("\nThe benchmark pipeline is ready for experiments.")
    else:
        print("\n✗ Some tests failed - check API key and connectivity")

    # Save results
    output_dir = Path("results/validation_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    output: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "tests_passed": matches,
        "tests_total": total,
        "success": matches == total,
        "results": results,
    }

    output_path = output_dir / "minimal_validation.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return output


def main() -> None:
    """Main entry point."""
    result = run_minimal_validation()
    sys.exit(0 if result.get("success", False) else 1)


if __name__ == "__main__":
    main()
