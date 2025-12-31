#!/usr/bin/env python3
"""Validate git-notes memory performance (git-notes only, no baseline comparison).

Quick test of git-notes adapter without running no-memory baseline.

Usage:
    # Default: 20 questions
    uv run python scripts/validate_git_notes.py

    # Custom question count
    uv run python scripts/validate_git_notes.py --questions 50

    # Verbose output (show each question)
    uv run python scripts/validate_git_notes.py --verbose
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Add git-notes-memory source for observability module (workaround for stale wheel)
_gnm_src = Path(__file__).parent.parent.parent / "git-notes-memory" / "src"
if _gnm_src.exists():
    sys.path.insert(0, str(_gnm_src))


def run_validation(
    num_questions: int = 20,
    verbose: bool = False,
    model: str = "gpt-5-mini",
) -> dict[str, Any]:
    """Run git-notes validation.

    Args:
        num_questions: Number of questions to test
        verbose: Show individual question results
        model: LLM model to use

    Returns:
        Validation results dict
    """
    # Load .env file first (for OPENAI_API_KEY etc)
    from dotenv import load_dotenv
    load_dotenv()

    # Enable auto-flush for telemetry (exports traces/metrics periodically)
    try:
        from git_notes_memory.observability import enable_auto_flush
        enable_auto_flush()
    except ImportError:
        pass  # Telemetry not available

    from src.adapters.git_notes import GitNotesAdapter
    from src.benchmarks.locomo.dataset import load_locomo
    from src.benchmarks.locomo.wrapper import LoCoMoAgent
    from src.clients.openai_client import OpenAIClient
    from src.evaluation.judge import LLMJudge

    print("=" * 60)
    print("Git-Notes Memory Validation")
    print("=" * 60)

    # Check API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        return {"success": False, "error": "OPENAI_API_KEY not set"}

    print(f"\nConfig:")
    print(f"  Questions: {num_questions}")
    print(f"  Model: {model}")
    print(f"  API Key: ...{api_key[-8:]}")

    # Load dataset
    print("\n[1/4] Loading LoCoMo dataset...")
    dataset = load_locomo()
    conv = dataset.conversations[0]
    questions = conv.questions[:num_questions]
    print(f"  Loaded {len(questions)} questions from conversation '{conv.sample_id}'")

    # Setup LLM and judge
    print("\n[2/4] Initializing LLM and judge...")
    llm = OpenAIClient(model=model)
    judge = LLMJudge(model=model)
    print(f"  LLM: {model}")
    print(f"  Judge: {model}")

    results: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "config": {
            "num_questions": num_questions,
            "model": model,
            "sample_id": conv.sample_id,
        },
    }

    # Test git-notes adapter
    print("\n[3/4] Testing git-notes...")

    # Create fresh adapter in temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # git-notes-memory requires a git repository
        subprocess.run(
            ["git", "init"],
            cwd=tmpdir,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"],
            cwd=tmpdir,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test User"],
            cwd=tmpdir,
            capture_output=True,
            check=True,
        )
        adapter = GitNotesAdapter(repo_path=tmpdir)

        agent = LoCoMoAgent(adapter=adapter, llm=llm, memory_search_limit=45)

        # Ingest conversation
        print(f"  Ingesting conversation...")
        agent.ingest_conversation(conv)
        stats = adapter.get_stats()
        mem_count = stats.get('memory_count', stats.get('total_memories', 'N/A'))
        print(f"  Ingested {mem_count} memories")

        # Answer questions
        scores: list[float] = []
        correct = 0
        failures: list[dict[str, Any]] = []

        for i, q in enumerate(questions):
            answer = agent.answer_question(q)
            judgment = judge.judge(
                question=q.question,
                reference_answer=q.answer,
                model_answer=answer.answer,
            )
            scores.append(judgment.score)
            if judgment.score >= 0.5:
                correct += 1
            else:
                failures.append({
                    "index": i + 1,
                    "question": q.question,
                    "category": q.category.name if hasattr(q.category, 'name') else str(q.category),
                    "expected": q.answer,
                    "got": answer.answer,
                    "score": judgment.score,
                })

            if verbose:
                status = "✓" if judgment.score >= 0.5 else "✗"
                print(f"  [{i+1:2d}] {status} score={judgment.score:.2f} {q.question[:50]}...")

        avg_score = sum(scores) / len(scores) if scores else 0
        accuracy = correct / len(questions) if questions else 0

        print(f"  Results: {correct}/{len(questions)} correct ({accuracy:.0%})")
        print(f"  Avg score: {avg_score:.2f}")

        results["correct"] = correct
        results["total"] = len(questions)
        results["accuracy"] = accuracy
        results["avg_score"] = avg_score
        results["scores"] = scores
        results["failures"] = failures

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\n  Correct:   {correct}/{len(questions)}")
    print(f"  Accuracy:  {accuracy:.0%}")
    print(f"  Avg Score: {avg_score:.2f}")

    if failures:
        print(f"\n  Failures ({len(failures)}):")
        for f in failures:
            print(f"    [{f['index']:2d}] [{f['category']}] {f['question'][:45]}...")

    print("\n" + "=" * 60)
    if accuracy >= 0.7:
        print("✓ GOOD: 70%+ accuracy achieved")
        results["verdict"] = "good"
        results["success"] = True
    elif accuracy >= 0.5:
        print("~ OK: 50-70% accuracy")
        results["verdict"] = "ok"
        results["success"] = True
    else:
        print("✗ NEEDS WORK: <50% accuracy")
        results["verdict"] = "needs_work"
        results["success"] = False

    print("=" * 60)

    # Save results
    output_dir = Path("results/validation")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"git_notes_{timestamp}.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    # Flush telemetry to OTLP if configured
    print("\n[4/4] Flushing telemetry...")
    try:
        from git_notes_memory.observability import flush_telemetry

        flush_telemetry()
        print("  Telemetry flushed to OTLP")
    except ImportError as e:
        print(f"  Telemetry flush skipped (missing deps): {e}")
    except Exception as e:
        print(f"  Telemetry flush failed: {e}")

    return results


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate git-notes memory performance"
    )
    parser.add_argument(
        "-n", "--questions",
        type=int,
        default=20,
        help="Number of questions to test (default: 20)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show individual question results",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-mini",
        help="LLM model to use (default: gpt-5-mini)",
    )

    args = parser.parse_args()

    result = run_validation(
        num_questions=args.questions,
        verbose=args.verbose,
        model=args.model,
    )

    sys.exit(0 if result.get("success", False) else 1)


if __name__ == "__main__":
    main()
