#!/usr/bin/env python3
"""Run all benchmarks locally with verbose feedback.

Runs all available benchmarks (LoCoMo, LongMemEval, Context-Bench, MemoryAgentBench)
with real-time progress output.

Usage:
    # Run all benchmarks with defaults
    uv run python scripts/run_all_benchmarks.py

    # Limit questions per benchmark
    uv run python scripts/run_all_benchmarks.py --questions 20

    # Run specific benchmarks only
    uv run python scripts/run_all_benchmarks.py --benchmarks locomo,longmemeval

    # Run specific adapter only
    uv run python scripts/run_all_benchmarks.py --adapter git-notes
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Add git-notes-memory source for observability module
_gnm_src = Path(__file__).parent.parent.parent / "git-notes-memory" / "src"
if _gnm_src.exists():
    sys.path.insert(0, str(_gnm_src))


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""

    name: str
    adapter: str
    correct: int = 0
    total: int = 0
    accuracy: float = 0.0
    avg_score: float = 0.0
    scores: list[float] = field(default_factory=list)
    failures: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    skipped: bool = False


def _init_git_repo(tmpdir: str) -> None:
    """Initialize a git repo in tmpdir for adapters that need it."""
    subprocess.run(["git", "init"], cwd=tmpdir, capture_output=True, check=True)
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


def _create_adapter(adapter_name: str, tmpdir: str) -> Any:
    """Create the appropriate adapter for the given name."""
    from src.adapters.git_notes import GitNotesAdapter
    from src.adapters.no_memory import NoMemoryAdapter
    from src.adapters.subcog import SubcogAdapter

    if adapter_name == "git-notes":
        _init_git_repo(tmpdir)
        return GitNotesAdapter(repo_path=tmpdir)
    elif adapter_name == "subcog":
        _init_git_repo(tmpdir)
        return SubcogAdapter(repo_path=tmpdir)
    else:
        return NoMemoryAdapter()


def run_locomo(
    adapter_name: str,
    llm: Any,
    judge: Any,
    num_questions: int,
    verbose: bool,
    tmpdir: str,
) -> BenchmarkResult:
    """Run LoCoMo benchmark."""
    from src.benchmarks.locomo.dataset import load_locomo
    from src.benchmarks.locomo.wrapper import LoCoMoAgent

    result = BenchmarkResult(name="locomo", adapter=adapter_name)

    try:
        dataset = load_locomo()
        conv = dataset.conversations[0]
        questions = conv.questions[:num_questions]
        result.total = len(questions)

        print(f"    Loaded {len(questions)} questions from '{conv.sample_id}'")

        # Create adapter
        adapter = _create_adapter(adapter_name, tmpdir)

        agent = LoCoMoAgent(adapter=adapter, llm=llm, memory_search_limit=45)

        print(f"    Ingesting conversation...")
        agent.ingest_conversation(conv)
        stats = adapter.get_stats()
        mem_count = stats.get("memory_count", stats.get("total_memories", "N/A"))
        print(f"    Ingested {mem_count} memories")

        for i, q in enumerate(questions):
            answer = agent.answer_question(q)
            judgment = judge.judge(
                question=q.question,
                reference_answer=q.answer,
                model_answer=answer.answer,
            )
            result.scores.append(judgment.score)
            if judgment.score >= 0.5:
                result.correct += 1
            else:
                result.failures.append({
                    "index": i + 1,
                    "question": q.question,
                    "category": q.category.name if hasattr(q.category, "name") else str(q.category),
                    "expected": q.answer,
                    "got": answer.answer,
                    "score": judgment.score,
                })

            if verbose:
                status = "✓" if judgment.score >= 0.5 else "✗"
                print(f"    [{i+1:3d}] {status} score={judgment.score:.2f} {q.question[:45]}...")

        result.accuracy = result.correct / result.total if result.total else 0
        result.avg_score = sum(result.scores) / len(result.scores) if result.scores else 0

    except Exception as e:
        result.error = str(e)
        print(f"    ERROR: {e}")

    return result


def run_longmemeval(
    adapter_name: str,
    llm: Any,
    judge: Any,
    num_questions: int,
    verbose: bool,
    tmpdir: str,
) -> BenchmarkResult:
    """Run LongMemEval benchmark.

    LongMemEval uses 'oracle' format where each question has its own haystack_sessions.
    We ingest sessions per-question to avoid impractical bulk ingestion of 19,195+ sessions.
    """
    result = BenchmarkResult(name="longmemeval", adapter=adapter_name)

    try:
        from src.benchmarks.longmemeval.dataset import (
            LongMemEvalSession,
            Message,
            load_longmemeval,
        )
        from src.benchmarks.longmemeval.wrapper import LongMemEvalAgent

        dataset = load_longmemeval()
        questions = dataset.questions[:num_questions]
        result.total = len(questions)

        print(f"    Loaded {len(questions)} questions (oracle mode: per-question sessions)")

        # Create adapter
        adapter = _create_adapter(adapter_name, tmpdir)

        # Use the official agent wrapper
        agent = LongMemEvalAgent(adapter=adapter, llm=llm, memory_search_limit=30)

        # Process each question with its own session context (oracle mode)
        for i, q in enumerate(questions):
            # Clear memory before each question for oracle evaluation
            agent.clear_memory()

            # Build sessions from question metadata if available (oracle format)
            q_sessions: list[LongMemEvalSession] = []

            # Check for embedded haystack_sessions in metadata
            if q.metadata.get("haystack_sessions"):
                haystack = q.metadata["haystack_sessions"]
                haystack_dates = q.metadata.get("haystack_dates", [])
                for sidx, session_msgs in enumerate(haystack):
                    if isinstance(session_msgs, list) and session_msgs:
                        messages = []
                        for msg in session_msgs:
                            if isinstance(msg, dict):
                                messages.append(Message(
                                    role=msg.get("role", "user"),
                                    content=msg.get("content", ""),
                                ))
                        if messages:
                            ts = haystack_dates[sidx] if sidx < len(haystack_dates) else None
                            q_sessions.append(LongMemEvalSession(
                                session_id=f"q{i}_s{sidx}",
                                messages=messages,
                                timestamp=ts,
                            ))

            # Fall back to dataset sessions if no embedded sessions
            if not q_sessions and q.relevant_session_ids:
                for sid in q.relevant_session_ids:
                    session = dataset.get_session(sid)
                    if session:
                        q_sessions.append(session)

            # Ingest sessions for this question
            if q_sessions:
                for session in q_sessions:
                    agent.ingest_session(session)

            # Answer using the ingested context
            answer = agent.answer_question(q, relevant_sessions_only=False)

            # ground_truth is a list, use first as reference
            reference = q.ground_truth[0] if q.ground_truth else ""

            judgment = judge.judge(
                question=q.question_text,
                reference_answer=reference,
                model_answer=answer.answer,
            )

            result.scores.append(judgment.score)
            if judgment.score >= 0.5:
                result.correct += 1
            else:
                result.failures.append({
                    "index": i + 1,
                    "question": q.question_text,
                    "expected": reference,
                    "got": answer.answer,
                    "score": judgment.score,
                })

            if verbose:
                status = "✓" if judgment.score >= 0.5 else "✗"
                sess_count = len(q_sessions)
                print(f"    [{i+1:3d}] {status} score={judgment.score:.2f} ({sess_count} sessions) {q.question_text[:35]}...")

        result.accuracy = result.correct / result.total if result.total else 0
        result.avg_score = sum(result.scores) / len(result.scores) if result.scores else 0

    except FileNotFoundError as e:
        result.skipped = True
        result.error = f"Dataset not found: {e}"
        print(f"    SKIPPED: {result.error}")
    except Exception as e:
        result.error = str(e)
        import traceback
        print(f"    ERROR: {e}")
        if verbose:
            traceback.print_exc()

    return result


def run_contextbench(
    adapter_name: str,
    llm: Any,
    judge: Any,
    num_questions: int,
    verbose: bool,
    tmpdir: str,
) -> BenchmarkResult:
    """Run Context-Bench benchmark."""
    result = BenchmarkResult(name="contextbench", adapter=adapter_name)

    try:
        from src.benchmarks.contextbench.dataset import (
            generate_synthetic_dataset,
            load_contextbench,
        )
        from src.benchmarks.contextbench.wrapper import ContextBenchAgent

        # Try to load real dataset, fall back to synthetic
        try:
            dataset = load_contextbench()
            if not dataset.questions or not dataset.files:
                raise FileNotFoundError("Empty dataset")
        except FileNotFoundError:
            print("    Using synthetic dataset (letta-evals data not found)")
            dataset = generate_synthetic_dataset(n_files=50, n_questions=num_questions * 2)

        questions = dataset.questions[:num_questions]
        result.total = len(questions)

        print(f"    Loaded {len(questions)} questions, {len(dataset.files)} files")

        # Create adapter
        adapter = _create_adapter(adapter_name, tmpdir)

        # Create agent with memory adapter
        agent = ContextBenchAgent(
            adapter=adapter,
            llm=llm,
            dataset=dataset,
            use_memory_for_navigation=True,
            retrieval_limit=10,
        )

        # Index files into memory
        print("    Indexing files into memory...")
        indexed = agent.index_files()
        print(f"    Indexed {indexed} files")

        # Answer questions using simple (single-shot) mode for efficiency
        for i, q in enumerate(questions):
            answer_result = agent.answer_question_simple(q)

            judgment = judge.judge(
                question=q.question_text,
                reference_answer=q.answer,
                model_answer=answer_result.answer,
            )
            result.scores.append(judgment.score)
            if judgment.score >= 0.5:
                result.correct += 1
            else:
                result.failures.append({
                    "index": i + 1,
                    "question": q.question_text,
                    "category": q.category.value,
                    "expected": q.answer,
                    "got": answer_result.answer,
                    "score": judgment.score,
                })

            if verbose:
                status = "✓" if judgment.score >= 0.5 else "✗"
                print(f"    [{i+1:3d}] {status} score={judgment.score:.2f} {q.question_text[:45]}...")

        result.accuracy = result.correct / result.total if result.total else 0
        result.avg_score = sum(result.scores) / len(result.scores) if result.scores else 0

    except FileNotFoundError as e:
        result.skipped = True
        result.error = f"Dataset not found: {e}"
        print(f"    SKIPPED: {result.error}")
    except Exception as e:
        result.error = str(e)
        import traceback
        print(f"    ERROR: {e}")
        if verbose:
            traceback.print_exc()

    return result


def run_memoryagentbench(
    adapter_name: str,
    llm: Any,
    judge: Any,
    num_questions: int,
    verbose: bool,
    tmpdir: str,
) -> BenchmarkResult:
    """Run MemoryAgentBench benchmark."""
    result = BenchmarkResult(name="memoryagentbench", adapter=adapter_name)

    try:
        from src.benchmarks.memoryagentbench.dataset import load_memoryagentbench
        from src.benchmarks.memoryagentbench.wrapper import MemoryAgentBenchAgent

        dataset = load_memoryagentbench()

        # Flatten all questions across splits, distributing evenly
        all_questions: list = []
        questions_per_split = max(1, num_questions // len(dataset.splits))
        for split in dataset.splits.values():
            all_questions.extend(split.questions[:questions_per_split])

        questions = all_questions[:num_questions]
        result.total = len(questions)

        print(f"    Loaded {len(questions)} questions from {len(dataset.splits)} competency splits")

        # Create adapter
        adapter = _create_adapter(adapter_name, tmpdir)

        # Create agent with memory adapter
        agent = MemoryAgentBenchAgent(
            adapter=adapter,
            llm=llm,
            use_version_history=True,
            retrieval_limit=10,
            min_relevance_score=0.3,
        )

        # Process each question
        for i, q in enumerate(questions):
            # Answer the question (this also ingests context)
            answer_result = agent.answer_question(q, ingest=True)

            # Judge answer against all acceptable answers
            # Use the first answer as reference, but any match should work
            best_score = 0.0
            for acceptable_answer in q.answers:
                judgment = judge.judge(
                    question=q.question_text,
                    reference_answer=acceptable_answer,
                    model_answer=answer_result.answer,
                )
                if judgment.score > best_score:
                    best_score = judgment.score

            result.scores.append(best_score)
            if best_score >= 0.5:
                result.correct += 1
            else:
                result.failures.append({
                    "index": i + 1,
                    "question": q.question_text,
                    "competency": q.competency.value,
                    "expected": q.answers[0] if q.answers else "",
                    "got": answer_result.answer,
                    "score": best_score,
                })

            if verbose:
                status = "✓" if best_score >= 0.5 else "✗"
                comp = q.competency.short_name
                print(f"    [{i+1:3d}] {status} score={best_score:.2f} [{comp}] {q.question_text[:40]}...")

        result.accuracy = result.correct / result.total if result.total else 0
        result.avg_score = sum(result.scores) / len(result.scores) if result.scores else 0

    except ImportError as e:
        result.skipped = True
        result.error = f"Missing dependency: {e}"
        print(f"    SKIPPED: {result.error}")
    except Exception as e:
        result.error = str(e)
        import traceback
        print(f"    ERROR: {e}")
        if verbose:
            traceback.print_exc()

    return result


BENCHMARK_RUNNERS = {
    "locomo": run_locomo,
    "longmemeval": run_longmemeval,
    "contextbench": run_contextbench,
    "memoryagentbench": run_memoryagentbench,
}


def run_all_benchmarks(
    benchmarks: list[str],
    adapters: list[str],
    num_questions: int = 20,
    verbose: bool = True,
    model: str = "gpt-5-mini",
) -> dict[str, Any]:
    """Run all specified benchmarks.

    Args:
        benchmarks: List of benchmark names to run
        adapters: List of adapter names to test
        num_questions: Number of questions per benchmark
        verbose: Show individual question results
        model: LLM model to use

    Returns:
        Results dict with all benchmark results
    """
    from dotenv import load_dotenv

    load_dotenv()

    # Enable auto-flush for telemetry
    try:
        from git_notes_memory.observability import enable_auto_flush

        enable_auto_flush()
    except ImportError:
        pass

    from src.clients.openai_client import OpenAIClient
    from src.evaluation.judge import LLMJudge

    print("=" * 70)
    print("ALL BENCHMARKS - Local Run")
    print("=" * 70)

    # Check API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        return {"success": False, "error": "OPENAI_API_KEY not set"}

    print(f"\nConfig:")
    print(f"  Benchmarks: {', '.join(benchmarks)}")
    print(f"  Adapters: {', '.join(adapters)}")
    print(f"  Questions per benchmark: {num_questions}")
    print(f"  Model: {model}")
    print(f"  API Key: ...{api_key[-8:]}")

    llm = OpenAIClient(model=model)
    judge = LLMJudge(model=model)

    results: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "config": {
            "benchmarks": benchmarks,
            "adapters": adapters,
            "num_questions": num_questions,
            "model": model,
        },
        "benchmarks": {},
    }

    total_correct = 0
    total_questions = 0
    total_skipped = 0

    step = 0
    total_steps = len(benchmarks) * len(adapters)

    for bench_name in benchmarks:
        if bench_name not in BENCHMARK_RUNNERS:
            print(f"\n[?/?] Unknown benchmark: {bench_name}")
            continue

        runner = BENCHMARK_RUNNERS[bench_name]
        results["benchmarks"][bench_name] = {}

        for adapter_name in adapters:
            step += 1
            print(f"\n[{step}/{total_steps}] {bench_name.upper()} with {adapter_name}...")

            # Reset git-notes-memory singletons before each run to avoid conflicts
            if adapter_name == "git-notes":
                try:
                    from git_notes_memory.registry import ServiceRegistry
                    from git_notes_memory.git_ops import GitOpsFactory

                    ServiceRegistry.reset()
                    GitOpsFactory.clear_cache()
                except ImportError:
                    pass  # git-notes-memory not installed

            with tempfile.TemporaryDirectory() as tmpdir:
                bench_result = runner(
                    adapter_name=adapter_name,
                    llm=llm,
                    judge=judge,
                    num_questions=num_questions,
                    verbose=verbose,
                    tmpdir=tmpdir,
                )

            results["benchmarks"][bench_name][adapter_name] = {
                "correct": bench_result.correct,
                "total": bench_result.total,
                "accuracy": bench_result.accuracy,
                "avg_score": bench_result.avg_score,
                "failures": bench_result.failures,
                "error": bench_result.error,
                "skipped": bench_result.skipped,
            }

            if bench_result.skipped:
                total_skipped += 1
                print(f"    SKIPPED")
            elif bench_result.error and bench_result.total == 0:
                print(f"    FAILED: {bench_result.error}")
            else:
                total_correct += bench_result.correct
                total_questions += bench_result.total
                print(f"    Results: {bench_result.correct}/{bench_result.total} ({bench_result.accuracy:.0%})")
                print(f"    Avg score: {bench_result.avg_score:.2f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n{'Benchmark':<20} {'Adapter':<12} {'Correct':<12} {'Accuracy':<12} {'Avg Score':<12}")
    print("-" * 68)

    for bench_name in benchmarks:
        if bench_name not in results["benchmarks"]:
            continue
        for adapter_name in adapters:
            if adapter_name not in results["benchmarks"][bench_name]:
                continue
            r = results["benchmarks"][bench_name][adapter_name]
            if r["skipped"]:
                print(f"{bench_name:<20} {adapter_name:<12} {'SKIPPED':<12}")
            elif r["error"] and r["total"] == 0:
                print(f"{bench_name:<20} {adapter_name:<12} {'ERROR':<12}")
            else:
                acc_str = f"{r['correct']}/{r['total']}"
                print(f"{bench_name:<20} {adapter_name:<12} {acc_str:<12} {r['accuracy']:.0%}{'':8} {r['avg_score']:.2f}")

    print("-" * 68)

    if total_questions > 0:
        overall_accuracy = total_correct / total_questions
        print(f"\n{'TOTAL':<20} {'':<12} {total_correct}/{total_questions:<8} {overall_accuracy:.0%}")
        results["overall"] = {
            "correct": total_correct,
            "total": total_questions,
            "accuracy": overall_accuracy,
            "skipped": total_skipped,
        }
        results["success"] = True
    else:
        print("\nNo questions answered successfully.")
        results["success"] = False

    print("=" * 70)

    # Save results
    output_dir = Path("results/all_benchmarks")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"all_benchmarks_{timestamp}.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    # Flush telemetry
    print("\nFlushing telemetry...")
    try:
        from git_notes_memory.observability import flush_telemetry

        flush_telemetry()
        print("  Telemetry flushed to OTLP")
    except ImportError as e:
        print(f"  Telemetry flush skipped: {e}")
    except Exception as e:
        print(f"  Telemetry flush failed: {e}")

    return results


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run all benchmarks locally with verbose feedback"
    )
    parser.add_argument(
        "-n",
        "--questions",
        type=int,
        default=20,
        help="Number of questions per benchmark (default: 20)",
    )
    parser.add_argument(
        "-b",
        "--benchmarks",
        type=str,
        default="locomo,longmemeval",
        help="Comma-separated list of benchmarks (default: locomo,longmemeval)",
    )
    parser.add_argument(
        "-a",
        "--adapter",
        type=str,
        default="subcog,no-memory",
        help="Comma-separated list of adapters (default: subcog,no-memory)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=True,
        help="Show individual question results (default: True)",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress individual question output",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-mini",
        help="LLM model to use (default: gpt-5-mini)",
    )

    args = parser.parse_args()

    benchmarks = [b.strip() for b in args.benchmarks.split(",")]
    adapters = [a.strip() for a in args.adapter.split(",")]

    result = run_all_benchmarks(
        benchmarks=benchmarks,
        adapters=adapters,
        num_questions=args.questions,
        verbose=not args.quiet,
        model=args.model,
    )

    sys.exit(0 if result.get("success", False) else 1)


if __name__ == "__main__":
    main()
