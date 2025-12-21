#!/usr/bin/env python3
"""Quick test of the experiment runner with LongMemEval local data.

This verifies the experiment runner API works end-to-end with local JSON files,
bypassing the HuggingFace pyarrow compatibility issue.

Uses only 1 trial with mock adapter and limits to first 5 questions.

Usage:
    uv run python scripts/quick_longmemeval_test.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add src to path for imports
_root = Path(__file__).parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.clients.openai_client import OpenAIClient  # noqa: E402
from src.experiments.runner import (  # noqa: E402
    AdapterCondition,
    ExperimentConfig,
    ExperimentRunner,
)


def main() -> None:
    """Run a quick experiment test."""
    print("=" * 60)
    print("Quick LongMemEval Experiment Test")
    print("=" * 60)

    # Check API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)
    print(f"\n1. API Key: ...{api_key[-8:]}")

    # Check data file exists
    data_path = Path("data/longmemeval/longmemeval_s_cleaned.json")
    if not data_path.exists():
        print(f"ERROR: Data file not found: {data_path}")
        sys.exit(1)

    print(f"\n2. Data file: {data_path}")

    # Create OpenAI client
    llm_client = OpenAIClient(model="gpt-4o")
    print("\n3. OpenAI client created (gpt-4o)")

    # Create config
    config = ExperimentConfig(
        benchmark="longmemeval",
        adapters=[AdapterCondition.MOCK],
        num_trials=1,
        dataset_path=str(data_path),
        llm_client=llm_client,
        output_dir="results/quick_test",
    )

    print("\n4. Config created:")
    print(f"   - Benchmark: {config.benchmark}")
    print(f"   - Adapters: {[a.value for a in config.adapters]}")
    print(f"   - Trials: {config.num_trials}")

    # Create runner
    runner = ExperimentRunner(config)
    print("\n5. Runner created")

    # Test dataset loading
    print("\n6. Testing dataset loading...")
    try:
        dataset, benchmark_type = runner._load_dataset()
        print(f"   Dataset loaded: {benchmark_type}")
        print(f"   Sessions: {dataset.session_count}")
        print(f"   Questions: {dataset.question_count}")
        print(f"   Total messages: {dataset.total_messages}")
    except Exception as e:
        print(f"   ERROR loading dataset: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # Test pipeline creation
    print("\n7. Testing pipeline creation...")
    try:
        from src.adapters.mock import MockAdapter
        from src.benchmarks.longmemeval import BenchmarkPipeline
        from src.evaluation.judge import LLMJudge

        adapter = MockAdapter()
        judge = LLMJudge()  # Uses OPENAI_API_KEY from env

        pipeline = BenchmarkPipeline(adapter, llm_client, judge)
        print("   BenchmarkPipeline created successfully")

    except Exception as e:
        print(f"   ERROR creating pipeline: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # Run a minimal assessment (first 5 questions)
    print("\n8. Running minimal assessment (5 questions)...")
    try:
        from src.benchmarks.longmemeval.dataset import LongMemEvalDataset

        # Create limited dataset with first 5 questions
        limited_questions = dataset.questions[:5]

        # Get relevant sessions for these questions
        relevant_session_ids = set()
        for q in limited_questions:
            relevant_session_ids.update(q.relevant_session_ids)

        limited_sessions = [s for s in dataset.sessions if s.session_id in relevant_session_ids]

        limited_dataset = LongMemEvalDataset(
            subset=dataset.subset,
            sessions=limited_sessions,
            questions=limited_questions,
            metadata=dataset.metadata,
        )

        print(f"   Limited to {len(limited_questions)} questions")
        print(f"   Relevant sessions: {len(limited_sessions)}")
        print("   Running pipeline.run()...")

        result = pipeline.run(limited_dataset)

        print("   Assessment complete!")
        print(f"   Accuracy: {result.accuracy:.2%}")
        print(f"   Mean score: {result.mean_score:.3f}")
        print(f"   Total questions: {result.total_questions}")
        print(f"   Correct: {result.correct_count}")
        print(f"   Partial: {result.partial_count}")

    except Exception as e:
        print(f"   ERROR during assessment: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    print("\n" + "=" * 60)
    print("Quick test complete - LongMemEval Pipeline works end-to-end!")
    print("=" * 60)


if __name__ == "__main__":
    main()
