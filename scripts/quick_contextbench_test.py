#!/usr/bin/env python3
"""Quick test of the Context-Bench pipeline with synthetic data.

This verifies the Context-Bench evaluation pipeline works end-to-end
using the synthetic dataset generator (no external data needed).

Usage:
    uv run python scripts/quick_contextbench_test.py
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


class LLMClientAdapter:
    """Adapter to make OpenAIClient compatible with Context-Bench LLMClient protocol.

    Context-Bench expects `llm.generate(prompt, system_prompt, max_tokens)`.
    OpenAIClient provides `llm.complete(system, messages, temperature)`.
    """

    def __init__(self, client: OpenAIClient) -> None:
        self.client = client

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 1024,
    ) -> str:
        """Generate a response from the LLM."""
        system = system_prompt or "You are a helpful assistant."
        messages = [{"role": "user", "content": prompt}]
        response = self.client.complete(system=system, messages=messages)
        return response.content


def main() -> None:
    """Run a quick Context-Bench test."""
    print("=" * 60)
    print("Quick Context-Bench Experiment Test")
    print("=" * 60)

    # Check API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)
    print(f"\n1. API Key: ...{api_key[-8:]}")

    # Generate synthetic dataset
    print("\n2. Generating synthetic Context-Bench dataset...")
    from src.benchmarks.contextbench import generate_synthetic_dataset

    dataset = generate_synthetic_dataset(n_files=20, n_questions=10, seed=42)
    print(f"   Files: {dataset.file_count}")
    print(f"   Questions: {dataset.question_count}")
    stats = dataset.get_stats()
    print(f"   Category distribution: {stats['category_distribution']}")

    # Create components
    print("\n3. Creating pipeline components...")
    from src.adapters.mock import MockAdapter
    from src.benchmarks.contextbench import ContextBenchAgent, ContextBenchPipeline
    from src.evaluation.judge import LLMJudge

    openai_client = OpenAIClient(model="gpt-4o")
    llm_adapter = LLMClientAdapter(openai_client)
    adapter = MockAdapter()
    judge = LLMJudge()

    # Create agent and pipeline
    agent = ContextBenchAgent(
        adapter=adapter,
        llm=llm_adapter,
        dataset=dataset,
        max_operations=5,  # Limit for quick test
    )

    pipeline = ContextBenchPipeline(
        agent=agent,
        judge=judge,
    )
    print("   ContextBenchPipeline created successfully")

    # Run evaluation on limited questions
    print("\n4. Running evaluation (5 questions, simple mode)...")
    try:
        # Limit to 5 questions for quick test
        limited_questions = dataset.questions[:5]

        result = pipeline.evaluate(
            dataset=dataset,
            questions=limited_questions,
            use_simple_mode=True,  # Single-shot mode for speed
            index_files=True,
        )

        print("   Evaluation complete!")
        print(f"   Accuracy: {result.accuracy:.2%}")
        print(f"   Correct: {result.correct_count}/{result.total_questions}")
        print(f"   Total cost estimate: {result.total_cost:.4f}")
        print(f"   Category breakdown: {list(result.category_breakdown.keys())}")

    except Exception as e:
        print(f"   ERROR during evaluation: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    print("\n" + "=" * 60)
    print("Quick test complete - Context-Bench Pipeline works end-to-end!")
    print("=" * 60)


if __name__ == "__main__":
    main()
