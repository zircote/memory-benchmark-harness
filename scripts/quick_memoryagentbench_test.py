#!/usr/bin/env python3
"""Quick test of the MemoryAgentBench pipeline.

This verifies the MemoryAgentBench evaluation pipeline works end-to-end
using programmatically created test data.

Usage:
    uv run python scripts/quick_memoryagentbench_test.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add src to path for imports
_root = Path(__file__).parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


def main() -> None:
    """Run a quick MemoryAgentBench test."""
    print("=" * 60)
    print("Quick MemoryAgentBench Experiment Test")
    print("=" * 60)

    # Check API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)
    print(f"\n1. API Key: ...{api_key[-8:]}")

    # Create sample dataset programmatically
    print("\n2. Creating sample MemoryAgentBench dataset...")
    from src.benchmarks.memoryagentbench.dataset import (
        Competency,
        DifficultyLevel,
        MemoryAgentBenchDataset,
        MemoryAgentBenchQuestion,
        MemoryAgentBenchSplit,
    )

    # Create questions for each competency
    questions_by_competency: dict[Competency, list[MemoryAgentBenchQuestion]] = {}

    for comp in Competency:
        questions = []
        for i in range(2):  # 2 questions per competency for quick test
            questions.append(
                MemoryAgentBenchQuestion(
                    question_id=f"{comp.short_name}_{i}",
                    question_text=f"Test question {i} for {comp.value}?",
                    answers=[f"answer_{comp.short_name}_{i}"],
                    competency=comp,
                    context=f"This is the context for {comp.value}. The answer is answer_{comp.short_name}_{i}.",
                    difficulty=DifficultyLevel.SINGLE_HOP,
                )
            )
        questions_by_competency[comp] = questions

    # Create splits as dict
    splits = {
        comp: MemoryAgentBenchSplit(competency=comp, questions=qs)
        for comp, qs in questions_by_competency.items()
    }

    dataset = MemoryAgentBenchDataset(splits=splits)
    print(f"   Competencies: {len(dataset.competencies)}")
    print(f"   Total questions: {dataset.total_questions}")
    stats = dataset.get_stats()
    # Show questions per competency from splits stats
    questions_per_comp = {k: v["question_count"] for k, v in stats["splits"].items()}
    print(f"   Questions per competency: {questions_per_comp}")

    # Create components
    print("\n3. Creating pipeline components...")
    from src.adapters.mock import MockAdapter
    from src.benchmarks.memoryagentbench import MemoryAgentBenchAgent, MemoryAgentBenchPipeline
    from src.clients.openai_client import OpenAIClient
    from src.evaluation.judge import LLMJudge

    # Create LLM client adapter for MemoryAgentBench protocol
    class LLMClientAdapter:
        """Adapter for MemoryAgentBench LLMClient protocol."""

        def __init__(self, client: OpenAIClient) -> None:
            self.client = client

        def generate(
            self,
            prompt: str,
            system_prompt: str | None = None,
            max_tokens: int = 1024,
        ) -> str:
            system = system_prompt or "You are a helpful assistant."
            messages = [{"role": "user", "content": prompt}]
            response = self.client.complete(system=system, messages=messages)
            return response.content

    openai_client = OpenAIClient(model="gpt-4o")
    llm_adapter = LLMClientAdapter(openai_client)
    adapter = MockAdapter()
    judge = LLMJudge()

    # Create agent and pipeline
    agent = MemoryAgentBenchAgent(
        adapter=adapter,
        llm=llm_adapter,
        retrieval_limit=5,  # Match actual dataclass attribute
    )

    pipeline = MemoryAgentBenchPipeline(
        agent=agent,
        judge=judge,
    )
    print("   MemoryAgentBenchPipeline created successfully")

    # Run evaluation on limited competencies
    print("\n4. Running evaluation (Accurate Retrieval only, 2 questions)...")
    try:
        # Get just the AR split
        ar_split = dataset.get_split(Competency.ACCURATE_RETRIEVAL)
        if ar_split is None:
            print("   ERROR: AR split not found")
            sys.exit(1)

        # Evaluate just this split
        result = pipeline.evaluate_split(ar_split)

        print("   Evaluation complete!")
        print(f"   Accuracy: {result.accuracy:.2%}")
        print(f"   Correct: {result.correct_count}/{result.total_questions}")
        print(f"   Competency: {result.competency.value}")

    except Exception as e:
        print(f"   ERROR during evaluation: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    print("\n" + "=" * 60)
    print("Quick test complete - MemoryAgentBench Pipeline works end-to-end!")
    print("=" * 60)


if __name__ == "__main__":
    main()
