"""Mock LLM Judge for testing without API calls.

This module provides a mock judge that returns deterministic judgments
for testing the benchmark pipeline without making external API calls.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.evaluation.judge import Judgment, JudgmentResult


class MockLLMJudge:
    """Mock LLM judge for testing without API calls.

    Returns deterministic judgments that allow the pipeline to complete
    without making external API calls. Useful for:
    - Testing the benchmark infrastructure
    - CI/CD smoke tests
    - Development without API key

    The mock judge scores based on simple heuristics:
    - Empty answers get score 0.0 (incorrect)
    - Answers containing "mock" get score 0.5 (partial)
    - All other answers get score 1.0 (correct)

    Example:
        ```python
        judge = MockLLMJudge()
        judgment = judge.judge("What is X?", "X is Y", "X is Y")
        print(judgment.score)  # 1.0
        print(judgment.result)  # JudgmentResult.CORRECT
        ```
    """

    def __init__(
        self,
        model: str = "mock-judge",
        _cache_dir: Path | str = ".cache/judgments",  # noqa: ARG002 - API compat
        _cache_ttl_days: int = 30,  # noqa: ARG002 - API compat
        _prompt_template: str | None = None,  # noqa: ARG002 - API compat
        _temperature: float = 0.0,  # noqa: ARG002 - API compat
        _max_tokens: int = 500,  # noqa: ARG002 - API compat
    ) -> None:
        """Initialize the mock judge.

        Args match LLMJudge for drop-in compatibility, but are ignored.
        """
        self.model = model
        self._call_count = 0
        self._error_count = 0

    def judge(
        self,
        question: str,
        reference_answer: str,
        model_answer: str,
        *,
        skip_cache: bool = False,  # noqa: ARG002 - API compat
        metadata: dict[str, Any] | None = None,  # noqa: ARG002 - API compat
    ) -> Judgment:
        """Return a deterministic mock judgment.

        Args:
            question: The question being evaluated
            reference_answer: The expected/correct answer
            model_answer: The model's response to evaluate
            skip_cache: Ignored in mock
            metadata: Ignored in mock

        Returns:
            Judgment with deterministic scoring
        """
        self._call_count += 1

        # Simple heuristic scoring for testing
        answer_lower = model_answer.lower().strip()

        if not answer_lower:
            result = JudgmentResult.INCORRECT
            score = 0.0
            reasoning = "Mock judgment: Empty answer"
        elif "mock" in answer_lower:
            # Mock responses get partial credit
            result = JudgmentResult.PARTIAL
            score = 0.5
            reasoning = "Mock judgment: Mock response detected, partial credit"
        else:
            # All other answers get full credit in mock mode
            result = JudgmentResult.CORRECT
            score = 1.0
            reasoning = "Mock judgment: Answer provided, assuming correct for testing"

        return Judgment(
            result=result,
            score=score,
            reasoning=reasoning,
            question=question,
            reference_answer=reference_answer,
            model_answer=model_answer,
            metadata={"mock": True, "call_count": self._call_count},
            cached=False,
            timestamp=datetime.now(UTC),
        )

    def batch_judge(
        self,
        items: list[tuple[str, str, str]],
        *,
        skip_cache: bool = False,
        progress_callback: Any | None = None,  # noqa: ARG002 - API compat
    ) -> list[Judgment]:
        """Judge a batch of items.

        Args:
            items: List of (question, reference, model_answer) tuples
            skip_cache: Ignored in mock
            progress_callback: Ignored in mock

        Returns:
            List of mock Judgments
        """
        return [
            self.judge(question, reference, model_answer, skip_cache=skip_cache)
            for question, reference, model_answer in items
        ]

    @property
    def call_count(self) -> int:
        """Number of mock judgments made."""
        return self._call_count

    @property
    def error_count(self) -> int:
        """Number of errors (always 0 for mock)."""
        return self._error_count
