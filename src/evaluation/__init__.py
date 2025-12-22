"""Evaluation module for benchmark harness.

This module provides evaluation components including:
- LLM-as-Judge for semantic correctness evaluation
- MockLLMJudge for testing without API calls
- Statistical analysis for significance testing
"""

from src.evaluation.judge import (
    DEFAULT_JUDGE_PROMPT,
    Judgment,
    JudgmentCache,
    JudgmentResult,
    LLMJudge,
)
from src.evaluation.mock_judge import MockLLMJudge
from src.evaluation.statistics import (
    ComparisonResult,
    ConfidenceInterval,
    StatisticalAnalyzer,
)

__all__ = [
    # Judge module
    "Judgment",
    "JudgmentCache",
    "JudgmentResult",
    "LLMJudge",
    "MockLLMJudge",
    "DEFAULT_JUDGE_PROMPT",
    # Statistics module
    "ConfidenceInterval",
    "ComparisonResult",
    "StatisticalAnalyzer",
]
