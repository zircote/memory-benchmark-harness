"""LongMemEval benchmark implementation.

This module provides the dataset loader, agent wrapper, evaluation pipeline,
and metrics calculation for the LongMemEval benchmark (ICLR 2025).

LongMemEval evaluates long-term memory capabilities through:
- Single-session questions (user preferences, assistant facts)
- Multi-session temporal reasoning
- Knowledge update tracking
- Abstention detection

Dataset: xiaowu0162/longmemeval-cleaned from HuggingFace
Paper: "LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory"
"""

from src.benchmarks.longmemeval.dataset import (
    LongMemEvalDataset,
    LongMemEvalQuestion,
    LongMemEvalSession,
    Message,
    QuestionType,
    load_longmemeval,
    load_longmemeval_from_file,
)
from src.benchmarks.longmemeval.metrics import (
    AbilityMetrics,
    AbstentionMetrics,
    LongMemEvalMetrics,
    MetricsCalculator,
    compare_results,
)
from src.benchmarks.longmemeval.pipeline import (
    AssessmentResult,
    BenchmarkPipeline,
    QuestionResult,
)
from src.benchmarks.longmemeval.wrapper import (
    AgentAnswer,
    LLMClient,
    LLMResponse,
    LongMemEvalAgent,
)

__all__ = [
    # Dataset classes
    "LongMemEvalDataset",
    "LongMemEvalSession",
    "LongMemEvalQuestion",
    "Message",
    "QuestionType",
    "load_longmemeval",
    "load_longmemeval_from_file",
    # Agent wrapper
    "LongMemEvalAgent",
    "AgentAnswer",
    "LLMClient",
    "LLMResponse",
    # Pipeline
    "BenchmarkPipeline",
    "AssessmentResult",
    "QuestionResult",
    # Metrics
    "AbilityMetrics",
    "AbstentionMetrics",
    "LongMemEvalMetrics",
    "MetricsCalculator",
    "compare_results",
]
