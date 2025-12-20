"""Context-Bench benchmark implementation.

This module provides the dataset loader, agent wrapper, evaluation pipeline,
and metrics calculation for the Context-Bench benchmark from Letta.

Context-Bench evaluates agents on multi-hop information retrieval tasks
requiring file navigation and entity relationship tracing.

Source: https://github.com/letta-ai/letta-evals
Blog: https://www.letta.com/blog/context-bench
"""

from src.benchmarks.contextbench.dataset import (
    ContextBenchDataset,
    ContextBenchFile,
    ContextBenchQuestion,
    load_contextbench,
    load_contextbench_from_file,
)
from src.benchmarks.contextbench.metrics import (
    ContextBenchMetrics,
    MetricsCalculator,
)
from src.benchmarks.contextbench.pipeline import (
    ContextBenchPipeline,
    EvaluationResult,
    QuestionResult,
)
from src.benchmarks.contextbench.wrapper import (
    ContextBenchAgent,
    FileOperation,
    OperationResult,
)

__all__ = [
    # Dataset
    "ContextBenchDataset",
    "ContextBenchQuestion",
    "ContextBenchFile",
    "load_contextbench",
    "load_contextbench_from_file",
    # Wrapper
    "ContextBenchAgent",
    "FileOperation",
    "OperationResult",
    # Pipeline
    "ContextBenchPipeline",
    "QuestionResult",
    "EvaluationResult",
    # Metrics
    "ContextBenchMetrics",
    "MetricsCalculator",
]
