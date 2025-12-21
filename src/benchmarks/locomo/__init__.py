"""LoCoMo benchmark implementation.

This module provides the dataset loader, agent wrapper, evaluation pipeline,
and metrics calculation for the LoCoMo benchmark (Snap Research).

LoCoMo evaluates long-term conversational memory through:
- 10 multi-session conversations (300 turns, 9K tokens avg)
- 5 QA categories testing different memory aspects
- Event summarization capabilities
- Temporal reasoning assessment

Dataset: snap-research/locomo from GitHub
Paper: "Evaluating Very Long-Term Conversational Memory of LLM Agents"
"""

from src.benchmarks.locomo.dataset import (
    LoCoMoConversation,
    LoCoMoDataset,
    LoCoMoQuestion,
    LoCoMoSession,
    LoCoMoTurn,
    QACategory,
    load_locomo,
    load_locomo_from_file,
)
from src.benchmarks.locomo.metrics import (
    AdversarialMetrics,
    CategoryMetricsReport,
    ConversationDifficultyMetrics,
    LoCoMoMetrics,
    LoCoMoMetricsCalculator,
    calculate_metrics,
    compare_metrics,
)
from src.benchmarks.locomo.pipeline import (
    AssessmentResult,
    CategoryMetrics,
    ConversationMetrics,
    LoCoMoPipeline,
    QuestionResult,
)
from src.benchmarks.locomo.wrapper import (
    IngestionResult,
    LLMClient,
    LLMResponse,
    LoCoMoAgent,
    LoCoMoAnswer,
)

__all__ = [
    # Dataset classes
    "LoCoMoDataset",
    "LoCoMoConversation",
    "LoCoMoSession",
    "LoCoMoTurn",
    "LoCoMoQuestion",
    "QACategory",
    "load_locomo",
    "load_locomo_from_file",
    # Agent wrapper
    "LoCoMoAgent",
    "LoCoMoAnswer",
    "IngestionResult",
    "LLMClient",
    "LLMResponse",
    # Pipeline
    "LoCoMoPipeline",
    "QuestionResult",
    "CategoryMetrics",
    "ConversationMetrics",
    "AssessmentResult",
    # Metrics
    "LoCoMoMetrics",
    "LoCoMoMetricsCalculator",
    "CategoryMetricsReport",
    "AdversarialMetrics",
    "ConversationDifficultyMetrics",
    "calculate_metrics",
    "compare_metrics",
]
