"""LoCoMo metrics calculation.

This module provides structured metrics calculation and reporting for
LoCoMo benchmark results, including:
- Per-category accuracy breakdown (5 QA categories)
- Aggregate accuracy with confidence intervals
- Per-conversation difficulty analysis
- Adversarial question handling metrics
- Statistical analysis integration

The module follows the metrics requirements from IMPLEMENTATION_PLAN.md Task 1.4.4.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from src.benchmarks.locomo.pipeline import AssessmentResult, QuestionResult
    from src.evaluation.statistics import StatisticalAnalyzer

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CategoryMetricsReport:
    """Detailed metrics for a single QA category.

    LoCoMo has 5 QA categories testing different memory aspects:
    - Category 1 (IDENTITY): Basic identity/background facts about speakers
    - Category 2 (TEMPORAL): When events occurred (temporal reasoning)
    - Category 3 (INFERENCE): Prediction/inference based on context
    - Category 4 (CONTEXTUAL): Detailed contextual questions about events
    - Category 5 (ADVERSARIAL): Questions with intentionally incorrect premises

    Attributes:
        category_name: The QA category name
        category_number: The category number (1-5)
        total_questions: Number of questions in this category
        correct_count: Number answered correctly
        partial_count: Number answered partially correctly
        accuracy: Proportion correct
        mean_score: Mean score (0.0 to 1.0)
        mean_latency_ms: Average response latency
        confidence_interval: Optional 95% CI tuple (lower, upper)
    """

    category_name: str
    category_number: int
    total_questions: int
    correct_count: int
    partial_count: int
    accuracy: float
    mean_score: float
    mean_latency_ms: float
    confidence_interval: tuple[float, float] | None = None


@dataclass(slots=True)
class AdversarialMetrics:
    """Metrics specifically for adversarial question handling (Category 5).

    LoCoMo's Category 5 contains questions with false premises that
    the agent should identify rather than answer directly.

    Attributes:
        total_adversarial: Total adversarial questions
        correctly_identified: Questions where false premise was identified
        incorrectly_answered: Questions where agent answered despite false premise
        identification_rate: Rate of correctly identifying false premises
        false_acceptance_rate: Rate of incorrectly accepting false premises
    """

    total_adversarial: int
    correctly_identified: int
    incorrectly_answered: int
    identification_rate: float
    false_acceptance_rate: float


@dataclass(slots=True)
class ConversationDifficultyMetrics:
    """Metrics for conversation difficulty analysis.

    Analyzes how conversation properties affect performance.

    Attributes:
        conversation_id: The conversation identifier
        num_sessions: Number of sessions in the conversation
        num_turns: Total turns across all sessions
        token_count_estimate: Estimated token count
        questions_assessed: Number of questions from this conversation
        accuracy: Accuracy for this conversation
        mean_score: Mean score for this conversation
        difficulty_score: Computed difficulty (0-1, higher = harder)
    """

    conversation_id: str
    num_sessions: int
    num_turns: int
    token_count_estimate: int
    questions_assessed: int
    accuracy: float
    mean_score: float
    difficulty_score: float


@dataclass(slots=True)
class LoCoMoMetrics:
    """Complete metrics for a LoCoMo benchmark run.

    Provides structured access to all metrics with optional statistical
    analysis integration for confidence intervals.

    Attributes:
        total_questions: Total number of questions assessed
        total_conversations: Number of conversations assessed
        aggregate_accuracy: Overall accuracy
        aggregate_mean_score: Overall mean score
        category_metrics: Per-category breakdown
        adversarial_metrics: Adversarial question handling analysis
        conversation_difficulty: Per-conversation difficulty analysis
        aggregate_ci: Optional 95% CI for aggregate accuracy
        latency_stats: Latency statistics
        metadata: Additional metadata
    """

    total_questions: int
    total_conversations: int
    aggregate_accuracy: float
    aggregate_mean_score: float
    category_metrics: dict[str, CategoryMetricsReport]
    adversarial_metrics: AdversarialMetrics
    conversation_difficulty: list[ConversationDifficultyMetrics]
    aggregate_ci: tuple[float, float] | None = None
    latency_stats: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to a dictionary for serialization."""
        return {
            "total_questions": self.total_questions,
            "total_conversations": self.total_conversations,
            "aggregate_accuracy": self.aggregate_accuracy,
            "aggregate_mean_score": self.aggregate_mean_score,
            "aggregate_ci": self.aggregate_ci,
            "category_metrics": {
                name: {
                    "category_name": m.category_name,
                    "category_number": m.category_number,
                    "total_questions": m.total_questions,
                    "correct_count": m.correct_count,
                    "partial_count": m.partial_count,
                    "accuracy": m.accuracy,
                    "mean_score": m.mean_score,
                    "mean_latency_ms": m.mean_latency_ms,
                    "confidence_interval": m.confidence_interval,
                }
                for name, m in self.category_metrics.items()
            },
            "adversarial_metrics": {
                "total_adversarial": self.adversarial_metrics.total_adversarial,
                "correctly_identified": self.adversarial_metrics.correctly_identified,
                "incorrectly_answered": self.adversarial_metrics.incorrectly_answered,
                "identification_rate": self.adversarial_metrics.identification_rate,
                "false_acceptance_rate": self.adversarial_metrics.false_acceptance_rate,
            },
            "conversation_difficulty": [
                {
                    "conversation_id": c.conversation_id,
                    "num_sessions": c.num_sessions,
                    "num_turns": c.num_turns,
                    "token_count_estimate": c.token_count_estimate,
                    "questions_assessed": c.questions_assessed,
                    "accuracy": c.accuracy,
                    "mean_score": c.mean_score,
                    "difficulty_score": c.difficulty_score,
                }
                for c in self.conversation_difficulty
            ],
            "latency_stats": self.latency_stats,
            "metadata": self.metadata,
        }

    def get_summary(self) -> str:
        """Get a human-readable summary of the metrics."""
        lines = [
            "LoCoMo Benchmark Results",
            "=" * 50,
            f"Total Questions: {self.total_questions}",
            f"Total Conversations: {self.total_conversations}",
            f"Aggregate Accuracy: {self.aggregate_accuracy:.2%}",
            f"Aggregate Mean Score: {self.aggregate_mean_score:.3f}",
        ]

        if self.aggregate_ci:
            lines.append(f"95% CI: [{self.aggregate_ci[0]:.2%}, {self.aggregate_ci[1]:.2%}]")

        lines.append("")
        lines.append("Per-Category Results:")
        lines.append("-" * 50)
        for name, metrics in sorted(
            self.category_metrics.items(), key=lambda x: x[1].category_number
        ):
            ci_str = ""
            if metrics.confidence_interval:
                ci_str = (
                    f" (CI: [{metrics.confidence_interval[0]:.2%}, "
                    f"{metrics.confidence_interval[1]:.2%}])"
                )
            lines.append(
                f"  {name}: {metrics.accuracy:.2%} "
                f"({metrics.correct_count}/{metrics.total_questions}){ci_str}"
            )

        lines.append("")
        lines.append("Adversarial Question Handling:")
        lines.append("-" * 50)
        adv = self.adversarial_metrics
        lines.append(f"  Total Adversarial: {adv.total_adversarial}")
        lines.append(
            f"  Correctly Identified: {adv.correctly_identified} ({adv.identification_rate:.2%})"
        )
        lines.append(f"  False Acceptance Rate: {adv.false_acceptance_rate:.2%}")

        if self.conversation_difficulty:
            lines.append("")
            lines.append("Conversation Difficulty Analysis:")
            lines.append("-" * 50)
            # Sort by difficulty score descending
            sorted_convs = sorted(
                self.conversation_difficulty, key=lambda x: x.difficulty_score, reverse=True
            )
            for conv in sorted_convs[:5]:  # Top 5 hardest
                lines.append(
                    f"  {conv.conversation_id}: "
                    f"difficulty={conv.difficulty_score:.3f}, "
                    f"accuracy={conv.accuracy:.2%}, "
                    f"sessions={conv.num_sessions}"
                )

        return "\n".join(lines)


class LoCoMoMetricsCalculator:
    """Calculator for LoCoMo benchmark metrics.

    Processes assessment results to compute comprehensive metrics
    with optional statistical analysis for confidence intervals.

    Example:
        ```python
        from src.benchmarks.locomo import LoCoMoPipeline
        from src.benchmarks.locomo.metrics import LoCoMoMetricsCalculator
        from src.evaluation.statistics import StatisticalAnalyzer

        # Run pipeline
        results = pipeline.run(dataset)

        # Calculate metrics
        calculator = LoCoMoMetricsCalculator()
        metrics = calculator.calculate(results)

        # With statistical analysis
        analyzer = StatisticalAnalyzer()
        metrics_with_ci = calculator.calculate(results, analyzer)
        ```
    """

    # Category name to number mapping (matches QACategory enum)
    CATEGORY_NUMBERS: dict[str, int] = {
        "IDENTITY": 1,
        "TEMPORAL": 2,
        "INFERENCE": 3,
        "CONTEXTUAL": 4,
        "ADVERSARIAL": 5,
    }

    def calculate(
        self,
        results: AssessmentResult,
        analyzer: StatisticalAnalyzer | None = None,
    ) -> LoCoMoMetrics:
        """Calculate comprehensive metrics from assessment results.

        Args:
            results: Assessment results from LoCoMoPipeline.run()
            analyzer: Optional statistical analyzer for confidence intervals

        Returns:
            LoCoMoMetrics with all computed metrics
        """
        # Calculate category metrics
        category_metrics = self._calculate_category_metrics(results.question_results, analyzer)

        # Calculate adversarial metrics
        adversarial_metrics = self._calculate_adversarial_metrics(results.question_results)

        # Calculate conversation difficulty
        conversation_difficulty = self._calculate_conversation_difficulty(results)

        # Calculate latency statistics
        latency_stats = self._calculate_latency_stats(results.question_results)

        # Calculate aggregate CI if analyzer provided
        aggregate_ci = None
        if analyzer and results.question_results:
            scores = [r.score for r in results.question_results]
            ci = analyzer.bootstrap_confidence_interval(np.array(scores), statistic=np.mean)
            aggregate_ci = (ci.lower, ci.upper)

        return LoCoMoMetrics(
            total_questions=results.total_questions,
            total_conversations=len(results.conversation_metrics),
            aggregate_accuracy=results.accuracy,
            aggregate_mean_score=results.mean_score,
            category_metrics=category_metrics,
            adversarial_metrics=adversarial_metrics,
            conversation_difficulty=conversation_difficulty,
            aggregate_ci=aggregate_ci,
            latency_stats=latency_stats,
            metadata={
                "ingestion_time_ms": results.ingestion_time_ms,
                "assessment_time_ms": results.assessment_time_ms,
                "started_at": results.started_at.isoformat(),
                "completed_at": results.completed_at.isoformat(),
            },
        )

    def _calculate_category_metrics(
        self,
        question_results: list[QuestionResult],
        analyzer: StatisticalAnalyzer | None = None,
    ) -> dict[str, CategoryMetricsReport]:
        """Calculate per-category metrics."""
        from collections import defaultdict

        # Group results by category
        by_category: dict[str, list[QuestionResult]] = defaultdict(list)
        for result in question_results:
            by_category[result.category_name].append(result)

        category_metrics = {}
        for cat_name, results in by_category.items():
            total = len(results)
            correct = sum(1 for r in results if r.is_correct)
            partial = sum(1 for r in results if r.is_partial)
            scores = [r.score for r in results]
            latencies = [r.latency_ms for r in results]

            accuracy = correct / total if total > 0 else 0.0
            mean_score = float(np.mean(scores)) if scores else 0.0
            mean_latency = float(np.mean(latencies)) if latencies else 0.0

            # Calculate CI if analyzer provided
            ci = None
            if analyzer and scores:
                ci_result = analyzer.bootstrap_confidence_interval(
                    np.array(scores), statistic=np.mean
                )
                ci = (ci_result.lower, ci_result.upper)

            category_metrics[cat_name] = CategoryMetricsReport(
                category_name=cat_name,
                category_number=self.CATEGORY_NUMBERS.get(cat_name, 0),
                total_questions=total,
                correct_count=correct,
                partial_count=partial,
                accuracy=accuracy,
                mean_score=mean_score,
                mean_latency_ms=mean_latency,
                confidence_interval=ci,
            )

        return category_metrics

    def _calculate_adversarial_metrics(
        self,
        question_results: list[QuestionResult],
    ) -> AdversarialMetrics:
        """Calculate adversarial question handling metrics."""
        adversarial_results = [r for r in question_results if r.is_adversarial]

        total = len(adversarial_results)
        if total == 0:
            return AdversarialMetrics(
                total_adversarial=0,
                correctly_identified=0,
                incorrectly_answered=0,
                identification_rate=1.0,  # No adversarial = perfect by default
                false_acceptance_rate=0.0,
            )

        correctly_identified = sum(1 for r in adversarial_results if r.adversarial_handled)
        incorrectly_answered = total - correctly_identified

        return AdversarialMetrics(
            total_adversarial=total,
            correctly_identified=correctly_identified,
            incorrectly_answered=incorrectly_answered,
            identification_rate=correctly_identified / total,
            false_acceptance_rate=incorrectly_answered / total,
        )

    def _calculate_conversation_difficulty(
        self,
        results: AssessmentResult,
    ) -> list[ConversationDifficultyMetrics]:
        """Calculate per-conversation difficulty metrics.

        Difficulty is computed based on:
        - Number of sessions (more sessions = harder)
        - Number of turns (more turns = harder)
        - Token count estimate (more tokens = harder)
        - Accuracy achieved (lower accuracy = harder)
        """
        difficulty_metrics = []

        for conv_id, conv_metrics in results.conversation_metrics.items():
            # Compute difficulty score (0-1, higher = harder)
            # Weighted combination of factors
            sessions_factor = min(conv_metrics.sessions_ingested / 35, 1.0)  # Max 35 sessions
            turns_factor = min(conv_metrics.turns_ingested / 500, 1.0)  # Max ~500 turns

            # Token estimate (rough: ~4 chars per token, ~50 chars per turn)
            token_estimate = conv_metrics.turns_ingested * 50

            # Accuracy inversely correlates with difficulty
            accuracy_factor = 1.0 - conv_metrics.accuracy

            # Weighted difficulty score
            difficulty_score = 0.3 * sessions_factor + 0.3 * turns_factor + 0.4 * accuracy_factor

            difficulty_metrics.append(
                ConversationDifficultyMetrics(
                    conversation_id=conv_id,
                    num_sessions=conv_metrics.sessions_ingested,
                    num_turns=conv_metrics.turns_ingested,
                    token_count_estimate=token_estimate,
                    questions_assessed=conv_metrics.questions_assessed,
                    accuracy=conv_metrics.accuracy,
                    mean_score=conv_metrics.mean_score,
                    difficulty_score=difficulty_score,
                )
            )

        return difficulty_metrics

    def _calculate_latency_stats(
        self,
        question_results: list[QuestionResult],
    ) -> dict[str, float]:
        """Calculate latency statistics."""
        if not question_results:
            return {
                "mean_ms": 0.0,
                "median_ms": 0.0,
                "p95_ms": 0.0,
                "p99_ms": 0.0,
                "min_ms": 0.0,
                "max_ms": 0.0,
            }

        latencies = [r.latency_ms for r in question_results]
        return {
            "mean_ms": float(np.mean(latencies)),
            "median_ms": float(np.median(latencies)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "p99_ms": float(np.percentile(latencies, 99)),
            "min_ms": float(np.min(latencies)),
            "max_ms": float(np.max(latencies)),
        }


def calculate_metrics(
    results: AssessmentResult,
    analyzer: StatisticalAnalyzer | None = None,
) -> LoCoMoMetrics:
    """Convenience function to calculate LoCoMo metrics.

    Args:
        results: Assessment results from LoCoMoPipeline.run()
        analyzer: Optional statistical analyzer for confidence intervals

    Returns:
        LoCoMoMetrics with all computed metrics
    """
    calculator = LoCoMoMetricsCalculator()
    return calculator.calculate(results, analyzer)


def compare_metrics(
    metrics_a: LoCoMoMetrics,
    metrics_b: LoCoMoMetrics,
    label_a: str = "Condition A",
    label_b: str = "Condition B",
) -> dict[str, Any]:
    """Compare two sets of LoCoMo metrics.

    Useful for comparing git-notes vs no-memory conditions.

    Args:
        metrics_a: First set of metrics
        metrics_b: Second set of metrics
        label_a: Label for first condition
        label_b: Label for second condition

    Returns:
        Dictionary with comparison results
    """
    by_category: dict[str, dict[str, float | None]] = {}

    # Compare by category
    all_categories = set(metrics_a.category_metrics.keys()) | set(metrics_b.category_metrics.keys())
    for cat in all_categories:
        cat_a = metrics_a.category_metrics.get(cat)
        cat_b = metrics_b.category_metrics.get(cat)

        accuracy_a = cat_a.accuracy if cat_a else None
        accuracy_b = cat_b.accuracy if cat_b else None
        accuracy_diff = None
        if accuracy_a is not None and accuracy_b is not None:
            accuracy_diff = accuracy_a - accuracy_b

        by_category[cat] = {
            "accuracy_a": accuracy_a,
            "accuracy_b": accuracy_b,
            "accuracy_diff": accuracy_diff,
        }

    comparison: dict[str, Any] = {
        "labels": {"a": label_a, "b": label_b},
        "aggregate": {
            "accuracy_a": metrics_a.aggregate_accuracy,
            "accuracy_b": metrics_b.aggregate_accuracy,
            "accuracy_diff": metrics_a.aggregate_accuracy - metrics_b.aggregate_accuracy,
            "mean_score_a": metrics_a.aggregate_mean_score,
            "mean_score_b": metrics_b.aggregate_mean_score,
            "mean_score_diff": (metrics_a.aggregate_mean_score - metrics_b.aggregate_mean_score),
        },
        "by_category": by_category,
        "adversarial": {
            "identification_rate_a": metrics_a.adversarial_metrics.identification_rate,
            "identification_rate_b": metrics_b.adversarial_metrics.identification_rate,
            "identification_rate_diff": (
                metrics_a.adversarial_metrics.identification_rate
                - metrics_b.adversarial_metrics.identification_rate
            ),
        },
    }

    return comparison
