"""Context-Bench metrics calculation.

This module provides metrics calculation and comparison functionality
for Context-Bench evaluation results.

Key metrics:
- Overall accuracy
- Per-category accuracy
- Multi-hop performance
- Cost efficiency (accuracy per dollar)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.benchmarks.contextbench.dataset import QuestionCategory
from src.benchmarks.contextbench.pipeline import EvaluationResult, QuestionResult
from src.evaluation.statistics import (
    ComparisonResult,
    ConfidenceInterval,
    StatisticalAnalyzer,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ContextBenchMetrics:
    """Complete metrics for Context-Bench evaluation.

    Attributes:
        accuracy: Overall accuracy
        accuracy_ci: Bootstrap confidence interval
        total_questions: Total questions evaluated
        correct_count: Number correct
        total_cost: Total estimated cost
        cost_efficiency: Accuracy per dollar spent
        multi_hop_accuracy: Accuracy on multi-hop questions
        single_hop_accuracy: Accuracy on single-hop questions
        category_accuracies: Accuracy per category
        avg_operations: Average operations per question
    """

    accuracy: float
    accuracy_ci: ConfidenceInterval | None
    total_questions: int
    correct_count: int
    total_cost: float
    cost_efficiency: float
    multi_hop_accuracy: float | None
    single_hop_accuracy: float | None
    category_accuracies: dict[str, float]
    avg_operations: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "accuracy": self.accuracy,
            "accuracy_ci": self.accuracy_ci.to_dict() if self.accuracy_ci else None,
            "total_questions": self.total_questions,
            "correct_count": self.correct_count,
            "total_cost": self.total_cost,
            "cost_efficiency": self.cost_efficiency,
            "multi_hop_accuracy": self.multi_hop_accuracy,
            "single_hop_accuracy": self.single_hop_accuracy,
            "category_accuracies": self.category_accuracies,
            "avg_operations": self.avg_operations,
        }


@dataclass(slots=True)
class MetricsCalculator:
    """Calculator for Context-Bench metrics.

    Attributes:
        analyzer: Statistical analyzer for CI and comparisons
        n_bootstrap: Number of bootstrap iterations
        confidence_level: Confidence level for intervals
    """

    analyzer: StatisticalAnalyzer = field(default_factory=StatisticalAnalyzer)
    n_bootstrap: int = 2000
    confidence_level: float = 0.95

    def _compute_accuracy_ci(
        self,
        results: list[QuestionResult],
    ) -> ConfidenceInterval | None:
        """Compute bootstrap CI for accuracy."""
        if len(results) < 2:
            return None

        scores = [1.0 if r.correct else 0.0 for r in results]

        try:
            return self.analyzer.bootstrap_ci(np.array(scores))
        except Exception as e:
            logger.warning(f"Failed to compute CI: {e}")
            return None

    def calculate_metrics(
        self,
        result: EvaluationResult,
    ) -> ContextBenchMetrics:
        """Calculate comprehensive metrics from evaluation results.

        Args:
            result: Complete evaluation result

        Returns:
            ContextBenchMetrics with all computed metrics
        """
        # Compute accuracy CI
        accuracy_ci = self._compute_accuracy_ci(result.question_results)

        # Cost efficiency (accuracy per dollar, higher is better)
        cost_efficiency = 0.0
        if result.total_cost > 0:
            cost_efficiency = result.accuracy / result.total_cost

        # Compute hop-specific accuracies
        single_hop = [r for r in result.question_results if r.hop_count == 1]
        multi_hop = [r for r in result.question_results if r.hop_count > 1]

        single_hop_accuracy = None
        if single_hop:
            single_hop_accuracy = sum(1 for r in single_hop if r.correct) / len(single_hop)

        multi_hop_accuracy = None
        if multi_hop:
            multi_hop_accuracy = sum(1 for r in multi_hop if r.correct) / len(multi_hop)

        # Category accuracies
        category_accuracies: dict[str, float] = {}
        for cat in QuestionCategory:
            cat_results = [r for r in result.question_results if r.category == cat]
            if cat_results:
                category_accuracies[cat.value] = sum(1 for r in cat_results if r.correct) / len(
                    cat_results
                )

        return ContextBenchMetrics(
            accuracy=result.accuracy,
            accuracy_ci=accuracy_ci,
            total_questions=result.total_questions,
            correct_count=result.correct_count,
            total_cost=result.total_cost,
            cost_efficiency=cost_efficiency,
            multi_hop_accuracy=multi_hop_accuracy,
            single_hop_accuracy=single_hop_accuracy,
            category_accuracies=category_accuracies,
            avg_operations=result.avg_operations,
        )

    def compare_conditions(
        self,
        result_a: EvaluationResult,
        result_b: EvaluationResult,
        condition_a_name: str = "condition_a",
        condition_b_name: str = "condition_b",
    ) -> dict[str, ComparisonResult]:
        """Compare two experimental conditions.

        Args:
            result_a: Results from first condition
            result_b: Results from second condition
            condition_a_name: Label for first condition
            condition_b_name: Label for second condition

        Returns:
            Dictionary mapping comparison type to ComparisonResult
        """
        comparisons: dict[str, ComparisonResult] = {}

        # Overall comparison
        scores_a = [1.0 if r.correct else 0.0 for r in result_a.question_results]
        scores_b = [1.0 if r.correct else 0.0 for r in result_b.question_results]

        if len(scores_a) >= 2 and len(scores_b) >= 2:
            comparisons["overall"] = self.analyzer.compare_conditions(
                condition_a=scores_a,
                condition_b=scores_b,
                condition_a_name=condition_a_name,
                condition_b_name=condition_b_name,
            )

        # Multi-hop comparison
        multi_a = [r for r in result_a.question_results if r.hop_count > 1]
        multi_b = [r for r in result_b.question_results if r.hop_count > 1]

        if len(multi_a) >= 2 and len(multi_b) >= 2:
            scores_a = [1.0 if r.correct else 0.0 for r in multi_a]
            scores_b = [1.0 if r.correct else 0.0 for r in multi_b]
            comparisons["multi_hop"] = self.analyzer.compare_conditions(
                condition_a=scores_a,
                condition_b=scores_b,
                condition_a_name=condition_a_name,
                condition_b_name=condition_b_name,
            )

        # Per-category comparisons
        for cat in QuestionCategory:
            cat_a = [r for r in result_a.question_results if r.category == cat]
            cat_b = [r for r in result_b.question_results if r.category == cat]

            if len(cat_a) >= 2 and len(cat_b) >= 2:
                scores_a = [1.0 if r.correct else 0.0 for r in cat_a]
                scores_b = [1.0 if r.correct else 0.0 for r in cat_b]
                comparisons[cat.value] = self.analyzer.compare_conditions(
                    condition_a=scores_a,
                    condition_b=scores_b,
                    condition_a_name=condition_a_name,
                    condition_b_name=condition_b_name,
                )

        return comparisons

    def compare_cost_efficiency(
        self,
        result_a: EvaluationResult,
        result_b: EvaluationResult,
    ) -> dict[str, Any]:
        """Compare cost efficiency between conditions.

        Args:
            result_a: First condition results
            result_b: Second condition results

        Returns:
            Dictionary with cost efficiency comparison
        """
        metrics_a = self.calculate_metrics(result_a)
        metrics_b = self.calculate_metrics(result_b)

        return {
            "condition_a": {
                "accuracy": metrics_a.accuracy,
                "total_cost": metrics_a.total_cost,
                "cost_efficiency": metrics_a.cost_efficiency,
            },
            "condition_b": {
                "accuracy": metrics_b.accuracy,
                "total_cost": metrics_b.total_cost,
                "cost_efficiency": metrics_b.cost_efficiency,
            },
            "efficiency_ratio": (
                metrics_a.cost_efficiency / metrics_b.cost_efficiency
                if metrics_b.cost_efficiency > 0
                else float("inf")
            ),
        }

    def format_summary(
        self,
        metrics: ContextBenchMetrics,
    ) -> str:
        """Format metrics as human-readable summary.

        Args:
            metrics: Computed metrics

        Returns:
            Formatted summary string
        """
        lines: list[str] = []
        lines.append("# Context-Bench Results\n")

        # Overall
        lines.append("## Overall Performance\n")
        ci_str = ""
        if metrics.accuracy_ci:
            ci_str = f" [{metrics.accuracy_ci.lower:.1%}, {metrics.accuracy_ci.upper:.1%}]"
        lines.append(f"- **Accuracy**: {metrics.accuracy:.1%}{ci_str}")
        lines.append(f"- **Total Questions**: {metrics.total_questions}")
        lines.append(f"- **Total Cost**: ${metrics.total_cost:.4f}")
        lines.append(f"- **Cost Efficiency**: {metrics.cost_efficiency:.2f} acc/$")
        lines.append(f"- **Avg Operations**: {metrics.avg_operations:.1f}")
        lines.append("")

        # Hop breakdown
        lines.append("## Performance by Hop Count\n")
        if metrics.single_hop_accuracy is not None:
            lines.append(f"- **Single-hop**: {metrics.single_hop_accuracy:.1%}")
        if metrics.multi_hop_accuracy is not None:
            lines.append(f"- **Multi-hop**: {metrics.multi_hop_accuracy:.1%}")
        lines.append("")

        # Category breakdown
        lines.append("## Performance by Category\n")
        lines.append("| Category | Accuracy |")
        lines.append("|----------|----------|")
        for cat, acc in sorted(metrics.category_accuracies.items()):
            lines.append(f"| {cat} | {acc:.1%} |")
        lines.append("")

        return "\n".join(lines)
