"""MemoryAgentBench metrics calculation.

This module provides metrics calculation and comparison functionality
for MemoryAgentBench evaluation results.

Key metrics:
- Per-competency accuracy
- Conflict resolution accuracy (primary metric)
- Difficulty-stratified performance
- Statistical comparisons between conditions
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.benchmarks.memoryagentbench.dataset import Competency, DifficultyLevel
from src.benchmarks.memoryagentbench.pipeline import (
    CompetencyResult,
    QuestionResult,
    SplitResult,
)
from src.evaluation.statistics import (
    ComparisonResult,
    ConfidenceInterval,
    StatisticalAnalyzer,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CompetencyMetrics:
    """Metrics for a single competency.

    Attributes:
        competency: The competency measured
        accuracy: Overall accuracy for this competency
        accuracy_ci: Bootstrap confidence interval for accuracy
        total_questions: Number of questions evaluated
        correct_count: Number of correct answers
        single_hop_accuracy: Accuracy on single-hop questions
        multi_hop_accuracy: Accuracy on multi-hop questions
        retrieval_utilization: Avg retrieved memories per question
    """

    competency: Competency
    accuracy: float
    accuracy_ci: ConfidenceInterval | None
    total_questions: int
    correct_count: int
    single_hop_accuracy: float | None
    multi_hop_accuracy: float | None
    retrieval_utilization: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "competency": self.competency.value,
            "competency_short": self.competency.short_name,
            "accuracy": self.accuracy,
            "accuracy_ci": self.accuracy_ci.to_dict() if self.accuracy_ci else None,
            "total_questions": self.total_questions,
            "correct_count": self.correct_count,
            "single_hop_accuracy": self.single_hop_accuracy,
            "multi_hop_accuracy": self.multi_hop_accuracy,
            "retrieval_utilization": self.retrieval_utilization,
        }


@dataclass(frozen=True, slots=True)
class MemoryAgentBenchMetrics:
    """Complete metrics for MemoryAgentBench evaluation.

    Attributes:
        overall_accuracy: Accuracy across all competencies
        overall_accuracy_ci: Bootstrap CI for overall accuracy
        conflict_resolution_metrics: Metrics for CR (primary focus)
        competency_metrics: Metrics per competency
        total_questions: Total questions evaluated
        single_hop_overall: Overall single-hop accuracy
        multi_hop_overall: Overall multi-hop accuracy
    """

    overall_accuracy: float
    overall_accuracy_ci: ConfidenceInterval | None
    conflict_resolution_metrics: CompetencyMetrics | None
    competency_metrics: dict[Competency, CompetencyMetrics]
    total_questions: int
    single_hop_overall: float | None
    multi_hop_overall: float | None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "overall_accuracy": self.overall_accuracy,
            "overall_accuracy_ci": (
                self.overall_accuracy_ci.to_dict() if self.overall_accuracy_ci else None
            ),
            "conflict_resolution_metrics": (
                self.conflict_resolution_metrics.to_dict()
                if self.conflict_resolution_metrics
                else None
            ),
            "competency_metrics": {
                c.value: m.to_dict() for c, m in self.competency_metrics.items()
            },
            "total_questions": self.total_questions,
            "single_hop_overall": self.single_hop_overall,
            "multi_hop_overall": self.multi_hop_overall,
        }


@dataclass(slots=True)
class MetricsCalculator:
    """Calculator for MemoryAgentBench metrics.

    Computes accuracy metrics, confidence intervals, and statistical
    comparisons between experimental conditions.

    Attributes:
        analyzer: Statistical analyzer for CI and comparisons
        n_bootstrap: Number of bootstrap iterations for CI
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

        # Convert to binary scores
        scores = [1.0 if r.correct else 0.0 for r in results]

        try:
            return self.analyzer.bootstrap_ci(np.array(scores))
        except Exception as e:
            logger.warning(f"Failed to compute accuracy CI: {e}")
            return None

    def _compute_competency_metrics(
        self,
        result: CompetencyResult,
    ) -> CompetencyMetrics:
        """Compute metrics for a single competency."""
        # Compute accuracy CI
        accuracy_ci = self._compute_accuracy_ci(result.question_results)

        # Compute difficulty-specific accuracies
        single_hop_results = [
            r for r in result.question_results if r.difficulty == DifficultyLevel.SINGLE_HOP
        ]
        multi_hop_results = [
            r for r in result.question_results if r.difficulty == DifficultyLevel.MULTI_HOP
        ]

        single_hop_accuracy = None
        if single_hop_results:
            single_hop_accuracy = sum(1 for r in single_hop_results if r.correct) / len(
                single_hop_results
            )

        multi_hop_accuracy = None
        if multi_hop_results:
            multi_hop_accuracy = sum(1 for r in multi_hop_results if r.correct) / len(
                multi_hop_results
            )

        # Compute retrieval utilization
        total_retrieved = sum(r.retrieved_count for r in result.question_results)
        retrieval_utilization = (
            total_retrieved / result.total_questions if result.total_questions > 0 else 0
        )

        return CompetencyMetrics(
            competency=result.competency,
            accuracy=result.accuracy,
            accuracy_ci=accuracy_ci,
            total_questions=result.total_questions,
            correct_count=result.correct_count,
            single_hop_accuracy=single_hop_accuracy,
            multi_hop_accuracy=multi_hop_accuracy,
            retrieval_utilization=retrieval_utilization,
        )

    def calculate_metrics(
        self,
        result: SplitResult,
    ) -> MemoryAgentBenchMetrics:
        """Calculate comprehensive metrics from evaluation results.

        Args:
            result: Complete evaluation result

        Returns:
            MemoryAgentBenchMetrics with all computed metrics
        """
        # Compute per-competency metrics
        competency_metrics: dict[Competency, CompetencyMetrics] = {}
        for competency, comp_result in result.competency_results.items():
            competency_metrics[competency] = self._compute_competency_metrics(comp_result)

        # Get conflict resolution metrics specifically
        cr_metrics = competency_metrics.get(Competency.CONFLICT_RESOLUTION)

        # Collect all results for overall CI
        all_results: list[QuestionResult] = []
        for comp_result in result.competency_results.values():
            all_results.extend(comp_result.question_results)

        # Compute overall accuracy CI
        overall_ci = self._compute_accuracy_ci(all_results)

        # Compute overall difficulty-specific accuracies
        single_hop_all = [r for r in all_results if r.difficulty == DifficultyLevel.SINGLE_HOP]
        multi_hop_all = [r for r in all_results if r.difficulty == DifficultyLevel.MULTI_HOP]

        single_hop_overall = None
        if single_hop_all:
            single_hop_overall = sum(1 for r in single_hop_all if r.correct) / len(single_hop_all)

        multi_hop_overall = None
        if multi_hop_all:
            multi_hop_overall = sum(1 for r in multi_hop_all if r.correct) / len(multi_hop_all)

        return MemoryAgentBenchMetrics(
            overall_accuracy=result.overall_accuracy,
            overall_accuracy_ci=overall_ci,
            conflict_resolution_metrics=cr_metrics,
            competency_metrics=competency_metrics,
            total_questions=result.total_questions,
            single_hop_overall=single_hop_overall,
            multi_hop_overall=multi_hop_overall,
        )

    def compare_conditions(
        self,
        result_a: SplitResult,
        result_b: SplitResult,
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

        # Compare overall accuracy
        scores_a = [
            1.0 if r.correct else 0.0
            for comp_result in result_a.competency_results.values()
            for r in comp_result.question_results
        ]
        scores_b = [
            1.0 if r.correct else 0.0
            for comp_result in result_b.competency_results.values()
            for r in comp_result.question_results
        ]

        if len(scores_a) >= 2 and len(scores_b) >= 2:
            comparisons["overall"] = self.analyzer.compare_conditions(
                condition_a=scores_a,
                condition_b=scores_b,
                condition_a_name=condition_a_name,
                condition_b_name=condition_b_name,
            )

        # Compare conflict resolution specifically
        cr_a = result_a.competency_results.get(Competency.CONFLICT_RESOLUTION)
        cr_b = result_b.competency_results.get(Competency.CONFLICT_RESOLUTION)

        if cr_a and cr_b:
            cr_scores_a = [1.0 if r.correct else 0.0 for r in cr_a.question_results]
            cr_scores_b = [1.0 if r.correct else 0.0 for r in cr_b.question_results]

            if len(cr_scores_a) >= 2 and len(cr_scores_b) >= 2:
                comparisons["conflict_resolution"] = self.analyzer.compare_conditions(
                    condition_a=cr_scores_a,
                    condition_b=cr_scores_b,
                    condition_a_name=condition_a_name,
                    condition_b_name=condition_b_name,
                )

        # Compare per-competency
        for competency in Competency:
            comp_a = result_a.competency_results.get(competency)
            comp_b = result_b.competency_results.get(competency)

            if comp_a and comp_b:
                comp_scores_a = [1.0 if r.correct else 0.0 for r in comp_a.question_results]
                comp_scores_b = [1.0 if r.correct else 0.0 for r in comp_b.question_results]

                if len(comp_scores_a) >= 2 and len(comp_scores_b) >= 2:
                    comparisons[competency.short_name] = self.analyzer.compare_conditions(
                        condition_a=comp_scores_a,
                        condition_b=comp_scores_b,
                        condition_a_name=condition_a_name,
                        condition_b_name=condition_b_name,
                    )

        return comparisons

    def compare_difficulty_levels(
        self,
        result: SplitResult,
    ) -> ComparisonResult | None:
        """Compare single-hop vs multi-hop performance.

        This is particularly interesting for conflict resolution,
        where multi-hop is much harder.

        Args:
            result: Evaluation result to analyze

        Returns:
            ComparisonResult comparing difficulty levels, or None if insufficient data
        """
        single_hop_scores: list[float] = []
        multi_hop_scores: list[float] = []

        for comp_result in result.competency_results.values():
            for r in comp_result.question_results:
                if r.difficulty == DifficultyLevel.SINGLE_HOP:
                    single_hop_scores.append(1.0 if r.correct else 0.0)
                elif r.difficulty == DifficultyLevel.MULTI_HOP:
                    multi_hop_scores.append(1.0 if r.correct else 0.0)

        if len(single_hop_scores) < 2 or len(multi_hop_scores) < 2:
            return None

        return self.analyzer.compare_conditions(
            condition_a=single_hop_scores,
            condition_b=multi_hop_scores,
            condition_a_name="single_hop",
            condition_b_name="multi_hop",
        )

    def format_summary(
        self,
        metrics: MemoryAgentBenchMetrics,
    ) -> str:
        """Format metrics as human-readable summary.

        Args:
            metrics: Computed metrics

        Returns:
            Formatted summary string
        """
        lines: list[str] = []
        lines.append("# MemoryAgentBench Results\n")

        # Overall
        lines.append("## Overall Performance\n")
        ci_str = ""
        if metrics.overall_accuracy_ci:
            ci_str = (
                f" [{metrics.overall_accuracy_ci.lower:.1%}, "
                f"{metrics.overall_accuracy_ci.upper:.1%}]"
            )
        lines.append(f"- **Accuracy**: {metrics.overall_accuracy:.1%}{ci_str}")
        lines.append(f"- **Total Questions**: {metrics.total_questions}")

        if metrics.single_hop_overall is not None:
            lines.append(f"- **Single-hop Accuracy**: {metrics.single_hop_overall:.1%}")
        if metrics.multi_hop_overall is not None:
            lines.append(f"- **Multi-hop Accuracy**: {metrics.multi_hop_overall:.1%}")

        lines.append("")

        # Conflict Resolution (primary)
        if metrics.conflict_resolution_metrics:
            cr = metrics.conflict_resolution_metrics
            lines.append("## Conflict Resolution (Primary Metric)\n")
            cr_ci_str = ""
            if cr.accuracy_ci:
                cr_ci_str = f" [{cr.accuracy_ci.lower:.1%}, {cr.accuracy_ci.upper:.1%}]"
            lines.append(f"- **Accuracy**: {cr.accuracy:.1%}{cr_ci_str}")
            if cr.single_hop_accuracy is not None:
                lines.append(f"- **Single-hop**: {cr.single_hop_accuracy:.1%}")
            if cr.multi_hop_accuracy is not None:
                lines.append(f"- **Multi-hop**: {cr.multi_hop_accuracy:.1%}")
            lines.append("")

        # Per-competency
        lines.append("## Per-Competency Results\n")
        lines.append("| Competency | Accuracy | Questions |")
        lines.append("|------------|----------|-----------|")

        for comp, comp_metrics in sorted(
            metrics.competency_metrics.items(),
            key=lambda x: x[0].value,
        ):
            lines.append(
                f"| {comp.short_name} | {comp_metrics.accuracy:.1%} | "
                f"{comp_metrics.total_questions} |"
            )

        lines.append("")

        return "\n".join(lines)
