"""LongMemEval metrics calculation.

This module provides structured metrics calculation and reporting for
LongMemEval benchmark results, including:
- Per-ability accuracy breakdown (5 memory abilities)
- Aggregate accuracy with confidence intervals
- Per-subset breakdown (S vs M)
- Statistical analysis integration

The module follows the metrics requirements from IMPLEMENTATION_PLAN.md Task 1.3.4.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from src.benchmarks.longmemeval.pipeline import AssessmentResult, QuestionResult
    from src.evaluation.statistics import StatisticalAnalyzer


@dataclass(slots=True)
class AbilityMetrics:
    """Metrics for a single memory ability (question type).

    Attributes:
        ability_name: The question type/ability name
        total_questions: Number of questions for this ability
        correct_count: Number answered correctly
        partial_count: Number answered partially correctly
        accuracy: Proportion correct
        mean_score: Mean score (0.0 to 1.0)
        confidence_interval: Optional 95% CI tuple (lower, upper)
    """

    ability_name: str
    total_questions: int
    correct_count: int
    partial_count: int
    accuracy: float
    mean_score: float
    confidence_interval: tuple[float, float] | None = None


@dataclass(slots=True)
class AbstentionMetrics:
    """Metrics specifically for abstention behavior.

    Attributes:
        total_abstention_expected: Questions where abstention was correct
        correct_abstentions: Agent correctly abstained
        false_abstentions: Agent abstained when shouldn't have
        missed_abstentions: Agent answered when should have abstained
        abstention_precision: TP / (TP + FP)
        abstention_recall: TP / (TP + FN)
        abstention_f1: Harmonic mean of precision and recall
    """

    total_abstention_expected: int
    correct_abstentions: int
    false_abstentions: int
    missed_abstentions: int
    abstention_precision: float
    abstention_recall: float
    abstention_f1: float


@dataclass(slots=True)
class LongMemEvalMetrics:
    """Complete metrics for a LongMemEval benchmark run.

    Provides structured access to all metrics with optional statistical
    analysis integration for confidence intervals.

    Attributes:
        subset: Dataset subset (S or M)
        total_questions: Total number of questions assessed
        aggregate_accuracy: Overall accuracy
        aggregate_mean_score: Overall mean score
        ability_metrics: Per-ability breakdown
        abstention_metrics: Abstention behavior analysis
        aggregate_ci: Optional 95% CI for aggregate accuracy
        latency_stats: Latency statistics
        metadata: Additional metadata
    """

    subset: str
    total_questions: int
    aggregate_accuracy: float
    aggregate_mean_score: float
    ability_metrics: dict[str, AbilityMetrics]
    abstention_metrics: AbstentionMetrics
    aggregate_ci: tuple[float, float] | None = None
    latency_stats: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary for serialization."""
        return {
            "subset": self.subset,
            "total_questions": self.total_questions,
            "aggregate_accuracy": self.aggregate_accuracy,
            "aggregate_mean_score": self.aggregate_mean_score,
            "aggregate_ci": self.aggregate_ci,
            "ability_metrics": {
                name: {
                    "ability_name": m.ability_name,
                    "total_questions": m.total_questions,
                    "correct_count": m.correct_count,
                    "partial_count": m.partial_count,
                    "accuracy": m.accuracy,
                    "mean_score": m.mean_score,
                    "confidence_interval": m.confidence_interval,
                }
                for name, m in self.ability_metrics.items()
            },
            "abstention_metrics": {
                "total_abstention_expected": self.abstention_metrics.total_abstention_expected,
                "correct_abstentions": self.abstention_metrics.correct_abstentions,
                "false_abstentions": self.abstention_metrics.false_abstentions,
                "missed_abstentions": self.abstention_metrics.missed_abstentions,
                "abstention_precision": self.abstention_metrics.abstention_precision,
                "abstention_recall": self.abstention_metrics.abstention_recall,
                "abstention_f1": self.abstention_metrics.abstention_f1,
            },
            "latency_stats": self.latency_stats,
            "metadata": self.metadata,
        }

    def format_report(self) -> str:
        """Generate a human-readable metrics report."""
        lines = [
            f"LongMemEval Metrics Report - Subset {self.subset}",
            "=" * 50,
            "",
            "Aggregate Metrics:",
            f"  Total Questions: {self.total_questions}",
            f"  Accuracy: {self.aggregate_accuracy:.2%}",
            f"  Mean Score: {self.aggregate_mean_score:.3f}",
        ]

        if self.aggregate_ci:
            lines.append(f"  95% CI: [{self.aggregate_ci[0]:.3f}, {self.aggregate_ci[1]:.3f}]")

        lines.extend(["", "Per-Ability Breakdown:", "-" * 40])

        for name, ability in sorted(self.ability_metrics.items()):
            ci_str = ""
            if ability.confidence_interval:
                ci_str = f" CI: [{ability.confidence_interval[0]:.3f}, {ability.confidence_interval[1]:.3f}]"
            lines.append(
                f"  {name}: {ability.accuracy:.2%} ({ability.correct_count}/{ability.total_questions}){ci_str}"
            )

        lines.extend(["", "Abstention Analysis:", "-" * 40])
        abst = self.abstention_metrics
        lines.extend(
            [
                f"  Expected Abstentions: {abst.total_abstention_expected}",
                f"  Correct Abstentions: {abst.correct_abstentions}",
                f"  False Abstentions: {abst.false_abstentions}",
                f"  Missed Abstentions: {abst.missed_abstentions}",
                f"  Precision: {abst.abstention_precision:.2%}",
                f"  Recall: {abst.abstention_recall:.2%}",
                f"  F1 Score: {abst.abstention_f1:.3f}",
            ]
        )

        if self.latency_stats:
            lines.extend(["", "Latency Statistics:", "-" * 40])
            for key, value in sorted(self.latency_stats.items()):
                lines.append(f"  {key}: {value:.1f}ms")

        return "\n".join(lines)


class MetricsCalculator:
    """Calculator for LongMemEval benchmark metrics.

    Transforms raw AssessmentResult into structured LongMemEvalMetrics
    with optional statistical analysis for confidence intervals.

    Example:
        ```python
        from src.benchmarks.longmemeval import BenchmarkPipeline, MetricsCalculator
        from src.statistics.analyzer import StatisticalAnalyzer

        # Run pipeline
        pipeline = BenchmarkPipeline(adapter, llm_client, judge)
        result = pipeline.run(dataset)

        # Calculate metrics
        analyzer = StatisticalAnalyzer()
        calculator = MetricsCalculator(analyzer)
        metrics = calculator.calculate(result)

        print(metrics.format_report())
        ```
    """

    def __init__(self, analyzer: StatisticalAnalyzer | None = None) -> None:
        """Initialize the metrics calculator.

        Args:
            analyzer: Optional statistical analyzer for confidence intervals
        """
        self._analyzer = analyzer

    def calculate(
        self,
        result: AssessmentResult,
        *,
        compute_confidence_intervals: bool = True,
    ) -> LongMemEvalMetrics:
        """Calculate comprehensive metrics from assessment results.

        Args:
            result: The assessment result to analyze
            compute_confidence_intervals: Whether to compute CIs (requires analyzer)

        Returns:
            LongMemEvalMetrics with all calculated metrics
        """
        # Calculate per-ability metrics
        ability_metrics = self._calculate_ability_metrics(
            result.question_results,
            compute_ci=compute_confidence_intervals and self._analyzer is not None,
        )

        # Calculate abstention metrics
        abstention_metrics = self._calculate_abstention_metrics(result.question_results)

        # Calculate latency statistics
        latency_stats = self._calculate_latency_stats(result.question_results)

        # Calculate aggregate CI if analyzer available
        aggregate_ci = None
        if compute_confidence_intervals and self._analyzer is not None:
            scores = np.array([r.score for r in result.question_results])
            if len(scores) > 0:
                ci_result = self._analyzer.bootstrap_ci(scores)
                aggregate_ci = (ci_result.lower, ci_result.upper)

        return LongMemEvalMetrics(
            subset=result.dataset_subset,
            total_questions=result.total_questions,
            aggregate_accuracy=result.accuracy,
            aggregate_mean_score=result.mean_score,
            ability_metrics=ability_metrics,
            abstention_metrics=abstention_metrics,
            aggregate_ci=aggregate_ci,
            latency_stats=latency_stats,
            metadata={
                "ingestion_time_ms": result.ingestion_time_ms,
                "assessment_time_ms": result.assessment_time_ms,
                "started_at": result.started_at.isoformat(),
                "completed_at": result.completed_at.isoformat(),
                **result.metadata,
            },
        )

    def _calculate_ability_metrics(
        self,
        results: list[QuestionResult],
        *,
        compute_ci: bool = False,
    ) -> dict[str, AbilityMetrics]:
        """Calculate metrics grouped by ability (question type)."""
        # Group by question type
        by_type: dict[str, list[QuestionResult]] = {}
        for r in results:
            if r.question_type not in by_type:
                by_type[r.question_type] = []
            by_type[r.question_type].append(r)

        ability_metrics: dict[str, AbilityMetrics] = {}

        for ability_name, ability_results in by_type.items():
            correct = sum(1 for r in ability_results if r.is_correct)
            partial = sum(1 for r in ability_results if r.is_partial)
            total = len(ability_results)
            scores = [r.score for r in ability_results]

            ci = None
            if compute_ci and self._analyzer is not None and scores:
                ci_result = self._analyzer.bootstrap_ci(np.array(scores))
                ci = (ci_result.lower, ci_result.upper)

            ability_metrics[ability_name] = AbilityMetrics(
                ability_name=ability_name,
                total_questions=total,
                correct_count=correct,
                partial_count=partial,
                accuracy=correct / total if total > 0 else 0.0,
                mean_score=sum(scores) / len(scores) if scores else 0.0,
                confidence_interval=ci,
            )

        return ability_metrics

    def _calculate_abstention_metrics(self, results: list[QuestionResult]) -> AbstentionMetrics:
        """Calculate abstention-specific metrics."""
        # Count abstention cases
        total_expected = sum(1 for r in results if r.is_abstention_expected)
        correct = sum(1 for r in results if r.is_abstention_expected and r.is_abstention_actual)
        false_positives = sum(
            1 for r in results if not r.is_abstention_expected and r.is_abstention_actual
        )
        missed = sum(1 for r in results if r.is_abstention_expected and not r.is_abstention_actual)

        # Calculate precision, recall, F1
        precision = (
            correct / (correct + false_positives) if (correct + false_positives) > 0 else 0.0
        )
        recall = correct / total_expected if total_expected > 0 else 1.0  # Perfect if none expected
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        return AbstentionMetrics(
            total_abstention_expected=total_expected,
            correct_abstentions=correct,
            false_abstentions=false_positives,
            missed_abstentions=missed,
            abstention_precision=precision,
            abstention_recall=recall,
            abstention_f1=f1,
        )

    def _calculate_latency_stats(self, results: list[QuestionResult]) -> dict[str, float]:
        """Calculate latency statistics."""
        if not results:
            return {}

        latencies = [r.latency_ms for r in results]
        sorted_latencies = sorted(latencies)
        n = len(sorted_latencies)

        return {
            "min": min(latencies),
            "max": max(latencies),
            "mean": sum(latencies) / n,
            "median": sorted_latencies[n // 2],
            "p95": sorted_latencies[int(n * 0.95)] if n >= 20 else sorted_latencies[-1],
            "p99": sorted_latencies[int(n * 0.99)] if n >= 100 else sorted_latencies[-1],
        }


def compare_results(
    results: list[AssessmentResult],
    labels: list[str],
    *,
    analyzer: StatisticalAnalyzer | None = None,
) -> dict[str, Any]:
    """Compare multiple assessment results (e.g., git-notes vs no-memory).

    Args:
        results: List of AssessmentResult to compare
        labels: Labels for each result (e.g., ["git-notes", "no-memory"])
        analyzer: Optional analyzer for statistical significance testing

    Returns:
        Dictionary with comparison metrics and optional significance tests
    """
    if len(results) != len(labels):
        raise ValueError("Number of results must match number of labels")

    comparison: dict[str, Any] = {
        "conditions": {},
        "differences": {},
    }

    # Calculate metrics for each condition
    calculator = MetricsCalculator(analyzer)
    for result, label in zip(results, labels, strict=True):
        metrics = calculator.calculate(result)
        comparison["conditions"][label] = {
            "accuracy": metrics.aggregate_accuracy,
            "mean_score": metrics.aggregate_mean_score,
            "ability_scores": {name: m.mean_score for name, m in metrics.ability_metrics.items()},
            "abstention_f1": metrics.abstention_metrics.abstention_f1,
        }

    # Calculate pairwise differences
    if len(results) >= 2:
        base_label = labels[0]
        base_accuracy = comparison["conditions"][base_label]["accuracy"]

        for label in labels[1:]:
            other_accuracy = comparison["conditions"][label]["accuracy"]
            diff = base_accuracy - other_accuracy
            comparison["differences"][f"{base_label}_vs_{label}"] = {
                "accuracy_diff": diff,
                "relative_improvement": diff / other_accuracy
                if other_accuracy > 0
                else float("inf"),
            }

            # Statistical significance if analyzer provided
            if analyzer is not None:
                # Get scores for paired comparison
                base_scores = [r.score for r in results[labels.index(base_label)].question_results]
                other_scores = [r.score for r in results[labels.index(label)].question_results]

                if len(base_scores) == len(other_scores):
                    comparison_result = analyzer.paired_comparison(
                        np.array(base_scores),
                        np.array(other_scores),
                        comparison_name=f"{base_label}_vs_{label}",
                    )
                    comparison["differences"][f"{base_label}_vs_{label}"]["statistical_test"] = {
                        "p_value": comparison_result.p_value,
                        "effect_size": comparison_result.effect_size,
                        "effect_interpretation": comparison_result.effect_size_interpretation(),
                        "is_significant": comparison_result.is_significant,
                        "ci_difference": comparison_result.ci_difference.to_dict(),
                    }

    return comparison
