"""Metrics calculation for Terminal-Bench 2.0 evaluations.

This module provides metrics calculation and comparison for
Terminal-Bench 2.0 evaluation results.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from src.evaluation.statistics import StatisticalAnalyzer

from src.benchmarks.terminalbench.runner import TaskResult, TrialResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TaskMetrics:
    """Metrics for a specific task category.

    Attributes:
        category: Task category or relevance type
        total_attempts: Total number of task attempts
        successful_attempts: Number of successful attempts
        success_rate: Proportion of successful attempts
        avg_execution_time: Average execution time in seconds
        memory_usage_rate: Proportion of attempts that used memory
    """

    category: str
    total_attempts: int
    successful_attempts: int
    success_rate: float
    avg_execution_time: float
    memory_usage_rate: float


@dataclass(frozen=True, slots=True)
class TerminalBenchMetrics:
    """Aggregated metrics for Terminal-Bench evaluation.

    Attributes:
        adapter_name: Name of the adapter used
        total_tasks: Total number of tasks attempted
        successful_tasks: Total successful tasks
        overall_success_rate: Overall success rate
        avg_execution_time: Average execution time
        memory_usage_rate: Rate of memory context usage
        by_category: Metrics broken down by category
        by_relevance: Metrics broken down by memory relevance
        by_difficulty: Metrics broken down by difficulty level
        ci_lower: Lower bound of 95% confidence interval
        ci_upper: Upper bound of 95% confidence interval
        metadata: Additional metrics metadata
    """

    adapter_name: str
    total_tasks: int
    successful_tasks: int
    overall_success_rate: float
    avg_execution_time: float
    memory_usage_rate: float
    by_category: dict[str, TaskMetrics]
    by_relevance: dict[str, TaskMetrics]
    by_difficulty: dict[int, TaskMetrics]
    ci_lower: float | None = None
    ci_upper: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    """Result of comparing two adapters.

    Attributes:
        adapter_a: First adapter name
        adapter_b: Second adapter name
        success_diff: Difference in success rates (A - B)
        time_diff: Difference in execution times (A - B)
        p_value: Statistical significance p-value
        effect_size: Effect size (Cohen's d)
        is_significant: Whether difference is statistically significant
        confidence_interval: 95% CI for the difference
    """

    adapter_a: str
    adapter_b: str
    success_diff: float
    time_diff: float
    p_value: float
    effect_size: float
    is_significant: bool
    confidence_interval: tuple[float, float]


class TerminalBenchMetricsCalculator:
    """Calculator for Terminal-Bench metrics.

    Computes aggregate metrics from trial results and supports
    statistical comparison between adapters.
    """

    def __init__(
        self,
        statistical_analyzer: StatisticalAnalyzer | None = None,
        confidence_level: float = 0.95,
    ) -> None:
        """Initialize the metrics calculator.

        Args:
            statistical_analyzer: Optional analyzer for statistical tests
            confidence_level: Confidence level for CIs
        """
        self.analyzer = statistical_analyzer
        self.confidence_level = confidence_level

    def calculate(self, trial_results: list[TrialResult]) -> TerminalBenchMetrics:
        """Calculate aggregate metrics from trial results.

        Args:
            trial_results: List of trial results

        Returns:
            TerminalBenchMetrics
        """
        if not trial_results:
            return TerminalBenchMetrics(
                adapter_name="unknown",
                total_tasks=0,
                successful_tasks=0,
                overall_success_rate=0.0,
                avg_execution_time=0.0,
                memory_usage_rate=0.0,
                by_category={},
                by_relevance={},
                by_difficulty={},
            )

        # Collect all task results
        all_results: list[TaskResult] = []
        for trial in trial_results:
            all_results.extend(trial.task_results)

        adapter_name = trial_results[0].adapter_name

        # Calculate overall metrics
        total = len(all_results)
        successful = sum(1 for r in all_results if r.success)
        success_rate = successful / total if total > 0 else 0.0

        # Average execution time
        exec_times = [r.execution_time_seconds for r in all_results]
        avg_time = sum(exec_times) / len(exec_times) if exec_times else 0.0

        # Memory usage rate
        memory_used = sum(1 for r in all_results if r.memory_context)
        memory_rate = memory_used / total if total > 0 else 0.0

        # Calculate by category
        by_category = self._calculate_by_category(all_results)

        # Calculate by relevance
        by_relevance = self._calculate_by_relevance(all_results)

        # Calculate by difficulty
        by_difficulty = self._calculate_by_difficulty(all_results)

        # Calculate confidence interval if analyzer available
        ci_lower = None
        ci_upper = None
        if self.analyzer is not None and len(trial_results) >= 3:
            trial_rates = [t.success_rate for t in trial_results]
            try:
                ci = self.analyzer.bootstrap_ci(np.array(trial_rates))
                ci_lower = ci.lower
                ci_upper = ci.upper
            except Exception as e:
                logger.warning(f"Could not compute CI: {e}")

        return TerminalBenchMetrics(
            adapter_name=adapter_name,
            total_tasks=total,
            successful_tasks=successful,
            overall_success_rate=success_rate,
            avg_execution_time=avg_time,
            memory_usage_rate=memory_rate,
            by_category=by_category,
            by_relevance=by_relevance,
            by_difficulty=by_difficulty,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            metadata={
                "n_trials": len(trial_results),
                "confidence_level": self.confidence_level,
            },
        )

    def _calculate_by_category(
        self,
        results: list[TaskResult],
    ) -> dict[str, TaskMetrics]:
        """Calculate metrics grouped by category.

        Args:
            results: Task results

        Returns:
            Metrics by category
        """
        # Group by category
        by_cat: dict[str, list[TaskResult]] = {}
        for r in results:
            cat = r.metadata.get("category", "unknown")
            if cat not in by_cat:
                by_cat[cat] = []
            by_cat[cat].append(r)

        return {cat: self._compute_group_metrics(cat, group) for cat, group in by_cat.items()}

    def _calculate_by_relevance(
        self,
        results: list[TaskResult],
    ) -> dict[str, TaskMetrics]:
        """Calculate metrics grouped by memory relevance.

        Args:
            results: Task results

        Returns:
            Metrics by relevance level
        """
        # Group by relevance
        by_rel: dict[str, list[TaskResult]] = {}
        for r in results:
            rel = r.metadata.get("memory_relevance", "unknown")
            if rel not in by_rel:
                by_rel[rel] = []
            by_rel[rel].append(r)

        return {rel: self._compute_group_metrics(rel, group) for rel, group in by_rel.items()}

    def _calculate_by_difficulty(
        self,
        results: list[TaskResult],
    ) -> dict[int, TaskMetrics]:
        """Calculate metrics grouped by difficulty level.

        Args:
            results: Task results

        Returns:
            Metrics by difficulty level
        """
        # Group by difficulty
        by_diff: dict[int, list[TaskResult]] = {}
        for r in results:
            diff = r.metadata.get("difficulty", 3)
            if diff not in by_diff:
                by_diff[diff] = []
            by_diff[diff].append(r)

        return {
            diff: self._compute_group_metrics(str(diff), group) for diff, group in by_diff.items()
        }

    def _compute_group_metrics(
        self,
        group_name: str,
        results: list[TaskResult],
    ) -> TaskMetrics:
        """Compute metrics for a group of results.

        Args:
            group_name: Name of the group
            results: Results in the group

        Returns:
            TaskMetrics for the group
        """
        total = len(results)
        successful = sum(1 for r in results if r.success)
        success_rate = successful / total if total > 0 else 0.0

        exec_times = [r.execution_time_seconds for r in results]
        avg_time = sum(exec_times) / len(exec_times) if exec_times else 0.0

        memory_used = sum(1 for r in results if r.memory_context)
        memory_rate = memory_used / total if total > 0 else 0.0

        return TaskMetrics(
            category=group_name,
            total_attempts=total,
            successful_attempts=successful,
            success_rate=success_rate,
            avg_execution_time=avg_time,
            memory_usage_rate=memory_rate,
        )

    def compare(
        self,
        results_a: list[TrialResult],
        results_b: list[TrialResult],
    ) -> ComparisonResult:
        """Compare two sets of trial results.

        Args:
            results_a: First adapter's results
            results_b: Second adapter's results

        Returns:
            ComparisonResult
        """
        metrics_a = self.calculate(results_a)
        metrics_b = self.calculate(results_b)

        success_diff = metrics_a.overall_success_rate - metrics_b.overall_success_rate
        time_diff = metrics_a.avg_execution_time - metrics_b.avg_execution_time

        # Statistical comparison
        p_value = 1.0
        effect_size = 0.0
        is_significant = False
        ci = (success_diff, success_diff)

        if self.analyzer is not None:
            rates_a = [t.success_rate for t in results_a]
            rates_b = [t.success_rate for t in results_b]

            if len(rates_a) >= 3 and len(rates_b) >= 3:
                try:
                    comparison = self.analyzer.paired_comparison(rates_a, rates_b)
                    p_value = comparison.p_value
                    effect_size = comparison.effect_size
                    is_significant = comparison.is_significant
                    ci = (comparison.ci_difference.lower, comparison.ci_difference.upper)
                except Exception as e:
                    logger.warning(f"Statistical comparison failed: {e}")

        return ComparisonResult(
            adapter_a=metrics_a.adapter_name,
            adapter_b=metrics_b.adapter_name,
            success_diff=success_diff,
            time_diff=time_diff,
            p_value=p_value,
            effect_size=effect_size,
            is_significant=is_significant,
            confidence_interval=ci,
        )

    def format_report(self, metrics: TerminalBenchMetrics) -> str:
        """Format metrics as a human-readable report.

        Args:
            metrics: Metrics to format

        Returns:
            Formatted report string
        """
        lines = [
            f"# Terminal-Bench 2.0 Results: {metrics.adapter_name}",
            "",
            "## Overall Metrics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total Tasks | {metrics.total_tasks} |",
            f"| Successful | {metrics.successful_tasks} |",
            f"| Success Rate | {metrics.overall_success_rate:.1%} |",
            f"| Avg Execution Time | {metrics.avg_execution_time:.2f}s |",
            f"| Memory Usage Rate | {metrics.memory_usage_rate:.1%} |",
        ]

        if metrics.ci_lower is not None and metrics.ci_upper is not None:
            lines.append(f"| 95% CI | [{metrics.ci_lower:.1%}, {metrics.ci_upper:.1%}] |")

        # By category
        if metrics.by_category:
            lines.extend(
                [
                    "",
                    "## By Category",
                    "",
                    "| Category | Tasks | Success Rate | Avg Time |",
                    "|----------|-------|--------------|----------|",
                ]
            )
            for cat, m in sorted(metrics.by_category.items()):
                lines.append(
                    f"| {cat} | {m.total_attempts} | {m.success_rate:.1%} | {m.avg_execution_time:.2f}s |"
                )

        # By relevance
        if metrics.by_relevance:
            lines.extend(
                [
                    "",
                    "## By Memory Relevance",
                    "",
                    "| Relevance | Tasks | Success Rate | Memory Used |",
                    "|-----------|-------|--------------|-------------|",
                ]
            )
            for rel, m in sorted(metrics.by_relevance.items()):
                lines.append(
                    f"| {rel} | {m.total_attempts} | {m.success_rate:.1%} | {m.memory_usage_rate:.1%} |"
                )

        # By difficulty
        if metrics.by_difficulty:
            lines.extend(
                [
                    "",
                    "## By Difficulty",
                    "",
                    "| Difficulty | Tasks | Success Rate |",
                    "|------------|-------|--------------|",
                ]
            )
            for diff, m in sorted(metrics.by_difficulty.items()):
                lines.append(f"| {diff} | {m.total_attempts} | {m.success_rate:.1%} |")

        return "\n".join(lines)

    def format_comparison(self, comparison: ComparisonResult) -> str:
        """Format comparison result as a report.

        Args:
            comparison: Comparison result

        Returns:
            Formatted comparison report
        """
        sig_marker = "*" if comparison.is_significant else ""

        lines = [
            f"# Comparison: {comparison.adapter_a} vs {comparison.adapter_b}",
            "",
            "| Metric | Difference | p-value | Significant |",
            "|--------|------------|---------|-------------|",
            f"| Success Rate | {comparison.success_diff:+.1%} | {comparison.p_value:.4f} | {sig_marker} |",
            f"| Execution Time | {comparison.time_diff:+.2f}s | - | - |",
            "",
            f"Effect Size (Cohen's d): {comparison.effect_size:.3f}",
            f"95% CI: [{comparison.confidence_interval[0]:.1%}, {comparison.confidence_interval[1]:.1%}]",
        ]

        return "\n".join(lines)
