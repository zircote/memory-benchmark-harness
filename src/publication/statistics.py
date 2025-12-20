"""Unified statistics for publication.

This module aggregates results across all benchmarks into a unified
format suitable for publication tables and figures.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.evaluation.statistics import StatisticalAnalyzer

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class UnifiedMetrics:
    """Unified metrics across all benchmarks.

    Attributes:
        accuracy: Overall accuracy (0-1)
        accuracy_ci: 95% confidence interval for accuracy
        precision: Precision for correct answers
        recall: Recall for correct answers
        f1_score: F1 score
        abstention_rate: Rate of abstained answers
        avg_latency_ms: Average response latency
        cost_per_1k: Cost per 1000 queries (USD)
    """

    accuracy: float
    accuracy_ci: tuple[float, float] = (0.0, 0.0)
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    abstention_rate: float = 0.0
    avg_latency_ms: float = 0.0
    cost_per_1k: float = 0.0


@dataclass(frozen=True, slots=True)
class BenchmarkSummary:
    """Summary of results for a single benchmark.

    Attributes:
        benchmark_name: Name of the benchmark
        adapter_name: Name of the adapter/condition
        total_samples: Total samples evaluated
        metrics: Unified metrics for this benchmark
        category_metrics: Per-category metrics if applicable
        metadata: Additional benchmark-specific data
    """

    benchmark_name: str
    adapter_name: str
    total_samples: int
    metrics: UnifiedMetrics
    category_metrics: dict[str, UnifiedMetrics] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AblationResult:
    """Result from an ablation study.

    Attributes:
        ablation_name: Name of the ablation (e.g., "no_semantic_search")
        baseline_accuracy: Accuracy with full system
        ablated_accuracy: Accuracy with ablated component
        delta: Change in accuracy (ablated - baseline)
        delta_pct: Percentage change
        p_value: Statistical significance
        effect_size: Cohen's d effect size
    """

    ablation_name: str
    baseline_accuracy: float
    ablated_accuracy: float
    delta: float
    delta_pct: float
    p_value: float = 0.0
    effect_size: float = 0.0


@dataclass
class PublicationStatistics:
    """Aggregates and computes final statistics for publication.

    This class:
    1. Loads results from all benchmarks
    2. Computes unified metrics
    3. Performs statistical comparisons
    4. Prepares data for tables and figures
    """

    summaries: list[BenchmarkSummary] = field(default_factory=list)
    ablation_results: list[AblationResult] = field(default_factory=list)
    analyzer: StatisticalAnalyzer = field(default_factory=StatisticalAnalyzer)

    def add_summary(self, summary: BenchmarkSummary) -> None:
        """Add a benchmark summary.

        Args:
            summary: Benchmark summary to add
        """
        self.summaries.append(summary)

    def add_ablation(self, result: AblationResult) -> None:
        """Add an ablation result.

        Args:
            result: Ablation result to add
        """
        self.ablation_results.append(result)

    def load_results_file(
        self,
        filepath: Path | str,
        benchmark_name: str,
        adapter_name: str,
    ) -> BenchmarkSummary | None:
        """Load results from a JSON file.

        Args:
            filepath: Path to results file
            benchmark_name: Name of the benchmark
            adapter_name: Name of the adapter

        Returns:
            BenchmarkSummary or None if loading fails
        """
        filepath = Path(filepath)
        if not filepath.exists():
            logger.warning(f"Results file not found: {filepath}")
            return None

        try:
            with open(filepath) as f:
                data = json.load(f)

            # Extract metrics based on common result formats
            metrics = self._extract_metrics(data)
            category_metrics = self._extract_category_metrics(data)

            total_samples = data.get("total_samples", 0)
            if not total_samples:
                # Try alternative field names
                total_samples = data.get("num_samples", data.get("count", 0))

            summary = BenchmarkSummary(
                benchmark_name=benchmark_name,
                adapter_name=adapter_name,
                total_samples=total_samples,
                metrics=metrics,
                category_metrics=category_metrics,
                metadata=data.get("metadata", {}),
            )

            self.summaries.append(summary)
            return summary

        except Exception as e:
            logger.warning(f"Failed to load results from {filepath}: {e}")
            return None

    def _extract_metrics(self, data: dict[str, Any]) -> UnifiedMetrics:
        """Extract unified metrics from result data.

        Args:
            data: Raw result data

        Returns:
            UnifiedMetrics
        """
        # Handle various result formats
        accuracy = data.get("accuracy", data.get("score", 0.0))
        if isinstance(accuracy, dict):
            accuracy = accuracy.get("mean", accuracy.get("value", 0.0))

        # Extract confidence interval
        ci_data = data.get("accuracy_ci", data.get("confidence_interval", {}))
        if isinstance(ci_data, dict):
            ci = (ci_data.get("lower", 0.0), ci_data.get("upper", 0.0))
        elif isinstance(ci_data, (list, tuple)) and len(ci_data) >= 2:
            ci = (float(ci_data[0]), float(ci_data[1]))
        else:
            ci = (0.0, 0.0)

        # Extract other metrics with fallbacks
        precision = data.get("precision", 0.0)
        recall = data.get("recall", 0.0)
        f1 = data.get("f1_score", data.get("f1", 0.0))

        # Calculate F1 if not provided
        if f1 == 0.0 and precision > 0 and recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)

        abstention = data.get(
            "abstention_rate",
            data.get("abstention", data.get("unknown_rate", 0.0)),
        )
        latency = data.get(
            "avg_latency_ms",
            data.get("latency_ms", data.get("latency", 0.0)),
        )
        cost = data.get("cost_per_1k", data.get("cost", 0.0))

        return UnifiedMetrics(
            accuracy=float(accuracy),
            accuracy_ci=ci,
            precision=float(precision),
            recall=float(recall),
            f1_score=float(f1),
            abstention_rate=float(abstention),
            avg_latency_ms=float(latency),
            cost_per_1k=float(cost),
        )

    def _extract_category_metrics(self, data: dict[str, Any]) -> dict[str, UnifiedMetrics]:
        """Extract per-category metrics.

        Args:
            data: Raw result data

        Returns:
            Dictionary mapping category names to metrics
        """
        category_data = data.get(
            "by_category",
            data.get("categories", data.get("per_category", {})),
        )

        result = {}
        for category, cat_data in category_data.items():
            if isinstance(cat_data, dict):
                result[category] = self._extract_metrics(cat_data)
            elif isinstance(cat_data, (int, float)):
                result[category] = UnifiedMetrics(accuracy=float(cat_data))

        return result

    def compute_aggregate_metrics(self, adapter_name: str | None = None) -> UnifiedMetrics:
        """Compute aggregate metrics across benchmarks.

        Args:
            adapter_name: Filter by adapter name (None for all)

        Returns:
            Aggregate UnifiedMetrics
        """
        summaries = self.summaries
        if adapter_name:
            summaries = [s for s in summaries if s.adapter_name == adapter_name]

        if not summaries:
            return UnifiedMetrics(accuracy=0.0)

        # Weighted average by sample count
        total_weight = sum(s.total_samples for s in summaries)
        if total_weight == 0:
            total_weight = len(summaries)

        accuracy = sum(s.metrics.accuracy * s.total_samples for s in summaries) / total_weight

        precision = sum(s.metrics.precision * s.total_samples for s in summaries) / total_weight

        recall = sum(s.metrics.recall * s.total_samples for s in summaries) / total_weight

        f1 = sum(s.metrics.f1_score * s.total_samples for s in summaries) / total_weight

        abstention = (
            sum(s.metrics.abstention_rate * s.total_samples for s in summaries) / total_weight
        )

        latency = sum(s.metrics.avg_latency_ms * s.total_samples for s in summaries) / total_weight

        cost = sum(s.metrics.cost_per_1k * s.total_samples for s in summaries) / total_weight

        return UnifiedMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            abstention_rate=abstention,
            avg_latency_ms=latency,
            cost_per_1k=cost,
        )

    def compare_adapters(
        self,
        adapter_a: str,
        adapter_b: str,
    ) -> dict[str, Any]:
        """Compare two adapters statistically.

        Args:
            adapter_a: First adapter name
            adapter_b: Second adapter name

        Returns:
            Comparison results including p-value and effect size
        """
        summaries_a = [s for s in self.summaries if s.adapter_name == adapter_a]
        summaries_b = [s for s in self.summaries if s.adapter_name == adapter_b]

        if not summaries_a or not summaries_b:
            return {
                "valid": False,
                "error": "Insufficient data for comparison",
            }

        # Collect accuracy scores by benchmark
        scores_a = [s.metrics.accuracy for s in summaries_a]
        scores_b = [s.metrics.accuracy for s in summaries_b]

        mean_a = sum(scores_a) / len(scores_a)
        mean_b = sum(scores_b) / len(scores_b)
        mean_diff = mean_a - mean_b

        # Paired comparison if same benchmarks
        result = self.analyzer.paired_comparison(scores_a, scores_b)

        return {
            "valid": True,
            "adapter_a": adapter_a,
            "adapter_b": adapter_b,
            "mean_a": mean_a,
            "mean_b": mean_b,
            "mean_diff": mean_diff,
            "p_value": result.p_value,
            "effect_size": result.effect_size,
            "significant": result.is_significant,
        }

    def get_main_results_data(self) -> list[dict[str, Any]]:
        """Get data for main results table.

        Returns:
            List of row data for main results table
        """
        # Group by adapter
        by_adapter: dict[str, list[BenchmarkSummary]] = {}
        for s in self.summaries:
            if s.adapter_name not in by_adapter:
                by_adapter[s.adapter_name] = []
            by_adapter[s.adapter_name].append(s)

        results = []
        for adapter, summaries in sorted(by_adapter.items()):
            aggregate = self.compute_aggregate_metrics(adapter)

            # Per-benchmark accuracies
            benchmarks = {s.benchmark_name: s.metrics.accuracy for s in summaries}

            results.append(
                {
                    "adapter": adapter,
                    "overall_accuracy": aggregate.accuracy,
                    "f1_score": aggregate.f1_score,
                    "abstention_rate": aggregate.abstention_rate,
                    "benchmarks": benchmarks,
                }
            )

        return results

    def get_ablation_data(self) -> list[dict[str, Any]]:
        """Get data for ablation table.

        Returns:
            List of row data for ablation table
        """
        return [
            {
                "ablation": r.ablation_name,
                "baseline": r.baseline_accuracy,
                "ablated": r.ablated_accuracy,
                "delta": r.delta,
                "delta_pct": r.delta_pct,
                "p_value": r.p_value,
                "effect_size": r.effect_size,
                "significant": r.p_value < 0.05,
            }
            for r in self.ablation_results
        ]

    def get_category_data(self, benchmark_name: str | None = None) -> list[dict[str, Any]]:
        """Get data for category breakdown table.

        Args:
            benchmark_name: Filter by benchmark (None for all)

        Returns:
            List of row data for category table
        """
        results = []

        summaries = self.summaries
        if benchmark_name:
            summaries = [s for s in summaries if s.benchmark_name == benchmark_name]

        # Aggregate categories across summaries
        category_accuracies: dict[str, dict[str, list[float]]] = {}

        for s in summaries:
            for category, metrics in s.category_metrics.items():
                if category not in category_accuracies:
                    category_accuracies[category] = {}
                if s.adapter_name not in category_accuracies[category]:
                    category_accuracies[category][s.adapter_name] = []
                category_accuracies[category][s.adapter_name].append(metrics.accuracy)

        for category, by_adapter in sorted(category_accuracies.items()):
            row = {"category": category}
            for adapter, scores in sorted(by_adapter.items()):
                row[adapter] = sum(scores) / len(scores) if scores else 0.0
            results.append(row)

        return results

    def export_json(self, output_path: Path | str) -> None:
        """Export all statistics to JSON.

        Args:
            output_path: Path to output file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "main_results": self.get_main_results_data(),
            "ablation": self.get_ablation_data(),
            "category_breakdown": self.get_category_data(),
            "summaries": [
                {
                    "benchmark": s.benchmark_name,
                    "adapter": s.adapter_name,
                    "total_samples": s.total_samples,
                    "accuracy": s.metrics.accuracy,
                    "accuracy_ci": s.metrics.accuracy_ci,
                    "f1_score": s.metrics.f1_score,
                    "abstention_rate": s.metrics.abstention_rate,
                }
                for s in self.summaries
            ],
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported statistics to {output_path}")
