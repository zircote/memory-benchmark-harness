"""Tests for publication statistics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.publication.statistics import (
    AblationResult,
    BenchmarkSummary,
    PublicationStatistics,
    UnifiedMetrics,
)


class TestUnifiedMetrics:
    """Tests for UnifiedMetrics dataclass."""

    def test_metrics_creation(self) -> None:
        """Test creating metrics."""
        metrics = UnifiedMetrics(
            accuracy=0.85,
            accuracy_ci=(0.82, 0.88),
            precision=0.87,
            recall=0.83,
            f1_score=0.85,
            abstention_rate=0.05,
            avg_latency_ms=150.0,
            cost_per_1k=2.50,
        )

        assert metrics.accuracy == 0.85
        assert metrics.accuracy_ci == (0.82, 0.88)
        assert metrics.precision == 0.87
        assert metrics.f1_score == 0.85
        assert metrics.abstention_rate == 0.05

    def test_metrics_defaults(self) -> None:
        """Test default values."""
        metrics = UnifiedMetrics(accuracy=0.75)

        assert metrics.accuracy_ci == (0.0, 0.0)
        assert metrics.precision == 0.0
        assert metrics.recall == 0.0


class TestBenchmarkSummary:
    """Tests for BenchmarkSummary dataclass."""

    def test_summary_creation(self) -> None:
        """Test creating a summary."""
        metrics = UnifiedMetrics(accuracy=0.85)
        summary = BenchmarkSummary(
            benchmark_name="longmemeval",
            adapter_name="git_notes",
            total_samples=500,
            metrics=metrics,
        )

        assert summary.benchmark_name == "longmemeval"
        assert summary.adapter_name == "git_notes"
        assert summary.total_samples == 500
        assert summary.metrics.accuracy == 0.85

    def test_summary_with_categories(self) -> None:
        """Test summary with category metrics."""
        summary = BenchmarkSummary(
            benchmark_name="locomo",
            adapter_name="test",
            total_samples=200,
            metrics=UnifiedMetrics(accuracy=0.80),
            category_metrics={
                "single_hop": UnifiedMetrics(accuracy=0.90),
                "multi_hop": UnifiedMetrics(accuracy=0.70),
            },
        )

        assert len(summary.category_metrics) == 2
        assert summary.category_metrics["single_hop"].accuracy == 0.90


class TestAblationResult:
    """Tests for AblationResult dataclass."""

    def test_ablation_creation(self) -> None:
        """Test creating an ablation result."""
        result = AblationResult(
            ablation_name="no_semantic_search",
            baseline_accuracy=0.85,
            ablated_accuracy=0.70,
            delta=-0.15,
            delta_pct=-17.6,
            p_value=0.001,
            effect_size=0.8,
        )

        assert result.ablation_name == "no_semantic_search"
        assert result.delta == -0.15
        assert result.delta_pct == -17.6
        assert result.p_value == 0.001


class TestPublicationStatistics:
    """Tests for PublicationStatistics class."""

    @pytest.fixture
    def stats(self) -> PublicationStatistics:
        """Create publication statistics."""
        return PublicationStatistics()

    @pytest.fixture
    def sample_summary(self) -> BenchmarkSummary:
        """Create a sample summary."""
        return BenchmarkSummary(
            benchmark_name="longmemeval",
            adapter_name="git_notes",
            total_samples=100,
            metrics=UnifiedMetrics(
                accuracy=0.85,
                precision=0.87,
                recall=0.83,
                f1_score=0.85,
            ),
            category_metrics={
                "qa": UnifiedMetrics(accuracy=0.90),
                "generation": UnifiedMetrics(accuracy=0.80),
            },
        )

    def test_add_summary(
        self, stats: PublicationStatistics, sample_summary: BenchmarkSummary
    ) -> None:
        """Test adding a summary."""
        stats.add_summary(sample_summary)

        assert len(stats.summaries) == 1
        assert stats.summaries[0].benchmark_name == "longmemeval"

    def test_add_ablation(self, stats: PublicationStatistics) -> None:
        """Test adding an ablation result."""
        result = AblationResult(
            ablation_name="no_version_history",
            baseline_accuracy=0.85,
            ablated_accuracy=0.82,
            delta=-0.03,
            delta_pct=-3.5,
        )
        stats.add_ablation(result)

        assert len(stats.ablation_results) == 1
        assert stats.ablation_results[0].ablation_name == "no_version_history"

    def test_load_results_file(self, stats: PublicationStatistics, tmp_path) -> None:
        """Test loading results from file."""
        data = {
            "accuracy": 0.82,
            "precision": 0.85,
            "recall": 0.79,
            "total_samples": 150,
            "by_category": {
                "qa": {"accuracy": 0.88},
                "summary": {"accuracy": 0.76},
            },
        }
        filepath = tmp_path / "results.json"
        with open(filepath, "w") as f:
            json.dump(data, f)

        summary = stats.load_results_file(filepath, "locomo", "semantic_search")

        assert summary is not None
        assert summary.benchmark_name == "locomo"
        assert summary.adapter_name == "semantic_search"
        assert summary.metrics.accuracy == 0.82
        assert len(summary.category_metrics) == 2

    def test_load_results_file_missing(self, stats: PublicationStatistics) -> None:
        """Test loading from missing file."""
        summary = stats.load_results_file(Path("/nonexistent/file.json"), "test", "test")

        assert summary is None
        assert len(stats.summaries) == 0

    def test_load_results_alternative_fields(self, stats: PublicationStatistics, tmp_path) -> None:
        """Test loading with alternative field names."""
        data = {
            "score": 0.78,
            "num_samples": 200,
            "confidence_interval": {"lower": 0.75, "upper": 0.81},
        }
        filepath = tmp_path / "alt_results.json"
        with open(filepath, "w") as f:
            json.dump(data, f)

        summary = stats.load_results_file(filepath, "test", "test")

        assert summary is not None
        assert summary.metrics.accuracy == 0.78
        assert summary.metrics.accuracy_ci == (0.75, 0.81)
        assert summary.total_samples == 200

    def test_compute_aggregate_metrics(self, stats: PublicationStatistics) -> None:
        """Test computing aggregate metrics."""
        stats.add_summary(
            BenchmarkSummary(
                benchmark_name="lme",
                adapter_name="adapter_a",
                total_samples=100,
                metrics=UnifiedMetrics(
                    accuracy=0.80,
                    f1_score=0.78,
                ),
            )
        )
        stats.add_summary(
            BenchmarkSummary(
                benchmark_name="locomo",
                adapter_name="adapter_a",
                total_samples=100,
                metrics=UnifiedMetrics(
                    accuracy=0.90,
                    f1_score=0.88,
                ),
            )
        )

        aggregate = stats.compute_aggregate_metrics("adapter_a")

        # Weighted average: (0.80 * 100 + 0.90 * 100) / 200 = 0.85
        assert aggregate.accuracy == 0.85
        assert aggregate.f1_score == 0.83

    def test_compute_aggregate_filtered(self, stats: PublicationStatistics) -> None:
        """Test aggregate metrics with adapter filter."""
        stats.add_summary(
            BenchmarkSummary(
                benchmark_name="lme",
                adapter_name="adapter_a",
                total_samples=100,
                metrics=UnifiedMetrics(accuracy=0.80),
            )
        )
        stats.add_summary(
            BenchmarkSummary(
                benchmark_name="lme",
                adapter_name="adapter_b",
                total_samples=100,
                metrics=UnifiedMetrics(accuracy=0.70),
            )
        )

        aggregate_a = stats.compute_aggregate_metrics("adapter_a")
        aggregate_b = stats.compute_aggregate_metrics("adapter_b")

        assert aggregate_a.accuracy == 0.80
        assert aggregate_b.accuracy == 0.70

    def test_compute_aggregate_empty(self, stats: PublicationStatistics) -> None:
        """Test aggregate with no data."""
        aggregate = stats.compute_aggregate_metrics()

        assert aggregate.accuracy == 0.0

    def test_compare_adapters(self, stats: PublicationStatistics) -> None:
        """Test comparing two adapters."""
        # Need at least 2 benchmarks for paired comparison
        stats.add_summary(
            BenchmarkSummary(
                benchmark_name="lme",
                adapter_name="a",
                total_samples=100,
                metrics=UnifiedMetrics(accuracy=0.85),
            )
        )
        stats.add_summary(
            BenchmarkSummary(
                benchmark_name="locomo",
                adapter_name="a",
                total_samples=100,
                metrics=UnifiedMetrics(accuracy=0.88),
            )
        )
        stats.add_summary(
            BenchmarkSummary(
                benchmark_name="lme",
                adapter_name="b",
                total_samples=100,
                metrics=UnifiedMetrics(accuracy=0.75),
            )
        )
        stats.add_summary(
            BenchmarkSummary(
                benchmark_name="locomo",
                adapter_name="b",
                total_samples=100,
                metrics=UnifiedMetrics(accuracy=0.72),
            )
        )

        result = stats.compare_adapters("a", "b")

        assert result["valid"]
        assert result["adapter_a"] == "a"
        assert result["adapter_b"] == "b"
        assert "p_value" in result
        assert "effect_size" in result

    def test_compare_adapters_missing(self, stats: PublicationStatistics) -> None:
        """Test comparison with missing adapter."""
        result = stats.compare_adapters("nonexistent", "also_nonexistent")

        assert not result["valid"]
        assert "error" in result

    def test_get_main_results_data(self, stats: PublicationStatistics) -> None:
        """Test getting main results data."""
        stats.add_summary(
            BenchmarkSummary(
                benchmark_name="lme",
                adapter_name="git_notes",
                total_samples=100,
                metrics=UnifiedMetrics(accuracy=0.85, f1_score=0.83),
            )
        )
        stats.add_summary(
            BenchmarkSummary(
                benchmark_name="locomo",
                adapter_name="git_notes",
                total_samples=100,
                metrics=UnifiedMetrics(accuracy=0.80, f1_score=0.78),
            )
        )

        data = stats.get_main_results_data()

        assert len(data) == 1
        assert data[0]["adapter"] == "git_notes"
        assert "lme" in data[0]["benchmarks"]
        assert "locomo" in data[0]["benchmarks"]
        assert data[0]["benchmarks"]["lme"] == 0.85

    def test_get_ablation_data(self, stats: PublicationStatistics) -> None:
        """Test getting ablation data."""
        stats.add_ablation(
            AblationResult(
                ablation_name="no_semantic",
                baseline_accuracy=0.85,
                ablated_accuracy=0.70,
                delta=-0.15,
                delta_pct=-17.6,
                p_value=0.001,
                effect_size=0.8,
            )
        )

        data = stats.get_ablation_data()

        assert len(data) == 1
        assert data[0]["ablation"] == "no_semantic"
        assert data[0]["delta"] == -0.15
        assert data[0]["significant"]

    def test_get_category_data(self, stats: PublicationStatistics) -> None:
        """Test getting category breakdown data."""
        stats.add_summary(
            BenchmarkSummary(
                benchmark_name="lme",
                adapter_name="a",
                total_samples=100,
                metrics=UnifiedMetrics(accuracy=0.85),
                category_metrics={
                    "qa": UnifiedMetrics(accuracy=0.90),
                    "gen": UnifiedMetrics(accuracy=0.80),
                },
            )
        )

        data = stats.get_category_data()

        assert len(data) == 2
        categories = {row["category"] for row in data}
        assert categories == {"qa", "gen"}

    def test_export_json(self, stats: PublicationStatistics, tmp_path) -> None:
        """Test exporting to JSON."""
        stats.add_summary(
            BenchmarkSummary(
                benchmark_name="test",
                adapter_name="adapter",
                total_samples=50,
                metrics=UnifiedMetrics(accuracy=0.75),
            )
        )

        output_path = tmp_path / "export" / "stats.json"
        stats.export_json(output_path)

        assert output_path.exists()
        with open(output_path) as f:
            data = json.load(f)

        assert "main_results" in data
        assert "ablation" in data
        assert "summaries" in data
