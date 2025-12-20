"""Tests for publication figure generators."""

from __future__ import annotations

from pathlib import Path

import pytest

# Check if matplotlib is available
try:
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend for testing
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from src.publication.figures import (
    MATPLOTLIB_AVAILABLE as MODULE_MATPLOTLIB,
)
from src.publication.figures import (
    AblationHeatmap,
    CategoryRadarPlot,
    ConfidenceIntervalPlot,
    HumanAgreementPlot,
    PerformanceBarChart,
)


@pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="matplotlib not installed")
class TestPerformanceBarChart:
    """Tests for PerformanceBarChart generator."""

    @pytest.fixture
    def chart(self) -> PerformanceBarChart:
        """Create a bar chart generator."""
        return PerformanceBarChart()

    @pytest.fixture
    def sample_data(self) -> list[dict]:
        """Sample performance data."""
        return [
            {
                "adapter": "git_notes",
                "overall_accuracy": 0.85,
                "benchmarks": {
                    "longmemeval": 0.88,
                    "locomo": 0.82,
                    "contextbench": 0.85,
                },
            },
            {
                "adapter": "no_memory",
                "overall_accuracy": 0.70,
                "benchmarks": {
                    "longmemeval": 0.72,
                    "locomo": 0.68,
                    "contextbench": 0.70,
                },
            },
        ]

    def test_generate(self, chart: PerformanceBarChart, sample_data: list[dict]) -> None:
        """Test figure generation."""
        fig = chart.generate(sample_data)

        assert fig is not None
        # Check figure has axes
        assert len(fig.axes) > 0

    def test_generate_empty(self, chart: PerformanceBarChart) -> None:
        """Test with empty data."""
        fig = chart.generate([])
        assert fig is None

    def test_save(self, chart: PerformanceBarChart, sample_data: list[dict], tmp_path) -> None:
        """Test saving figure."""
        output_path = tmp_path / "chart.png"
        success = chart.save(sample_data, output_path, format="png")

        assert success
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_save_pdf(self, chart: PerformanceBarChart, sample_data: list[dict], tmp_path) -> None:
        """Test saving as PDF."""
        output_path = tmp_path / "chart.pdf"
        success = chart.save(sample_data, output_path, format="pdf")

        assert success
        assert output_path.exists()


@pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="matplotlib not installed")
class TestAblationHeatmap:
    """Tests for AblationHeatmap generator."""

    @pytest.fixture
    def heatmap(self) -> AblationHeatmap:
        """Create a heatmap generator."""
        return AblationHeatmap()

    @pytest.fixture
    def sample_data(self) -> list[dict]:
        """Sample ablation data."""
        return [
            {"ablation": "no_semantic_search", "delta": -0.15, "delta_pct": -17.6},
            {"ablation": "no_metadata_filter", "delta": -0.08, "delta_pct": -9.4},
            {"ablation": "no_version_history", "delta": -0.03, "delta_pct": -3.5},
            {"ablation": "recency_only", "delta": -0.20, "delta_pct": -23.5},
        ]

    def test_generate(self, heatmap: AblationHeatmap, sample_data: list[dict]) -> None:
        """Test figure generation."""
        fig = heatmap.generate(sample_data)

        assert fig is not None
        assert len(fig.axes) > 0

    def test_generate_empty(self, heatmap: AblationHeatmap) -> None:
        """Test with empty data."""
        fig = heatmap.generate([])
        assert fig is None

    def test_ablation_formatting(self, heatmap: AblationHeatmap, sample_data: list[dict]) -> None:
        """Test ablation name formatting in figure."""
        fig = heatmap.generate(sample_data)
        ax = fig.axes[0]

        # Check y-axis labels
        labels = [t.get_text() for t in ax.get_yticklabels()]
        assert any("Semantic Search" in l for l in labels)


@pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="matplotlib not installed")
class TestCategoryRadarPlot:
    """Tests for CategoryRadarPlot generator."""

    @pytest.fixture
    def radar(self) -> CategoryRadarPlot:
        """Create a radar plot generator."""
        return CategoryRadarPlot()

    @pytest.fixture
    def sample_data(self) -> list[dict]:
        """Sample category data."""
        return [
            {"category": "Single-Hop", "git_notes": 0.92, "no_memory": 0.75},
            {"category": "Multi-Hop", "git_notes": 0.78, "no_memory": 0.55},
            {"category": "Temporal", "git_notes": 0.85, "no_memory": 0.62},
            {"category": "Adversarial", "git_notes": 0.70, "no_memory": 0.50},
        ]

    def test_generate(self, radar: CategoryRadarPlot, sample_data: list[dict]) -> None:
        """Test figure generation."""
        fig = radar.generate(sample_data)

        assert fig is not None
        assert len(fig.axes) > 0

    def test_generate_insufficient_categories(self, radar: CategoryRadarPlot) -> None:
        """Test with too few categories."""
        data = [
            {"category": "A", "adapter": 0.8},
            {"category": "B", "adapter": 0.7},
        ]
        fig = radar.generate(data)
        # Should return None or handle gracefully
        assert fig is None


@pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="matplotlib not installed")
class TestConfidenceIntervalPlot:
    """Tests for ConfidenceIntervalPlot generator."""

    @pytest.fixture
    def ci_plot(self) -> ConfidenceIntervalPlot:
        """Create a CI plot generator."""
        return ConfidenceIntervalPlot()

    @pytest.fixture
    def sample_data(self) -> list[dict]:
        """Sample CI data."""
        return [
            {"adapter": "git_notes", "accuracy": 0.85, "ci_lower": 0.82, "ci_upper": 0.88},
            {"adapter": "semantic", "accuracy": 0.80, "ci_lower": 0.76, "ci_upper": 0.84},
            {"adapter": "no_memory", "accuracy": 0.70, "ci_lower": 0.65, "ci_upper": 0.75},
        ]

    def test_generate(self, ci_plot: ConfidenceIntervalPlot, sample_data: list[dict]) -> None:
        """Test figure generation."""
        fig = ci_plot.generate(sample_data)

        assert fig is not None
        assert len(fig.axes) > 0


@pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="matplotlib not installed")
class TestHumanAgreementPlot:
    """Tests for HumanAgreementPlot generator."""

    @pytest.fixture
    def plot(self) -> HumanAgreementPlot:
        """Create an agreement plot generator."""
        return HumanAgreementPlot()

    @pytest.fixture
    def sample_data(self) -> list[dict]:
        """Sample agreement data."""
        return [
            {"benchmark": "LongMemEval", "agreement_rate": 0.92, "kappa": 0.85},
            {"benchmark": "LoCoMo", "agreement_rate": 0.88, "kappa": 0.78},
            {"benchmark": "Context-Bench", "agreement_rate": 0.90, "kappa": 0.82},
        ]

    def test_generate(self, plot: HumanAgreementPlot, sample_data: list[dict]) -> None:
        """Test figure generation."""
        fig = plot.generate(sample_data)

        assert fig is not None
        assert len(fig.axes) > 0


class TestFiguresWithoutMatplotlib:
    """Tests for figure generators when matplotlib is unavailable."""

    def test_module_reports_availability(self) -> None:
        """Test that module correctly reports matplotlib availability."""
        # The module-level constant should match actual availability
        assert MODULE_MATPLOTLIB == MATPLOTLIB_AVAILABLE

    def test_generate_without_matplotlib(self) -> None:
        """Test graceful handling when matplotlib unavailable."""
        # This test runs regardless of matplotlib availability
        # The generators should handle the case gracefully
        chart = PerformanceBarChart()
        data = [{"adapter": "test", "benchmarks": {"a": 0.8}}]

        if not MATPLOTLIB_AVAILABLE:
            fig = chart.generate(data)
            assert fig is None

            saved = chart.save(data, Path("/tmp/test.png"))
            assert not saved
