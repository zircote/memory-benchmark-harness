"""Tests for the Terminal-Bench metrics module."""

from __future__ import annotations

from datetime import datetime

import pytest

from src.benchmarks.terminalbench.metrics import (
    ComparisonResult,
    TaskMetrics,
    TerminalBenchMetrics,
    TerminalBenchMetricsCalculator,
)
from src.benchmarks.terminalbench.runner import TaskResult, TrialResult


class TestTaskMetrics:
    """Tests for the TaskMetrics dataclass."""

    def test_creation(self) -> None:
        """Test creating task metrics."""
        metrics = TaskMetrics(
            category="debugging",
            total_attempts=10,
            successful_attempts=7,
            success_rate=0.7,
            avg_execution_time=5.5,
            memory_usage_rate=0.8,
        )
        assert metrics.category == "debugging"
        assert metrics.success_rate == 0.7


class TestTerminalBenchMetrics:
    """Tests for the TerminalBenchMetrics dataclass."""

    def test_creation(self) -> None:
        """Test creating metrics."""
        metrics = TerminalBenchMetrics(
            adapter_name="TestAdapter",
            total_tasks=100,
            successful_tasks=70,
            overall_success_rate=0.7,
            avg_execution_time=3.0,
            memory_usage_rate=0.6,
            by_category={},
            by_relevance={},
            by_difficulty={},
        )
        assert metrics.adapter_name == "TestAdapter"
        assert metrics.overall_success_rate == 0.7

    def test_with_confidence_interval(self) -> None:
        """Test metrics with confidence interval."""
        metrics = TerminalBenchMetrics(
            adapter_name="TestAdapter",
            total_tasks=100,
            successful_tasks=70,
            overall_success_rate=0.7,
            avg_execution_time=3.0,
            memory_usage_rate=0.6,
            by_category={},
            by_relevance={},
            by_difficulty={},
            ci_lower=0.65,
            ci_upper=0.75,
        )
        assert metrics.ci_lower == 0.65
        assert metrics.ci_upper == 0.75


class TestComparisonResult:
    """Tests for the ComparisonResult dataclass."""

    def test_creation(self) -> None:
        """Test creating comparison result."""
        result = ComparisonResult(
            adapter_a="AdapterA",
            adapter_b="AdapterB",
            success_diff=0.1,
            time_diff=-0.5,
            p_value=0.03,
            effect_size=0.5,
            is_significant=True,
            confidence_interval=(0.05, 0.15),
        )
        assert result.adapter_a == "AdapterA"
        assert result.success_diff == 0.1
        assert result.is_significant is True


class TestTerminalBenchMetricsCalculator:
    """Tests for the TerminalBenchMetricsCalculator class."""

    @pytest.fixture
    def calculator(self) -> TerminalBenchMetricsCalculator:
        """Create a calculator for testing."""
        return TerminalBenchMetricsCalculator()

    @pytest.fixture
    def sample_task_results(self) -> list[TaskResult]:
        """Create sample task results."""
        results = []
        for i in range(10):
            results.append(
                TaskResult(
                    task_id=f"task_{i}",
                    task_description=f"Task {i}",
                    augmented_description="aug",
                    success=i < 7,  # 70% success
                    output="output",
                    execution_time_seconds=1.0 + i * 0.1,
                    memory_context="context" if i < 6 else "",  # 60% memory usage
                    metadata={
                        "category": "debugging" if i % 2 == 0 else "software",
                        "memory_relevance": "high" if i < 5 else "medium",
                        "difficulty": (i % 5) + 1,
                    },
                )
            )
        return results

    @pytest.fixture
    def sample_trial(self, sample_task_results: list[TaskResult]) -> TrialResult:
        """Create a sample trial result."""
        return TrialResult(
            trial_id="trial_001",
            adapter_name="TestAdapter",
            task_results=tuple(sample_task_results),
            total_tasks=len(sample_task_results),
            successful_tasks=sum(1 for r in sample_task_results if r.success),
            start_time=datetime.now(),
            end_time=datetime.now(),
        )

    def test_calculate_empty(self, calculator: TerminalBenchMetricsCalculator) -> None:
        """Test calculating with empty results."""
        metrics = calculator.calculate([])
        assert metrics.adapter_name == "unknown"
        assert metrics.total_tasks == 0
        assert metrics.overall_success_rate == 0.0

    def test_calculate_basic(
        self,
        calculator: TerminalBenchMetricsCalculator,
        sample_trial: TrialResult,
    ) -> None:
        """Test basic metrics calculation."""
        metrics = calculator.calculate([sample_trial])

        assert metrics.adapter_name == "TestAdapter"
        assert metrics.total_tasks == 10
        assert metrics.successful_tasks == 7
        assert metrics.overall_success_rate == 0.7

    def test_calculate_memory_usage(
        self,
        calculator: TerminalBenchMetricsCalculator,
        sample_trial: TrialResult,
    ) -> None:
        """Test memory usage calculation."""
        metrics = calculator.calculate([sample_trial])
        assert metrics.memory_usage_rate == 0.6

    def test_calculate_by_category(
        self,
        calculator: TerminalBenchMetricsCalculator,
        sample_trial: TrialResult,
    ) -> None:
        """Test metrics by category."""
        metrics = calculator.calculate([sample_trial])

        assert "debugging" in metrics.by_category
        assert "software" in metrics.by_category

        debug_metrics = metrics.by_category["debugging"]
        assert debug_metrics.total_attempts == 5  # Even indices 0,2,4,6,8

    def test_calculate_by_relevance(
        self,
        calculator: TerminalBenchMetricsCalculator,
        sample_trial: TrialResult,
    ) -> None:
        """Test metrics by relevance."""
        metrics = calculator.calculate([sample_trial])

        assert "high" in metrics.by_relevance
        assert "medium" in metrics.by_relevance

    def test_calculate_by_difficulty(
        self,
        calculator: TerminalBenchMetricsCalculator,
        sample_trial: TrialResult,
    ) -> None:
        """Test metrics by difficulty."""
        metrics = calculator.calculate([sample_trial])

        assert len(metrics.by_difficulty) > 0
        assert all(1 <= d <= 5 for d in metrics.by_difficulty.keys())

    def test_compare(
        self,
        calculator: TerminalBenchMetricsCalculator,
        sample_trial: TrialResult,
    ) -> None:
        """Test comparing two adapters."""
        trial_a = sample_trial
        trial_b = TrialResult(
            trial_id="trial_002",
            adapter_name="OtherAdapter",
            task_results=tuple(
                TaskResult(
                    task_id=r.task_id,
                    task_description=r.task_description,
                    augmented_description=r.augmented_description,
                    success=not r.success,  # Inverse success
                    output=r.output,
                    execution_time_seconds=r.execution_time_seconds + 1.0,
                    metadata=r.metadata,
                )
                for r in sample_trial.task_results
            ),
            total_tasks=sample_trial.total_tasks,
            successful_tasks=sample_trial.total_tasks - sample_trial.successful_tasks,
            start_time=datetime.now(),
            end_time=datetime.now(),
        )

        result = calculator.compare([trial_a], [trial_b])

        assert result.adapter_a == "TestAdapter"
        assert result.adapter_b == "OtherAdapter"
        assert abs(result.success_diff - 0.4) < 0.01  # 0.7 - 0.3 = 0.4

    def test_format_report(
        self,
        calculator: TerminalBenchMetricsCalculator,
        sample_trial: TrialResult,
    ) -> None:
        """Test formatting report."""
        metrics = calculator.calculate([sample_trial])
        report = calculator.format_report(metrics)

        assert "TestAdapter" in report
        assert "Overall Metrics" in report
        assert "70.0%" in report  # Success rate
        assert "By Category" in report

    def test_format_comparison(self, calculator: TerminalBenchMetricsCalculator) -> None:
        """Test formatting comparison."""
        result = ComparisonResult(
            adapter_a="AdapterA",
            adapter_b="AdapterB",
            success_diff=0.1,
            time_diff=-0.5,
            p_value=0.03,
            effect_size=0.5,
            is_significant=True,
            confidence_interval=(0.05, 0.15),
        )

        report = calculator.format_comparison(result)

        assert "AdapterA" in report
        assert "AdapterB" in report
        assert "+10.0%" in report  # Success diff
        assert "*" in report  # Significant marker
