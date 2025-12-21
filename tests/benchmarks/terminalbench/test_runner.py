"""Tests for the Terminal-Bench runner module."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from src.adapters.base import MemoryItem, MemoryOperationResult
from src.benchmarks.terminalbench.runner import (
    TaskResult,
    TerminalBenchConfig,
    TerminalBenchRunner,
    TrialResult,
)


class MockAdapter:
    """Mock memory adapter for testing."""

    def __init__(self) -> None:
        self.memories: list[tuple[str, dict[str, Any]]] = []
        self.search_results: list[MemoryItem] = []
        self.add_calls = 0
        self.search_calls = 0

    def add(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryOperationResult:
        self.add_calls += 1
        mem_id = f"mem_{self.add_calls}"
        self.memories.append((content, metadata or {}))
        return MemoryOperationResult(success=True, memory_id=mem_id)

    def search(
        self,
        query: str,
        limit: int = 10,
        min_score: float = 0.0,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        self.search_calls += 1
        return self.search_results[:limit]


class TestTaskResult:
    """Tests for the TaskResult dataclass."""

    def test_creation(self) -> None:
        """Test creating a task result."""
        result = TaskResult(
            task_id="task_001",
            task_description="Debug service",
            augmented_description="...",
            success=True,
            output="Success!",
            execution_time_seconds=5.5,
            memory_context="Previous fix...",
        )
        assert result.task_id == "task_001"
        assert result.success is True
        assert result.execution_time_seconds == 5.5

    def test_with_error(self) -> None:
        """Test result with error."""
        result = TaskResult(
            task_id="task_002",
            task_description="Fix bug",
            augmented_description="...",
            success=False,
            output="",
            execution_time_seconds=2.0,
            error="Timeout expired",
        )
        assert result.success is False
        assert result.error == "Timeout expired"


class TestTrialResult:
    """Tests for the TrialResult dataclass."""

    @pytest.fixture
    def sample_results(self) -> list[TaskResult]:
        """Create sample task results."""
        return [
            TaskResult(
                task_id=f"task_{i}",
                task_description="desc",
                augmented_description="aug",
                success=i % 2 == 0,  # 50% success
                output="out",
                execution_time_seconds=1.0,
            )
            for i in range(10)
        ]

    def test_creation(self, sample_results: list[TaskResult]) -> None:
        """Test creating a trial result."""
        start = datetime.now()
        result = TrialResult(
            trial_id="trial_001",
            adapter_name="TestAdapter",
            task_results=tuple(sample_results),
            total_tasks=10,
            successful_tasks=5,
            start_time=start,
            end_time=start,
        )
        assert result.trial_id == "trial_001"
        assert len(result.task_results) == 10

    def test_success_rate(self, sample_results: list[TaskResult]) -> None:
        """Test success rate calculation."""
        result = TrialResult(
            trial_id="trial_001",
            adapter_name="TestAdapter",
            task_results=tuple(sample_results),
            total_tasks=10,
            successful_tasks=5,
            start_time=datetime.now(),
            end_time=datetime.now(),
        )
        assert result.success_rate == 0.5

    def test_success_rate_empty(self) -> None:
        """Test success rate with no tasks."""
        result = TrialResult(
            trial_id="trial_001",
            adapter_name="TestAdapter",
            task_results=(),
            total_tasks=0,
            successful_tasks=0,
            start_time=datetime.now(),
            end_time=datetime.now(),
        )
        assert result.success_rate == 0.0

    def test_duration(self) -> None:
        """Test duration calculation."""
        start = datetime(2024, 1, 1, 10, 0, 0)
        end = datetime(2024, 1, 1, 10, 1, 30)  # 90 seconds later

        result = TrialResult(
            trial_id="trial_001",
            adapter_name="TestAdapter",
            task_results=(),
            total_tasks=0,
            successful_tasks=0,
            start_time=start,
            end_time=end,
        )
        assert result.duration_seconds == 90.0


class TestTerminalBenchConfig:
    """Tests for the TerminalBenchConfig dataclass."""

    def test_defaults(self, tmp_path: Path) -> None:
        """Test default configuration."""
        config = TerminalBenchConfig(tasks_dir=tmp_path)
        assert config.docker_image == "terminal-bench:latest"
        assert config.timeout_seconds == 600
        assert config.n_trials == 1
        assert config.store_results_in_memory is True

    def test_custom_values(self, tmp_path: Path) -> None:
        """Test custom configuration."""
        config = TerminalBenchConfig(
            tasks_dir=tmp_path,
            results_dir=tmp_path / "results",
            timeout_seconds=300,
            n_trials=5,
            seed=123,
        )
        assert config.timeout_seconds == 300
        assert config.n_trials == 5
        assert config.seed == 123


class TestTerminalBenchRunner:
    """Tests for the TerminalBenchRunner class."""

    @pytest.fixture
    def runner(self, tmp_path: Path) -> TerminalBenchRunner:
        """Create a runner for testing."""
        config = TerminalBenchConfig(
            tasks_dir=tmp_path / "tasks",
            results_dir=tmp_path / "results",
            n_trials=2,
            seed=42,
        )
        return TerminalBenchRunner(
            adapter=MockAdapter(),
            config=config,
        )

    def test_init(self, runner: TerminalBenchRunner) -> None:
        """Test runner initialization."""
        assert runner.results_dir.exists()
        assert runner.agent is not None
        assert runner.task_selector is not None

    def test_run_with_synthetic_tasks(self, runner: TerminalBenchRunner) -> None:
        """Test running with synthetic tasks."""
        results = runner.run()
        assert len(results) == 2  # n_trials
        for trial in results:
            assert trial.total_tasks > 0
            assert trial.adapter_name == "MockAdapter"

    def test_run_with_progress_callback(self, runner: TerminalBenchRunner) -> None:
        """Test running with progress callback."""
        progress_calls = []

        def callback(current: int, total: int, message: str) -> None:
            progress_calls.append((current, total, message))

        results = runner.run(progress_callback=callback)
        assert len(results) == 2
        assert len(progress_calls) > 0

    def test_simulate_execution_with_memory(self, runner: TerminalBenchRunner) -> None:
        """Test simulation with memory context."""
        from src.benchmarks.terminalbench.agent import MemoryAugmentedTask

        task = MemoryAugmentedTask(
            task_id="test",
            description="Test task",
            augmented_description="Augmented",
            memory_context="Previous solution...",
        )
        output, success, error = runner._simulate_execution(task)
        assert "Simulated execution" in output
        assert "memory context" in output

    def test_save_and_load_trial_result(self, runner: TerminalBenchRunner) -> None:
        """Test saving and loading trial results."""
        results = runner.run()
        trial = results[0]

        # Check result was saved
        result_path = runner.results_dir / f"{trial.trial_id}.json"
        assert result_path.exists()

        # Load and verify
        loaded = TerminalBenchRunner.load_trial_result(result_path)
        assert loaded.trial_id == trial.trial_id
        assert loaded.adapter_name == trial.adapter_name
        assert loaded.total_tasks == trial.total_tasks

    def test_get_stats(self, runner: TerminalBenchRunner) -> None:
        """Test getting runner statistics."""
        stats = runner.get_stats()
        assert "adapter" in stats
        assert stats["adapter"] == "MockAdapter"
        assert "n_trials" in stats
        assert stats["n_trials"] == 2

    def test_results_stored_in_memory(self, runner: TerminalBenchRunner) -> None:
        """Test that results are stored in memory when configured."""
        runner.run()
        adapter = runner.adapter
        # Results should be stored (one per task * n_trials)
        assert adapter.add_calls > 0  # type: ignore
