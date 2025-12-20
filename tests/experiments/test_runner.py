"""Tests for ExperimentRunner and related classes."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.experiments.runner import (
    AdapterCondition,
    ExperimentConfig,
    ExperimentResults,
    ExperimentRunner,
    TrialResult,
)


class TestAdapterCondition:
    """Tests for AdapterCondition enum."""

    def test_git_notes_value(self) -> None:
        """Git notes condition has correct value."""
        assert AdapterCondition.GIT_NOTES.value == "git-notes"

    def test_no_memory_value(self) -> None:
        """No memory condition has correct value."""
        assert AdapterCondition.NO_MEMORY.value == "no-memory"

    def test_mock_value(self) -> None:
        """Mock condition has correct value."""
        assert AdapterCondition.MOCK.value == "mock"

    def test_from_string(self) -> None:
        """Can create condition from string value."""
        assert AdapterCondition("git-notes") == AdapterCondition.GIT_NOTES
        assert AdapterCondition("no-memory") == AdapterCondition.NO_MEMORY


class TestTrialResult:
    """Tests for TrialResult dataclass."""

    def test_success_when_no_error(self) -> None:
        """Trial is successful when error is None."""
        result = TrialResult(
            trial_id="test_01",
            adapter=AdapterCondition.NO_MEMORY,
            seed=42,
            metrics={"accuracy": 0.85},
            raw_results={},
            duration_seconds=1.5,
            timestamp="2025-01-01T00:00:00Z",
        )
        assert result.success is True

    def test_failure_when_error_present(self) -> None:
        """Trial is failed when error is set."""
        result = TrialResult(
            trial_id="test_01",
            adapter=AdapterCondition.NO_MEMORY,
            seed=42,
            metrics={},
            raw_results={},
            duration_seconds=0.1,
            timestamp="2025-01-01T00:00:00Z",
            error="Something went wrong",
        )
        assert result.success is False

    def test_to_dict_serialization(self) -> None:
        """Trial result serializes to dict correctly."""
        result = TrialResult(
            trial_id="test_01",
            adapter=AdapterCondition.GIT_NOTES,
            seed=12345,
            metrics={"accuracy": 0.9},
            raw_results={"questions": 10},
            duration_seconds=2.5,
            timestamp="2025-01-01T12:00:00Z",
        )
        data = result.to_dict()
        assert data["trial_id"] == "test_01"
        assert data["adapter"] == "git-notes"
        assert data["seed"] == 12345
        assert data["metrics"]["accuracy"] == 0.9
        assert data["success"] is True


class TestExperimentResults:
    """Tests for ExperimentResults dataclass."""

    def test_total_trials_empty(self) -> None:
        """Total trials is 0 when no trials exist."""
        results = ExperimentResults(
            experiment_id="exp_001",
            benchmark="locomo",
            config={},
        )
        assert results.total_trials == 0

    def test_total_trials_with_data(self) -> None:
        """Total trials counts across all conditions."""
        results = ExperimentResults(
            experiment_id="exp_001",
            benchmark="locomo",
            config={},
        )
        results.trials["git-notes"] = [
            TrialResult("t1", AdapterCondition.GIT_NOTES, 1, {}, {}, 1.0, ""),
            TrialResult("t2", AdapterCondition.GIT_NOTES, 2, {}, {}, 1.0, ""),
        ]
        results.trials["no-memory"] = [
            TrialResult("t3", AdapterCondition.NO_MEMORY, 3, {}, {}, 1.0, ""),
        ]
        assert results.total_trials == 3

    def test_successful_trials_count(self) -> None:
        """Successful trials counts only non-error trials."""
        results = ExperimentResults(
            experiment_id="exp_001",
            benchmark="locomo",
            config={},
        )
        results.trials["git-notes"] = [
            TrialResult("t1", AdapterCondition.GIT_NOTES, 1, {}, {}, 1.0, ""),
            TrialResult("t2", AdapterCondition.GIT_NOTES, 2, {}, {}, 1.0, "", "error"),
        ]
        assert results.successful_trials == 1

    def test_get_metrics_by_condition(self) -> None:
        """Can retrieve metrics for specific condition."""
        results = ExperimentResults(
            experiment_id="exp_001",
            benchmark="locomo",
            config={},
        )
        results.trials["git-notes"] = [
            TrialResult("t1", AdapterCondition.GIT_NOTES, 1, {"acc": 0.8}, {}, 1.0, ""),
            TrialResult("t2", AdapterCondition.GIT_NOTES, 2, {"acc": 0.9}, {}, 1.0, ""),
        ]
        metrics = results.get_metrics_by_condition(AdapterCondition.GIT_NOTES)
        assert len(metrics) == 2
        assert metrics[0]["acc"] == 0.8
        assert metrics[1]["acc"] == 0.9

    def test_save_creates_file(self) -> None:
        """Save creates JSON file with results."""
        results = ExperimentResults(
            experiment_id="exp_test",
            benchmark="locomo",
            config={"num_trials": 5},
            started_at="2025-01-01T00:00:00Z",
            completed_at="2025-01-01T01:00:00Z",
        )
        results.trials["no-memory"] = [
            TrialResult("t1", AdapterCondition.NO_MEMORY, 42, {"x": 1}, {}, 1.0, ""),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "results.json"
            results.save(path)

            assert path.exists()
            with path.open() as f:
                data = json.load(f)
            assert data["experiment_id"] == "exp_test"
            assert data["benchmark"] == "locomo"
            assert data["total_trials"] == 1

    def test_to_dict_serialization(self) -> None:
        """Results serialize to dict correctly."""
        results = ExperimentResults(
            experiment_id="exp_001",
            benchmark="longmemeval",
            config={"adapters": ["no-memory"]},
        )
        data = results.to_dict()
        assert data["experiment_id"] == "exp_001"
        assert data["benchmark"] == "longmemeval"
        assert "trials" in data


class TestExperimentConfig:
    """Tests for ExperimentConfig dataclass."""

    def test_valid_config(self) -> None:
        """Valid config is accepted."""
        config = ExperimentConfig(
            benchmark="locomo",
            adapters=[AdapterCondition.NO_MEMORY],
        )
        assert config.benchmark == "locomo"
        assert config.num_trials == 5

    def test_invalid_benchmark_raises(self) -> None:
        """Invalid benchmark name raises ValueError."""
        with pytest.raises(ValueError, match="Invalid benchmark"):
            ExperimentConfig(
                benchmark="invalid",
                adapters=[AdapterCondition.NO_MEMORY],
            )

    def test_zero_trials_raises(self) -> None:
        """Zero trials raises ValueError."""
        with pytest.raises(ValueError, match="num_trials must be at least 1"):
            ExperimentConfig(
                benchmark="locomo",
                adapters=[AdapterCondition.NO_MEMORY],
                num_trials=0,
            )

    def test_empty_adapters_raises(self) -> None:
        """Empty adapters list raises ValueError."""
        with pytest.raises(ValueError, match="At least one adapter"):
            ExperimentConfig(
                benchmark="locomo",
                adapters=[],
            )

    def test_get_seeds_reproducible(self) -> None:
        """Seeds are reproducible with same base_seed."""
        config1 = ExperimentConfig(
            benchmark="locomo",
            adapters=[AdapterCondition.NO_MEMORY],
            num_trials=5,
            base_seed=42,
        )
        config2 = ExperimentConfig(
            benchmark="locomo",
            adapters=[AdapterCondition.NO_MEMORY],
            num_trials=5,
            base_seed=42,
        )
        assert config1.get_seeds() == config2.get_seeds()

    def test_get_seeds_different_with_different_base(self) -> None:
        """Different base_seed produces different seeds."""
        config1 = ExperimentConfig(
            benchmark="locomo",
            adapters=[AdapterCondition.NO_MEMORY],
            base_seed=42,
        )
        config2 = ExperimentConfig(
            benchmark="locomo",
            adapters=[AdapterCondition.NO_MEMORY],
            base_seed=123,
        )
        assert config1.get_seeds() != config2.get_seeds()

    def test_get_seeds_returns_correct_count(self) -> None:
        """get_seeds returns num_trials seeds."""
        config = ExperimentConfig(
            benchmark="locomo",
            adapters=[AdapterCondition.NO_MEMORY],
            num_trials=10,
        )
        assert len(config.get_seeds()) == 10


class TestExperimentRunner:
    """Tests for ExperimentRunner class."""

    def test_init_with_config(self) -> None:
        """Runner initializes with config."""
        config = ExperimentConfig(
            benchmark="locomo",
            adapters=[AdapterCondition.NO_MEMORY],
        )
        runner = ExperimentRunner(config)
        assert runner.config == config

    def test_register_adapter_factory(self) -> None:
        """Can register custom adapter factory."""
        config = ExperimentConfig(
            benchmark="locomo",
            adapters=[AdapterCondition.GIT_NOTES],
        )
        runner = ExperimentRunner(config)

        mock_adapter = MagicMock()
        factory = lambda: mock_adapter

        runner.register_adapter_factory(AdapterCondition.GIT_NOTES, factory)

        adapter = runner._create_adapter(AdapterCondition.GIT_NOTES)
        assert adapter is mock_adapter

    def test_create_adapter_no_memory_default(self) -> None:
        """NO_MEMORY condition uses default NoMemoryAdapter."""
        config = ExperimentConfig(
            benchmark="locomo",
            adapters=[AdapterCondition.NO_MEMORY],
        )
        runner = ExperimentRunner(config)

        from src.adapters.no_memory import NoMemoryAdapter

        adapter = runner._create_adapter(AdapterCondition.NO_MEMORY)
        assert isinstance(adapter, NoMemoryAdapter)

    def test_create_adapter_mock_default(self) -> None:
        """MOCK condition uses default MockAdapter."""
        config = ExperimentConfig(
            benchmark="locomo",
            adapters=[AdapterCondition.MOCK],
        )
        runner = ExperimentRunner(config)

        from src.adapters.mock import MockAdapter

        adapter = runner._create_adapter(AdapterCondition.MOCK)
        assert isinstance(adapter, MockAdapter)

    def test_create_adapter_unregistered_raises(self) -> None:
        """Unregistered condition without default raises ValueError."""
        config = ExperimentConfig(
            benchmark="locomo",
            adapters=[AdapterCondition.GIT_NOTES],
        )
        runner = ExperimentRunner(config)

        with pytest.raises(ValueError, match="No adapter factory registered"):
            runner._create_adapter(AdapterCondition.GIT_NOTES)


class TestExperimentRunnerAsync:
    """Async tests for ExperimentRunner."""

    @pytest.fixture
    def mock_llm_client(self) -> MagicMock:
        """Create a mock LLM client."""
        client = MagicMock()
        client.generate = AsyncMock(
            return_value=MagicMock(
                text="The answer is correct.",
                confidence=0.9,
            )
        )
        return client

    @pytest.mark.asyncio
    async def test_run_trial_success(self) -> None:
        """Single trial runs successfully with mock adapter."""
        config = ExperimentConfig(
            benchmark="locomo",
            adapters=[AdapterCondition.MOCK],
            num_trials=1,
        )
        runner = ExperimentRunner(config)

        # Mock the pipeline creation to avoid full benchmark execution
        mock_assessment = MagicMock()
        mock_assessment.overall_accuracy = 0.85
        mock_assessment.adversarial_accuracy = 0.80
        mock_assessment.category_metrics = {}
        mock_assessment.conversation_count = 10
        mock_assessment.question_count = 50
        mock_assessment.conversation_results = []

        mock_pipeline = MagicMock()
        mock_pipeline.run_assessment = AsyncMock(return_value=mock_assessment)

        runner._create_benchmark_pipeline = MagicMock(return_value=mock_pipeline)

        result = await runner._run_trial(AdapterCondition.MOCK, 0, 42)

        assert result.success is True
        assert result.metrics["overall_accuracy"] == 0.85
        assert result.adapter == AdapterCondition.MOCK

    @pytest.mark.asyncio
    async def test_run_trial_captures_error(self) -> None:
        """Trial captures error when pipeline fails."""
        config = ExperimentConfig(
            benchmark="locomo",
            adapters=[AdapterCondition.MOCK],
            num_trials=1,
        )
        runner = ExperimentRunner(config)

        mock_pipeline = MagicMock()
        mock_pipeline.run_assessment = AsyncMock(side_effect=RuntimeError("Pipeline crashed"))

        runner._create_benchmark_pipeline = MagicMock(return_value=mock_pipeline)

        result = await runner._run_trial(AdapterCondition.MOCK, 0, 42)

        assert result.success is False
        assert "Pipeline crashed" in result.error

    @pytest.mark.asyncio
    async def test_run_executes_all_trials(self) -> None:
        """Run executes all trials for all conditions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ExperimentConfig(
                benchmark="locomo",
                adapters=[AdapterCondition.MOCK, AdapterCondition.NO_MEMORY],
                num_trials=2,
                output_dir=tmpdir,
            )
            runner = ExperimentRunner(config)

            # Mock pipeline
            mock_assessment = MagicMock()
            mock_assessment.overall_accuracy = 0.85
            mock_assessment.adversarial_accuracy = 0.80
            mock_assessment.category_metrics = {}
            mock_assessment.conversation_count = 10
            mock_assessment.question_count = 50
            mock_assessment.conversation_results = []

            mock_pipeline = MagicMock()
            mock_pipeline.run_assessment = AsyncMock(return_value=mock_assessment)
            runner._create_benchmark_pipeline = MagicMock(return_value=mock_pipeline)

            results = await runner.run()

            # 2 conditions x 2 trials = 4 total
            assert results.total_trials == 4
            assert len(results.trials["mock"]) == 2
            assert len(results.trials["no-memory"]) == 2

    @pytest.mark.asyncio
    async def test_run_calls_progress_callback(self) -> None:
        """Run calls progress callback for each trial."""
        progress_calls: list[tuple[str, int, int]] = []

        def progress_cb(msg: str, completed: int, total: int) -> None:
            progress_calls.append((msg, completed, total))

        with tempfile.TemporaryDirectory() as tmpdir:
            config = ExperimentConfig(
                benchmark="locomo",
                adapters=[AdapterCondition.MOCK],
                num_trials=3,
                output_dir=tmpdir,
                progress_callback=progress_cb,
            )
            runner = ExperimentRunner(config)

            mock_assessment = MagicMock()
            mock_assessment.overall_accuracy = 0.85
            mock_assessment.adversarial_accuracy = 0.80
            mock_assessment.category_metrics = {}
            mock_assessment.conversation_count = 10
            mock_assessment.question_count = 50
            mock_assessment.conversation_results = []

            mock_pipeline = MagicMock()
            mock_pipeline.run_assessment = AsyncMock(return_value=mock_assessment)
            runner._create_benchmark_pipeline = MagicMock(return_value=mock_pipeline)

            await runner.run()

            assert len(progress_calls) == 3
            # Check progress increments
            assert progress_calls[0][1] == 0  # First call: 0 completed
            assert progress_calls[1][1] == 1  # Second call: 1 completed
            assert progress_calls[2][1] == 2  # Third call: 2 completed

    @pytest.mark.asyncio
    async def test_run_saves_results(self) -> None:
        """Run saves results to output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ExperimentConfig(
                benchmark="locomo",
                adapters=[AdapterCondition.MOCK],
                num_trials=1,
                output_dir=tmpdir,
            )
            runner = ExperimentRunner(config)

            mock_assessment = MagicMock()
            mock_assessment.overall_accuracy = 0.85
            mock_assessment.adversarial_accuracy = 0.80
            mock_assessment.category_metrics = {}
            mock_assessment.conversation_count = 10
            mock_assessment.question_count = 50
            mock_assessment.conversation_results = []

            mock_pipeline = MagicMock()
            mock_pipeline.run_assessment = AsyncMock(return_value=mock_assessment)
            runner._create_benchmark_pipeline = MagicMock(return_value=mock_pipeline)

            results = await runner.run()

            # Check file was created
            output_path = Path(tmpdir) / f"{results.experiment_id}.json"
            assert output_path.exists()

            # Verify content
            with output_path.open() as f:
                saved_data = json.load(f)
            assert saved_data["experiment_id"] == results.experiment_id
            assert saved_data["benchmark"] == "locomo"


class TestLongMemEvalExperiment:
    """Tests specific to LongMemEval benchmark experiments."""

    @pytest.mark.asyncio
    async def test_longmemeval_metrics_extraction(self) -> None:
        """LongMemEval assessment metrics are correctly extracted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ExperimentConfig(
                benchmark="longmemeval",
                adapters=[AdapterCondition.MOCK],
                num_trials=1,
                output_dir=tmpdir,
            )
            runner = ExperimentRunner(config)

            # Mock LongMemEval assessment result
            mock_assessment = MagicMock()
            mock_assessment.accuracy = 0.75
            mock_assessment.abstention_rate = 0.10
            mock_assessment.answered_correctly = 15
            mock_assessment.total_questions = 20
            mock_assessment.session_count = 5
            mock_assessment.question_results = []

            mock_pipeline = MagicMock()
            mock_pipeline.run_assessment = AsyncMock(return_value=mock_assessment)
            runner._create_benchmark_pipeline = MagicMock(return_value=mock_pipeline)

            result = await runner._run_trial(AdapterCondition.MOCK, 0, 42)

            assert result.success is True
            assert result.metrics["accuracy"] == 0.75
            assert result.metrics["abstention_rate"] == 0.10
            assert result.metrics["answered_correctly"] == 15
            assert result.metrics["total_questions"] == 20
