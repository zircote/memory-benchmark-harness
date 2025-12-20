"""Experiment runner for benchmark assessment.

This module provides the core experiment orchestration, managing multiple
benchmark trials across different memory adapter conditions with reproducible
seeds and structured result persistence.

The runner follows the spec requirements:
- 5 runs per condition with different seeds
- Both git-notes and no-memory conditions
- Raw result recording for statistical analysis
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from src.adapters.base import MemorySystemAdapter


class AdapterCondition(Enum):
    """Memory adapter conditions for A/B comparison."""

    GIT_NOTES = "git-notes"
    NO_MEMORY = "no-memory"
    MOCK = "mock"


class BenchmarkProtocol(Protocol):
    """Protocol for benchmark pipelines.

    Both LongMemEval and LoCoMo pipelines implement this interface.
    """

    async def run_assessment(self) -> Any:
        """Run the full benchmark assessment."""
        ...


class LLMClientProtocol(Protocol):
    """Protocol for LLM clients used by benchmarks."""

    async def generate(self, prompt: str, **kwargs: Any) -> Any:
        """Generate a response from the LLM."""
        ...


@dataclass(frozen=True, slots=True)
class TrialResult:
    """Result from a single benchmark trial.

    Attributes:
        trial_id: Unique identifier for this trial
        adapter: The memory adapter condition used
        seed: Random seed for reproducibility
        metrics: Computed metrics from the assessment
        raw_results: Full assessment results object (serialized)
        duration_seconds: Time taken to complete the trial
        timestamp: When the trial completed
        error: Error message if trial failed
    """

    trial_id: str
    adapter: AdapterCondition
    seed: int
    metrics: dict[str, Any]
    raw_results: dict[str, Any]
    duration_seconds: float
    timestamp: str
    error: str | None = None

    @property
    def success(self) -> bool:
        """Whether the trial completed successfully."""
        return self.error is None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON export."""
        return {
            "trial_id": self.trial_id,
            "adapter": self.adapter.value,
            "seed": self.seed,
            "metrics": self.metrics,
            "raw_results": self.raw_results,
            "duration_seconds": self.duration_seconds,
            "timestamp": self.timestamp,
            "error": self.error,
            "success": self.success,
        }


@dataclass(slots=True)
class ExperimentResults:
    """Results from a complete experiment run.

    Attributes:
        experiment_id: Unique experiment identifier
        benchmark: Name of the benchmark (longmemeval, locomo)
        config: Experiment configuration used
        trials: All trial results by condition
        started_at: Experiment start timestamp
        completed_at: Experiment completion timestamp
    """

    experiment_id: str
    benchmark: str
    config: dict[str, Any]
    trials: dict[str, list[TrialResult]] = field(default_factory=dict)
    started_at: str = ""
    completed_at: str = ""

    @property
    def total_trials(self) -> int:
        """Total number of trials across all conditions."""
        return sum(len(t) for t in self.trials.values())

    @property
    def successful_trials(self) -> int:
        """Number of successful trials."""
        return sum(sum(1 for t in trials if t.success) for trials in self.trials.values())

    def get_metrics_by_condition(self, condition: AdapterCondition) -> list[dict[str, Any]]:
        """Get all metrics for a specific condition."""
        return [t.metrics for t in self.trials.get(condition.value, []) if t.success]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON export."""
        return {
            "experiment_id": self.experiment_id,
            "benchmark": self.benchmark,
            "config": self.config,
            "trials": {
                condition: [t.to_dict() for t in trials]
                for condition, trials in self.trials.items()
            },
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "total_trials": self.total_trials,
            "successful_trials": self.successful_trials,
        }

    def save(self, path: Path) -> None:
        """Save results to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


@dataclass(slots=True)
class ExperimentConfig:
    """Configuration for a benchmark experiment.

    Attributes:
        benchmark: Benchmark to run (longmemeval, locomo)
        adapters: Memory adapter conditions to test
        num_trials: Number of trials per condition (default: 5)
        base_seed: Starting seed for reproducibility
        output_dir: Directory to save results
        dataset_path: Path to benchmark dataset
        llm_client: LLM client for benchmark judging
        progress_callback: Optional callback for progress updates
    """

    benchmark: str
    adapters: list[AdapterCondition]
    num_trials: int = 5
    base_seed: int = 42
    output_dir: str = "results"
    dataset_path: str | None = None
    llm_client: LLMClientProtocol | None = None
    progress_callback: Callable[[str, int, int], None] | None = None

    def __post_init__(self) -> None:
        """Validate configuration."""
        valid_benchmarks = {"longmemeval", "locomo"}
        if self.benchmark not in valid_benchmarks:
            raise ValueError(
                f"Invalid benchmark: {self.benchmark}. " f"Must be one of: {valid_benchmarks}"
            )
        if self.num_trials < 1:
            raise ValueError("num_trials must be at least 1")
        if not self.adapters:
            raise ValueError("At least one adapter condition must be specified")

    def get_seeds(self) -> list[int]:
        """Generate reproducible seeds for all trials."""
        rng = random.Random(self.base_seed)
        return [rng.randint(0, 2**31 - 1) for _ in range(self.num_trials)]


class ExperimentRunner:
    """Orchestrates benchmark experiments across conditions.

    The runner manages:
    - Creating appropriate memory adapters for each condition
    - Running multiple trials with different seeds
    - Collecting and persisting results
    - Progress reporting

    Example:
        ```python
        config = ExperimentConfig(
            benchmark="longmemeval",
            adapters=[AdapterCondition.GIT_NOTES, AdapterCondition.NO_MEMORY],
            num_trials=5,
        )
        runner = ExperimentRunner(config)
        results = await runner.run()
        results.save(Path("results/experiment.json"))
        ```
    """

    def __init__(self, config: ExperimentConfig) -> None:
        """Initialize the experiment runner.

        Args:
            config: Experiment configuration
        """
        self.config = config
        self._adapter_factories: dict[AdapterCondition, Callable[[], MemorySystemAdapter]] = {}

    def register_adapter_factory(
        self,
        condition: AdapterCondition,
        factory: Callable[[], MemorySystemAdapter],
    ) -> None:
        """Register a factory function for creating adapters.

        Args:
            condition: The adapter condition this factory creates
            factory: Callable that returns a new adapter instance
        """
        self._adapter_factories[condition] = factory

    def _create_adapter(self, condition: AdapterCondition) -> MemorySystemAdapter:
        """Create an adapter for the given condition.

        Args:
            condition: The adapter condition to create

        Returns:
            New memory adapter instance

        Raises:
            ValueError: If no factory registered for condition
        """
        if condition not in self._adapter_factories:
            # Fall back to default adapters
            from src.adapters.mock import MockAdapter
            from src.adapters.no_memory import NoMemoryAdapter

            if condition == AdapterCondition.NO_MEMORY:
                return NoMemoryAdapter()
            elif condition == AdapterCondition.MOCK:
                return MockAdapter()
            else:
                raise ValueError(
                    f"No adapter factory registered for {condition}. "
                    "Use register_adapter_factory() to register one."
                )
        return self._adapter_factories[condition]()

    def _create_benchmark_pipeline(
        self,
        adapter: MemorySystemAdapter,
        seed: int,
    ) -> BenchmarkProtocol:
        """Create a benchmark pipeline for the given adapter.

        Args:
            adapter: Memory adapter to use
            seed: Random seed for the trial

        Returns:
            Benchmark pipeline instance
        """
        if self.config.benchmark == "longmemeval":
            from src.benchmarks.longmemeval import (
                BenchmarkPipeline,
                load_longmemeval,
                load_longmemeval_from_file,
            )

            # Load dataset
            if self.config.dataset_path:
                dataset = load_longmemeval_from_file(self.config.dataset_path)
            else:
                dataset = load_longmemeval()

            # Create pipeline
            return BenchmarkPipeline(
                dataset=dataset,
                adapter=adapter,
                llm_client=self.config.llm_client,
                seed=seed,
            )

        elif self.config.benchmark == "locomo":
            from src.benchmarks.locomo import LoCoMoPipeline, load_locomo, load_locomo_from_file

            # Load dataset
            if self.config.dataset_path:
                dataset = load_locomo_from_file(self.config.dataset_path)
            else:
                dataset = load_locomo()

            # Create pipeline
            return LoCoMoPipeline(
                dataset=dataset,
                adapter=adapter,
                llm_client=self.config.llm_client,
                seed=seed,
            )

        else:
            raise ValueError(f"Unknown benchmark: {self.config.benchmark}")

    async def _run_trial(
        self,
        condition: AdapterCondition,
        trial_num: int,
        seed: int,
    ) -> TrialResult:
        """Run a single trial.

        Args:
            condition: Memory adapter condition
            trial_num: Trial number (0-indexed)
            seed: Random seed for this trial

        Returns:
            TrialResult with metrics and raw results
        """
        import time

        trial_id = f"{condition.value}_{trial_num:02d}_{seed}"
        start_time = time.monotonic()

        try:
            # Create fresh adapter and pipeline for this trial
            adapter = self._create_adapter(condition)
            pipeline = self._create_benchmark_pipeline(adapter, seed)

            # Run the assessment
            assessment = await pipeline.run_assessment()

            # Extract metrics based on benchmark type
            if self.config.benchmark == "longmemeval":
                metrics = {
                    "accuracy": assessment.accuracy,
                    "abstention_rate": assessment.abstention_rate,
                    "answered_correctly": assessment.answered_correctly,
                    "total_questions": assessment.total_questions,
                }
                raw_results = {
                    "session_count": assessment.session_count,
                    "question_results": [
                        {
                            "question_id": r.question_id,
                            "correct": r.correct,
                            "abstained": r.abstained,
                            "predicted": r.predicted_answer,
                        }
                        for r in assessment.question_results
                    ],
                }
            elif self.config.benchmark == "locomo":
                metrics = {
                    "overall_accuracy": assessment.overall_accuracy,
                    "adversarial_accuracy": assessment.adversarial_accuracy,
                    "category_accuracies": {
                        cat.name: m.accuracy for cat, m in assessment.category_metrics.items()
                    },
                }
                raw_results = {
                    "conversation_count": assessment.conversation_count,
                    "question_count": assessment.question_count,
                    "conversation_results": [
                        {
                            "sample_id": cr.sample_id,
                            "accuracy": cr.accuracy,
                        }
                        for cr in assessment.conversation_results
                    ],
                }
            else:
                metrics = {}
                raw_results = {}

            duration = time.monotonic() - start_time

            return TrialResult(
                trial_id=trial_id,
                adapter=condition,
                seed=seed,
                metrics=metrics,
                raw_results=raw_results,
                duration_seconds=duration,
                timestamp=datetime.now(UTC).isoformat(),
            )

        except Exception as e:
            duration = time.monotonic() - start_time
            return TrialResult(
                trial_id=trial_id,
                adapter=condition,
                seed=seed,
                metrics={},
                raw_results={},
                duration_seconds=duration,
                timestamp=datetime.now(UTC).isoformat(),
                error=str(e),
            )

    async def run(self) -> ExperimentResults:
        """Run the complete experiment.

        Executes all trials across all adapter conditions with reproducible
        seeds, collecting results for statistical analysis.

        Returns:
            ExperimentResults containing all trial outcomes
        """
        import uuid

        experiment_id = f"exp_{self.config.benchmark}_{uuid.uuid4().hex[:8]}"
        results = ExperimentResults(
            experiment_id=experiment_id,
            benchmark=self.config.benchmark,
            config={
                "adapters": [a.value for a in self.config.adapters],
                "num_trials": self.config.num_trials,
                "base_seed": self.config.base_seed,
                "dataset_path": self.config.dataset_path,
            },
            started_at=datetime.now(UTC).isoformat(),
        )

        seeds = self.config.get_seeds()
        total_trials = len(self.config.adapters) * self.config.num_trials
        completed = 0

        for condition in self.config.adapters:
            results.trials[condition.value] = []

            for trial_num, seed in enumerate(seeds):
                # Progress callback
                if self.config.progress_callback:
                    self.config.progress_callback(
                        f"Running {condition.value} trial {trial_num + 1}/{self.config.num_trials}",
                        completed,
                        total_trials,
                    )

                trial_result = await self._run_trial(condition, trial_num, seed)
                results.trials[condition.value].append(trial_result)
                completed += 1

        results.completed_at = datetime.now(UTC).isoformat()

        # Save results
        output_path = Path(self.config.output_dir) / f"{experiment_id}.json"
        results.save(output_path)

        return results


async def run_experiment(
    benchmark: str,
    conditions: list[str] | None = None,
    num_trials: int = 5,
    output_dir: str = "results",
    dataset_path: str | None = None,
    llm_client: LLMClientProtocol | None = None,
) -> ExperimentResults:
    """Convenience function to run an experiment.

    Args:
        benchmark: Benchmark to run (longmemeval, locomo)
        conditions: Adapter conditions (default: git-notes and no-memory)
        num_trials: Number of trials per condition
        output_dir: Directory to save results
        dataset_path: Optional path to dataset
        llm_client: LLM client for judging

    Returns:
        ExperimentResults containing all trial outcomes
    """
    if conditions is None:
        conditions = ["git-notes", "no-memory"]

    adapters = [AdapterCondition(c) for c in conditions]

    config = ExperimentConfig(
        benchmark=benchmark,
        adapters=adapters,
        num_trials=num_trials,
        output_dir=output_dir,
        dataset_path=dataset_path,
        llm_client=llm_client,
    )

    runner = ExperimentRunner(config)
    return await runner.run()
