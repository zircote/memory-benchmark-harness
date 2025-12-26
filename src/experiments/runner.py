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
import logging
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from src.adapters.base import MemorySystemAdapter

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.evaluation.judge import LLMJudge
    from src.evaluation.mock_judge import MockLLMJudge


class AdapterCondition(Enum):
    """Memory adapter conditions for A/B comparison."""

    GIT_NOTES = "git-notes"
    NO_MEMORY = "no-memory"
    MOCK = "mock"


class BenchmarkProtocol(Protocol):
    """Protocol for benchmark pipelines.

    Both LongMemEval and LoCoMo pipelines implement this interface.
    This protocol is relaxed since actual implementations have more specific signatures.
    """

    def run(self, dataset: Any, *args: Any, **kwargs: Any) -> Any:
        """Run the full benchmark assessment."""
        ...


class LLMClientProtocol(Protocol):
    """Protocol for LLM clients used by benchmarks."""

    async def generate(self, prompt: str, **kwargs: Any) -> Any:
        """Generate a response from the LLM."""
        ...


class _ContextBenchWrapper:
    """Wrapper to adapt ContextBenchPipeline to BenchmarkProtocol.

    ContextBenchPipeline uses evaluate() instead of run(), and needs
    the agent+pipeline created with the dataset at run time.
    """

    def __init__(self, adapter: Any, llm_client: Any, judge: Any) -> None:
        self._adapter = adapter
        self._llm_client = llm_client
        self._judge = judge

    def run(self, dataset: Any, *_args: Any, **_kwargs: Any) -> Any:
        """Run evaluation with dataset."""
        from src.benchmarks.contextbench import ContextBenchAgent, ContextBenchPipeline

        # Create agent with dataset
        agent = ContextBenchAgent(
            adapter=self._adapter,
            dataset=dataset,
            llm=self._llm_client,
        )
        # Create pipeline with agent and judge
        pipeline = ContextBenchPipeline(agent=agent, judge=self._judge)
        # Call evaluate() which returns EvaluationResult
        return pipeline.evaluate(dataset=dataset)


class _MemoryAgentBenchWrapper:
    """Wrapper to adapt MemoryAgentBenchPipeline to BenchmarkProtocol.

    MemoryAgentBenchPipeline uses evaluate_dataset() instead of run().
    """

    def __init__(self, pipeline: Any) -> None:
        self._pipeline = pipeline

    def run(self, dataset: Any, *_args: Any, **_kwargs: Any) -> Any:
        """Run evaluation with dataset."""
        # Call evaluate_dataset() which returns SplitResult
        return self._pipeline.evaluate_dataset(dataset)


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
        valid_benchmarks = {
            "longmemeval",
            "locomo",
            "contextbench",
            "memoryagentbench",
            # "terminalbench" requires Docker/Harbor setup - not yet integrated
        }
        if self.benchmark not in valid_benchmarks:
            raise ValueError(
                f"Invalid benchmark: {self.benchmark}. Must be one of: {valid_benchmarks}"
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
        self._judge: LLMJudge | None = None
        self._dataset: Any = None  # Cached dataset to avoid reloading

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

    def _get_or_create_judge(self) -> LLMJudge | MockLLMJudge:
        """Get existing judge or create a new one.

        Uses MockLLMJudge when the LLM client is a MockLLMClient,
        otherwise creates a real LLMJudge that uses OpenAI API.
        """
        if self._judge is None:
            # Check if we're using MockLLMClient by checking class name
            # (avoids circular import by not importing MockLLMClient directly)
            client_class = type(self.config.llm_client).__name__
            if client_class == "MockLLMClient":
                from src.evaluation.mock_judge import MockLLMJudge

                self._judge = MockLLMJudge()  # type: ignore[assignment]
            else:
                from src.evaluation.judge import LLMJudge

                self._judge = LLMJudge()  # Uses OPENAI_API_KEY from env
        return self._judge

    def _load_dataset(self) -> tuple[Any, str]:
        """Load the benchmark dataset.

        Returns:
            Tuple of (dataset, benchmark_type)
        """
        if self.config.benchmark == "longmemeval":
            from src.benchmarks.longmemeval import load_longmemeval, load_longmemeval_from_file

            if self.config.dataset_path:
                return load_longmemeval_from_file(self.config.dataset_path), "longmemeval"
            return load_longmemeval(), "longmemeval"

        elif self.config.benchmark == "locomo":
            from src.benchmarks.locomo import load_locomo, load_locomo_from_file

            if self.config.dataset_path:
                return load_locomo_from_file(self.config.dataset_path), "locomo"
            return load_locomo(), "locomo"

        elif self.config.benchmark == "contextbench":
            from src.benchmarks.contextbench import (
                load_contextbench,
                load_contextbench_from_file,
            )

            if self.config.dataset_path:
                return load_contextbench_from_file(self.config.dataset_path), "contextbench"
            return load_contextbench(), "contextbench"

        elif self.config.benchmark == "memoryagentbench":
            from src.benchmarks.memoryagentbench import load_memoryagentbench

            # Note: load_memoryagentbench_from_file requires competency param,
            # so we only support HuggingFace loading for now (dataset_path ignored)
            return load_memoryagentbench(), "memoryagentbench"

        else:
            raise ValueError(f"Unknown benchmark: {self.config.benchmark}")

    def _create_benchmark_pipeline(
        self,
        adapter: MemorySystemAdapter,
    ) -> BenchmarkProtocol:
        """Create a benchmark pipeline for the given adapter.

        Args:
            adapter: Memory adapter to use

        Returns:
            Benchmark pipeline instance

        Raises:
            ValueError: If llm_client is not configured
        """
        if self.config.llm_client is None:
            raise ValueError(
                "llm_client is required for running experiments. "
                "Pass an LLM client to ExperimentConfig."
            )

        judge = self._get_or_create_judge()

        if self.config.benchmark == "longmemeval":
            from src.benchmarks.longmemeval import BenchmarkPipeline

            # Create pipeline with correct positional args: (adapter, llm_client, judge)
            return BenchmarkPipeline(
                adapter,
                self.config.llm_client,  # type: ignore[arg-type]
                judge,
            )

        elif self.config.benchmark == "locomo":
            from src.benchmarks.locomo import LoCoMoPipeline

            # Create pipeline with correct positional args: (adapter, llm_client, judge)
            return LoCoMoPipeline(
                adapter,
                self.config.llm_client,  # type: ignore[arg-type]
                judge,
            )

        elif self.config.benchmark == "contextbench":
            # ContextBench pipeline will create its agent internally
            # We pass adapter and llm_client via a deferred wrapper
            return _ContextBenchWrapper(
                adapter=adapter,
                llm_client=self.config.llm_client,
                judge=judge,
            )

        elif self.config.benchmark == "memoryagentbench":
            from src.benchmarks.memoryagentbench import (
                MemoryAgentBenchAgent,
                MemoryAgentBenchPipeline,
            )

            # MemoryAgentBench needs agent wrapping adapter+llm, then pipeline wrapping agent+judge
            agent = MemoryAgentBenchAgent(
                adapter=adapter,
                llm=self.config.llm_client,  # type: ignore[arg-type]
            )
            pipeline = MemoryAgentBenchPipeline(agent=agent, judge=judge)  # type: ignore[arg-type]
            return _MemoryAgentBenchWrapper(pipeline)

        else:
            raise ValueError(f"Unknown benchmark: {self.config.benchmark}")

    def _run_trial(
        self,
        condition: AdapterCondition,
        trial_num: int,
        seed: int,
        dataset: Any,
    ) -> TrialResult:
        """Run a single trial.

        Args:
            condition: Memory adapter condition
            trial_num: Trial number (0-indexed)
            seed: Random seed for this trial
            dataset: The benchmark dataset to use

        Returns:
            TrialResult with metrics and raw results
        """
        import time

        trial_id = f"{condition.value}_{trial_num:02d}_{seed}"
        start_time = time.monotonic()

        try:
            # Set random seed for reproducibility
            random.seed(seed)

            # Create fresh adapter and pipeline for this trial
            adapter = self._create_adapter(condition)

            # CRITICAL: Clear adapter index before each trial to prevent accumulation
            # For git-notes adapters, multiple adapter instances share the same SQLite
            # index (same repo path). Without clearing, each trial adds 200k messages
            # on top of previous trials, causing memory to grow: 200k -> 400k -> 600k
            clear_result = adapter.clear()
            if not clear_result.success:
                logger.warning(
                    "Failed to clear adapter before trial %d: %s",
                    trial_num,
                    clear_result.error,
                )

            pipeline = self._create_benchmark_pipeline(adapter)

            # Run the assessment (synchronous)
            assessment = pipeline.run(dataset)

            # Extract metrics based on benchmark type
            if self.config.benchmark == "longmemeval":
                metrics = {
                    "accuracy": assessment.accuracy,
                    "mean_score": assessment.mean_score,
                    "abstention_accuracy": assessment.abstention_accuracy,
                    "correct_count": assessment.correct_count,
                    "partial_count": assessment.partial_count,
                    "total_questions": assessment.total_questions,
                }
                raw_results = {
                    "dataset_subset": assessment.dataset_subset,
                    "ingestion_time_ms": assessment.ingestion_time_ms,
                    "assessment_time_ms": assessment.assessment_time_ms,
                    "scores_by_type": assessment.scores_by_type(),
                    "question_results": [
                        {
                            "question_id": r.question_id,
                            "is_correct": r.is_correct,
                            "is_partial": r.is_partial,
                            "score": r.score,
                            "is_abstention_actual": r.is_abstention_actual,
                            "is_abstention_expected": r.is_abstention_expected,
                            "agent_answer": r.agent_answer[:200],  # Truncate for storage
                        }
                        for r in assessment.question_results
                    ],
                }
            elif self.config.benchmark == "locomo":
                metrics = {
                    "accuracy": assessment.accuracy,
                    "mean_score": assessment.mean_score,
                    "adversarial_accuracy": assessment.adversarial_accuracy,
                    "correct_count": assessment.correct_count,
                    "partial_count": assessment.partial_count,
                    "total_questions": assessment.total_questions,
                    "category_accuracies": assessment.accuracy_by_category(),
                }
                raw_results = {
                    "ingestion_time_ms": assessment.ingestion_time_ms,
                    "assessment_time_ms": assessment.assessment_time_ms,
                    "scores_by_category": assessment.scores_by_category(),
                    "conversation_count": len(assessment.conversation_metrics),
                    "conversation_results": [
                        {
                            "conversation_id": conv_id,
                            "accuracy": m.accuracy,
                            "questions_assessed": m.questions_assessed,
                        }
                        for conv_id, m in assessment.conversation_metrics.items()
                    ],
                }
            elif self.config.benchmark == "contextbench":
                # EvaluationResult from ContextBenchPipeline
                metrics = {
                    "accuracy": assessment.accuracy,
                    "correct_count": assessment.correct_count,
                    "total_questions": assessment.total_questions,
                    "total_cost": assessment.total_cost,
                    "avg_operations": assessment.avg_operations,
                }
                raw_results = {
                    "started_at": assessment.started_at,
                    "completed_at": assessment.completed_at,
                    "category_breakdown": assessment.category_breakdown,
                    "hop_breakdown": assessment.hop_breakdown,
                    "question_results": [
                        {
                            "question_id": r.question_id,
                            "correct": r.correct,
                            "category": r.category.value,
                            "hop_count": r.hop_count,
                            "operations_count": r.operations_count,
                        }
                        for r in assessment.question_results
                    ],
                }
            elif self.config.benchmark == "memoryagentbench":
                # SplitResult from MemoryAgentBenchPipeline
                metrics = {
                    "accuracy": assessment.overall_accuracy,
                    "total_questions": assessment.total_questions,
                    "conflict_resolution_accuracy": assessment.conflict_resolution_accuracy,
                }
                raw_results = {
                    "started_at": assessment.started_at,
                    "completed_at": assessment.completed_at,
                    "competency_results": {
                        c.value: {
                            "accuracy": r.accuracy,
                            "total_questions": r.total_questions,
                            "correct_count": r.correct_count,
                        }
                        for c, r in assessment.competency_results.items()
                    },
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

    def run(self) -> ExperimentResults:
        """Run the complete experiment.

        Executes all trials across all adapter conditions with reproducible
        seeds, collecting results for statistical analysis.

        Returns:
            ExperimentResults containing all trial outcomes
        """
        import uuid

        experiment_id = f"exp_{self.config.benchmark}_{uuid.uuid4().hex[:8]}"

        # Capture adapter versions for cross-version comparison
        adapter_versions: dict[str, str] = {}
        for condition in self.config.adapters:
            try:
                adapter = self._create_adapter(condition)
                version = adapter.get_version()
                if version:
                    adapter_versions[condition.value] = version
            except Exception:
                pass  # Adapter creation may fail if not configured

        results = ExperimentResults(
            experiment_id=experiment_id,
            benchmark=self.config.benchmark,
            config={
                "adapters": [a.value for a in self.config.adapters],
                "num_trials": self.config.num_trials,
                "base_seed": self.config.base_seed,
                "dataset_path": self.config.dataset_path,
                "adapter_versions": adapter_versions,
            },
            started_at=datetime.now(UTC).isoformat(),
        )

        # Load dataset once for all trials
        dataset, _ = self._load_dataset()

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

                trial_result = self._run_trial(condition, trial_num, seed, dataset)
                results.trials[condition.value].append(trial_result)
                completed += 1

        results.completed_at = datetime.now(UTC).isoformat()

        # Save results
        output_path = Path(self.config.output_dir) / f"{experiment_id}.json"
        results.save(output_path)

        return results


def run_experiment(
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
        llm_client: LLM client for answer generation

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
    return runner.run()
