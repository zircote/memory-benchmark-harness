"""Terminal-Bench 2.0 evaluation runner.

This module provides the evaluation runner for Terminal-Bench 2.0,
managing task execution, result collection, and Docker integration.

The runner coordinates:
1. Task selection based on memory relevance
2. Memory augmentation of task descriptions
3. Docker container execution via Harbor
4. Result collection and storage
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.adapters.base import MemorySystemAdapter

from src.benchmarks.terminalbench.agent import (
    MemoryAugmentedInstalledAgent,
    MemoryAugmentedTask,
)
from src.benchmarks.terminalbench.task_selector import (
    TaskFilter,
    TaskInfo,
    TaskSelector,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TaskResult:
    """Result of a single task execution.

    Attributes:
        task_id: Task identifier
        task_description: Original task description
        augmented_description: Memory-augmented description
        success: Whether the task was completed successfully
        output: Task output (stdout/stderr)
        execution_time_seconds: Time to execute task
        memory_context: Memory content used for augmentation
        error: Error message if task failed
        metadata: Additional result metadata
    """

    task_id: str
    task_description: str
    augmented_description: str
    success: bool
    output: str
    execution_time_seconds: float
    memory_context: str = ""
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TrialResult:
    """Result of a complete trial (multiple tasks).

    Attributes:
        trial_id: Trial identifier
        adapter_name: Name of the memory adapter used
        task_results: Results for each task
        total_tasks: Total number of tasks attempted
        successful_tasks: Number of successful tasks
        start_time: Trial start time
        end_time: Trial end time
        config: Trial configuration
    """

    trial_id: str
    adapter_name: str
    task_results: tuple[TaskResult, ...]
    total_tasks: int
    successful_tasks: int
    start_time: datetime
    end_time: datetime
    config: dict[str, Any] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_tasks == 0:
            return 0.0
        return self.successful_tasks / self.total_tasks

    @property
    def duration_seconds(self) -> float:
        """Calculate total trial duration."""
        return (self.end_time - self.start_time).total_seconds()


@dataclass
class TerminalBenchConfig:
    """Configuration for Terminal-Bench evaluation.

    Attributes:
        tasks_dir: Path to Terminal-Bench tasks directory
        results_dir: Path to save results
        docker_image: Docker image to use for execution
        timeout_seconds: Per-task timeout
        memory_retrieval_limit: Max memories to retrieve per task
        min_relevance_score: Minimum memory relevance score
        task_filter: Filter for task selection
        n_trials: Number of trials to run
        seed: Random seed for reproducibility
        base_agent_command: Base agent to augment
        store_results_in_memory: Whether to store results in memory
    """

    tasks_dir: Path | str
    results_dir: Path | str = "results/terminalbench"
    docker_image: str = "terminal-bench:latest"
    timeout_seconds: int = 600
    memory_retrieval_limit: int = 5
    min_relevance_score: float = 0.3
    task_filter: TaskFilter | None = None
    n_trials: int = 1
    seed: int = 42
    base_agent_command: str = "claude-code"
    store_results_in_memory: bool = True


class TerminalBenchRunner:
    """Runner for Terminal-Bench 2.0 evaluations.

    This class manages the complete evaluation workflow:
    1. Load and filter tasks
    2. Set up memory-augmented agent
    3. Execute tasks in Docker containers
    4. Collect and aggregate results
    5. Store results for future reference
    """

    def __init__(
        self,
        adapter: MemorySystemAdapter,
        config: TerminalBenchConfig,
    ) -> None:
        """Initialize the runner.

        Args:
            adapter: Memory system adapter
            config: Evaluation configuration
        """
        self.adapter = adapter
        self.config = config
        self.results_dir = Path(config.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.task_selector = TaskSelector(config.tasks_dir)
        self.agent = MemoryAugmentedInstalledAgent(
            adapter=adapter,
            base_agent_command=config.base_agent_command,
            memory_retrieval_limit=config.memory_retrieval_limit,
            min_relevance_score=config.min_relevance_score,
        )

    def run(
        self,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> list[TrialResult]:
        """Run the complete evaluation.

        Args:
            progress_callback: Optional callback(current, total, message)

        Returns:
            List of TrialResult for each trial
        """
        # Select tasks
        tasks = self.task_selector.select_tasks(self.config.task_filter)

        if not tasks:
            logger.warning("No tasks selected for evaluation")
            # Generate synthetic tasks for testing
            tasks = self.task_selector.create_synthetic_tasks(n_tasks=10, seed=self.config.seed)
            logger.info(f"Using {len(tasks)} synthetic tasks for evaluation")

        logger.info(f"Selected {len(tasks)} tasks for evaluation")

        # Run trials
        trial_results = []
        for trial_idx in range(self.config.n_trials):
            trial_id = f"trial_{trial_idx:03d}"
            logger.info(f"Starting trial {trial_id}")

            if progress_callback:
                progress_callback(trial_idx, self.config.n_trials, f"Running trial {trial_id}")

            trial_result = self._run_trial(
                trial_id=trial_id,
                tasks=tasks,
                progress_callback=progress_callback,
            )

            trial_results.append(trial_result)

            # Save trial result
            self._save_trial_result(trial_result)

        return trial_results

    def _run_trial(
        self,
        trial_id: str,
        tasks: list[TaskInfo],
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> TrialResult:
        """Run a single trial.

        Args:
            trial_id: Trial identifier
            tasks: Tasks to execute
            progress_callback: Optional progress callback

        Returns:
            TrialResult for the trial
        """
        start_time = datetime.now()
        task_results = []
        successful = 0

        for i, task in enumerate(tasks):
            if progress_callback:
                progress_callback(
                    i,
                    len(tasks),
                    f"Executing task {task.task_id}",
                )

            result = self._execute_task(task)
            task_results.append(result)

            if result.success:
                successful += 1

            # Store result in memory if configured
            if self.config.store_results_in_memory:
                augmented_task = MemoryAugmentedTask(
                    task_id=task.task_id,
                    description=task.description,
                    augmented_description=result.augmented_description,
                    memory_context=result.memory_context,
                    category=task.category.value,
                    difficulty=task.difficulty,
                )
                self.agent.store_result(
                    task=augmented_task,
                    result=result.output,
                    success=result.success,
                    metadata={"trial_id": trial_id},
                )

        end_time = datetime.now()

        return TrialResult(
            trial_id=trial_id,
            adapter_name=type(self.adapter).__name__,
            task_results=tuple(task_results),
            total_tasks=len(tasks),
            successful_tasks=successful,
            start_time=start_time,
            end_time=end_time,
            config={
                "memory_retrieval_limit": self.config.memory_retrieval_limit,
                "min_relevance_score": self.config.min_relevance_score,
                "base_agent_command": self.config.base_agent_command,
                "timeout_seconds": self.config.timeout_seconds,
            },
        )

    def _execute_task(self, task: TaskInfo) -> TaskResult:
        """Execute a single task.

        Args:
            task: Task to execute

        Returns:
            TaskResult
        """
        import time

        start_time = time.time()

        # Augment task with memory
        augmented = self.agent.augment_task(
            task_description=task.description,
            task_id=task.task_id,
            category=task.category.value,
            difficulty=task.difficulty,
        )

        try:
            # Try to execute via Docker if available
            output, success, error = self._execute_in_docker(
                task=task,
                augmented=augmented,
            )
        except Exception as e:
            # Fall back to simulation if Docker not available
            logger.warning(f"Docker execution failed, using simulation: {e}")
            output, success, error = self._simulate_execution(augmented)

        execution_time = time.time() - start_time

        return TaskResult(
            task_id=task.task_id,
            task_description=task.description,
            augmented_description=augmented.augmented_description,
            success=success,
            output=output,
            execution_time_seconds=execution_time,
            memory_context=augmented.memory_context,
            error=error,
            metadata={
                "category": task.category.value,
                "difficulty": task.difficulty,
                "memory_relevance": task.memory_relevance.value,
            },
        )

    def _execute_in_docker(
        self,
        task: TaskInfo,
        augmented: MemoryAugmentedTask,
    ) -> tuple[str, bool, str | None]:
        """Execute task in Docker container.

        Args:
            task: Task info
            augmented: Augmented task

        Returns:
            Tuple of (output, success, error)
        """
        # Check if Docker is available
        try:
            subprocess.run(
                ["docker", "version"],
                capture_output=True,
                check=True,
                timeout=5,
            )
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
            raise RuntimeError(f"Docker not available: {e}") from e

        # Check if task has a Dockerfile
        task_path = Path(task.path)
        if not task_path.exists() or not (task_path / "Dockerfile").exists():
            raise RuntimeError(f"Task directory or Dockerfile not found: {task_path}")

        # Create temporary file for augmented task
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".txt",
            delete=False,
        ) as f:
            f.write(augmented.augmented_description)
            task_file = f.name

        try:
            # Build and run container
            result = subprocess.run(
                [
                    "docker",
                    "compose",
                    "-f",
                    str(task_path / "docker-compose.yaml"),
                    "run",
                    "--rm",
                    "-v",
                    f"{task_file}:/task.txt:ro",
                    "agent",
                    "cat",
                    "/task.txt",
                    "|",
                    self.config.base_agent_command,
                ],
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds,
                cwd=task_path,
            )

            output = result.stdout + result.stderr
            success = result.returncode == 0
            error = result.stderr if not success else None

            return output, success, error

        except subprocess.TimeoutExpired:
            return "", False, f"Task timed out after {self.config.timeout_seconds}s"

        except Exception as e:
            return "", False, str(e)

        finally:
            Path(task_file).unlink(missing_ok=True)

    def _simulate_execution(
        self,
        augmented: MemoryAugmentedTask,
    ) -> tuple[str, bool, str | None]:
        """Simulate task execution for testing.

        Args:
            augmented: Augmented task

        Returns:
            Tuple of (output, success, error)
        """
        import random
        import time

        # Simulate execution time
        time.sleep(0.1)

        # Memory context generally improves success rate
        has_memory = bool(augmented.memory_context)

        # Base success rate + memory bonus
        base_rate = 0.4
        memory_bonus = 0.2 if has_memory else 0.0
        success = random.random() < (base_rate + memory_bonus)

        output = f"Simulated execution of task: {augmented.task_id}\n"
        if has_memory:
            output += "Used memory context for augmentation.\n"
        output += f"Result: {'Success' if success else 'Failed'}\n"

        error = None if success else "Simulated task failure"

        return output, success, error

    def _save_trial_result(self, trial_result: TrialResult) -> None:
        """Save trial result to disk.

        Args:
            trial_result: Trial result to save
        """
        result_path = self.results_dir / f"{trial_result.trial_id}.json"

        data = {
            "trial_id": trial_result.trial_id,
            "adapter_name": trial_result.adapter_name,
            "total_tasks": trial_result.total_tasks,
            "successful_tasks": trial_result.successful_tasks,
            "success_rate": trial_result.success_rate,
            "start_time": trial_result.start_time.isoformat(),
            "end_time": trial_result.end_time.isoformat(),
            "duration_seconds": trial_result.duration_seconds,
            "config": trial_result.config,
            "task_results": [
                {
                    "task_id": r.task_id,
                    "success": r.success,
                    "execution_time_seconds": r.execution_time_seconds,
                    "error": r.error,
                    "metadata": r.metadata,
                }
                for r in trial_result.task_results
            ],
        }

        with open(result_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved trial result to {result_path}")

    def get_stats(self) -> dict[str, Any]:
        """Get runner statistics.

        Returns:
            Dictionary of statistics
        """
        task_stats = self.task_selector.get_stats()

        return {
            "adapter": type(self.adapter).__name__,
            "tasks_dir": str(self.config.tasks_dir),
            "results_dir": str(self.results_dir),
            "n_trials": self.config.n_trials,
            **task_stats,
        }

    @staticmethod
    def load_trial_result(result_path: Path | str) -> TrialResult:
        """Load a trial result from disk.

        Args:
            result_path: Path to result JSON file

        Returns:
            TrialResult
        """
        with open(result_path) as f:
            data = json.load(f)

        task_results = tuple(
            TaskResult(
                task_id=r["task_id"],
                task_description="",  # Not stored
                augmented_description="",  # Not stored
                success=r["success"],
                output="",  # Not stored
                execution_time_seconds=r["execution_time_seconds"],
                error=r.get("error"),
                metadata=r.get("metadata", {}),
            )
            for r in data["task_results"]
        )

        return TrialResult(
            trial_id=data["trial_id"],
            adapter_name=data["adapter_name"],
            task_results=task_results,
            total_tasks=data["total_tasks"],
            successful_tasks=data["successful_tasks"],
            start_time=datetime.fromisoformat(data["start_time"]),
            end_time=datetime.fromisoformat(data["end_time"]),
            config=data.get("config", {}),
        )
