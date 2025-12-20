"""Experiment orchestration for benchmark evaluation.

This module provides the ExperimentRunner for executing multiple benchmark
trials across different memory system conditions with reproducible seeds.

Typical usage:
    ```python
    from src.experiments import ExperimentRunner, ExperimentConfig

    config = ExperimentConfig(
        benchmark="longmemeval",
        adapters=["git-notes", "no-memory"],
        num_trials=5,
        output_dir="results/experiment_001",
    )
    runner = ExperimentRunner(config)
    results = await runner.run()
    ```
"""

from src.experiments.runner import (
    AdapterCondition,
    ExperimentConfig,
    ExperimentResults,
    ExperimentRunner,
    TrialResult,
)

__all__ = [
    "ExperimentConfig",
    "ExperimentRunner",
    "TrialResult",
    "ExperimentResults",
    "AdapterCondition",
]
