"""Terminal-Bench 2.0 evaluation module.

This module provides integration with Terminal-Bench 2.0 for evaluating
memory-augmented agents on real-world terminal tasks.

Terminal-Bench 2.0 evaluates agents on complex terminal-based tasks including:
- Software development and debugging
- System administration
- Data processing and analysis
- Security and compliance tasks

Integration approaches:
1. AbstractInstalledAgent - For CLI-installable agents
2. BaseAgent - For custom agent implementations

Reference: https://www.tbench.ai/docs/agent-introduction
"""

from __future__ import annotations

from src.benchmarks.terminalbench.agent import (
    MemoryAugmentedInstalledAgent,
    MemoryAugmentedTask,
    create_memory_agent,
)
from src.benchmarks.terminalbench.metrics import (
    TaskMetrics,
    TerminalBenchMetrics,
    TerminalBenchMetricsCalculator,
)
from src.benchmarks.terminalbench.runner import (
    TerminalBenchConfig,
    TerminalBenchRunner,
    TrialResult,
)
from src.benchmarks.terminalbench.task_selector import (
    MemoryRelevance,
    TaskCategory,
    TaskFilter,
    TaskInfo,
    TaskSelector,
)

__all__ = [
    # Agent
    "MemoryAugmentedInstalledAgent",
    "MemoryAugmentedTask",
    "create_memory_agent",
    # Task Selector
    "TaskSelector",
    "TaskInfo",
    "TaskCategory",
    "TaskFilter",
    "MemoryRelevance",
    # Runner
    "TerminalBenchRunner",
    "TerminalBenchConfig",
    "TrialResult",
    # Metrics
    "TerminalBenchMetricsCalculator",
    "TerminalBenchMetrics",
    "TaskMetrics",
]
