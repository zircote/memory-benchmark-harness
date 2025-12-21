#!/usr/bin/env python3
"""Quick test of the Terminal-Bench pipeline.

This verifies the Terminal-Bench evaluation infrastructure works
using synthetic task data (no Docker execution required).

Usage:
    uv run python scripts/quick_terminalbench_test.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add src to path for imports
_root = Path(__file__).parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


def main() -> None:
    """Run a quick Terminal-Bench test."""
    print("=" * 60)
    print("Quick Terminal-Bench Infrastructure Test")
    print("=" * 60)

    # Check API key (for future LLM integration)
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)
    print(f"\n1. API Key: ...{api_key[-8:]}")

    # Test task selector
    print("\n2. Testing TaskSelector...")
    from src.benchmarks.terminalbench import (
        MemoryRelevance,
        TaskCategory,
        TaskFilter,
        TaskInfo,
        TaskSelector,
    )

    # Create some synthetic tasks
    tasks = [
        TaskInfo(
            task_id="task_1",
            name="Configure SSH Keys",
            description="Set up SSH keys for GitHub access",
            category=TaskCategory.SYSTEM_ADMINISTRATION,
            memory_relevance=MemoryRelevance.HIGH,
            difficulty=3,
            keywords=["ssh", "github", "keys", "authentication"],
        ),
        TaskInfo(
            task_id="task_2",
            name="Parse JSON Log File",
            description="Extract error entries from JSON logs",
            category=TaskCategory.DATA_PROCESSING,
            memory_relevance=MemoryRelevance.MEDIUM,
            difficulty=2,
            keywords=["json", "logs", "parsing", "errors"],
        ),
        TaskInfo(
            task_id="task_3",
            name="Debug Python Script",
            description="Find and fix a bug in a Python script",
            category=TaskCategory.SOFTWARE_DEVELOPMENT,
            memory_relevance=MemoryRelevance.HIGH,
            difficulty=4,
            keywords=["python", "debug", "bug", "script"],
        ),
        TaskInfo(
            task_id="task_4",
            name="Install Docker",
            description="Install and configure Docker on the system",
            category=TaskCategory.SYSTEM_ADMINISTRATION,
            memory_relevance=MemoryRelevance.LOW,
            difficulty=2,
            keywords=["docker", "install", "container"],
        ),
    ]

    # TaskSelector takes a tasks_dir path; inject synthetic tasks via cache
    selector = TaskSelector(tasks_dir=None)
    selector._tasks_cache = tasks  # Inject synthetic tasks for testing
    print(f"   Total tasks: {len(selector.load_tasks())}")

    # Test filtering with select_tasks
    high_relevance = selector.select_tasks(TaskFilter(memory_relevance={MemoryRelevance.HIGH}))
    print(f"   High memory relevance tasks: {len(high_relevance)}")

    dev_tasks = selector.select_tasks(TaskFilter(categories={TaskCategory.SOFTWARE_DEVELOPMENT}))
    print(f"   Software development tasks: {len(dev_tasks)}")

    # Test MemoryAugmentedInstalledAgent creation
    print("\n3. Testing MemoryAugmentedInstalledAgent...")
    from src.adapters.mock import MockAdapter
    from src.benchmarks.terminalbench import MemoryAugmentedInstalledAgent

    adapter = MockAdapter()
    agent = MemoryAugmentedInstalledAgent(
        adapter=adapter,
        base_agent_command="/usr/bin/false",  # Dummy path
        memory_retrieval_limit=5,
    )
    print(f"   Agent created with adapter: {type(adapter).__name__}")
    print(f"   Retrieval limit: {agent.memory_retrieval_limit}")

    # Test TerminalBenchRunner configuration
    print("\n4. Testing TerminalBenchRunner configuration...")
    from src.benchmarks.terminalbench import TerminalBenchConfig, TerminalBenchRunner

    # Config requires tasks_dir; use dummy path for quick test
    config = TerminalBenchConfig(
        tasks_dir="/tmp/terminalbench-tasks",  # Dummy path
        task_filter=TaskFilter(memory_relevance={MemoryRelevance.MEDIUM, MemoryRelevance.HIGH}),
        timeout_seconds=30,
    )
    print("   Config created:")
    print(f"   - Tasks dir: {config.tasks_dir}")
    print(f"   - Timeout: {config.timeout_seconds}s")
    print(f"   - Memory retrieval limit: {config.memory_retrieval_limit}")

    runner = TerminalBenchRunner(config=config, adapter=adapter)
    print("   TerminalBenchRunner created successfully")

    # Test metrics calculator
    print("\n5. Testing metrics calculator...")
    from datetime import datetime

    from src.benchmarks.terminalbench import (
        TerminalBenchMetricsCalculator,
    )
    from src.benchmarks.terminalbench.runner import TaskResult, TrialResult

    # Create sample task results
    task_results = (
        TaskResult(
            task_id="task_1",
            task_description="Configure SSH Keys",
            augmented_description="Configure SSH Keys [with memory context]",
            success=True,
            output="SSH keys configured",
            execution_time_seconds=15.5,
            metadata={
                "category": "system_administration",
                "memory_relevance": "high",
                "difficulty": 3,
            },
        ),
        TaskResult(
            task_id="task_2",
            task_description="Parse JSON Log File",
            augmented_description="Parse JSON Log File [with memory context]",
            success=False,
            output="",
            execution_time_seconds=25.0,
            error="Parse error",
            metadata={"category": "data_processing", "memory_relevance": "medium", "difficulty": 2},
        ),
    )

    # Create a trial result
    trial = TrialResult(
        trial_id="trial_1",
        adapter_name="MockAdapter",
        task_results=task_results,
        total_tasks=2,
        successful_tasks=1,
        start_time=datetime.now(),
        end_time=datetime.now(),
    )

    calculator = TerminalBenchMetricsCalculator()
    result = calculator.calculate([trial])

    print(f"   Overall success rate: {result.overall_success_rate:.2%}")
    print(f"   Total tasks: {result.total_tasks}")
    print(f"   Successful: {result.successful_tasks}")

    print("\n" + "=" * 60)
    print("Quick test complete - Terminal-Bench infrastructure works!")
    print("=" * 60)


if __name__ == "__main__":
    main()
