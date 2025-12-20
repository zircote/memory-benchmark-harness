"""Tests for the Terminal-Bench agent module."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from src.adapters.base import MemoryItem, MemoryOperationResult
from src.benchmarks.terminalbench.agent import (
    MemoryAugmentedInstalledAgent,
    MemoryAugmentedTask,
    SimpleTerminalCommand,
    create_memory_agent,
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


class TestMemoryAugmentedTask:
    """Tests for MemoryAugmentedTask dataclass."""

    def test_creation(self) -> None:
        """Test creating a task."""
        task = MemoryAugmentedTask(
            task_id="test_001",
            description="Debug the failing service",
            augmented_description="## Context\n...\n## Task\nDebug the failing service",
            memory_context="Previous fix: ...",
            category="debugging",
            difficulty=3,
        )
        assert task.task_id == "test_001"
        assert "Context" in task.augmented_description

    def test_frozen(self) -> None:
        """Test that task is frozen."""
        task = MemoryAugmentedTask(
            task_id="test",
            description="test",
            augmented_description="test",
            memory_context="",
        )
        with pytest.raises(AttributeError):
            task.task_id = "new_id"  # type: ignore


class TestSimpleTerminalCommand:
    """Tests for SimpleTerminalCommand."""

    def test_creation(self) -> None:
        """Test creating a command."""
        cmd = SimpleTerminalCommand(command="echo hello", timeout=30)
        assert cmd.command == "echo hello"
        assert cmd.timeout == 30

    def test_default_timeout(self) -> None:
        """Test default timeout."""
        cmd = SimpleTerminalCommand(command="ls")
        assert cmd.timeout is None


class TestMemoryAugmentedInstalledAgent:
    """Tests for the MemoryAugmentedInstalledAgent class."""

    @pytest.fixture
    def agent(self) -> MemoryAugmentedInstalledAgent:
        """Create an agent for testing."""
        return MemoryAugmentedInstalledAgent(
            adapter=MockAdapter(),
            base_agent_command="claude-code",
            memory_retrieval_limit=5,
            min_relevance_score=0.3,
        )

    def test_name(self) -> None:
        """Test agent name."""
        assert MemoryAugmentedInstalledAgent.name() == "memory-augmented-agent"

    def test_env(self, agent: MemoryAugmentedInstalledAgent) -> None:
        """Test environment variables."""
        env = agent._env
        assert "MEMORY_RETRIEVAL_LIMIT" in env
        assert env["MEMORY_RETRIEVAL_LIMIT"] == "5"
        assert env["MEMORY_MIN_SCORE"] == "0.3"

    def test_env_with_custom_vars(self) -> None:
        """Test environment with custom variables."""
        agent = MemoryAugmentedInstalledAgent(
            adapter=MockAdapter(),
            env_vars={"API_KEY": "test123"},
        )
        env = agent._env
        assert env["API_KEY"] == "test123"

    def test_install_script_path_creates_file(self, agent: MemoryAugmentedInstalledAgent) -> None:
        """Test that install script path creates a file."""
        script_path = Path(agent._install_agent_script_path)
        try:
            assert script_path.exists()
            assert script_path.suffix == ".sh"

            # Check script content
            content = script_path.read_text()
            assert "#!/bin/bash" in content
            assert "claude-code" in content
        finally:
            agent.cleanup()

    def test_install_script_cached(self, agent: MemoryAugmentedInstalledAgent) -> None:
        """Test that install script is cached."""
        path1 = agent._install_agent_script_path
        path2 = agent._install_agent_script_path
        assert path1 == path2
        agent.cleanup()

    def test_run_agent_commands(self, agent: MemoryAugmentedInstalledAgent) -> None:
        """Test generating run commands."""
        commands = agent._run_agent_commands("Debug the service")
        assert len(commands) == 1
        assert "claude-code" in commands[0].command
        assert commands[0].timeout == 600

    def test_augment_task_no_memory(self, agent: MemoryAugmentedInstalledAgent) -> None:
        """Test augmenting task without memory."""
        task = agent.augment_task(
            task_description="Fix the bug",
            task_id="task_001",
            category="debugging",
        )
        assert task.task_id == "task_001"
        assert task.description == "Fix the bug"
        assert task.memory_context == ""
        assert task.augmented_description == "Fix the bug"

    def test_augment_task_with_memory(self) -> None:
        """Test augmenting task with memory context."""
        adapter = MockAdapter()
        adapter.search_results = [
            MemoryItem(
                memory_id="mem1",
                content="Previous fix: check null pointers",
                score=0.9,
                metadata={},
                created_at=datetime.now(),
            ),
            MemoryItem(
                memory_id="mem2",
                content="Common issue: missing config",
                score=0.8,
                metadata={},
                created_at=datetime.now(),
            ),
        ]

        agent = MemoryAugmentedInstalledAgent(adapter=adapter)
        task = agent.augment_task("Debug the service")

        assert adapter.search_calls == 1
        assert "Previous fix" in task.memory_context
        assert "Common issue" in task.memory_context
        assert "Relevant Context" in task.augmented_description
        assert "Debug the service" in task.augmented_description

    def test_store_result(self, agent: MemoryAugmentedInstalledAgent) -> None:
        """Test storing task results."""
        task = MemoryAugmentedTask(
            task_id="task_001",
            description="Fix the bug",
            augmented_description="...",
            memory_context="",
            category="debugging",
            difficulty=3,
        )

        agent.store_result(
            task=task,
            result="Bug fixed successfully",
            success=True,
            metadata={"extra": "data"},
        )

        assert agent.adapter.add_calls == 1  # type: ignore
        content, metadata = agent.adapter.memories[0]  # type: ignore
        assert "Fix the bug" in content
        assert metadata["success"] is True
        assert metadata["source"] == "terminal-bench"

    def test_cleanup(self, agent: MemoryAugmentedInstalledAgent) -> None:
        """Test cleanup removes temp file."""
        # Create the script file
        script_path = Path(agent._install_agent_script_path)
        assert script_path.exists()

        # Cleanup
        agent.cleanup()
        assert not script_path.exists()
        assert agent._install_script_path is None


class TestCreateMemoryAgent:
    """Tests for the factory function."""

    def test_create_basic(self) -> None:
        """Test creating a basic agent."""
        agent = create_memory_agent(
            adapter=MockAdapter(),
        )
        assert isinstance(agent, MemoryAugmentedInstalledAgent)
        assert agent.base_agent_command == "claude-code"

    def test_create_with_options(self) -> None:
        """Test creating agent with options."""
        agent = create_memory_agent(
            adapter=MockAdapter(),
            base_agent="codex",
            name="custom-agent",
            memory_retrieval_limit=10,
        )
        assert agent.base_agent_command == "codex"
        assert agent.agent_name == "custom-agent"
        assert agent.memory_retrieval_limit == 10
