"""Memory-augmented agent for Terminal-Bench 2.0.

This module implements the AbstractInstalledAgent interface for integrating
git-notes memory system with Terminal-Bench 2.0 evaluation.

The agent wraps a base LLM agent (like Claude Code or Codex) and augments
it with persistent memory retrieved from git-notes.

Reference: https://www.tbench.ai/docs/agent-introduction
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from src.adapters.base import MemorySystemAdapter

logger = logging.getLogger(__name__)


class TerminalCommand(Protocol):
    """Protocol matching Terminal-Bench's TerminalCommand."""

    command: str
    timeout: int | None


@dataclass(frozen=True, slots=True)
class SimpleTerminalCommand:
    """Simple implementation of TerminalCommand for testing."""

    command: str
    timeout: int | None = None


@dataclass(frozen=True, slots=True)
class MemoryAugmentedTask:
    """A Terminal-Bench task augmented with memory context.

    Attributes:
        task_id: Unique identifier for the task
        description: Original task description
        augmented_description: Description with memory context prepended
        memory_context: Retrieved memory content used for augmentation
        category: Task category (e.g., "software", "sysadmin")
        difficulty: Estimated task difficulty (1-5)
    """

    task_id: str
    description: str
    augmented_description: str
    memory_context: str
    category: str = "unknown"
    difficulty: int = 3


@dataclass
class MemoryAugmentedInstalledAgent:
    """Memory-augmented agent implementing AbstractInstalledAgent interface.

    This agent wraps a base agent (specified by base_agent_command) and
    augments task descriptions with relevant memories before execution.

    The implementation follows the Terminal-Bench AbstractInstalledAgent
    interface pattern:
    - name: Static method returning agent identifier
    - _env: Property returning environment variables
    - _install_agent_script_path: Property returning setup script path
    - _run_agent_commands: Method returning execution commands

    Attributes:
        adapter: Memory system adapter for retrieval
        base_agent_command: Command to run the base agent (e.g., "claude-code")
        agent_name: Name identifier for this agent
        memory_retrieval_limit: Maximum memories to retrieve per task
        min_relevance_score: Minimum score for memory inclusion
        install_script_content: Content for the installation script
        env_vars: Additional environment variables
    """

    adapter: MemorySystemAdapter
    base_agent_command: str = "claude-code"
    agent_name: str = "memory-augmented-agent"
    memory_retrieval_limit: int = 5
    min_relevance_score: float = 0.3
    install_script_content: str = ""
    env_vars: dict[str, str] = field(default_factory=dict)
    _install_script_path: Path | None = field(default=None, init=False)

    @staticmethod
    def name() -> str:
        """Return the agent name for Terminal-Bench registration.

        Returns:
            Agent identifier string
        """
        return "memory-augmented-agent"

    @property
    def _env(self) -> dict[str, str]:
        """Return environment variables for the agent container.

        Returns:
            Dictionary of environment variables
        """
        base_env = {
            "MEMORY_RETRIEVAL_LIMIT": str(self.memory_retrieval_limit),
            "MEMORY_MIN_SCORE": str(self.min_relevance_score),
        }
        base_env.update(self.env_vars)
        return base_env

    @property
    def _install_agent_script_path(self) -> os.PathLike[str]:
        """Return path to the installation script.

        Creates a temporary script file if not already created.

        Returns:
            Path to the installation script
        """
        if self._install_script_path is None:
            # Create installation script
            script_content = self._generate_install_script()

            # Write to temp file
            fd, path = tempfile.mkstemp(suffix=".sh", prefix="memory_agent_install_")
            with os.fdopen(fd, "w") as f:
                f.write(script_content)

            self._install_script_path = Path(path)
            os.chmod(self._install_script_path, 0o755)

        return self._install_script_path

    def _generate_install_script(self) -> str:
        """Generate the installation script content.

        Returns:
            Bash script content for agent installation
        """
        base_script = """#!/bin/bash
set -e

# Install base agent dependencies
echo "Installing memory-augmented agent..."

# Install the base agent command if needed
if ! command -v {base_agent} &> /dev/null; then
    echo "Base agent {base_agent} not found, attempting installation..."
    # Try common installation methods
    if command -v npm &> /dev/null; then
        npm install -g {base_agent} 2>/dev/null || true
    fi
    if command -v pip &> /dev/null; then
        pip install {base_agent} 2>/dev/null || true
    fi
fi

# Additional custom installation steps
{custom_install}

echo "Memory-augmented agent installation complete."
""".format(
            base_agent=self.base_agent_command,
            custom_install=self.install_script_content or "# No custom installation steps",
        )
        return base_script

    def _run_agent_commands(
        self,
        task_description: str,
    ) -> list[SimpleTerminalCommand]:
        """Generate commands to run the agent with the task.

        The task description is augmented with relevant memories before
        being passed to the base agent.

        Args:
            task_description: The original task description

        Returns:
            List of TerminalCommand objects to execute
        """
        # Augment task with memory
        augmented_task = self.augment_task(task_description)

        # Build the command to run the base agent
        # Escape the description for shell
        escaped_desc = augmented_task.augmented_description.replace("'", "'\\''")

        commands = [
            SimpleTerminalCommand(
                command=f"{self.base_agent_command} '{escaped_desc}'",
                timeout=600,  # 10 minute timeout
            ),
        ]

        return commands

    def augment_task(
        self,
        task_description: str,
        task_id: str = "",
        category: str = "unknown",
        difficulty: int = 3,
    ) -> MemoryAugmentedTask:
        """Augment a task description with relevant memories.

        Args:
            task_description: Original task description
            task_id: Optional task identifier
            category: Task category
            difficulty: Task difficulty level

        Returns:
            MemoryAugmentedTask with memory context
        """
        # Retrieve relevant memories
        memories = self.adapter.search(
            query=task_description,
            limit=self.memory_retrieval_limit,
            min_score=self.min_relevance_score,
        )

        # Build memory context
        if memories:
            memory_lines = [
                "## Relevant Context from Previous Sessions\n",
                "The following information may be helpful for this task:\n",
            ]

            for i, mem in enumerate(memories, 1):
                memory_lines.append(f"{i}. {mem.content}")

            memory_lines.append("\n---\n")
            memory_context = "\n".join(memory_lines)
        else:
            memory_context = ""

        # Augment the description
        if memory_context:
            augmented = f"{memory_context}\n## Task\n\n{task_description}"
        else:
            augmented = task_description

        return MemoryAugmentedTask(
            task_id=task_id or f"task_{hash(task_description) % 10000}",
            description=task_description,
            augmented_description=augmented,
            memory_context=memory_context,
            category=category,
            difficulty=difficulty,
        )

    def store_result(
        self,
        task: MemoryAugmentedTask,
        result: str,
        success: bool,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store task result in memory for future reference.

        Args:
            task: The augmented task that was executed
            result: Output or result from task execution
            success: Whether the task was successful
            metadata: Additional metadata to store
        """
        # Build memory content
        content = f"""Task: {task.description}

Category: {task.category}
Result: {"Success" if success else "Failed"}

Output Summary:
{result[:500] if len(result) > 500 else result}
"""

        # Build metadata
        mem_metadata = {
            "task_id": task.task_id,
            "category": task.category,
            "difficulty": task.difficulty,
            "success": success,
            "source": "terminal-bench",
        }
        if metadata:
            mem_metadata.update(metadata)

        # Store in memory
        self.adapter.add(content=content, metadata=mem_metadata)

    def cleanup(self) -> None:
        """Clean up temporary resources."""
        if self._install_script_path and self._install_script_path.exists():
            self._install_script_path.unlink()
            self._install_script_path = None


def create_memory_agent(
    adapter: MemorySystemAdapter,
    base_agent: str = "claude-code",
    name: str = "memory-augmented-agent",
    **kwargs: Any,
) -> MemoryAugmentedInstalledAgent:
    """Factory function to create a memory-augmented agent.

    Args:
        adapter: Memory system adapter
        base_agent: Base agent command to wrap
        name: Agent name identifier
        **kwargs: Additional configuration options

    Returns:
        Configured MemoryAugmentedInstalledAgent
    """
    return MemoryAugmentedInstalledAgent(
        adapter=adapter,
        base_agent_command=base_agent,
        agent_name=name,
        **kwargs,
    )
