"""Task selection for Terminal-Bench 2.0 memory evaluation.

This module provides utilities for selecting memory-relevant tasks from
the Terminal-Bench 2.0 benchmark suite.

Memory-relevant tasks are those where persistent memory of previous
sessions would provide meaningful benefit, such as:
- Multi-session debugging workflows
- Iterative development tasks
- Configuration management across environments
- Pattern recognition from historical solutions
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class TaskCategory(Enum):
    """Categories of Terminal-Bench tasks."""

    SOFTWARE_DEVELOPMENT = "software"
    SYSTEM_ADMINISTRATION = "sysadmin"
    DATA_PROCESSING = "data"
    SECURITY = "security"
    DEBUGGING = "debugging"
    CONFIGURATION = "config"
    INSTALLATION = "install"
    NETWORK = "network"
    DATABASE = "database"
    UNKNOWN = "unknown"

    @classmethod
    def from_string(cls, value: str) -> TaskCategory:
        """Parse category from string.

        Args:
            value: String representation of category

        Returns:
            Matching TaskCategory enum value
        """
        normalized = value.lower().strip().replace("-", "_")

        for category in cls:
            if category.value == normalized or category.name.lower() == normalized:
                return category

        # Try fuzzy matching
        if "software" in normalized or "dev" in normalized or "code" in normalized:
            return cls.SOFTWARE_DEVELOPMENT
        if "sys" in normalized or "admin" in normalized:
            return cls.SYSTEM_ADMINISTRATION
        if "data" in normalized or "process" in normalized:
            return cls.DATA_PROCESSING
        if "debug" in normalized:
            return cls.DEBUGGING
        if "config" in normalized:
            return cls.CONFIGURATION
        if "install" in normalized or "setup" in normalized:
            return cls.INSTALLATION
        if "network" in normalized or "net" in normalized:
            return cls.NETWORK
        if "database" in normalized or "db" in normalized or "sql" in normalized:
            return cls.DATABASE
        if "security" in normalized or "auth" in normalized:
            return cls.SECURITY

        return cls.UNKNOWN


class MemoryRelevance(Enum):
    """Degree of relevance for memory augmentation."""

    HIGH = "high"  # Task strongly benefits from memory
    MEDIUM = "medium"  # Task may benefit from memory
    LOW = "low"  # Task unlikely to benefit from memory


@dataclass(frozen=True, slots=True)
class TaskInfo:
    """Information about a Terminal-Bench task.

    Attributes:
        task_id: Unique task identifier
        name: Human-readable task name
        description: Full task description
        category: Task category
        difficulty: Difficulty level (1-5)
        memory_relevance: Estimated memory relevance
        keywords: Keywords extracted from description
        path: Path to task directory
        metadata: Additional task metadata
    """

    task_id: str
    name: str
    description: str
    category: TaskCategory
    difficulty: int = 3
    memory_relevance: MemoryRelevance = MemoryRelevance.MEDIUM
    keywords: tuple[str, ...] = ()
    path: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskFilter:
    """Filter configuration for task selection.

    Attributes:
        categories: Only include these categories (None = all)
        min_difficulty: Minimum difficulty level
        max_difficulty: Maximum difficulty level
        memory_relevance: Only include these relevance levels
        keywords: Include tasks matching any of these keywords
        exclude_keywords: Exclude tasks matching any of these keywords
        max_tasks: Maximum number of tasks to return
    """

    categories: set[TaskCategory] | None = None
    min_difficulty: int = 1
    max_difficulty: int = 5
    memory_relevance: set[MemoryRelevance] | None = None
    keywords: set[str] | None = None
    exclude_keywords: set[str] | None = None
    max_tasks: int | None = None


class TaskSelector:
    """Selects memory-relevant tasks from Terminal-Bench.

    This class analyzes Terminal-Bench tasks and identifies those
    where memory augmentation would provide meaningful benefit.
    """

    # Keywords that suggest memory relevance
    MEMORY_RELEVANT_KEYWORDS = {
        "previous",
        "earlier",
        "before",
        "history",
        "remember",
        "again",
        "similar",
        "like last",
        "as before",
        "continuing",
        "follow up",
        "based on",
        "learned",
        "pattern",
        "recurring",
        "repeat",
        "debug",
        "investigate",
        "troubleshoot",
        "diagnose",
        "analyze",
    }

    # Category-specific memory relevance
    CATEGORY_MEMORY_RELEVANCE = {
        TaskCategory.DEBUGGING: MemoryRelevance.HIGH,
        TaskCategory.CONFIGURATION: MemoryRelevance.HIGH,
        TaskCategory.SOFTWARE_DEVELOPMENT: MemoryRelevance.HIGH,
        TaskCategory.SYSTEM_ADMINISTRATION: MemoryRelevance.MEDIUM,
        TaskCategory.DATABASE: MemoryRelevance.MEDIUM,
        TaskCategory.DATA_PROCESSING: MemoryRelevance.MEDIUM,
        TaskCategory.INSTALLATION: MemoryRelevance.LOW,
        TaskCategory.NETWORK: MemoryRelevance.MEDIUM,
        TaskCategory.SECURITY: MemoryRelevance.MEDIUM,
        TaskCategory.UNKNOWN: MemoryRelevance.LOW,
    }

    def __init__(self, tasks_dir: Path | str | None = None) -> None:
        """Initialize the task selector.

        Args:
            tasks_dir: Path to Terminal-Bench tasks directory
        """
        self.tasks_dir = Path(tasks_dir) if tasks_dir else None
        self._tasks_cache: list[TaskInfo] | None = None

    def load_tasks(self) -> list[TaskInfo]:
        """Load all tasks from the tasks directory.

        Returns:
            List of TaskInfo objects
        """
        if self._tasks_cache is not None:
            return self._tasks_cache

        if self.tasks_dir is None:
            logger.warning("No tasks directory specified, returning empty list")
            return []

        if not self.tasks_dir.exists():
            logger.warning(f"Tasks directory does not exist: {self.tasks_dir}")
            return []

        tasks = []

        # Scan for task directories
        for task_path in self.tasks_dir.iterdir():
            if not task_path.is_dir():
                continue

            task_yaml = task_path / "task.yaml"
            if not task_yaml.exists():
                continue

            try:
                task_info = self._parse_task(task_path, task_yaml)
                tasks.append(task_info)
            except Exception as e:
                logger.warning(f"Failed to parse task {task_path}: {e}")

        self._tasks_cache = tasks
        return tasks

    def _parse_task(self, task_path: Path, task_yaml: Path) -> TaskInfo:
        """Parse a task from its YAML configuration.

        Args:
            task_path: Path to task directory
            task_yaml: Path to task.yaml file

        Returns:
            TaskInfo object
        """
        with open(task_yaml) as f:
            config = yaml.safe_load(f)

        task_id = task_path.name
        name = config.get("name", task_id)
        description = config.get("description", "")

        # Determine category
        category_str = config.get("category", "unknown")
        category = TaskCategory.from_string(category_str)

        # Determine difficulty
        difficulty = config.get("difficulty", 3)
        if isinstance(difficulty, str):
            difficulty = {"easy": 1, "medium": 3, "hard": 5}.get(difficulty.lower(), 3)

        # Extract keywords
        keywords = self._extract_keywords(description)

        # Determine memory relevance
        memory_relevance = self._assess_memory_relevance(description, category, keywords)

        return TaskInfo(
            task_id=task_id,
            name=name,
            description=description,
            category=category,
            difficulty=difficulty,
            memory_relevance=memory_relevance,
            keywords=tuple(keywords),
            path=str(task_path),
            metadata=config.get("metadata", {}),
        )

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract keywords from text.

        Args:
            text: Text to extract keywords from

        Returns:
            List of keywords
        """
        # Simple keyword extraction
        words = re.findall(r"\b\w+\b", text.lower())

        # Filter to meaningful keywords
        stopwords = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "shall",
            "can",
            "need",
            "dare",
            "ought",
            "used",
            "to",
            "of",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "as",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "under",
            "again",
            "further",
            "then",
            "once",
            "and",
            "but",
            "or",
            "so",
            "if",
            "this",
            "that",
            "these",
            "those",
            "it",
            "its",
        }

        keywords = [w for w in words if w not in stopwords and len(w) > 2]

        # Return unique keywords
        return list(dict.fromkeys(keywords))[:20]

    def _assess_memory_relevance(
        self,
        description: str,
        category: TaskCategory,
        keywords: list[str],
    ) -> MemoryRelevance:
        """Assess the memory relevance of a task.

        Args:
            description: Task description
            category: Task category
            keywords: Extracted keywords

        Returns:
            MemoryRelevance level
        """
        # Start with category-based relevance
        base_relevance = self.CATEGORY_MEMORY_RELEVANCE.get(category, MemoryRelevance.LOW)

        # Check for memory-relevant keywords
        desc_lower = description.lower()
        keyword_matches = sum(1 for kw in self.MEMORY_RELEVANT_KEYWORDS if kw in desc_lower)

        # Upgrade relevance based on keyword matches
        if keyword_matches >= 3:
            return MemoryRelevance.HIGH
        if keyword_matches >= 1 and base_relevance == MemoryRelevance.LOW:
            return MemoryRelevance.MEDIUM
        if keyword_matches >= 2 and base_relevance == MemoryRelevance.MEDIUM:
            return MemoryRelevance.HIGH

        return base_relevance

    def select_tasks(
        self,
        filter_config: TaskFilter | None = None,
    ) -> list[TaskInfo]:
        """Select tasks based on filter configuration.

        Args:
            filter_config: Filter configuration (None = select all memory-relevant)

        Returns:
            List of matching TaskInfo objects
        """
        tasks = self.load_tasks()

        if filter_config is None:
            # Default: select all HIGH and MEDIUM memory relevance tasks
            filter_config = TaskFilter(
                memory_relevance={MemoryRelevance.HIGH, MemoryRelevance.MEDIUM}
            )

        # Apply filters
        filtered = []
        for task in tasks:
            if not self._matches_filter(task, filter_config):
                continue
            filtered.append(task)

        # Apply max_tasks limit
        if filter_config.max_tasks is not None:
            # Sort by relevance (HIGH first) then difficulty
            relevance_order = {
                MemoryRelevance.HIGH: 0,
                MemoryRelevance.MEDIUM: 1,
                MemoryRelevance.LOW: 2,
            }
            filtered.sort(key=lambda t: (relevance_order[t.memory_relevance], -t.difficulty))
            filtered = filtered[: filter_config.max_tasks]

        return filtered

    def _matches_filter(self, task: TaskInfo, filter_config: TaskFilter) -> bool:
        """Check if a task matches the filter configuration.

        Args:
            task: Task to check
            filter_config: Filter configuration

        Returns:
            True if task matches filter
        """
        # Category filter
        if filter_config.categories is not None:
            if task.category not in filter_config.categories:
                return False

        # Difficulty filter
        if task.difficulty < filter_config.min_difficulty:
            return False
        if task.difficulty > filter_config.max_difficulty:
            return False

        # Memory relevance filter
        if filter_config.memory_relevance is not None:
            if task.memory_relevance not in filter_config.memory_relevance:
                return False

        # Keyword filters
        task_keywords_lower = {kw.lower() for kw in task.keywords}
        desc_lower = task.description.lower()

        if filter_config.keywords is not None:
            # Must match at least one keyword
            if not any(
                kw.lower() in task_keywords_lower or kw.lower() in desc_lower
                for kw in filter_config.keywords
            ):
                return False

        if filter_config.exclude_keywords is not None:
            # Must not match any exclude keyword
            if any(
                kw.lower() in task_keywords_lower or kw.lower() in desc_lower
                for kw in filter_config.exclude_keywords
            ):
                return False

        return True

    def get_task(self, task_id: str) -> TaskInfo | None:
        """Get a specific task by ID.

        Args:
            task_id: Task identifier

        Returns:
            TaskInfo or None if not found
        """
        tasks = self.load_tasks()
        for task in tasks:
            if task.task_id == task_id:
                return task
        return None

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about available tasks.

        Returns:
            Dictionary with task statistics
        """
        tasks = self.load_tasks()

        # Count by category
        category_counts: dict[str, int] = {}
        for task in tasks:
            cat = task.category.value
            category_counts[cat] = category_counts.get(cat, 0) + 1

        # Count by relevance
        relevance_counts: dict[str, int] = {}
        for task in tasks:
            rel = task.memory_relevance.value
            relevance_counts[rel] = relevance_counts.get(rel, 0) + 1

        # Difficulty distribution
        difficulty_counts: dict[int, int] = {}
        for task in tasks:
            difficulty_counts[task.difficulty] = difficulty_counts.get(task.difficulty, 0) + 1

        return {
            "total_tasks": len(tasks),
            "by_category": category_counts,
            "by_relevance": relevance_counts,
            "by_difficulty": difficulty_counts,
            "tasks_dir": str(self.tasks_dir) if self.tasks_dir else None,
        }

    def create_synthetic_tasks(self, n_tasks: int = 10, seed: int = 42) -> list[TaskInfo]:
        """Create synthetic tasks for testing.

        Args:
            n_tasks: Number of tasks to create
            seed: Random seed for reproducibility

        Returns:
            List of synthetic TaskInfo objects
        """
        import random

        rng = random.Random(seed)

        categories = list(TaskCategory)
        difficulties = [1, 2, 3, 4, 5]

        task_templates = [
            ("Debug the failing {component} service", TaskCategory.DEBUGGING),
            ("Configure {component} settings for production", TaskCategory.CONFIGURATION),
            ("Install and set up {component}", TaskCategory.INSTALLATION),
            ("Optimize database queries in {component}", TaskCategory.DATABASE),
            ("Implement feature in {component} module", TaskCategory.SOFTWARE_DEVELOPMENT),
            ("Troubleshoot network issues with {component}", TaskCategory.NETWORK),
            ("Set up monitoring for {component}", TaskCategory.SYSTEM_ADMINISTRATION),
            ("Process data from {component} logs", TaskCategory.DATA_PROCESSING),
            ("Secure {component} endpoints", TaskCategory.SECURITY),
        ]

        components = [
            "authentication",
            "payment",
            "user",
            "notification",
            "analytics",
            "cache",
            "queue",
            "storage",
            "api",
            "frontend",
        ]

        tasks = []
        for i in range(n_tasks):
            template, category = rng.choice(task_templates)
            component = rng.choice(components)

            description = template.format(component=component)
            difficulty = rng.choice(difficulties)

            keywords = self._extract_keywords(description)
            memory_relevance = self._assess_memory_relevance(description, category, keywords)

            task = TaskInfo(
                task_id=f"synthetic_{i:04d}",
                name=f"Task {i}: {description[:30]}...",
                description=description,
                category=category,
                difficulty=difficulty,
                memory_relevance=memory_relevance,
                keywords=tuple(keywords),
                path=f"/synthetic/task_{i:04d}",
                metadata={"synthetic": True, "seed": seed},
            )
            tasks.append(task)

        return tasks
