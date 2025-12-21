"""Tests for the Terminal-Bench task selector module."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.benchmarks.terminalbench.task_selector import (
    MemoryRelevance,
    TaskCategory,
    TaskFilter,
    TaskInfo,
    TaskSelector,
)


class TestTaskCategory:
    """Tests for the TaskCategory enum."""

    def test_all_categories_exist(self) -> None:
        """Test that all expected categories exist."""
        expected = [
            "software",
            "sysadmin",
            "data",
            "security",
            "debugging",
            "config",
            "install",
            "network",
            "database",
            "unknown",
        ]
        actual = [c.value for c in TaskCategory]
        assert set(actual) == set(expected)

    def test_from_string_exact(self) -> None:
        """Test exact string matching."""
        assert TaskCategory.from_string("software") == TaskCategory.SOFTWARE_DEVELOPMENT
        assert TaskCategory.from_string("debugging") == TaskCategory.DEBUGGING

    def test_from_string_fuzzy(self) -> None:
        """Test fuzzy string matching."""
        assert TaskCategory.from_string("development") == TaskCategory.SOFTWARE_DEVELOPMENT
        assert TaskCategory.from_string("db") == TaskCategory.DATABASE
        assert TaskCategory.from_string("net") == TaskCategory.NETWORK

    def test_from_string_unknown(self) -> None:
        """Test unknown category."""
        assert TaskCategory.from_string("random") == TaskCategory.UNKNOWN


class TestMemoryRelevance:
    """Tests for the MemoryRelevance enum."""

    def test_all_levels_exist(self) -> None:
        """Test all relevance levels exist."""
        assert MemoryRelevance.HIGH.value == "high"
        assert MemoryRelevance.MEDIUM.value == "medium"
        assert MemoryRelevance.LOW.value == "low"


class TestTaskInfo:
    """Tests for the TaskInfo dataclass."""

    def test_creation(self) -> None:
        """Test creating task info."""
        task = TaskInfo(
            task_id="task_001",
            name="Debug service",
            description="Debug the failing authentication service",
            category=TaskCategory.DEBUGGING,
            difficulty=3,
            memory_relevance=MemoryRelevance.HIGH,
            keywords=("debug", "service", "auth"),
        )
        assert task.task_id == "task_001"
        assert task.category == TaskCategory.DEBUGGING

    def test_frozen(self) -> None:
        """Test that task info is frozen."""
        task = TaskInfo(
            task_id="test",
            name="test",
            description="test",
            category=TaskCategory.UNKNOWN,
        )
        with pytest.raises(AttributeError):
            task.task_id = "new_id"  # type: ignore


class TestTaskFilter:
    """Tests for the TaskFilter dataclass."""

    def test_defaults(self) -> None:
        """Test default filter values."""
        f = TaskFilter()
        assert f.categories is None
        assert f.min_difficulty == 1
        assert f.max_difficulty == 5
        assert f.memory_relevance is None
        assert f.keywords is None
        assert f.max_tasks is None

    def test_with_values(self) -> None:
        """Test filter with values."""
        f = TaskFilter(
            categories={TaskCategory.DEBUGGING, TaskCategory.SOFTWARE_DEVELOPMENT},
            min_difficulty=2,
            max_difficulty=4,
            memory_relevance={MemoryRelevance.HIGH},
            max_tasks=10,
        )
        assert len(f.categories) == 2
        assert f.max_tasks == 10


class TestTaskSelector:
    """Tests for the TaskSelector class."""

    @pytest.fixture
    def selector(self) -> TaskSelector:
        """Create a selector for testing."""
        return TaskSelector()

    def test_init_without_tasks_dir(self, selector: TaskSelector) -> None:
        """Test initialization without tasks directory."""
        assert selector.tasks_dir is None

    def test_load_tasks_no_dir(self, selector: TaskSelector) -> None:
        """Test loading tasks without directory."""
        tasks = selector.load_tasks()
        assert tasks == []

    def test_load_tasks_nonexistent_dir(self, tmp_path: Path) -> None:
        """Test loading from non-existent directory."""
        selector = TaskSelector(tmp_path / "nonexistent")
        tasks = selector.load_tasks()
        assert tasks == []

    def test_extract_keywords(self, selector: TaskSelector) -> None:
        """Test keyword extraction."""
        text = "Debug the failing authentication service"
        keywords = selector._extract_keywords(text)
        assert "debug" in keywords
        assert "failing" in keywords
        assert "authentication" in keywords
        # Stopwords should be filtered
        assert "the" not in keywords

    def test_assess_memory_relevance_by_category(self, selector: TaskSelector) -> None:
        """Test relevance assessment by category."""
        # Debugging has high base relevance
        rel = selector._assess_memory_relevance(
            "Fix something",
            TaskCategory.DEBUGGING,
            ["fix"],
        )
        assert rel == MemoryRelevance.HIGH

        # Installation has low base relevance
        rel = selector._assess_memory_relevance(
            "Install package",
            TaskCategory.INSTALLATION,
            ["install"],
        )
        assert rel == MemoryRelevance.LOW

    def test_assess_memory_relevance_by_keywords(self, selector: TaskSelector) -> None:
        """Test relevance upgrade based on keywords."""
        # Multiple memory-relevant keywords upgrade relevance
        rel = selector._assess_memory_relevance(
            "debug and troubleshoot similar issue again",
            TaskCategory.UNKNOWN,
            [],
        )
        assert rel == MemoryRelevance.HIGH

    def test_create_synthetic_tasks(self, selector: TaskSelector) -> None:
        """Test synthetic task generation."""
        tasks = selector.create_synthetic_tasks(n_tasks=5, seed=42)
        assert len(tasks) == 5
        assert all(t.task_id.startswith("synthetic_") for t in tasks)

    def test_synthetic_tasks_reproducible(self, selector: TaskSelector) -> None:
        """Test that same seed produces same tasks."""
        tasks1 = selector.create_synthetic_tasks(n_tasks=5, seed=123)
        tasks2 = selector.create_synthetic_tasks(n_tasks=5, seed=123)
        assert [t.task_id for t in tasks1] == [t.task_id for t in tasks2]
        assert [t.description for t in tasks1] == [t.description for t in tasks2]

    def test_select_tasks_default_filter(self, selector: TaskSelector) -> None:
        """Test selecting with default filter."""
        # Generate synthetic tasks
        selector._tasks_cache = selector.create_synthetic_tasks(20, seed=42)

        # Default filter selects HIGH and MEDIUM relevance
        selected = selector.select_tasks()
        for task in selected:
            assert task.memory_relevance in {
                MemoryRelevance.HIGH,
                MemoryRelevance.MEDIUM,
            }

    def test_select_tasks_with_category_filter(self, selector: TaskSelector) -> None:
        """Test filtering by category."""
        selector._tasks_cache = selector.create_synthetic_tasks(20, seed=42)

        filter_config = TaskFilter(
            categories={TaskCategory.DEBUGGING},
        )
        selected = selector.select_tasks(filter_config)
        for task in selected:
            assert task.category == TaskCategory.DEBUGGING

    def test_select_tasks_with_difficulty_filter(self, selector: TaskSelector) -> None:
        """Test filtering by difficulty."""
        selector._tasks_cache = selector.create_synthetic_tasks(20, seed=42)

        filter_config = TaskFilter(
            min_difficulty=3,
            max_difficulty=4,
        )
        selected = selector.select_tasks(filter_config)
        for task in selected:
            assert 3 <= task.difficulty <= 4

    def test_select_tasks_with_max_tasks(self, selector: TaskSelector) -> None:
        """Test limiting number of tasks."""
        selector._tasks_cache = selector.create_synthetic_tasks(20, seed=42)

        filter_config = TaskFilter(max_tasks=5)
        selected = selector.select_tasks(filter_config)
        assert len(selected) <= 5

    def test_select_tasks_with_keyword_filter(self, selector: TaskSelector) -> None:
        """Test filtering by keywords."""
        selector._tasks_cache = selector.create_synthetic_tasks(20, seed=42)

        filter_config = TaskFilter(
            keywords={"authentication"},
        )
        selected = selector.select_tasks(filter_config)
        for task in selected:
            assert "authentication" in task.description.lower() or "authentication" in [
                k.lower() for k in task.keywords
            ]

    def test_select_tasks_with_exclude_keywords(self, selector: TaskSelector) -> None:
        """Test excluding by keywords."""
        selector._tasks_cache = selector.create_synthetic_tasks(20, seed=42)

        filter_config = TaskFilter(
            exclude_keywords={"authentication"},
        )
        selected = selector.select_tasks(filter_config)
        for task in selected:
            assert "authentication" not in task.description.lower()

    def test_get_task_found(self, selector: TaskSelector) -> None:
        """Test getting existing task."""
        selector._tasks_cache = selector.create_synthetic_tasks(5, seed=42)
        task = selector.get_task("synthetic_0000")
        assert task is not None
        assert task.task_id == "synthetic_0000"

    def test_get_task_not_found(self, selector: TaskSelector) -> None:
        """Test getting non-existent task."""
        selector._tasks_cache = selector.create_synthetic_tasks(5, seed=42)
        task = selector.get_task("nonexistent")
        assert task is None

    def test_get_stats(self, selector: TaskSelector) -> None:
        """Test getting statistics."""
        selector._tasks_cache = selector.create_synthetic_tasks(10, seed=42)
        stats = selector.get_stats()
        assert stats["total_tasks"] == 10
        assert "by_category" in stats
        assert "by_relevance" in stats
        assert "by_difficulty" in stats
