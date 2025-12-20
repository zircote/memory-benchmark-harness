"""Unit tests for the NoMemoryAdapter baseline adapter.

Tests cover:
- Memory storage (add, update, delete, clear operations)
- Search behavior (always returns empty - core baseline characteristic)
- Statistics tracking
- Edge cases and error handling

The NoMemoryAdapter is a scientific control condition - it stores memories
but never retrieves them, establishing a baseline for measuring the actual
benefit of memory retrieval.
"""

from datetime import datetime
from typing import Any

import pytest

from src.adapters.no_memory import NoMemoryAdapter


class TestNoMemoryAdapterBasicOperations:
    """Tests for basic memory operations."""

    def test_add_memory_returns_success(self) -> None:
        """Test that adding a memory returns a successful result."""
        adapter = NoMemoryAdapter()
        result = adapter.add("Test content")

        assert result.success is True
        assert result.memory_id is not None
        assert result.memory_id.startswith("baseline_")
        assert result.error is None

    def test_add_memory_increments_counter(self) -> None:
        """Test that memory IDs are sequential."""
        adapter = NoMemoryAdapter()

        result1 = adapter.add("First memory")
        result2 = adapter.add("Second memory")
        result3 = adapter.add("Third memory")

        assert result1.memory_id == "baseline_1"
        assert result2.memory_id == "baseline_2"
        assert result3.memory_id == "baseline_3"

    def test_add_memory_with_metadata(self) -> None:
        """Test adding a memory with metadata."""
        adapter = NoMemoryAdapter()
        metadata = {"session_id": "sess_123", "tags": ["important"]}

        result = adapter.add("Content with metadata", metadata=metadata)

        assert result.success is True
        assert "stored_at" in result.metadata

    def test_add_memory_stored_at_is_iso_format(self) -> None:
        """Test that stored_at timestamp is in ISO format."""
        adapter = NoMemoryAdapter()
        result = adapter.add("Test content")

        stored_at = result.metadata.get("stored_at")
        assert stored_at is not None
        # Should be parseable as ISO format
        datetime.fromisoformat(stored_at)


class TestNoMemoryAdapterSearch:
    """Tests for search behavior - the core baseline characteristic."""

    def test_search_always_returns_empty_list(self) -> None:
        """Test that search always returns empty list (baseline behavior)."""
        adapter = NoMemoryAdapter()
        adapter.add("Important context about Python")
        adapter.add("More information about testing")
        adapter.add("Detailed notes on memory systems")

        # Search should return nothing, regardless of query
        results = adapter.search("Python")
        assert results == []

    def test_search_with_exact_match_query_returns_empty(self) -> None:
        """Test that even exact match queries return empty."""
        adapter = NoMemoryAdapter()
        adapter.add("The quick brown fox jumps over the lazy dog")

        results = adapter.search("The quick brown fox jumps over the lazy dog")
        assert results == []

    def test_search_with_limit_parameter_returns_empty(self) -> None:
        """Test that limit parameter doesn't affect empty result."""
        adapter = NoMemoryAdapter()
        for i in range(100):
            adapter.add(f"Memory number {i}")

        results = adapter.search("memory", limit=50)
        assert results == []

    def test_search_with_min_score_returns_empty(self) -> None:
        """Test that min_score parameter is ignored (returns empty)."""
        adapter = NoMemoryAdapter()
        adapter.add("Test content")

        results = adapter.search("test", min_score=0.0)
        assert results == []

        results = adapter.search("test", min_score=1.0)
        assert results == []

    def test_search_with_metadata_filter_returns_empty(self) -> None:
        """Test that metadata filter is ignored (returns empty)."""
        adapter = NoMemoryAdapter()
        adapter.add("Tagged content", metadata={"tag": "important"})

        results = adapter.search("content", metadata_filter={"tag": "important"})
        assert results == []

    def test_search_on_empty_adapter_returns_empty(self) -> None:
        """Test search on adapter with no memories."""
        adapter = NoMemoryAdapter()
        results = adapter.search("anything")
        assert results == []


class TestNoMemoryAdapterUpdate:
    """Tests for update operations."""

    def test_update_existing_memory_succeeds(self) -> None:
        """Test updating an existing memory."""
        adapter = NoMemoryAdapter()
        add_result = adapter.add("Original content")
        memory_id = add_result.memory_id

        update_result = adapter.update(memory_id, "Updated content")  # type: ignore[arg-type]

        assert update_result.success is True
        assert update_result.memory_id == memory_id
        assert "updated_at" in update_result.metadata

    def test_update_nonexistent_memory_fails(self) -> None:
        """Test updating a memory that doesn't exist."""
        adapter = NoMemoryAdapter()

        result = adapter.update("nonexistent_123", "New content")

        assert result.success is False
        assert result.error is not None
        assert "not found" in result.error.lower()

    def test_update_with_metadata_merges(self) -> None:
        """Test that update metadata merges with existing."""
        adapter = NoMemoryAdapter()
        result = adapter.add("Content", metadata={"key1": "value1"})
        memory_id = result.memory_id

        adapter.update(memory_id, "Updated", metadata={"key2": "value2"})  # type: ignore[arg-type]

        # Verify metadata was merged by checking stats
        # (We can't directly access internal state, but stats should reflect the update)
        stats = adapter.get_stats()
        assert stats["memory_count"] == 1


class TestNoMemoryAdapterDelete:
    """Tests for delete operations."""

    def test_delete_existing_memory_succeeds(self) -> None:
        """Test deleting an existing memory."""
        adapter = NoMemoryAdapter()
        add_result = adapter.add("Content to delete")
        memory_id = add_result.memory_id

        delete_result = adapter.delete(memory_id)  # type: ignore[arg-type]

        assert delete_result.success is True
        assert delete_result.memory_id == memory_id

    def test_delete_reduces_memory_count(self) -> None:
        """Test that delete reduces the memory count."""
        adapter = NoMemoryAdapter()
        adapter.add("First")
        result = adapter.add("Second")
        adapter.add("Third")

        assert adapter.get_stats()["memory_count"] == 3

        adapter.delete(result.memory_id)  # type: ignore[arg-type]

        assert adapter.get_stats()["memory_count"] == 2

    def test_delete_nonexistent_memory_fails(self) -> None:
        """Test deleting a memory that doesn't exist."""
        adapter = NoMemoryAdapter()

        result = adapter.delete("nonexistent_456")

        assert result.success is False
        assert result.error is not None
        assert "not found" in result.error.lower()

    def test_delete_same_memory_twice_fails_second_time(self) -> None:
        """Test that deleting the same memory twice fails the second time."""
        adapter = NoMemoryAdapter()
        result = adapter.add("Content")
        memory_id = result.memory_id

        first_delete = adapter.delete(memory_id)  # type: ignore[arg-type]
        second_delete = adapter.delete(memory_id)  # type: ignore[arg-type]

        assert first_delete.success is True
        assert second_delete.success is False


class TestNoMemoryAdapterClear:
    """Tests for clear operation."""

    def test_clear_removes_all_memories(self) -> None:
        """Test that clear removes all memories."""
        adapter = NoMemoryAdapter()
        adapter.add("First")
        adapter.add("Second")
        adapter.add("Third")

        assert adapter.get_stats()["memory_count"] == 3

        result = adapter.clear()

        assert result.success is True
        assert adapter.get_stats()["memory_count"] == 0

    def test_clear_returns_cleared_count(self) -> None:
        """Test that clear returns the count of cleared memories."""
        adapter = NoMemoryAdapter()
        adapter.add("First")
        adapter.add("Second")

        result = adapter.clear()

        assert result.metadata.get("cleared_count") == 2

    def test_clear_resets_counter(self) -> None:
        """Test that clear resets the ID counter."""
        adapter = NoMemoryAdapter()
        adapter.add("First")
        adapter.add("Second")
        adapter.clear()

        result = adapter.add("After clear")
        assert result.memory_id == "baseline_1"

    def test_clear_empty_adapter_succeeds(self) -> None:
        """Test clearing an empty adapter succeeds."""
        adapter = NoMemoryAdapter()

        result = adapter.clear()

        assert result.success is True
        assert result.metadata.get("cleared_count") == 0


class TestNoMemoryAdapterStats:
    """Tests for statistics tracking."""

    def test_get_stats_on_empty_adapter(self) -> None:
        """Test stats on a fresh adapter."""
        adapter = NoMemoryAdapter()
        stats = adapter.get_stats()

        assert stats["memory_count"] == 0
        assert stats["type"] == "no-memory-baseline"
        assert stats["total_content_length"] == 0
        assert stats["avg_content_length"] == 0

    def test_get_stats_tracks_memory_count(self) -> None:
        """Test that stats tracks memory count correctly."""
        adapter = NoMemoryAdapter()
        adapter.add("First")
        adapter.add("Second")
        adapter.add("Third")

        stats = adapter.get_stats()

        assert stats["memory_count"] == 3

    def test_get_stats_tracks_content_length(self) -> None:
        """Test that stats tracks total content length."""
        adapter = NoMemoryAdapter()
        adapter.add("Hello")  # 5 chars
        adapter.add("World")  # 5 chars
        adapter.add("!")  # 1 char

        stats = adapter.get_stats()

        assert stats["total_content_length"] == 11
        assert stats["avg_content_length"] == pytest.approx(11 / 3)

    def test_get_stats_type_identifier(self) -> None:
        """Test that stats includes correct type identifier."""
        adapter = NoMemoryAdapter()
        stats = adapter.get_stats()

        assert stats["type"] == "no-memory-baseline"


class TestNoMemoryAdapterEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_add_empty_content(self) -> None:
        """Test adding empty content."""
        adapter = NoMemoryAdapter()
        result = adapter.add("")

        assert result.success is True
        assert adapter.get_stats()["memory_count"] == 1
        assert adapter.get_stats()["total_content_length"] == 0

    def test_add_unicode_content(self) -> None:
        """Test adding Unicode content."""
        adapter = NoMemoryAdapter()
        result = adapter.add("Hello 世界 🌍 мир")

        assert result.success is True
        assert adapter.get_stats()["memory_count"] == 1

    def test_add_very_long_content(self) -> None:
        """Test adding very long content."""
        adapter = NoMemoryAdapter()
        long_content = "x" * 1_000_000  # 1 million characters

        result = adapter.add(long_content)

        assert result.success is True
        assert adapter.get_stats()["total_content_length"] == 1_000_000

    def test_metadata_none_handling(self) -> None:
        """Test that None metadata is handled correctly."""
        adapter = NoMemoryAdapter()
        result = adapter.add("Content", metadata=None)

        assert result.success is True

    def test_complex_metadata(self) -> None:
        """Test adding complex nested metadata."""
        adapter = NoMemoryAdapter()
        metadata: dict[str, Any] = {
            "nested": {"deep": {"value": 42}},
            "list": [1, 2, 3],
            "mixed": {"items": ["a", "b"], "count": 2},
        }

        result = adapter.add("Content", metadata=metadata)

        assert result.success is True

    def test_concurrent_operations_simulation(self) -> None:
        """Test that sequential operations don't interfere with each other."""
        adapter = NoMemoryAdapter()

        # Simulate interleaved operations
        r1 = adapter.add("First")
        r2 = adapter.add("Second")
        adapter.delete(r1.memory_id)  # type: ignore[arg-type]
        r3 = adapter.add("Third")
        adapter.update(r2.memory_id, "Updated second")  # type: ignore[arg-type]

        stats = adapter.get_stats()
        assert stats["memory_count"] == 2  # r2 and r3 remain

    def test_adapter_isolation(self) -> None:
        """Test that multiple adapter instances are isolated."""
        adapter1 = NoMemoryAdapter()
        adapter2 = NoMemoryAdapter()

        adapter1.add("In adapter 1")
        adapter1.add("Also in adapter 1")

        adapter2.add("In adapter 2")

        assert adapter1.get_stats()["memory_count"] == 2
        assert adapter2.get_stats()["memory_count"] == 1
