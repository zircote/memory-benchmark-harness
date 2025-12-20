"""Unit tests for the MockAdapter test utility.

Tests cover:
- Basic CRUD operations (same as real adapters)
- Call history recording
- Configurable return values
- Failure simulation
- Callback support
- Test utilities (seed_memories, reset)
"""

from datetime import UTC, datetime
from typing import Any

from src.adapters.base import MemoryItem, MemoryOperationResult
from src.adapters.mock import MockAdapter


class TestMockAdapterBasicOperations:
    """Tests for basic memory operations (same as real adapters)."""

    def test_add_memory_returns_success(self) -> None:
        """Test that adding a memory returns a successful result."""
        adapter = MockAdapter()
        result = adapter.add("Test content")

        assert result.success is True
        assert result.memory_id is not None
        assert result.memory_id.startswith("mock_")

    def test_add_memory_increments_counter(self) -> None:
        """Test that memory IDs are sequential."""
        adapter = MockAdapter()

        result1 = adapter.add("First")
        result2 = adapter.add("Second")
        result3 = adapter.add("Third")

        assert result1.memory_id == "mock_1"
        assert result2.memory_id == "mock_2"
        assert result3.memory_id == "mock_3"

    def test_update_existing_memory_succeeds(self) -> None:
        """Test updating an existing memory."""
        adapter = MockAdapter()
        add_result = adapter.add("Original")

        update_result = adapter.update(add_result.memory_id, "Updated")  # type: ignore[arg-type]

        assert update_result.success is True
        assert "updated_at" in update_result.metadata

    def test_update_nonexistent_memory_fails(self) -> None:
        """Test updating a nonexistent memory fails."""
        adapter = MockAdapter()

        result = adapter.update("nonexistent", "Content")

        assert result.success is False
        assert "not found" in result.error.lower()  # type: ignore[union-attr]

    def test_delete_existing_memory_succeeds(self) -> None:
        """Test deleting an existing memory."""
        adapter = MockAdapter()
        add_result = adapter.add("To delete")

        delete_result = adapter.delete(add_result.memory_id)  # type: ignore[arg-type]

        assert delete_result.success is True
        assert adapter.get_stats()["memory_count"] == 0

    def test_delete_nonexistent_memory_fails(self) -> None:
        """Test deleting a nonexistent memory fails."""
        adapter = MockAdapter()

        result = adapter.delete("nonexistent")

        assert result.success is False
        assert "not found" in result.error.lower()  # type: ignore[union-attr]

    def test_clear_removes_all_memories(self) -> None:
        """Test that clear removes all memories."""
        adapter = MockAdapter()
        adapter.add("First")
        adapter.add("Second")

        result = adapter.clear()

        assert result.success is True
        assert result.metadata.get("cleared_count") == 2
        assert adapter.get_stats()["memory_count"] == 0

    def test_get_stats_returns_correct_values(self) -> None:
        """Test that stats are correct."""
        adapter = MockAdapter()
        adapter.add("Hello")
        adapter.add("World")

        stats = adapter.get_stats()

        assert stats["memory_count"] == 2
        assert stats["type"] == "mock"
        assert stats["total_content_length"] == 10  # "Hello" + "World"


class TestMockAdapterSearch:
    """Tests for search functionality."""

    def test_search_finds_matching_content(self) -> None:
        """Test that search finds content with substring match."""
        adapter = MockAdapter()
        adapter.add("Python programming guide")
        adapter.add("JavaScript tutorial")
        adapter.add("Advanced Python tips")

        results = adapter.search("Python")

        assert len(results) == 2
        assert all("Python" in item.content for item in results)

    def test_search_respects_limit(self) -> None:
        """Test that search respects the limit parameter."""
        adapter = MockAdapter()
        for i in range(10):
            adapter.add(f"Item {i} with keyword")

        results = adapter.search("keyword", limit=3)

        assert len(results) == 3

    def test_search_respects_min_score(self) -> None:
        """Test that search respects min_score threshold."""
        adapter = MockAdapter()
        adapter.add("Short")  # Low score (query length / content length)
        adapter.add("S")  # High score (query matches well)

        results = adapter.search("S", min_score=0.5)

        # Only "S" should match with score >= 0.5
        assert len(results) == 1
        assert results[0].content == "S"

    def test_search_with_metadata_filter(self) -> None:
        """Test that search filters by metadata."""
        adapter = MockAdapter()
        adapter.add("Doc 1", metadata={"category": "tutorial"})
        adapter.add("Doc 2", metadata={"category": "reference"})
        adapter.add("Doc 3", metadata={"category": "tutorial"})

        results = adapter.search("Doc", metadata_filter={"category": "tutorial"})

        assert len(results) == 2
        assert all(item.metadata.get("category") == "tutorial" for item in results)


class TestMockAdapterCallHistory:
    """Tests for call history recording."""

    def test_add_records_call_history(self) -> None:
        """Test that add() records call history."""
        adapter = MockAdapter()
        adapter.add("Content 1", metadata={"key": "value"})
        adapter.add("Content 2")

        history = adapter.get_call_history("add")

        assert len(history) == 2
        assert history[0] == {"content": "Content 1", "metadata": {"key": "value"}}
        assert history[1] == {"content": "Content 2", "metadata": None}

    def test_search_records_call_history(self) -> None:
        """Test that search() records call history."""
        adapter = MockAdapter()
        adapter.search("query1", limit=5, min_score=0.3)
        adapter.search("query2", metadata_filter={"tag": "important"})

        history = adapter.get_call_history("search")

        assert len(history) == 2
        assert history[0]["query"] == "query1"
        assert history[0]["limit"] == 5
        assert history[0]["min_score"] == 0.3
        assert history[1]["metadata_filter"] == {"tag": "important"}

    def test_update_records_call_history(self) -> None:
        """Test that update() records call history."""
        adapter = MockAdapter()
        result = adapter.add("Original")
        adapter.update(result.memory_id, "Updated", metadata={"v": 2})  # type: ignore[arg-type]

        history = adapter.get_call_history("update")

        assert len(history) == 1
        assert history[0]["memory_id"] == result.memory_id
        assert history[0]["content"] == "Updated"
        assert history[0]["metadata"] == {"v": 2}

    def test_delete_records_call_history(self) -> None:
        """Test that delete() records call history."""
        adapter = MockAdapter()
        result = adapter.add("To delete")
        adapter.delete(result.memory_id)  # type: ignore[arg-type]

        history = adapter.get_call_history("delete")

        assert len(history) == 1
        assert history[0]["memory_id"] == result.memory_id

    def test_clear_records_call_history(self) -> None:
        """Test that clear() records call history."""
        adapter = MockAdapter()
        adapter.clear()
        adapter.clear()

        history = adapter.get_call_history("clear")

        assert len(history) == 2
        assert history[0] == {}
        assert history[1] == {}

    def test_get_stats_records_call_history(self) -> None:
        """Test that get_stats() records call history."""
        adapter = MockAdapter()
        adapter.get_stats()

        history = adapter.get_call_history("get_stats")

        assert len(history) == 1
        assert history[0] == {}

    def test_get_all_call_history(self) -> None:
        """Test getting all call history at once."""
        adapter = MockAdapter()
        adapter.add("Content")
        adapter.search("query")
        adapter.get_stats()

        all_history = adapter.get_all_call_history()

        assert len(all_history["add"]) == 1
        assert len(all_history["search"]) == 1
        assert len(all_history["get_stats"]) == 1

    def test_reset_call_history(self) -> None:
        """Test resetting call history."""
        adapter = MockAdapter()
        adapter.add("Content")
        adapter.search("query")

        adapter.reset_call_history()

        assert adapter.get_call_history("add") == []
        assert adapter.get_call_history("search") == []


class TestMockAdapterConfiguredResults:
    """Tests for configurable return values."""

    def test_configure_search_returns_configured_results(self) -> None:
        """Test that configure_search() works."""
        adapter = MockAdapter()
        now = datetime.now(UTC)
        configured = [
            MemoryItem(
                memory_id="configured_1",
                content="Configured result",
                metadata={},
                score=0.99,
                created_at=now,
            )
        ]

        adapter.configure_search(configured)
        results = adapter.search("anything")

        assert len(results) == 1
        assert results[0].memory_id == "configured_1"
        assert results[0].content == "Configured result"

    def test_configure_search_respects_limit(self) -> None:
        """Test that configured search respects limit."""
        adapter = MockAdapter()
        now = datetime.now(UTC)
        configured = [MemoryItem(f"id_{i}", f"Content {i}", {}, 0.9, now) for i in range(10)]

        adapter.configure_search(configured)
        results = adapter.search("query", limit=3)

        assert len(results) == 3


class TestMockAdapterFailureSimulation:
    """Tests for failure simulation."""

    def test_configure_add_failure(self) -> None:
        """Test simulating add failure."""
        adapter = MockAdapter()
        adapter.configure_add_failure("Database connection failed")

        result = adapter.add("Content")

        assert result.success is False
        assert result.error == "Database connection failed"

    def test_configure_update_failure(self) -> None:
        """Test simulating update failure."""
        adapter = MockAdapter()
        result = adapter.add("Original")
        adapter.configure_update_failure("Update rejected")

        update_result = adapter.update(result.memory_id, "New")  # type: ignore[arg-type]

        assert update_result.success is False
        assert update_result.error == "Update rejected"

    def test_configure_delete_failure(self) -> None:
        """Test simulating delete failure."""
        adapter = MockAdapter()
        result = adapter.add("Content")
        adapter.configure_delete_failure("Delete not allowed")

        delete_result = adapter.delete(result.memory_id)  # type: ignore[arg-type]

        assert delete_result.success is False
        assert delete_result.error == "Delete not allowed"

    def test_configure_clear_failure(self) -> None:
        """Test simulating clear failure."""
        adapter = MockAdapter()
        adapter.add("Content")
        adapter.configure_clear_failure("Clear disabled")

        result = adapter.clear()

        assert result.success is False
        assert result.error == "Clear disabled"

    def test_reset_configuration_clears_failures(self) -> None:
        """Test that reset_configuration clears failure configs."""
        adapter = MockAdapter()
        adapter.configure_add_failure("Error")
        adapter.configure_update_failure("Error")
        adapter.configure_delete_failure("Error")
        adapter.configure_clear_failure("Error")

        adapter.reset_configuration()

        # All operations should now succeed
        result = adapter.add("Content")
        assert result.success is True


class TestMockAdapterCallbacks:
    """Tests for callback support."""

    def test_search_callback(self) -> None:
        """Test that search callback is called."""
        adapter = MockAdapter()
        now = datetime.now(UTC)

        def custom_search(
            query: str,
            limit: int,
            min_score: float,
            metadata_filter: dict[str, Any] | None,
        ) -> list[MemoryItem]:
            return [
                MemoryItem(
                    memory_id="callback_result",
                    content=f"Found: {query}",
                    metadata={"limit": limit},
                    score=0.95,
                    created_at=now,
                )
            ]

        adapter.configure_search_callback(custom_search)
        results = adapter.search("test query", limit=5)

        assert len(results) == 1
        assert results[0].memory_id == "callback_result"
        assert results[0].content == "Found: test query"
        assert results[0].metadata["limit"] == 5

    def test_add_callback(self) -> None:
        """Test that add callback is called."""
        adapter = MockAdapter()

        def custom_add(content: str, metadata: dict[str, Any] | None) -> MemoryOperationResult:
            return MemoryOperationResult(
                success=True,
                memory_id=f"custom_{len(content)}",
                metadata={"custom": True},
            )

        adapter.configure_add_callback(custom_add)
        result = adapter.add("Hello")

        assert result.success is True
        assert result.memory_id == "custom_5"
        assert result.metadata.get("custom") is True

    def test_callback_takes_priority_over_configured_results(self) -> None:
        """Test that callback has priority over configure_search."""
        adapter = MockAdapter()
        now = datetime.now(UTC)

        # Configure static results
        adapter.configure_search([MemoryItem("static", "Static result", {}, 0.5, now)])

        # Configure callback
        def callback(q: str, l: int, s: float, m: dict[str, Any] | None) -> list[MemoryItem]:
            return [MemoryItem("dynamic", "Dynamic result", {}, 0.9, now)]

        adapter.configure_search_callback(callback)

        results = adapter.search("query")

        # Callback should take priority
        assert len(results) == 1
        assert results[0].memory_id == "dynamic"

    def test_add_callback_takes_priority_over_failure(self) -> None:
        """Test that add callback has priority over failure config."""
        adapter = MockAdapter()
        adapter.configure_add_failure("Should not see this")

        def callback(content: str, metadata: dict[str, Any] | None) -> MemoryOperationResult:
            return MemoryOperationResult(success=True, memory_id="callback_id")

        adapter.configure_add_callback(callback)
        result = adapter.add("Content")

        assert result.success is True
        assert result.memory_id == "callback_id"


class TestMockAdapterUtilities:
    """Tests for test utility methods."""

    def test_seed_memories(self) -> None:
        """Test seeding memories for testing."""
        adapter = MockAdapter()

        memory_ids = adapter.seed_memories(
            [
                ("Content 1", {"tag": "first"}),
                ("Content 2", None),
                ("Content 3", {"tag": "third"}),
            ]
        )

        assert len(memory_ids) == 3
        assert adapter.get_stats()["memory_count"] == 3
        # Verify search finds seeded content
        results = adapter.search("Content")
        assert len(results) == 3

    def test_reset_clears_everything(self) -> None:
        """Test that reset() clears all state."""
        adapter = MockAdapter()

        # Set up various state
        adapter.add("Content")
        adapter.configure_add_failure("Error")
        adapter.configure_search([])
        adapter.search("query")

        adapter.reset()

        # Check all state is cleared
        assert adapter.get_stats()["memory_count"] == 0
        assert adapter.get_call_history("add") == []
        assert adapter.get_call_history("search") == []

        # Check operations work normally again
        result = adapter.add("New content")
        assert result.success is True
        assert result.memory_id == "mock_1"  # Counter reset

    def test_adapter_isolation(self) -> None:
        """Test that multiple adapter instances are isolated."""
        adapter1 = MockAdapter()
        adapter2 = MockAdapter()

        adapter1.add("In adapter 1")
        adapter1.configure_add_failure("Fail in 1")

        result2 = adapter2.add("In adapter 2")

        assert adapter1.get_stats()["memory_count"] == 1
        assert adapter2.get_stats()["memory_count"] == 1
        assert result2.success is True  # Not affected by adapter1's failure config


class TestMockAdapterEdgeCases:
    """Tests for edge cases."""

    def test_get_call_history_unknown_method(self) -> None:
        """Test getting call history for unknown method."""
        adapter = MockAdapter()

        history = adapter.get_call_history("unknown_method")

        assert history == []

    def test_empty_search_query(self) -> None:
        """Test search with empty query."""
        adapter = MockAdapter()
        adapter.add("Content")

        results = adapter.search("")

        # Empty string is substring of everything
        assert len(results) == 1

    def test_unicode_content(self) -> None:
        """Test with unicode content."""
        adapter = MockAdapter()
        adapter.add("Hello 世界 🌍")
        adapter.add("Привет мир")

        results = adapter.search("世界")

        assert len(results) == 1
        assert "世界" in results[0].content

    def test_metadata_filter_partial_match(self) -> None:
        """Test that metadata filter requires all keys to match."""
        adapter = MockAdapter()
        adapter.add("Content", metadata={"a": 1, "b": 2})

        # Should find when filter is subset
        results = adapter.search("Content", metadata_filter={"a": 1})
        assert len(results) == 1

        # Should not find when filter has extra key
        results = adapter.search("Content", metadata_filter={"a": 1, "c": 3})
        assert len(results) == 0
