"""Tests for the ablation adapters module."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

from src.adapters.ablation import (
    AblationType,
    FixedWindowAdapter,
    NoMetadataFilterAdapter,
    NoSemanticSearchAdapter,
    NoVersionHistoryAdapter,
    RecencyOnlyAdapter,
    create_ablation_adapter,
)
from src.adapters.base import MemoryItem, MemoryOperationResult


class MockBaseAdapter:
    """Mock base adapter for testing ablation wrappers."""

    def __init__(self) -> None:
        self.memories: dict[str, tuple[str, dict[str, Any]]] = {}
        self.add_calls = 0
        self.search_calls = 0
        self.search_kwargs: list[dict[str, Any]] = []
        self._next_id = 0

    def add(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryOperationResult:
        self.add_calls += 1
        mem_id = f"mem_{self._next_id}"
        self._next_id += 1
        self.memories[mem_id] = (content, metadata or {})
        return MemoryOperationResult(success=True, memory_id=mem_id)

    def search(
        self,
        query: str,
        limit: int = 10,
        min_score: float = 0.0,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        self.search_calls += 1
        self.search_kwargs.append(
            {
                "query": query,
                "limit": limit,
                "min_score": min_score,
                "metadata_filter": metadata_filter,
            }
        )

        # Return memories in order added
        results = []
        for i, (mem_id, (content, metadata)) in enumerate(self.memories.items()):
            if i >= limit:
                break
            results.append(
                MemoryItem(
                    memory_id=mem_id,
                    content=content,
                    score=0.9 - (i * 0.1),
                    metadata=metadata,
                    created_at=datetime.now(),
                )
            )
        return results

    def update(
        self,
        memory_id: str,
        content: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryOperationResult:
        if memory_id in self.memories:
            old_content, old_metadata = self.memories[memory_id]
            self.memories[memory_id] = (
                content or old_content,
                metadata or old_metadata,
            )
            return MemoryOperationResult(success=True, memory_id=memory_id)
        return MemoryOperationResult(success=False, error="Not found")

    def delete(self, memory_id: str) -> MemoryOperationResult:
        if memory_id in self.memories:
            del self.memories[memory_id]
            return MemoryOperationResult(success=True)
        return MemoryOperationResult(success=False, error="Not found")

    def clear(self) -> MemoryOperationResult:
        self.memories.clear()
        return MemoryOperationResult(success=True)

    def get_stats(self) -> dict[str, Any]:
        return {
            "memory_count": len(self.memories),
            "add_calls": self.add_calls,
            "search_calls": self.search_calls,
        }


class TestNoSemanticSearchAdapter:
    """Tests for the NoSemanticSearchAdapter."""

    @pytest.fixture
    def adapter(self) -> NoSemanticSearchAdapter:
        """Create adapter for testing."""
        return NoSemanticSearchAdapter(
            base_adapter=MockBaseAdapter(),
            seed=42,
        )

    def test_add_delegates_to_base(self, adapter: NoSemanticSearchAdapter) -> None:
        """Test that add delegates to base adapter."""
        result = adapter.add("test content", {"key": "value"})
        assert result.success
        assert adapter.base_adapter.add_calls == 1  # type: ignore

    def test_search_returns_random(self, adapter: NoSemanticSearchAdapter) -> None:
        """Test that search returns random results."""
        # Add some memories
        for i in range(5):
            adapter.add(f"content {i}")

        # Search should return random subset
        results = adapter.search("any query", limit=3)
        assert len(results) == 3
        # Scores should be random (not based on query)
        assert all(0 <= r.score <= 1 for r in results)

    def test_reproducibility_with_seed(self) -> None:
        """Test that same seed produces same random selection."""
        adapter1 = NoSemanticSearchAdapter(base_adapter=MockBaseAdapter(), seed=42)
        adapter2 = NoSemanticSearchAdapter(base_adapter=MockBaseAdapter(), seed=42)

        for i in range(5):
            adapter1.add(f"content {i}")
            adapter2.add(f"content {i}")

        results1 = adapter1.search("query", limit=3)
        results2 = adapter2.search("query", limit=3)

        # With same seed, should get same random selection
        # Memory IDs should match
        ids1 = [r.memory_id for r in results1]
        ids2 = [r.memory_id for r in results2]
        assert ids1 == ids2

    def test_clear(self, adapter: NoSemanticSearchAdapter) -> None:
        """Test clear functionality."""
        adapter.add("content")
        adapter.clear()
        results = adapter.search("query")
        assert len(results) == 0

    def test_stats_include_ablation_type(self, adapter: NoSemanticSearchAdapter) -> None:
        """Test that stats indicate ablation type."""
        stats = adapter.get_stats()
        assert stats["ablation"] == "no_semantic_search"


class TestNoMetadataFilterAdapter:
    """Tests for the NoMetadataFilterAdapter."""

    @pytest.fixture
    def adapter(self) -> NoMetadataFilterAdapter:
        """Create adapter for testing."""
        return NoMetadataFilterAdapter(base_adapter=MockBaseAdapter())

    def test_search_ignores_metadata_filter(self, adapter: NoMetadataFilterAdapter) -> None:
        """Test that metadata filter is ignored."""
        adapter.add("content", {"category": "test"})

        # Search with metadata filter
        adapter.search("query", metadata_filter={"category": "test"})

        # Base adapter should have been called without metadata filter
        search_kwargs = adapter.base_adapter.search_kwargs[-1]  # type: ignore
        assert search_kwargs["metadata_filter"] is None

    def test_add_preserves_metadata(self, adapter: NoMetadataFilterAdapter) -> None:
        """Test that add still stores metadata."""
        result = adapter.add("content", {"key": "value"})
        assert result.success

    def test_stats_include_ablation_type(self, adapter: NoMetadataFilterAdapter) -> None:
        """Test that stats indicate ablation type."""
        stats = adapter.get_stats()
        assert stats["ablation"] == "no_metadata_filter"


class TestNoVersionHistoryAdapter:
    """Tests for the NoVersionHistoryAdapter."""

    @pytest.fixture
    def adapter(self) -> NoVersionHistoryAdapter:
        """Create adapter for testing."""
        base = MockBaseAdapter()
        return NoVersionHistoryAdapter(base_adapter=base)

    def test_search_strips_version_history(self, adapter: NoVersionHistoryAdapter) -> None:
        """Test that version history is stripped from results."""
        adapter.add(
            "content",
            {
                "version_history": [{"v": 1}, {"v": 2}],
                "versions": [1, 2],
                "other_key": "preserved",
            },
        )

        results = adapter.search("query")
        assert len(results) == 1

        # Version history should be stripped
        metadata = results[0].metadata
        assert "version_history" not in metadata
        assert "versions" not in metadata
        assert metadata.get("other_key") == "preserved"

    def test_get_history_returns_empty(self, adapter: NoVersionHistoryAdapter) -> None:
        """Test that get_history returns empty list."""
        adapter.add("content")
        history = adapter.get_history("any_id")
        assert history == []

    def test_stats_include_ablation_type(self, adapter: NoVersionHistoryAdapter) -> None:
        """Test that stats indicate ablation type."""
        stats = adapter.get_stats()
        assert stats["ablation"] == "no_version_history"


class TestFixedWindowAdapter:
    """Tests for the FixedWindowAdapter."""

    @pytest.fixture
    def adapter(self) -> FixedWindowAdapter:
        """Create adapter for testing."""
        return FixedWindowAdapter(
            base_adapter=MockBaseAdapter(),
            window_size=3,
        )

    def test_search_returns_fixed_window(self, adapter: FixedWindowAdapter) -> None:
        """Test that search returns fixed window of first N memories."""
        for i in range(10):
            adapter.add(f"content {i}")

        # Query should be ignored, always return first 3
        results = adapter.search("anything", limit=5)
        assert len(results) == 3  # window_size

        # Should be first 3 memories
        contents = [r.content for r in results]
        assert contents == ["content 0", "content 1", "content 2"]

    def test_window_size_respected(self) -> None:
        """Test different window sizes."""
        adapter = FixedWindowAdapter(
            base_adapter=MockBaseAdapter(),
            window_size=5,
        )
        for i in range(10):
            adapter.add(f"content {i}")

        results = adapter.search("query")
        assert len(results) == 5

    def test_stats_include_window_size(self, adapter: FixedWindowAdapter) -> None:
        """Test that stats include window size."""
        stats = adapter.get_stats()
        assert stats["ablation"] == "fixed_window"
        assert stats["window_size"] == 3


class TestRecencyOnlyAdapter:
    """Tests for the RecencyOnlyAdapter."""

    @pytest.fixture
    def adapter(self) -> RecencyOnlyAdapter:
        """Create adapter for testing."""
        return RecencyOnlyAdapter(base_adapter=MockBaseAdapter())

    def test_search_returns_most_recent(self, adapter: RecencyOnlyAdapter) -> None:
        """Test that search returns most recent memories."""
        for i in range(5):
            adapter.add(f"content {i}")

        results = adapter.search("query", limit=3)
        assert len(results) == 3

        # Should be most recent 3 (in reverse order)
        contents = [r.content for r in results]
        assert contents == ["content 4", "content 3", "content 2"]

    def test_scores_decrease_by_recency(self, adapter: RecencyOnlyAdapter) -> None:
        """Test that scores decrease for older memories."""
        for i in range(5):
            adapter.add(f"content {i}")

        results = adapter.search("query", limit=5)

        # Scores should be decreasing
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score

    def test_update_moves_to_recent(self, adapter: RecencyOnlyAdapter) -> None:
        """Test that update moves memory to most recent."""
        adapter.add("content 0")
        result = adapter.add("content 1")
        adapter.add("content 2")

        # Update middle memory
        adapter.update(result.memory_id, content="updated content 1")

        # Now it should be most recent
        results = adapter.search("query", limit=3)
        assert results[0].content == "updated content 1"

    def test_stats_include_ablation_type(self, adapter: RecencyOnlyAdapter) -> None:
        """Test that stats indicate ablation type."""
        stats = adapter.get_stats()
        assert stats["ablation"] == "recency_only"


class TestCreateAblationAdapter:
    """Tests for the factory function."""

    def test_create_all_types(self) -> None:
        """Test creating all ablation types."""
        base = MockBaseAdapter()

        for ablation_type in [
            AblationType.NO_SEMANTIC_SEARCH,
            AblationType.NO_METADATA_FILTER,
            AblationType.NO_VERSION_HISTORY,
            AblationType.FIXED_WINDOW,
            AblationType.RECENCY_ONLY,
        ]:
            adapter = create_ablation_adapter(base, ablation_type)
            assert adapter is not None

    def test_create_with_kwargs(self) -> None:
        """Test creating with additional kwargs."""
        base = MockBaseAdapter()

        adapter = create_ablation_adapter(
            base,
            AblationType.FIXED_WINDOW,
            window_size=5,
        )
        assert adapter.window_size == 5  # type: ignore

    def test_create_invalid_type(self) -> None:
        """Test creating with invalid type raises error."""
        base = MockBaseAdapter()

        with pytest.raises(ValueError, match="Unknown ablation type"):
            create_ablation_adapter(base, "invalid_type")
