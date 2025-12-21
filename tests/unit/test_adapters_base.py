"""Unit tests for the base adapter interface and data classes.

Tests cover:
- MemoryItem dataclass validation and construction
- MemoryOperationResult dataclass validation and construction
- MemorySystemAdapter ABC interface enforcement
"""

from datetime import datetime

import pytest

from src.adapters.base import MemoryItem, MemoryOperationResult, MemorySystemAdapter


class TestMemoryItem:
    """Tests for MemoryItem dataclass."""

    def test_create_valid_memory_item(self) -> None:
        """Test creating a valid MemoryItem with all fields."""
        now = datetime.now()
        item = MemoryItem(
            memory_id="mem_123",
            content="Test memory content",
            metadata={"session_id": "sess_1", "tags": ["important"]},
            score=0.95,
            created_at=now,
            updated_at=now,
        )

        assert item.memory_id == "mem_123"
        assert item.content == "Test memory content"
        assert item.metadata == {"session_id": "sess_1", "tags": ["important"]}
        assert item.score == 0.95
        assert item.created_at == now
        assert item.updated_at == now

    def test_create_memory_item_without_updated_at(self) -> None:
        """Test creating a MemoryItem without updated_at (optional field)."""
        now = datetime.now()
        item = MemoryItem(
            memory_id="mem_456",
            content="Another memory",
            metadata={},
            score=0.5,
            created_at=now,
        )

        assert item.updated_at is None

    def test_score_boundary_zero(self) -> None:
        """Test that score=0.0 is valid."""
        item = MemoryItem(
            memory_id="test",
            content="test",
            metadata={},
            score=0.0,
            created_at=datetime.now(),
        )
        assert item.score == 0.0

    def test_score_boundary_one(self) -> None:
        """Test that score=1.0 is valid."""
        item = MemoryItem(
            memory_id="test",
            content="test",
            metadata={},
            score=1.0,
            created_at=datetime.now(),
        )
        assert item.score == 1.0

    def test_score_below_zero_raises_error(self) -> None:
        """Test that score < 0 raises ValueError."""
        with pytest.raises(ValueError, match="score must be between 0.0 and 1.0"):
            MemoryItem(
                memory_id="test",
                content="test",
                metadata={},
                score=-0.1,
                created_at=datetime.now(),
            )

    def test_score_above_one_raises_error(self) -> None:
        """Test that score > 1 raises ValueError."""
        with pytest.raises(ValueError, match="score must be between 0.0 and 1.0"):
            MemoryItem(
                memory_id="test",
                content="test",
                metadata={},
                score=1.1,
                created_at=datetime.now(),
            )

    def test_memory_item_uses_slots(self) -> None:
        """Test that MemoryItem uses __slots__ for memory efficiency."""
        assert hasattr(MemoryItem, "__slots__")


class TestMemoryOperationResult:
    """Tests for MemoryOperationResult dataclass."""

    def test_create_successful_result(self) -> None:
        """Test creating a successful operation result."""
        result = MemoryOperationResult(
            success=True,
            memory_id="mem_789",
        )

        assert result.success is True
        assert result.memory_id == "mem_789"
        assert result.error is None
        assert result.metadata == {}

    def test_create_successful_result_with_metadata(self) -> None:
        """Test creating a successful result with additional metadata."""
        result = MemoryOperationResult(
            success=True,
            memory_id="mem_123",
            metadata={"operation_time_ms": 42, "index_updated": True},
        )

        assert result.success is True
        assert result.metadata == {"operation_time_ms": 42, "index_updated": True}

    def test_create_failed_result_with_error(self) -> None:
        """Test creating a failed operation result with error message."""
        result = MemoryOperationResult(
            success=False,
            error="Memory not found",
        )

        assert result.success is False
        assert result.error == "Memory not found"
        assert result.memory_id is None

    def test_failed_result_without_error_raises(self) -> None:
        """Test that failed results must include an error message."""
        with pytest.raises(ValueError, match="Failed operations must include an error"):
            MemoryOperationResult(success=False)

    def test_failed_result_with_none_error_raises(self) -> None:
        """Test that failed results with None error raise ValueError."""
        with pytest.raises(ValueError, match="Failed operations must include an error"):
            MemoryOperationResult(success=False, error=None)

    def test_operation_result_uses_slots(self) -> None:
        """Test that MemoryOperationResult uses __slots__ for memory efficiency."""
        assert hasattr(MemoryOperationResult, "__slots__")


class TestMemorySystemAdapter:
    """Tests for MemorySystemAdapter ABC."""

    def test_cannot_instantiate_abstract_class(self) -> None:
        """Test that MemorySystemAdapter cannot be directly instantiated."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            MemorySystemAdapter()  # type: ignore[abstract]

    def test_incomplete_implementation_fails(self) -> None:
        """Test that incomplete implementations raise TypeError."""

        class IncompleteAdapter(MemorySystemAdapter):
            def add(self, content: str, metadata: dict | None = None) -> MemoryOperationResult:
                return MemoryOperationResult(success=True, memory_id="test")

            # Missing: search, update, delete, clear, get_stats

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteAdapter()  # type: ignore[abstract]

    def test_complete_implementation_succeeds(self) -> None:
        """Test that a complete implementation can be instantiated."""

        class CompleteAdapter(MemorySystemAdapter):
            def add(self, content: str, metadata: dict | None = None) -> MemoryOperationResult:
                return MemoryOperationResult(success=True, memory_id="test")

            def search(
                self,
                query: str,
                limit: int = 10,
                min_score: float = 0.0,
                metadata_filter: dict | None = None,
            ) -> list[MemoryItem]:
                return []

            def update(
                self, memory_id: str, content: str, metadata: dict | None = None
            ) -> MemoryOperationResult:
                return MemoryOperationResult(success=True, memory_id=memory_id)

            def delete(self, memory_id: str) -> MemoryOperationResult:
                return MemoryOperationResult(success=True, memory_id=memory_id)

            def clear(self) -> MemoryOperationResult:
                return MemoryOperationResult(success=True)

            def get_stats(self) -> dict:
                return {"memory_count": 0, "type": "test"}

        adapter = CompleteAdapter()
        assert adapter is not None

    def test_adapter_interface_methods_exist(self) -> None:
        """Test that all required abstract methods are defined."""
        required_methods = ["add", "search", "update", "delete", "clear", "get_stats"]

        for method in required_methods:
            assert hasattr(MemorySystemAdapter, method)
            assert callable(getattr(MemorySystemAdapter, method))
