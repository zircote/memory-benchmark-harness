"""Mock adapter for test isolation.

This module implements a configurable mock adapter that enables precise control
over memory system behavior during testing. Unlike the NoMemoryAdapter baseline,
this adapter is designed for test isolation - allowing tests to configure exact
return values and verify method calls.

Key features:
- Pre-configure return values for any method
- Record all method calls for verification
- Simulate failures on demand
- Support for callback functions for dynamic responses
"""

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from src.adapters.base import MemoryItem, MemoryOperationResult, MemorySystemAdapter


class MockAdapter(MemorySystemAdapter):
    """Mock adapter for test isolation.

    Allows tests to configure exact return values, record method calls,
    and simulate various scenarios including failures.

    Example:
        ```python
        # Basic usage - preconfigure search results
        adapter = MockAdapter()
        adapter.configure_search([
            MemoryItem(memory_id="1", content="Test", metadata={},
                      score=0.9, created_at=datetime.now(UTC))
        ])
        results = adapter.search("query")  # Returns configured items

        # Verify calls
        assert adapter.get_call_history("search") == [
            {"query": "query", "limit": 10, "min_score": 0.0, "metadata_filter": None}
        ]

        # Simulate failure
        adapter.configure_add_failure("Connection refused")
        result = adapter.add("content")  # Returns failure result
        assert not result.success
        ```
    """

    def __init__(self) -> None:
        """Initialize the mock adapter with empty state."""
        self._memories: dict[str, tuple[str, dict[str, Any], datetime, datetime | None]] = {}
        self._counter: int = 0
        self._call_history: dict[str, list[dict[str, Any]]] = {
            "add": [],
            "search": [],
            "update": [],
            "delete": [],
            "clear": [],
            "get_stats": [],
        }

        # Configurable return values
        self._search_results: list[MemoryItem] | None = None
        self._add_failure: str | None = None
        self._update_failure: str | None = None
        self._delete_failure: str | None = None
        self._clear_failure: str | None = None

        # Callbacks for dynamic behavior
        self._search_callback: (
            Callable[[str, int, float, dict[str, Any] | None], list[MemoryItem]] | None
        ) = None
        self._add_callback: Callable[[str, dict[str, Any] | None], MemoryOperationResult] | None = (
            None
        )

    def configure_search(self, results: list[MemoryItem]) -> None:
        """Configure what search() will return.

        Args:
            results: List of MemoryItems to return on search
        """
        self._search_results = results

    def configure_search_callback(
        self,
        callback: Callable[[str, int, float, dict[str, Any] | None], list[MemoryItem]],
    ) -> None:
        """Configure a callback for dynamic search behavior.

        Args:
            callback: Function that takes (query, limit, min_score, metadata_filter)
                     and returns list of MemoryItems
        """
        self._search_callback = callback

    def configure_add_failure(self, error: str) -> None:
        """Configure add() to fail with given error.

        Args:
            error: Error message to return
        """
        self._add_failure = error

    def configure_add_callback(
        self,
        callback: Callable[[str, dict[str, Any] | None], MemoryOperationResult],
    ) -> None:
        """Configure a callback for dynamic add behavior.

        Args:
            callback: Function that takes (content, metadata)
                     and returns MemoryOperationResult
        """
        self._add_callback = callback

    def configure_update_failure(self, error: str) -> None:
        """Configure update() to fail with given error.

        Args:
            error: Error message to return
        """
        self._update_failure = error

    def configure_delete_failure(self, error: str) -> None:
        """Configure delete() to fail with given error.

        Args:
            error: Error message to return
        """
        self._delete_failure = error

    def configure_clear_failure(self, error: str) -> None:
        """Configure clear() to fail with given error.

        Args:
            error: Error message to return
        """
        self._clear_failure = error

    def reset_configuration(self) -> None:
        """Reset all configured behaviors to defaults."""
        self._search_results = None
        self._add_failure = None
        self._update_failure = None
        self._delete_failure = None
        self._clear_failure = None
        self._search_callback = None
        self._add_callback = None

    def reset_call_history(self) -> None:
        """Clear all recorded method calls."""
        for key in self._call_history:
            self._call_history[key] = []

    def reset(self) -> None:
        """Reset all state including memories, configuration, and history."""
        self._memories.clear()
        self._counter = 0
        self.reset_configuration()
        self.reset_call_history()

    def get_call_history(self, method: str) -> list[dict[str, Any]]:
        """Get recorded calls for a specific method.

        Args:
            method: Method name ("add", "search", "update", "delete", "clear", "get_stats")

        Returns:
            List of call argument dictionaries
        """
        return self._call_history.get(method, [])

    def get_all_call_history(self) -> dict[str, list[dict[str, Any]]]:
        """Get all recorded method calls.

        Returns:
            Dictionary mapping method names to lists of call arguments
        """
        return self._call_history.copy()

    def add(self, content: str, metadata: dict[str, Any] | None = None) -> MemoryOperationResult:
        """Add a new memory entry.

        Records the call and either uses configured callback/failure
        or stores the memory normally.

        Args:
            content: The memory content to store
            metadata: Optional metadata

        Returns:
            MemoryOperationResult with the assigned memory_id or failure
        """
        self._call_history["add"].append(
            {
                "content": content,
                "metadata": metadata,
            }
        )

        # Check for callback first
        if self._add_callback is not None:
            return self._add_callback(content, metadata)

        # Check for configured failure
        if self._add_failure is not None:
            return MemoryOperationResult(
                success=False,
                error=self._add_failure,
            )

        # Normal operation
        self._counter += 1
        memory_id = f"mock_{self._counter}"
        now = datetime.now(UTC)
        self._memories[memory_id] = (content, metadata or {}, now, None)

        return MemoryOperationResult(
            success=True,
            memory_id=memory_id,
            metadata={"stored_at": now.isoformat()},
        )

    def search(
        self,
        query: str,
        limit: int = 10,
        min_score: float = 0.0,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        """Search memories.

        Returns configured results, callback results, or performs
        simple substring matching on stored memories.

        Args:
            query: The search query
            limit: Maximum number of results
            min_score: Minimum relevance score threshold
            metadata_filter: Optional metadata constraints

        Returns:
            List of MemoryItem ordered by relevance
        """
        self._call_history["search"].append(
            {
                "query": query,
                "limit": limit,
                "min_score": min_score,
                "metadata_filter": metadata_filter,
            }
        )

        # Check for callback first
        if self._search_callback is not None:
            return self._search_callback(query, limit, min_score, metadata_filter)

        # Return preconfigured results if set
        if self._search_results is not None:
            return self._search_results[:limit]

        # Default: simple substring matching
        results: list[MemoryItem] = []
        query_lower = query.lower()

        for memory_id, (content, metadata, created_at, updated_at) in self._memories.items():
            if query_lower in content.lower():
                # Simple scoring: length of match / length of content
                score = len(query) / max(len(content), 1)
                score = min(score, 1.0)  # Cap at 1.0

                if score >= min_score:
                    # Apply metadata filter if provided
                    if metadata_filter and not all(
                        metadata.get(k) == v for k, v in metadata_filter.items()
                    ):
                        continue

                    results.append(
                        MemoryItem(
                            memory_id=memory_id,
                            content=content,
                            metadata=metadata,
                            score=score,
                            created_at=created_at,
                            updated_at=updated_at,
                        )
                    )

        # Sort by score descending and limit
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:limit]

    def update(
        self,
        memory_id: str,
        content: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryOperationResult:
        """Update an existing memory entry.

        Args:
            memory_id: The ID of the memory to update
            content: New content (None means keep existing)
            metadata: Optional updated metadata (merges with existing)

        Returns:
            MemoryOperationResult indicating success/failure
        """
        self._call_history["update"].append(
            {
                "memory_id": memory_id,
                "content": content,
                "metadata": metadata,
            }
        )

        # Check for configured failure
        if self._update_failure is not None:
            return MemoryOperationResult(
                success=False,
                error=self._update_failure,
            )

        if memory_id not in self._memories:
            return MemoryOperationResult(
                success=False,
                error=f"Memory not found: {memory_id}",
            )

        old_content, old_metadata, created_at, _ = self._memories[memory_id]
        merged_metadata = {**old_metadata, **(metadata or {})}
        new_content = content if content is not None else old_content
        now = datetime.now(UTC)
        self._memories[memory_id] = (new_content, merged_metadata, created_at, now)

        return MemoryOperationResult(
            success=True,
            memory_id=memory_id,
            metadata={"updated_at": now.isoformat()},
        )

    def delete(self, memory_id: str) -> MemoryOperationResult:
        """Delete a memory entry.

        Args:
            memory_id: The ID of the memory to delete

        Returns:
            MemoryOperationResult indicating success/failure
        """
        self._call_history["delete"].append(
            {
                "memory_id": memory_id,
            }
        )

        # Check for configured failure
        if self._delete_failure is not None:
            return MemoryOperationResult(
                success=False,
                error=self._delete_failure,
            )

        if memory_id not in self._memories:
            return MemoryOperationResult(
                success=False,
                error=f"Memory not found: {memory_id}",
            )

        del self._memories[memory_id]
        return MemoryOperationResult(
            success=True,
            memory_id=memory_id,
        )

    def clear(self) -> MemoryOperationResult:
        """Clear all memories.

        Returns:
            MemoryOperationResult indicating success/failure
        """
        self._call_history["clear"].append({})

        # Check for configured failure
        if self._clear_failure is not None:
            return MemoryOperationResult(
                success=False,
                error=self._clear_failure,
            )

        count = len(self._memories)
        self._memories.clear()
        self._counter = 0

        return MemoryOperationResult(
            success=True,
            metadata={"cleared_count": count},
        )

    def get_stats(self) -> dict[str, Any]:
        """Get memory system statistics.

        Returns:
            Dictionary with memory count and type identifier
        """
        self._call_history["get_stats"].append({})

        total_content_length = sum(len(content) for content, _, _, _ in self._memories.values())

        return {
            "memory_count": len(self._memories),
            "type": "mock",
            "total_content_length": total_content_length,
            "avg_content_length": (
                total_content_length / len(self._memories) if self._memories else 0
            ),
        }

    def seed_memories(self, memories: list[tuple[str, dict[str, Any] | None]]) -> list[str]:
        """Seed the adapter with initial memories for testing.

        This is a convenience method for tests that need pre-populated state.

        Args:
            memories: List of (content, metadata) tuples

        Returns:
            List of assigned memory IDs
        """
        memory_ids = []
        for content, metadata in memories:
            result = self.add(content, metadata)
            if result.success and result.memory_id:
                memory_ids.append(result.memory_id)
        return memory_ids
