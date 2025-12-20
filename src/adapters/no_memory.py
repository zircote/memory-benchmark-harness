"""No-memory baseline adapter.

This module implements a no-retrieval baseline that stores memories but
never returns them on search. This is essential for measuring the actual
benefit of memory retrieval - without it, we can't distinguish whether
improvements come from storing information or from retrieving it.

See ADR-012 for the rationale behind the two-way comparison design.
"""

from datetime import UTC, datetime
from typing import Any

from src.adapters.base import MemoryItem, MemoryOperationResult, MemorySystemAdapter

# Unused args in search() are intentional - baseline ignores all search params
# ruff: noqa: ARG002


class NoMemoryAdapter(MemorySystemAdapter):
    """No-memory baseline adapter.

    Stores memories but search always returns empty. Used to measure
    the benefit of memory retrieval vs. just having context.

    This is the scientific control condition - it allows us to attribute
    performance differences specifically to memory retrieval capabilities
    rather than other factors.

    Example:
        ```python
        adapter = NoMemoryAdapter()
        result = adapter.add("Some important context")
        assert result.success

        # Key behavior: search never returns anything
        memories = adapter.search("context")
        assert memories == []  # Always empty!

        # But we still track what was stored for statistics
        stats = adapter.get_stats()
        assert stats["memory_count"] == 1
        ```
    """

    def __init__(self) -> None:
        """Initialize the no-memory baseline adapter."""
        self._memories: list[tuple[str, str, dict[str, Any], datetime]] = []
        self._counter: int = 0

    def add(self, content: str, metadata: dict[str, Any] | None = None) -> MemoryOperationResult:
        """Store a memory (but it will never be retrieved).

        Args:
            content: The memory content to store
            metadata: Optional metadata (ignored in searches)

        Returns:
            MemoryOperationResult with the assigned memory_id
        """
        self._counter += 1
        memory_id = f"baseline_{self._counter}"
        now = datetime.now(UTC)
        self._memories.append((memory_id, content, metadata or {}, now))
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
        """Search memories - always returns empty list.

        This is the key differentiator from real memory systems.
        By never returning memories, we establish a baseline for
        "what if the agent had no memory retrieval capability?"

        Args:
            query: The search query (ignored)
            limit: Maximum number of results (ignored)
            min_score: Minimum relevance score threshold (ignored)
            metadata_filter: Optional metadata constraints (ignored)

        Returns:
            Empty list - always. This is intentional.
        """
        # Intentionally return nothing - this is the baseline behavior
        return []

    def update(
        self, memory_id: str, content: str, metadata: dict[str, Any] | None = None
    ) -> MemoryOperationResult:
        """Update an existing memory entry.

        Args:
            memory_id: The ID of the memory to update
            content: New content
            metadata: Optional updated metadata

        Returns:
            MemoryOperationResult indicating success/failure
        """
        for i, (mid, _, old_metadata, created_at) in enumerate(self._memories):
            if mid == memory_id:
                merged_metadata = {**old_metadata, **(metadata or {})}
                self._memories[i] = (memory_id, content, merged_metadata, created_at)
                return MemoryOperationResult(
                    success=True,
                    memory_id=memory_id,
                    metadata={"updated_at": datetime.now(UTC).isoformat()},
                )

        return MemoryOperationResult(
            success=False,
            error=f"Memory not found: {memory_id}",
        )

    def delete(self, memory_id: str) -> MemoryOperationResult:
        """Delete a memory entry.

        Args:
            memory_id: The ID of the memory to delete

        Returns:
            MemoryOperationResult indicating success/failure
        """
        for i, (mid, _, _, _) in enumerate(self._memories):
            if mid == memory_id:
                self._memories.pop(i)
                return MemoryOperationResult(
                    success=True,
                    memory_id=memory_id,
                )

        return MemoryOperationResult(
            success=False,
            error=f"Memory not found: {memory_id}",
        )

    def clear(self) -> MemoryOperationResult:
        """Clear all memories.

        Returns:
            MemoryOperationResult indicating success
        """
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
        total_content_length = sum(len(content) for _, content, _, _ in self._memories)
        return {
            "memory_count": len(self._memories),
            "type": "no-memory-baseline",
            "total_content_length": total_content_length,
            "avg_content_length": (
                total_content_length / len(self._memories) if self._memories else 0
            ),
        }
