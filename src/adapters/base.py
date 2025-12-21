"""Base adapter interface for memory systems.

This module defines the abstract base class and data structures for all memory
system adapters used in benchmarking. All benchmark-specific wrappers delegate
to this interface, enabling consistent testing and baseline comparison.

See ADR-005 for design rationale on the unified memory interface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class MemoryItem:
    """Single memory item returned from search.

    Attributes:
        memory_id: Unique identifier for this memory
        content: The stored memory content
        metadata: Associated metadata (tags, session_id, etc.)
        score: Relevance score from search (0.0 - 1.0)
        created_at: When the memory was created
        updated_at: When the memory was last updated (if applicable)
    """

    memory_id: str
    content: str
    metadata: dict[str, Any]
    score: float
    created_at: datetime
    updated_at: datetime | None = None

    def __post_init__(self) -> None:
        """Validate score is within valid range."""
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"score must be between 0.0 and 1.0, got {self.score}")


@dataclass(slots=True)
class MemoryOperationResult:
    """Result of a memory operation (add, update, delete, clear).

    Attributes:
        success: Whether the operation succeeded
        memory_id: The memory ID (for add/update operations)
        error: Error message if operation failed
        metadata: Additional operation metadata (timing, etc.)
    """

    success: bool
    memory_id: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate result consistency."""
        if not self.success and self.error is None:
            raise ValueError("Failed operations must include an error message")


class MemorySystemAdapter(ABC):
    """Abstract base class for memory system adapters.

    All benchmark-specific wrappers delegate to this interface,
    enabling consistent testing and baseline comparison.

    Implementations:
        - GitNotesAdapter: Wraps git-notes-memory-manager plugin
        - NoMemoryAdapter: Baseline that stores but never retrieves
        - MockAdapter: For test isolation

    Example:
        ```python
        adapter = GitNotesAdapter(repo_path="/path/to/repo")
        result = adapter.add("Important context", {"session_id": "123"})
        if result.success:
            memories = adapter.search("context", limit=5)
            for mem in memories:
                print(f"{mem.memory_id}: {mem.content} (score: {mem.score})")
        ```
    """

    @abstractmethod
    def add(self, content: str, metadata: dict[str, Any] | None = None) -> MemoryOperationResult:
        """Add a new memory entry.

        Args:
            content: The memory content to store
            metadata: Optional metadata (timestamps, session_id, tags)

        Returns:
            MemoryOperationResult with the assigned memory_id
        """
        ...

    @abstractmethod
    def search(
        self,
        query: str,
        limit: int = 10,
        min_score: float = 0.0,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        """Search memories by semantic similarity.

        Args:
            query: The search query
            limit: Maximum number of results
            min_score: Minimum relevance score threshold
            metadata_filter: Optional metadata constraints

        Returns:
            List of MemoryItem ordered by relevance (highest first)
        """
        ...

    @abstractmethod
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
        ...

    @abstractmethod
    def delete(self, memory_id: str) -> MemoryOperationResult:
        """Delete a memory entry.

        Args:
            memory_id: The ID of the memory to delete

        Returns:
            MemoryOperationResult indicating success/failure
        """
        ...

    @abstractmethod
    def clear(self) -> MemoryOperationResult:
        """Clear all memories (for test isolation).

        This method is primarily used between benchmark runs to ensure
        clean state. Implementations should handle this efficiently.

        Returns:
            MemoryOperationResult indicating success/failure
        """
        ...

    @abstractmethod
    def get_stats(self) -> dict[str, Any]:
        """Get memory system statistics.

        Returns:
            Dictionary with at least:
                - memory_count: Number of stored memories
                - type: Adapter type identifier

            Implementations may include additional metrics like:
                - storage_bytes: Approximate storage used
                - avg_content_length: Average content length
                - index_status: Embedding index status
        """
        ...
