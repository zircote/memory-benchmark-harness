"""Ablation study adapters.

This module provides adapter variants that ablate specific components
of the memory system to understand their contribution to performance.

Ablation conditions:
1. NoSemanticSearchAdapter: Random retrieval instead of semantic
2. NoMetadataFilterAdapter: Ignores metadata in retrieval
3. NoVersionHistoryAdapter: Removes git version history support
4. FixedWindowAdapter: Fixed-size context window (no smart retrieval)
5. RecencyOnlyAdapter: Retrieves only most recent memories
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from src.adapters.base import (
    MemoryItem,
    MemoryOperationResult,
    MemorySystemAdapter,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class NoSemanticSearchAdapter(MemorySystemAdapter):
    """Adapter that returns random memories instead of semantic search.

    This ablates the semantic search component, providing a baseline
    to measure the value of semantic retrieval.

    Attributes:
        base_adapter: The underlying adapter to wrap
        seed: Random seed for reproducibility
    """

    base_adapter: MemorySystemAdapter
    seed: int | None = None
    _rng: random.Random = field(default_factory=random.Random, init=False)
    _memories: list[MemoryItem] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        """Initialize random generator."""
        if self.seed is not None:
            self._rng.seed(self.seed)

    def add(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryOperationResult:
        """Add content to memory."""
        result = self.base_adapter.add(content, metadata)

        if result.success and result.memory_id:
            # Track memory for random retrieval
            self._memories.append(
                MemoryItem(
                    memory_id=result.memory_id,
                    content=content,
                    score=1.0,
                    metadata=metadata or {},
                    created_at=datetime.now(),
                )
            )

        return result

    def search(
        self,
        query: str,
        limit: int = 10,
        min_score: float = 0.0,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        """Return random memories instead of semantic search."""
        if not self._memories:
            return []

        # Random selection
        k = min(limit, len(self._memories))
        selected = self._rng.sample(self._memories, k)

        # Assign random scores
        return [
            MemoryItem(
                memory_id=m.memory_id,
                content=m.content,
                score=self._rng.random(),  # Random score
                metadata=m.metadata,
                created_at=m.created_at,
            )
            for m in selected
        ]

    def update(
        self,
        memory_id: str,
        content: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryOperationResult:
        """Update a memory."""
        result = self.base_adapter.update(memory_id, content, metadata)

        if result.success:
            # Update tracked memory
            for i, m in enumerate(self._memories):
                if m.memory_id == memory_id:
                    self._memories[i] = MemoryItem(
                        memory_id=memory_id,
                        content=content or m.content,
                        score=m.score,
                        metadata=metadata or m.metadata,
                        created_at=m.created_at,
                        updated_at=datetime.now(),
                    )
                    break

        return result

    def delete(self, memory_id: str) -> MemoryOperationResult:
        """Delete a memory."""
        result = self.base_adapter.delete(memory_id)

        if result.success:
            self._memories = [m for m in self._memories if m.memory_id != memory_id]

        return result

    def clear(self) -> MemoryOperationResult:
        """Clear all memories."""
        self._memories.clear()
        return self.base_adapter.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get adapter statistics."""
        base_stats = self.base_adapter.get_stats()
        return {
            **base_stats,
            "ablation": "no_semantic_search",
            "tracked_memories": len(self._memories),
        }


@dataclass(slots=True)
class NoMetadataFilterAdapter(MemorySystemAdapter):
    """Adapter that ignores metadata filters in retrieval.

    This ablates the metadata filtering component.

    Attributes:
        base_adapter: The underlying adapter to wrap
    """

    base_adapter: MemorySystemAdapter

    def add(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryOperationResult:
        """Add content to memory."""
        return self.base_adapter.add(content, metadata)

    def search(
        self,
        query: str,
        limit: int = 10,
        min_score: float = 0.0,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        """Search without metadata filtering."""
        # Ignore metadata_filter
        return self.base_adapter.search(
            query=query,
            limit=limit,
            min_score=min_score,
            metadata_filter=None,  # Ablated
        )

    def update(
        self,
        memory_id: str,
        content: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryOperationResult:
        """Update a memory."""
        return self.base_adapter.update(memory_id, content, metadata)

    def delete(self, memory_id: str) -> MemoryOperationResult:
        """Delete a memory."""
        return self.base_adapter.delete(memory_id)

    def clear(self) -> MemoryOperationResult:
        """Clear all memories."""
        return self.base_adapter.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get adapter statistics."""
        base_stats = self.base_adapter.get_stats()
        return {
            **base_stats,
            "ablation": "no_metadata_filter",
        }


@dataclass(slots=True)
class NoVersionHistoryAdapter(MemorySystemAdapter):
    """Adapter that removes version history support.

    This ablates the git version history component, which is
    particularly important for conflict resolution tasks.

    Attributes:
        base_adapter: The underlying adapter to wrap
    """

    base_adapter: MemorySystemAdapter

    def add(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryOperationResult:
        """Add content to memory."""
        return self.base_adapter.add(content, metadata)

    def search(
        self,
        query: str,
        limit: int = 10,
        min_score: float = 0.0,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        """Search and strip version history from results."""
        results = self.base_adapter.search(
            query=query,
            limit=limit,
            min_score=min_score,
            metadata_filter=metadata_filter,
        )

        # Strip version history from metadata
        stripped_results: list[MemoryItem] = []
        for mem in results:
            metadata = dict(mem.metadata)
            metadata.pop("version_history", None)
            metadata.pop("versions", None)
            metadata.pop("git_history", None)

            stripped_results.append(
                MemoryItem(
                    memory_id=mem.memory_id,
                    content=mem.content,
                    score=mem.score,
                    metadata=metadata,
                    created_at=mem.created_at,
                    updated_at=mem.updated_at,
                )
            )

        return stripped_results

    def update(
        self,
        memory_id: str,
        content: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryOperationResult:
        """Update without preserving history."""
        return self.base_adapter.update(memory_id, content, metadata)

    def delete(self, memory_id: str) -> MemoryOperationResult:
        """Delete a memory."""
        return self.base_adapter.delete(memory_id)

    def clear(self) -> MemoryOperationResult:
        """Clear all memories."""
        return self.base_adapter.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get adapter statistics."""
        base_stats = self.base_adapter.get_stats()
        return {
            **base_stats,
            "ablation": "no_version_history",
        }

    # Override get_history to return nothing
    def get_history(self, memory_id: str) -> list[dict[str, Any]]:
        """Return empty history (ablated)."""
        return []


@dataclass(slots=True)
class FixedWindowAdapter(MemorySystemAdapter):
    """Adapter that uses fixed-size context window instead of smart retrieval.

    This ablates the intelligent retrieval component, always returning
    the same fixed window of content.

    Attributes:
        base_adapter: The underlying adapter to wrap
        window_size: Number of memories to return
    """

    base_adapter: MemorySystemAdapter
    window_size: int = 10
    _memories: list[MemoryItem] = field(default_factory=list, init=False)

    def add(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryOperationResult:
        """Add content to memory."""
        result = self.base_adapter.add(content, metadata)

        if result.success and result.memory_id:
            self._memories.append(
                MemoryItem(
                    memory_id=result.memory_id,
                    content=content,
                    score=1.0,
                    metadata=metadata or {},
                    created_at=datetime.now(),
                )
            )

        return result

    def search(
        self,
        query: str,
        limit: int = 10,
        min_score: float = 0.0,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        """Return fixed window regardless of query."""
        # Return first N memories (fixed window)
        window = self._memories[: self.window_size]

        return [
            MemoryItem(
                memory_id=m.memory_id,
                content=m.content,
                score=1.0,  # Fixed score
                metadata=m.metadata,
                created_at=m.created_at,
            )
            for m in window
        ]

    def update(
        self,
        memory_id: str,
        content: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryOperationResult:
        """Update a memory."""
        result = self.base_adapter.update(memory_id, content, metadata)

        if result.success:
            for i, m in enumerate(self._memories):
                if m.memory_id == memory_id:
                    self._memories[i] = MemoryItem(
                        memory_id=memory_id,
                        content=content or m.content,
                        score=m.score,
                        metadata=metadata or m.metadata,
                        created_at=m.created_at,
                        updated_at=datetime.now(),
                    )
                    break

        return result

    def delete(self, memory_id: str) -> MemoryOperationResult:
        """Delete a memory."""
        result = self.base_adapter.delete(memory_id)

        if result.success:
            self._memories = [m for m in self._memories if m.memory_id != memory_id]

        return result

    def clear(self) -> MemoryOperationResult:
        """Clear all memories."""
        self._memories.clear()
        return self.base_adapter.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get adapter statistics."""
        base_stats = self.base_adapter.get_stats()
        return {
            **base_stats,
            "ablation": "fixed_window",
            "window_size": self.window_size,
            "tracked_memories": len(self._memories),
        }


@dataclass(slots=True)
class RecencyOnlyAdapter(MemorySystemAdapter):
    """Adapter that retrieves only the most recent memories.

    This ablates relevance-based retrieval, using only recency.

    Attributes:
        base_adapter: The underlying adapter to wrap
    """

    base_adapter: MemorySystemAdapter
    _memories: list[MemoryItem] = field(default_factory=list, init=False)

    def add(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryOperationResult:
        """Add content to memory."""
        result = self.base_adapter.add(content, metadata)

        if result.success and result.memory_id:
            self._memories.append(
                MemoryItem(
                    memory_id=result.memory_id,
                    content=content,
                    score=1.0,
                    metadata=metadata or {},
                    created_at=datetime.now(),
                )
            )

        return result

    def search(
        self,
        query: str,
        limit: int = 10,
        min_score: float = 0.0,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        """Return most recent memories regardless of query."""
        # Return most recent N memories
        recent = list(reversed(self._memories[-limit:]))

        return [
            MemoryItem(
                memory_id=m.memory_id,
                content=m.content,
                score=1.0 - (i * 0.01),  # Slightly decreasing score by recency
                metadata=m.metadata,
                created_at=m.created_at,
            )
            for i, m in enumerate(recent)
        ]

    def update(
        self,
        memory_id: str,
        content: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryOperationResult:
        """Update a memory."""
        result = self.base_adapter.update(memory_id, content, metadata)

        if result.success:
            for i, m in enumerate(self._memories):
                if m.memory_id == memory_id:
                    # Move to end (most recent)
                    updated = MemoryItem(
                        memory_id=memory_id,
                        content=content or m.content,
                        score=m.score,
                        metadata=metadata or m.metadata,
                        created_at=m.created_at,
                        updated_at=datetime.now(),
                    )
                    self._memories.pop(i)
                    self._memories.append(updated)
                    break

        return result

    def delete(self, memory_id: str) -> MemoryOperationResult:
        """Delete a memory."""
        result = self.base_adapter.delete(memory_id)

        if result.success:
            self._memories = [m for m in self._memories if m.memory_id != memory_id]

        return result

    def clear(self) -> MemoryOperationResult:
        """Clear all memories."""
        self._memories.clear()
        return self.base_adapter.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get adapter statistics."""
        base_stats = self.base_adapter.get_stats()
        return {
            **base_stats,
            "ablation": "recency_only",
            "tracked_memories": len(self._memories),
        }


class AblationType:
    """Constants for ablation types."""

    NO_SEMANTIC_SEARCH = "no_semantic_search"
    NO_METADATA_FILTER = "no_metadata_filter"
    NO_VERSION_HISTORY = "no_version_history"
    FIXED_WINDOW = "fixed_window"
    RECENCY_ONLY = "recency_only"


def create_ablation_adapter(
    base_adapter: MemorySystemAdapter,
    ablation_type: str,
    **kwargs: Any,
) -> MemorySystemAdapter:
    """Factory function to create ablation adapters.

    Args:
        base_adapter: The adapter to wrap
        ablation_type: Type of ablation to apply
        **kwargs: Additional arguments for specific ablation types

    Returns:
        Wrapped adapter with ablation applied

    Raises:
        ValueError: If ablation_type is unknown
    """
    ablation_map = {
        AblationType.NO_SEMANTIC_SEARCH: NoSemanticSearchAdapter,
        AblationType.NO_METADATA_FILTER: NoMetadataFilterAdapter,
        AblationType.NO_VERSION_HISTORY: NoVersionHistoryAdapter,
        AblationType.FIXED_WINDOW: FixedWindowAdapter,
        AblationType.RECENCY_ONLY: RecencyOnlyAdapter,
    }

    if ablation_type not in ablation_map:
        raise ValueError(
            f"Unknown ablation type: {ablation_type}. Valid types: {list(ablation_map.keys())}"
        )

    adapter_class = ablation_map[ablation_type]
    return adapter_class(base_adapter=base_adapter, **kwargs)
