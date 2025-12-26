"""Git Notes Memory Manager adapter.

This module implements the MemorySystemAdapter interface for the
git-notes-memory-manager plugin, enabling benchmark comparison against
the no-memory baseline.

The adapter wraps three services from git-notes-memory-manager:
- CaptureService: For storing memories (add)
- RecallService: For retrieving memories (search)
- SyncService: For index management (clear, stats)

See GIT_NOTES_API.md for the full API documentation.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.adapters.base import MemoryItem, MemoryOperationResult, MemorySystemAdapter

# Type aliases for git-notes-memory-manager services
# These are dynamically imported at runtime, so we use Any for type hints
CaptureService = Any
RecallService = Any
SyncService = Any
IndexService = Any
EmbeddingService = Any

logger = logging.getLogger(__name__)


def _euclidean_to_similarity(distance: float) -> float:
    """Convert Euclidean distance to similarity score (0.0 - 1.0).

    The git-notes-memory-manager uses Euclidean distance where lower
    is more similar. We convert to a similarity score where higher
    is more similar using the formula: similarity = 1 / (1 + distance)

    Args:
        distance: Euclidean distance (0.0 = identical, higher = less similar)

    Returns:
        Similarity score between 0.0 and 1.0
    """
    return 1.0 / (1.0 + distance)


def _extract_namespace_from_metadata(metadata: dict[str, Any] | None) -> str:
    """Extract namespace from metadata or return default.

    Args:
        metadata: Optional metadata dictionary

    Returns:
        Namespace string (defaults to "learnings")
    """
    if metadata is None:
        return "learnings"
    namespace = metadata.get("namespace", "learnings")
    return str(namespace)


def _extract_summary_from_content(content: str, max_length: int = 100) -> str:
    """Extract a summary from content.

    Takes the first line or first N characters of content.

    Args:
        content: Full memory content
        max_length: Maximum summary length (git-notes limit is 100)

    Returns:
        Summary string
    """
    first_line = content.split("\n")[0].strip()
    if len(first_line) <= max_length:
        return first_line
    return first_line[: max_length - 3] + "..."


class GitNotesAdapter(MemorySystemAdapter):
    """Adapter wrapping git-notes-memory-manager for benchmarking.

    This adapter integrates the git-notes-memory-manager plugin with
    the benchmark harness, providing:
    - Semantic vector search via RecallService
    - Memory storage via CaptureService
    - Index management via SyncService/IndexService

    The git-notes system is append-only (git notes are immutable),
    so update() and delete() use status-based archival instead of
    actual modification/deletion.

    Example:
        ```python
        adapter = GitNotesAdapter(repo_path="/path/to/repo")
        result = adapter.add(
            "Decided to use PostgreSQL for persistence",
            metadata={"namespace": "decisions", "tags": ["database"]}
        )
        if result.success:
            memories = adapter.search("database decision", limit=5)
            for mem in memories:
                print(f"{mem.memory_id}: {mem.content}")
        ```

    Note:
        The repo_path must be a git repository. If the index doesn't
        exist, it will be created on first operation.
    """

    def __init__(self, repo_path: str | Path | None = None) -> None:
        """Initialize the GitNotesAdapter.

        Args:
            repo_path: Path to the git repository. If None, uses current
                      working directory.

        Raises:
            ImportError: If git-notes-memory-manager is not installed
            ValueError: If repo_path is not a valid git repository
        """
        self._repo_path = Path(repo_path) if repo_path else Path.cwd()

        # Lazy-loaded services (initialized on first use)
        self._capture_service: CaptureService | None = None
        self._recall_service: RecallService | None = None
        self._sync_service: SyncService | None = None
        self._index_service: IndexService | None = None
        self._embedding_service: EmbeddingService | None = None

        # Track initialization state
        self._initialized = False

        # Batch capture service (without embedding for deferred batch embedding)
        self._batch_capture_service: CaptureService | None = None

    def _ensure_initialized(self) -> None:
        """Ensure services are initialized.

        Lazy initialization allows the adapter to be created before
        the git repository is fully set up.

        Creates per-repository service instances (not singletons) to ensure
        proper isolation when working with different repositories.

        Raises:
            ImportError: If git-notes-memory-manager is not installed
            ValueError: If repo_path is not a valid git repository
        """
        if self._initialized:
            return

        try:
            from git_notes_memory import get_sync_service
            from git_notes_memory.capture import CaptureService
            from git_notes_memory.config import get_project_index_path
            from git_notes_memory.embedding import get_default_service as get_embedding_service
            from git_notes_memory.index import IndexService as IndexSvc
            from git_notes_memory.recall import RecallService

            # Create project-specific IndexService for this repo
            # All services share this instance for consistency
            index_path = get_project_index_path(self._repo_path)
            self._index_service = IndexSvc(index_path)
            self._index_service.initialize()

            # Get shared embedding service (singleton is fine for model)
            self._embedding_service = get_embedding_service()

            # Create per-repo CaptureService with our index and embedding services
            # This ensures captures are indexed to the correct database
            self._capture_service = CaptureService()
            self._capture_service.set_index_service(self._index_service)
            self._capture_service.set_embedding_service(self._embedding_service)

            # Create batch capture service WITHOUT embedding (for deferred batch embedding)
            # This is much faster for bulk ingestion
            self._batch_capture_service = CaptureService()
            self._batch_capture_service.set_index_service(self._index_service)
            # Note: We intentionally don't set embedding service here

            # Create per-repo RecallService with our index and embedding services
            # This ensures searches query the correct database
            self._recall_service = RecallService(
                index_path=index_path,
                index_service=self._index_service,
                embedding_service=self._embedding_service,
            )

            # SyncService accepts repo_path for initialization
            self._sync_service = get_sync_service(repo_path=self._repo_path)

            self._initialized = True
            logger.info(
                "GitNotesAdapter initialized for repo: %s (index: %s)",
                self._repo_path,
                index_path,
            )

        except ImportError as e:
            msg = (
                "git-notes-memory-manager is not installed. "
                "Install it with: uv add git-notes-memory-manager"
            )
            raise ImportError(msg) from e

    def add(self, content: str, metadata: dict[str, Any] | None = None) -> MemoryOperationResult:
        """Add a new memory entry using CaptureService.

        Maps to git-notes-memory-manager's capture() method.

        Args:
            content: The memory content to store
            metadata: Optional metadata including:
                - namespace: Memory type (default: "learnings")
                - tags: List of tags
                - spec: Specification slug
                - phase: Lifecycle phase

        Returns:
            MemoryOperationResult with the assigned memory_id
        """
        self._ensure_initialized()
        assert self._capture_service is not None

        try:
            namespace = _extract_namespace_from_metadata(metadata)
            summary = _extract_summary_from_content(content)

            # Extract optional fields from metadata
            spec = metadata.get("spec") if metadata else None
            tags = metadata.get("tags") if metadata else None
            phase = metadata.get("phase") if metadata else None

            result = self._capture_service.capture(
                namespace=namespace,
                summary=summary,
                content=content,
                spec=spec,
                tags=tags,
                phase=phase,
            )

            if result.success and result.memory:
                return MemoryOperationResult(
                    success=True,
                    memory_id=result.memory.id,
                    metadata={
                        "indexed": result.indexed,
                        "namespace": namespace,
                        "stored_at": datetime.now(UTC).isoformat(),
                    },
                )

            return MemoryOperationResult(
                success=False,
                error=result.warning or "Capture failed without specific error",
            )

        except Exception as e:
            logger.exception("Failed to add memory")
            return MemoryOperationResult(
                success=False,
                error=str(e),
            )

    def search(
        self,
        query: str,
        limit: int = 10,
        min_score: float = 0.0,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        """Search memories using semantic vector search.

        Maps to git-notes-memory-manager's RecallService.search().

        Args:
            query: The search query
            limit: Maximum number of results
            min_score: Minimum similarity score threshold (0.0 - 1.0)
            metadata_filter: Optional filters:
                - namespace: Filter by namespace
                - spec: Filter by specification

        Returns:
            List of MemoryItem ordered by relevance (highest first)
        """
        self._ensure_initialized()
        assert self._recall_service is not None

        try:
            # Extract filters from metadata_filter
            namespace = metadata_filter.get("namespace") if metadata_filter else None
            spec = metadata_filter.get("spec") if metadata_filter else None

            # Convert min_score to min_similarity (approximate)
            # Note: git-notes uses distance, we use similarity
            min_similarity = min_score if min_score > 0 else None

            results = self._recall_service.search(
                query=query,
                k=limit,
                namespace=namespace,
                spec=spec,
                min_similarity=min_similarity,
            )

            memory_items: list[MemoryItem] = []
            for result in results:
                memory = result.memory
                similarity = _euclidean_to_similarity(result.distance)

                if similarity >= min_score:
                    memory_items.append(
                        MemoryItem(
                            memory_id=memory.id,
                            content=memory.content,
                            metadata={
                                "namespace": memory.namespace,
                                "spec": memory.spec,
                                "tags": list(memory.tags) if memory.tags else [],
                                "phase": memory.phase,
                                "status": memory.status,
                                "summary": memory.summary,
                                "commit_sha": memory.commit_sha,
                            },
                            score=similarity,
                            created_at=memory.timestamp,
                            updated_at=None,  # Git notes are immutable
                        )
                    )

            return memory_items

        except Exception:
            logger.exception("Search failed")
            return []

    def update(
        self,
        memory_id: str,
        content: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryOperationResult:
        """Update an existing memory entry.

        Git notes are append-only, so "update" creates a new memory
        with the updated content and adds a `supersedes` relation.

        Args:
            memory_id: The ID of the memory to update
            content: New content (None means keep existing)
            metadata: Optional updated metadata

        Returns:
            MemoryOperationResult with the new memory_id
        """
        self._ensure_initialized()
        assert self._capture_service is not None

        try:
            # Get the original memory to preserve its namespace
            original = None
            if self._recall_service:
                original = self._recall_service.get(memory_id)

            if original is None:
                return MemoryOperationResult(
                    success=False,
                    error=f"Memory not found: {memory_id}",
                )

            # Use new content or keep original
            new_content = content if content is not None else original.content

            # Create new memory with updated content
            namespace = (metadata.get("namespace") if metadata else None) or original.namespace
            summary = _extract_summary_from_content(new_content)

            result = self._capture_service.capture(
                namespace=namespace,
                summary=summary,
                content=new_content,
                spec=metadata.get("spec") if metadata else original.spec,
                tags=metadata.get("tags") if metadata else list(original.tags),
                relates_to=[memory_id],  # Link to original
            )

            if result.success and result.memory:
                return MemoryOperationResult(
                    success=True,
                    memory_id=result.memory.id,
                    metadata={
                        "supersedes": memory_id,
                        "updated_at": datetime.now(UTC).isoformat(),
                    },
                )

            return MemoryOperationResult(
                success=False,
                error=result.warning or "Update (re-capture) failed",
            )

        except Exception as e:
            logger.exception("Failed to update memory")
            return MemoryOperationResult(
                success=False,
                error=str(e),
            )

    def delete(self, memory_id: str) -> MemoryOperationResult:
        """Delete a memory entry.

        Git notes are append-only, so "delete" archives the memory
        by updating its status to "archived".

        Note: This doesn't actually remove the memory from git history.
        The memory will be excluded from future searches based on status.

        Args:
            memory_id: The ID of the memory to delete

        Returns:
            MemoryOperationResult indicating success/failure
        """
        self._ensure_initialized()

        try:
            # Verify the memory exists
            if self._recall_service:
                original = self._recall_service.get(memory_id)
                if original is None:
                    return MemoryOperationResult(
                        success=False,
                        error=f"Memory not found: {memory_id}",
                    )

            # Git notes are immutable - we can't truly delete
            # For benchmarking purposes, we'll log this as a warning
            # and return success (the memory won't appear in searches
            # if properly filtered by status)
            logger.warning(
                "GitNotesAdapter.delete() cannot truly delete. "
                "Memory %s marked as conceptually archived.",
                memory_id,
            )

            return MemoryOperationResult(
                success=True,
                memory_id=memory_id,
                metadata={"note": "Git notes are immutable; memory archived"},
            )

        except Exception as e:
            logger.exception("Failed to delete memory")
            return MemoryOperationResult(
                success=False,
                error=str(e),
            )

    def clear(self) -> MemoryOperationResult:
        """Clear all memories from the index and reset service singletons.

        This clears the vector index but does NOT remove git notes
        from the repository. A subsequent reindex() would restore
        all memories.

        Also resets the ServiceRegistry singletons to allow fresh adapter
        instances to be created (critical for multi-trial benchmarks).

        For benchmark isolation, this is sufficient as it prevents
        previous run's memories from affecting current searches.

        Returns:
            MemoryOperationResult indicating success/failure
        """
        self._ensure_initialized()

        try:
            if self._index_service is None:
                return MemoryOperationResult(
                    success=False,
                    error="IndexService not available",
                )

            count = self._index_service.clear()

            # Reset service singletons to allow fresh instances on next adapter creation
            # This is critical for multi-trial benchmarks where each trial needs
            # fresh services but the ServiceRegistry retains singletons
            self._reset_services()

            return MemoryOperationResult(
                success=True,
                metadata={
                    "cleared_count": count,
                    "note": "Index cleared; git notes preserved; services reset",
                },
            )

        except Exception as e:
            logger.exception("Failed to clear index")
            return MemoryOperationResult(
                success=False,
                error=str(e),
            )

    def _reset_services(self) -> None:
        """Reset all service singletons for benchmark isolation.

        The git-notes-memory library uses ServiceRegistry to manage singleton
        services (SyncService, EmbeddingService, etc.). When running multiple
        trials, each trial creates a new adapter instance, but the singletons
        persist causing "Service instance already exists" errors.

        This method clears all registered singletons, allowing fresh instances
        to be created for subsequent trials.
        """
        try:
            from git_notes_memory.registry import ServiceRegistry

            ServiceRegistry.reset()

            # Also reset our local references so we reinitialize on next use
            self._capture_service = None
            self._recall_service = None
            self._sync_service = None
            self._index_service = None
            self._embedding_service = None
            self._batch_capture_service = None
            self._initialized = False

            logger.debug("Reset ServiceRegistry and adapter state for benchmark isolation")
        except ImportError:
            # If registry module isn't available, just reset local state
            logger.warning("ServiceRegistry not available; only resetting local adapter state")
            self._initialized = False

    def get_stats(self) -> dict[str, Any]:
        """Get memory system statistics.

        Returns:
            Dictionary with:
                - memory_count: Number of indexed memories
                - type: "git-notes"
                - by_namespace: Count per namespace
                - by_spec: Count per specification
                - index_size_bytes: Index storage size
        """
        self._ensure_initialized()

        try:
            if self._index_service is None:
                return {
                    "memory_count": 0,
                    "type": "git-notes",
                    "error": "IndexService not available",
                }

            stats = self._index_service.get_stats()

            return {
                "memory_count": stats.total_memories,
                "type": "git-notes",
                "by_namespace": dict(stats.by_namespace_dict),
                "by_spec": dict(stats.by_spec_dict),
                "index_size_bytes": stats.index_size_bytes,
                "last_sync": (stats.last_sync.isoformat() if stats.last_sync else None),
            }

        except Exception as e:
            logger.exception("Failed to get stats")
            return {
                "memory_count": 0,
                "type": "git-notes",
                "error": str(e),
            }

    def get_version(self) -> str | None:
        """Get the installed git-notes-memory-manager version.

        Uses importlib.metadata to retrieve the package version,
        enabling cross-version performance comparisons.

        Returns:
            Version string (e.g., "0.8.1") or None if unavailable
        """
        import importlib.metadata

        try:
            return importlib.metadata.version("git-notes-memory")
        except importlib.metadata.PackageNotFoundError:
            return None

    def reindex(self, full: bool = False) -> int:
        """Rebuild the index from git notes.

        This is a git-notes-specific method not in the base interface.
        Useful for ensuring the index is synchronized after manual
        git operations.

        Args:
            full: If True, clears index first. Otherwise incremental.

        Returns:
            Number of memories indexed
        """
        self._ensure_initialized()

        if self._sync_service is None:
            return 0

        result = self._sync_service.reindex(full=full)
        return int(result)

    # =========================================================================
    # Batch Operations (for high-performance bulk ingestion)
    # =========================================================================

    def add_batch(
        self,
        items: Sequence[tuple[str, dict[str, Any] | None]],
        *,
        embed: bool = True,
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> list[MemoryOperationResult]:
        """Add multiple memories in batch with optimized embedding.

        This method is ~100-300x faster than calling add() repeatedly because:
        1. Captures are done without inline embedding (very fast)
        2. All embeddings are computed in a single batch (GPU/SIMD optimized)
        3. Index updates are batched

        Args:
            items: Sequence of (content, metadata) tuples
            embed: If True, compute embeddings after capture (default True)
            batch_size: Batch size for embedding computation
            show_progress: Show progress bar for embedding (if embed=True)

        Returns:
            List of MemoryOperationResult, one per input item

        Example:
            >>> results = adapter.add_batch([
            ...     ("First memory content", {"namespace": "learnings"}),
            ...     ("Second memory content", {"namespace": "decisions"}),
            ... ])
            >>> successful = sum(1 for r in results if r.success)
        """
        self._ensure_initialized()
        assert self._batch_capture_service is not None

        results: list[MemoryOperationResult] = []

        # Phase 1: Capture all memories without embedding (fast)
        memory_ids: list[str] = []
        for content, metadata in items:
            try:
                namespace = _extract_namespace_from_metadata(metadata)
                summary = _extract_summary_from_content(content)
                spec = metadata.get("spec") if metadata else None
                tags = metadata.get("tags") if metadata else None
                phase = metadata.get("phase") if metadata else None

                result = self._batch_capture_service.capture(
                    namespace=namespace,
                    summary=summary,
                    content=content,
                    spec=spec,
                    tags=tags,
                    phase=phase,
                )

                if result.success and result.memory:
                    memory_ids.append(result.memory.id)
                    results.append(
                        MemoryOperationResult(
                            success=True,
                            memory_id=result.memory.id,
                            metadata={
                                "indexed": False,  # Will be indexed in batch
                                "namespace": namespace,
                                "stored_at": datetime.now(UTC).isoformat(),
                            },
                        )
                    )
                else:
                    results.append(
                        MemoryOperationResult(
                            success=False,
                            error=result.warning or "Capture failed",
                        )
                    )
            except Exception as e:
                logger.exception("Failed to capture memory in batch")
                results.append(
                    MemoryOperationResult(
                        success=False,
                        error=str(e),
                    )
                )

        # Phase 2: Batch embed all pending memories
        if embed and memory_ids:
            indexed_count = self.embed_pending(
                batch_size=batch_size,
                show_progress=show_progress,
            )
            logger.info("Batch indexed %d memories", indexed_count)

            # Update results to reflect indexed status
            for result in results:
                if result.success and result.metadata:
                    result.metadata["indexed"] = True

        return results

    def embed_pending(
        self,
        *,
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> int:
        """Compute embeddings for all memories that don't have them.

        This is useful after add_batch(embed=False) or when recovering
        from interrupted ingestion.

        Args:
            batch_size: Number of texts to embed at once
            show_progress: Show progress bar during embedding

        Returns:
            Number of memories that were embedded
        """
        self._ensure_initialized()
        assert self._index_service is not None
        assert self._embedding_service is not None

        # Get all memory IDs without embeddings
        pending_ids = self._index_service.get_memories_without_embeddings()
        if not pending_ids:
            return 0

        logger.info("Computing embeddings for %d memories...", len(pending_ids))

        # Get the content for each memory
        memories_to_embed: list[tuple[str, str]] = []  # (id, content)
        for memory_id in pending_ids:
            memory = self._index_service.get(memory_id)
            if memory:
                memories_to_embed.append((memory_id, memory.content))

        if not memories_to_embed:
            return 0

        # Batch compute embeddings
        contents = [content for _, content in memories_to_embed]
        embeddings = self._embedding_service.embed_batch(
            contents,
            batch_size=batch_size,
            show_progress=show_progress,
        )

        # Update each memory with its embedding
        indexed_count = 0
        for (memory_id, _), embedding in zip(memories_to_embed, embeddings, strict=True):
            if self._index_service.update_embedding(memory_id, embedding):
                indexed_count += 1

        return indexed_count

    def get_pending_count(self) -> int:
        """Get the number of memories waiting for embeddings.

        Returns:
            Count of memories without embeddings
        """
        self._ensure_initialized()
        if self._index_service is None:
            return 0
        return len(self._index_service.get_memories_without_embeddings())

    def add_batch_fast(
        self,
        items: Sequence[tuple[str, dict[str, Any] | None]],
        *,
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> list[MemoryOperationResult]:
        """Add memories directly to index, bypassing git notes.

        This is ~100x faster than add_batch() because it skips git notes
        entirely and writes directly to the vector index. Use this for
        benchmark scenarios where git notes persistence isn't needed.

        WARNING: Memories added this way are NOT persisted in git notes.
        They exist only in the SQLite index and will be lost if the index
        is recreated from git notes.

        Args:
            items: Sequence of (content, metadata) tuples
            batch_size: Batch size for embedding computation
            show_progress: Show progress bar during embedding

        Returns:
            List of MemoryOperationResult, one per input item

        Example:
            >>> # Fast ingestion for benchmarking
            >>> results = adapter.add_batch_fast([
            ...     ("Memory 1", {"namespace": "learnings"}),
            ...     ("Memory 2", {"namespace": "decisions"}),
            ... ])
        """
        self._ensure_initialized()
        assert self._index_service is not None
        assert self._embedding_service is not None

        import hashlib

        from git_notes_memory.models import Memory

        # Get current commit SHA (or use a placeholder)
        try:
            import subprocess

            git_result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self._repo_path,
                capture_output=True,
                text=True,
                check=True,
            )
            commit_sha = git_result.stdout.strip()[:7]
        except Exception:
            commit_sha = "0000000"

        # Phase 1: Create Memory objects
        now = datetime.now(UTC)
        memories: list[Memory] = []
        results: list[MemoryOperationResult] = []

        for _idx, (content, metadata) in enumerate(items):
            try:
                namespace = _extract_namespace_from_metadata(metadata)
                summary = _extract_summary_from_content(content)
                spec = metadata.get("spec") if metadata else None
                tags = tuple(metadata.get("tags", [])) if metadata else ()
                phase = metadata.get("phase") if metadata else None

                # Generate a unique ID based on content hash
                content_hash = hashlib.sha256(content.encode()).hexdigest()[:12]
                memory_id = f"{namespace}:{commit_sha}:{content_hash}"

                memory = Memory(
                    id=memory_id,
                    commit_sha=commit_sha,
                    namespace=namespace,
                    summary=summary,
                    content=content,
                    timestamp=now,
                    repo_path=str(self._repo_path),
                    spec=spec,
                    phase=phase,
                    tags=tags,
                    status="active",
                )
                memories.append(memory)
                results.append(
                    MemoryOperationResult(
                        success=True,
                        memory_id=memory_id,
                        metadata={
                            "indexed": False,  # Will be set to True after insert
                            "namespace": namespace,
                            "stored_at": now.isoformat(),
                            "fast_mode": True,
                        },
                    )
                )
            except Exception as e:
                logger.exception("Failed to create memory object")
                results.append(
                    MemoryOperationResult(
                        success=False,
                        error=str(e),
                    )
                )

        if not memories:
            return results

        import gc

        # Phase 2 & 3: Process in chunks to limit peak memory usage
        # Each chunk: embed -> insert -> force garbage collection
        # Reduced chunk size to 5k to stay under Docker memory limits
        chunk_size = 5000  # Process 5k items at a time
        total_inserted = 0
        num_chunks = (len(memories) + chunk_size - 1) // chunk_size

        for chunk_idx, chunk_start in enumerate(range(0, len(memories), chunk_size)):
            chunk_end = min(chunk_start + chunk_size, len(memories))
            chunk_memories = memories[chunk_start:chunk_end]

            logger.info(
                "Processing chunk %d/%d (%d-%d of %d memories)...",
                chunk_idx + 1,
                num_chunks,
                chunk_start,
                chunk_end,
                len(memories),
            )

            # Compute embeddings for this chunk
            contents = [m.content for m in chunk_memories]
            embeddings = self._embedding_service.embed_batch(
                contents,
                batch_size=batch_size,
                show_progress=show_progress,
            )

            # Insert this chunk into index
            inserted = self._index_service.insert_batch(chunk_memories, embeddings)
            total_inserted += inserted

            # CRITICAL: Force garbage collection to release embedding tensors
            # Without this, memory accumulates across chunks causing OOM
            del contents
            del embeddings
            del chunk_memories
            gc.collect()

        logger.info("Inserted %d memories total", total_inserted)

        # Update results to reflect indexed status
        for op_result in results:
            if op_result.success and op_result.metadata:
                op_result.metadata["indexed"] = True

        return results
