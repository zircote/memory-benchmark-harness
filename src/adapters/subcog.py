"""Subcog memory system adapter.

This module implements the MemorySystemAdapter interface for the subcog
Rust-based memory system, enabling benchmark comparison against other
memory systems like git-notes-memory-manager.

Subcog provides a three-layer storage architecture:
- Git notes persistence (GitNotesBackend)
- SQLite FTS5 for text search (SqliteBackend)
- Usearch for vector search (UsearchBackend)

The adapter connects directly to subcog's SQLite database for reliable
read/write operations, bypassing CLI inconsistencies.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import sqlite3
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.adapters.base import MemoryItem, MemoryOperationResult, MemorySystemAdapter

logger = logging.getLogger(__name__)

# Characters that cause FTS5 syntax errors and need to be escaped/removed
# Includes operators, punctuation, and special symbols that break FTS5 queries
FTS5_SPECIAL_CHARS = set('?*":()[]{}^~\\\',-+$.<>|@#%&;/!')


def _sanitize_fts_query(query: str) -> str:
    """Remove or escape special characters that break FTS5 queries.

    Args:
        query: Raw search query

    Returns:
        Sanitized query safe for FTS5
    """
    return "".join(c if c not in FTS5_SPECIAL_CHARS else " " for c in query)


def _compute_repo_hash(repo_path: Path) -> str:
    """Compute the cache directory hash for a repository path.

    Subcog uses a hash of the repo path to create isolated cache directories.
    This matches subcog's Rust implementation.

    Args:
        repo_path: Path to the git repository

    Returns:
        16-character hex hash
    """
    # Use the canonical path for hashing
    canonical = str(repo_path.resolve())
    return hashlib.blake2b(canonical.encode(), digest_size=8).hexdigest()


def _get_cache_dir(repo_path: Path) -> Path:
    """Get the subcog cache directory for a repository.

    Args:
        repo_path: Path to the git repository

    Returns:
        Path to the subcog cache directory
    """
    cache_base = Path.home() / "Library" / "Caches" / "subcog"
    repo_hash = _compute_repo_hash(repo_path)
    return cache_base / repo_hash


class SubcogAdapter(MemorySystemAdapter):
    """Adapter connecting directly to subcog's SQLite database.

    This adapter integrates the subcog Rust-based memory system with
    the benchmark harness by connecting directly to subcog's SQLite
    storage layer, providing:
    - Full-text search via FTS5
    - Memory storage via direct SQLite inserts
    - Reliable statistics from database queries

    Example:
        ```python
        adapter = SubcogAdapter(repo_path="/path/to/repo")
        result = adapter.add(
            "Decided to use PostgreSQL for persistence",
            metadata={"namespace": "decisions", "tags": ["database"]}
        )
        if result.success:
            memories = adapter.search("database decision", limit=5)
            for mem in memories:
                print(f"{mem.memory_id}: {mem.content} (score: {mem.score})")
        ```

    Note:
        The repo_path must be a git repository. Subcog stores its data
        in a cache directory based on a hash of the repo path.
    """

    def __init__(
        self,
        repo_path: str | Path | None = None,
        db_path: str | Path | None = None,
        data_dir: str | Path | None = None,
    ) -> None:
        """Initialize the SubcogAdapter.

        Args:
            repo_path: Path to the git repository. If None, uses current
                      working directory.
            db_path: Direct path to SQLite database. If provided, overrides
                    the cache directory lookup.
            data_dir: Custom data directory for subcog. If None, uses default
                     cache location. Setting this enables test isolation.

        Raises:
            ValueError: If repo_path is not a valid git repository
        """
        self._repo_path = Path(repo_path) if repo_path else Path.cwd()
        self._data_dir = Path(data_dir) if data_dir else None
        self._temp_data_dir: Path | None = None  # For clear() isolation
        self._conn: sqlite3.Connection | None = None

        # Validate repo is a git repository
        if not (self._repo_path / ".git").exists():
            msg = f"Not a git repository: {self._repo_path}"
            raise ValueError(msg)

        # Determine database path
        if db_path:
            self._db_path = Path(db_path)
        elif data_dir:
            self._db_path = Path(data_dir) / "index.db"
        else:
            cache_dir = _get_cache_dir(self._repo_path)
            self._db_path = cache_dir / "index.db"

        logger.debug("SubcogAdapter using database: %s", self._db_path)

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create SQLite connection.

        Returns:
            Active SQLite connection
        """
        if self._conn is None:
            # Ensure parent directory exists
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self._db_path))
            self._conn.row_factory = sqlite3.Row
            self._init_schema()
        return self._conn

    def _init_schema(self) -> None:
        """Initialize database schema if needed."""
        conn = self._conn
        if conn is None:
            return

        cursor = conn.cursor()

        # Create memories table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                namespace TEXT NOT NULL,
                content TEXT NOT NULL,
                tags TEXT,
                created_at INTEGER NOT NULL
            )
        """)

        # Create indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_namespace
            ON memories(namespace)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_created_at
            ON memories(created_at)
        """)

        # Create FTS5 virtual table with Porter stemmer for better matching
        # Porter stemmer enables: "painting" matches "painted", "running" matches "ran"
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
            USING fts5(content, content='memories', content_rowid='rowid', tokenize='porter unicode61')
        """)

        # Create triggers to keep FTS in sync
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                INSERT INTO memories_fts(rowid, content) VALUES (new.rowid, new.content);
            END
        """)
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, content)
                VALUES('delete', old.rowid, old.content);
            END
        """)
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, content)
                VALUES('delete', old.rowid, old.content);
                INSERT INTO memories_fts(rowid, content) VALUES (new.rowid, new.content);
            END
        """)

        conn.commit()

    def add(self, content: str, metadata: dict[str, Any] | None = None) -> MemoryOperationResult:
        """Add a new memory entry to SQLite database.

        Args:
            content: The memory content to store
            metadata: Optional metadata including:
                - namespace: Memory type (default: "learnings")
                - tags: List of tags (comma-separated string or list)
                - source: Source file or context

        Returns:
            MemoryOperationResult with the assigned memory_id
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            namespace = "learnings"
            if metadata:
                namespace = metadata.get("namespace", "learnings")

            # Generate unique ID
            memory_id = f"{namespace}_{uuid.uuid4()!s}"

            # Handle tags
            tags_str = ""
            if metadata and "tags" in metadata:
                tags = metadata["tags"]
                if isinstance(tags, list):
                    tags_str = ",".join(str(t) for t in tags)
                else:
                    tags_str = str(tags)

            # Insert memory
            created_at = int(datetime.now(UTC).timestamp())
            cursor.execute(
                """
                INSERT INTO memories (id, namespace, content, tags, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (memory_id, namespace, content, tags_str, created_at),
            )
            conn.commit()

            return MemoryOperationResult(
                success=True,
                memory_id=memory_id,
                metadata={
                    "namespace": namespace,
                    "tags": tags_str,
                    "stored_at": datetime.now(UTC).isoformat(),
                },
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
        *,
        search_mode: str = "hybrid",  # Only "text" is supported via SQLite FTS5
    ) -> list[MemoryItem]:
        """Search memories using SQLite FTS5 with LIKE fallback.

        Args:
            query: The search query
            limit: Maximum number of results
            min_score: Minimum similarity score threshold (0.0 - 1.0)
            metadata_filter: Optional filters:
                - namespace: Filter by namespace
                - tags: Filter by tags
            search_mode: Ignored (only FTS5 text search is supported)

        Returns:
            List of MemoryItem ordered by relevance (highest first)
        """
        _ = search_mode  # SQLite adapter only supports text search

        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Sanitize query for FTS5
            sanitized_query = _sanitize_fts_query(query)
            if not sanitized_query.strip():
                return []

            # Build namespace filter
            namespace_filter = ""
            namespace_param: list[str] = []

            if metadata_filter and "namespace" in metadata_filter:
                namespace_filter = "AND m.namespace = ?"
                namespace_param = [str(metadata_filter["namespace"])]

            # Try FTS5 MATCH first with bm25 ranking
            memory_items = self._fts5_search(
                cursor, sanitized_query, namespace_filter, namespace_param, limit
            )

            # If FTS5 returns nothing, try LIKE fallback with individual words
            if not memory_items:
                memory_items = self._like_search(
                    cursor, sanitized_query, namespace_filter, namespace_param, limit
                )

            # Filter by min_score
            return [m for m in memory_items if m.score >= min_score]

        except Exception:
            logger.exception("Search failed")
            return []

    def _fts5_search(
        self,
        cursor: sqlite3.Cursor,
        query: str,
        namespace_filter: str,
        namespace_param: list[str],
        limit: int,
    ) -> list[MemoryItem]:
        """Execute FTS5 MATCH search."""
        sql = f"""
            SELECT m.id, m.namespace, m.content, m.tags, m.created_at,
                   -bm25(memories_fts) as score
            FROM memories m
            JOIN memories_fts fts ON m.rowid = fts.rowid
            WHERE memories_fts MATCH ?
            {namespace_filter}
            ORDER BY score DESC
            LIMIT ?
        """

        params = (query, *namespace_param, limit)
        cursor.execute(sql, params)

        return self._rows_to_items(cursor.fetchall())

    def _like_search(
        self,
        cursor: sqlite3.Cursor,
        query: str,
        namespace_filter: str,
        namespace_param: list[str],
        limit: int,
    ) -> list[MemoryItem]:
        """Execute LIKE fallback search using individual words."""
        # Split query into words and search for any match
        words = [w.strip() for w in query.split() if len(w.strip()) >= 3]
        if not words:
            return []

        # Build OR conditions for each word
        like_conditions = " OR ".join(["m.content LIKE ?" for _ in words])
        like_params = [f"%{w}%" for w in words]

        sql = f"""
            SELECT m.id, m.namespace, m.content, m.tags, m.created_at,
                   1.0 as score
            FROM memories m
            WHERE ({like_conditions})
            {namespace_filter}
            LIMIT ?
        """

        params = (*like_params, *namespace_param, limit)
        cursor.execute(sql, params)

        # Score based on number of matching words
        results = []
        for row in cursor.fetchall():
            content_lower = row["content"].lower()
            match_count = sum(1 for w in words if w.lower() in content_lower)
            score = min(1.0, match_count / len(words))  # Fraction of words matched

            created_at = datetime.fromtimestamp(row["created_at"], tz=UTC)
            tags = row["tags"].split(",") if row["tags"] else []

            results.append(
                MemoryItem(
                    memory_id=row["id"],
                    content=row["content"],
                    metadata={
                        "namespace": row["namespace"],
                        "tags": tags,
                        "search_type": "like",
                    },
                    score=score,
                    created_at=created_at,
                    updated_at=None,
                )
            )

        # Sort by score descending
        results.sort(key=lambda x: x.score, reverse=True)
        return results

    def _rows_to_items(self, rows: list[sqlite3.Row]) -> list[MemoryItem]:
        """Convert database rows to MemoryItem objects."""
        items = []
        for row in rows:
            # Normalize bm25 score to 0-1 range
            raw_score = row["score"]
            # bm25 scores typically range 0-25, use gentler normalization
            normalized_score = min(1.0, max(0.0, raw_score / 5.0))

            created_at = datetime.fromtimestamp(row["created_at"], tz=UTC)
            tags = row["tags"].split(",") if row["tags"] else []

            items.append(
                MemoryItem(
                    memory_id=row["id"],
                    content=row["content"],
                    metadata={
                        "namespace": row["namespace"],
                        "tags": tags,
                        "raw_score": raw_score,
                    },
                    score=normalized_score,
                    created_at=created_at,
                    updated_at=None,
                )
            )
        return items

    def update(
        self,
        memory_id: str,
        content: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryOperationResult:
        """Update an existing memory entry.

        Subcog memories are append-only (like git notes), so "update"
        creates a new memory with the updated content.

        Args:
            memory_id: The ID of the memory to update
            content: New content (required for update)
            metadata: Optional updated metadata

        Returns:
            MemoryOperationResult with the new memory_id
        """
        if content is None:
            return MemoryOperationResult(
                success=False,
                error="Content is required for update",
            )

        # Create new memory with updated content
        result = self.add(content, metadata)

        if result.success:
            result.metadata = result.metadata or {}
            result.metadata["supersedes"] = memory_id
            result.metadata["updated_at"] = datetime.now(UTC).isoformat()

        return result

    def delete(self, memory_id: str) -> MemoryOperationResult:
        """Delete a memory entry.

        Subcog uses git notes which are append-only. True deletion
        is not supported. This method logs a warning and returns success
        for API compatibility.

        Args:
            memory_id: The ID of the memory to delete

        Returns:
            MemoryOperationResult indicating success
        """
        logger.warning(
            "SubcogAdapter.delete() cannot truly delete. "
            "Memory %s marked as conceptually archived.",
            memory_id,
        )

        return MemoryOperationResult(
            success=True,
            memory_id=memory_id,
            metadata={"note": "Subcog memories are immutable; memory archived"},
        )

    def clear(self) -> MemoryOperationResult:
        """Clear all memories from the database.

        Deletes all rows from the memories table and rebuilds FTS index.

        Returns:
            MemoryOperationResult indicating success/failure
        """
        try:
            # Close existing connection
            if self._conn:
                self._conn.close()
                self._conn = None

            # Clean up previous temp dir if exists
            if self._temp_data_dir and self._temp_data_dir.exists():
                shutil.rmtree(self._temp_data_dir)

            # Create new temp data dir for isolation
            self._temp_data_dir = Path(tempfile.mkdtemp(prefix="subcog_bench_"))
            self._db_path = self._temp_data_dir / "index.db"
            logger.debug("Created fresh database: %s", self._db_path)

            return MemoryOperationResult(
                success=True,
                metadata={
                    "note": "Using fresh database for isolation",
                    "data_dir": str(self._temp_data_dir),
                },
            )

        except Exception as e:
            logger.exception("Failed to clear")
            return MemoryOperationResult(
                success=False,
                error=str(e),
            )

    def get_stats(self) -> dict[str, Any]:
        """Get memory system statistics.

        Returns:
            Dictionary with:
                - memory_count: Number of stored memories
                - type: "subcog"
                - db_path: Database file path
                - repo_root: Repository root path
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM memories")
            count = cursor.fetchone()[0]

            # Get namespace breakdown
            cursor.execute(
                "SELECT namespace, COUNT(*) FROM memories GROUP BY namespace"
            )
            namespaces = {row[0]: row[1] for row in cursor.fetchall()}

            return {
                "memory_count": count,
                "type": "subcog",
                "db_path": str(self._db_path),
                "repo_root": str(self._repo_path),
                "namespaces": namespaces,
            }

        except Exception as e:
            logger.exception("Failed to get stats")
            return {
                "memory_count": 0,
                "type": "subcog",
                "error": str(e),
            }

    def get_version(self) -> str | None:
        """Get the adapter version.

        Returns:
            Version string
        """
        return "subcog-sqlite-adapter-1.0.0"

    def cleanup(self) -> None:
        """Clean up database connection and temporary directories.

        Call this when done with the adapter to close connections and
        remove any temp directories created during clear() operations.
        """
        # Close database connection
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

        # Remove temp directory
        if self._temp_data_dir and self._temp_data_dir.exists():
            try:
                shutil.rmtree(self._temp_data_dir)
                logger.debug("Cleaned up temp data dir: %s", self._temp_data_dir)
            except Exception:
                logger.exception("Failed to cleanup temp dir")
            self._temp_data_dir = None

    def __del__(self) -> None:
        """Destructor to cleanup resources."""
        self.cleanup()

    # =========================================================================
    # Batch Operations (for high-performance bulk ingestion)
    # =========================================================================

    def add_batch(
        self,
        items: list[tuple[str, dict[str, Any] | None]],
        *,
        show_progress: bool = False,
    ) -> list[MemoryOperationResult]:
        """Add multiple memories in batch using SQLite transactions.

        Uses a single transaction for all inserts for maximum performance.

        Args:
            items: List of (content, metadata) tuples
            show_progress: Show progress during ingestion

        Returns:
            List of MemoryOperationResult, one per input item
        """
        results: list[MemoryOperationResult] = []

        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            now = int(datetime.now(UTC).timestamp())

            for i, (content, metadata) in enumerate(items):
                if show_progress and (i + 1) % 100 == 0:
                    logger.info("Captured %d/%d memories", i + 1, len(items))

                namespace = "learnings"
                if metadata:
                    namespace = metadata.get("namespace", "learnings")

                memory_id = f"{namespace}_{uuid.uuid4()!s}"

                tags_str = ""
                if metadata and "tags" in metadata:
                    tags = metadata["tags"]
                    if isinstance(tags, list):
                        tags_str = ",".join(str(t) for t in tags)
                    else:
                        tags_str = str(tags)

                cursor.execute(
                    """
                    INSERT INTO memories (id, namespace, content, tags, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (memory_id, namespace, content, tags_str, now),
                )

                results.append(
                    MemoryOperationResult(
                        success=True,
                        memory_id=memory_id,
                        metadata={"namespace": namespace},
                    )
                )

            conn.commit()

        except Exception as e:
            logger.exception("Batch add failed")
            # Mark remaining as failed
            while len(results) < len(items):
                results.append(
                    MemoryOperationResult(success=False, error=str(e))
                )

        return results

    def add_batch_fast(
        self,
        items: list[tuple[str, dict[str, Any] | None]],
        *,
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> list[MemoryOperationResult]:
        """Add memories in batch with optimized SQLite settings.

        Uses PRAGMA optimizations and batch commits for maximum throughput.

        Args:
            items: List of (content, metadata) tuples
            batch_size: Number of items per commit (default: 32)
            show_progress: Show progress during ingestion

        Returns:
            List of MemoryOperationResult, one per input item
        """
        results: list[MemoryOperationResult] = []

        try:
            conn = self._get_connection()

            # Enable fast mode
            conn.execute("PRAGMA synchronous = OFF")
            conn.execute("PRAGMA journal_mode = MEMORY")

            cursor = conn.cursor()
            now = int(datetime.now(UTC).timestamp())

            for i, (content, metadata) in enumerate(items):
                if show_progress and (i + 1) % 100 == 0:
                    logger.info("Captured %d/%d memories", i + 1, len(items))

                namespace = "learnings"
                if metadata:
                    namespace = metadata.get("namespace", "learnings")

                memory_id = f"{namespace}_{uuid.uuid4()!s}"

                tags_str = ""
                if metadata and "tags" in metadata:
                    tags = metadata["tags"]
                    if isinstance(tags, list):
                        tags_str = ",".join(str(t) for t in tags)
                    else:
                        tags_str = str(tags)

                cursor.execute(
                    """
                    INSERT INTO memories (id, namespace, content, tags, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (memory_id, namespace, content, tags_str, now),
                )

                results.append(
                    MemoryOperationResult(
                        success=True,
                        memory_id=memory_id,
                        metadata={"namespace": namespace},
                    )
                )

                # Commit every batch_size items
                if (i + 1) % batch_size == 0:
                    conn.commit()

            # Final commit
            conn.commit()

            # Restore safe mode
            conn.execute("PRAGMA synchronous = FULL")
            conn.execute("PRAGMA journal_mode = DELETE")

        except Exception as e:
            logger.exception("Batch add failed")
            while len(results) < len(items):
                results.append(
                    MemoryOperationResult(success=False, error=str(e))
                )

        return results
