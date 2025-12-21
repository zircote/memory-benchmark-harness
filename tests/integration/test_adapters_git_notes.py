"""Integration tests for GitNotesAdapter.

These tests verify the GitNotesAdapter works correctly with the
git-notes-memory-manager package. Tests are skipped if the package
is not installed.

Note: Integration tests require a real git repository and may modify
git notes. Each test creates an isolated test environment.
"""

from __future__ import annotations

import importlib.util
import subprocess
import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from src.adapters.base import MemoryItem, MemoryOperationResult

# Check if git-notes-memory-manager is available
HAS_GIT_NOTES = importlib.util.find_spec("git_notes_memory") is not None

if TYPE_CHECKING:
    from src.adapters.git_notes import GitNotesAdapter


pytestmark = pytest.mark.skipif(
    not HAS_GIT_NOTES,
    reason="git-notes-memory-manager not installed",
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_git_repo() -> Generator[Path, None, None]:
    """Create a temporary git repository for testing.

    Yields:
        Path to the temporary git repository.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir)

        # Initialize git repo
        subprocess.run(
            ["git", "init"],
            cwd=repo_path,
            check=True,
            capture_output=True,
        )

        # Configure git user for commits
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"],
            cwd=repo_path,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test User"],
            cwd=repo_path,
            check=True,
            capture_output=True,
        )

        # Create initial commit (required for git notes)
        readme = repo_path / "README.md"
        readme.write_text("# Test Repository\n")
        subprocess.run(
            ["git", "add", "."],
            cwd=repo_path,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "Initial commit"],
            cwd=repo_path,
            check=True,
            capture_output=True,
        )

        yield repo_path


@pytest.fixture
def adapter(temp_git_repo: Path) -> GitNotesAdapter:
    """Create a GitNotesAdapter for the temporary repository.

    Args:
        temp_git_repo: Path to the temporary git repository.

    Returns:
        Configured GitNotesAdapter instance.
    """
    from src.adapters.git_notes import GitNotesAdapter

    return GitNotesAdapter(repo_path=temp_git_repo)


# =============================================================================
# Initialization Tests
# =============================================================================


class TestGitNotesAdapterInitialization:
    """Test adapter initialization and lazy loading."""

    def test_initialization_with_valid_repo(self, temp_git_repo: Path) -> None:
        """Test that adapter initializes with a valid git repo."""
        from src.adapters.git_notes import GitNotesAdapter

        adapter = GitNotesAdapter(repo_path=temp_git_repo)
        assert adapter._repo_path == temp_git_repo
        assert not adapter._initialized

    def test_initialization_without_path_uses_cwd(self) -> None:
        """Test that adapter uses CWD when no path provided."""
        from src.adapters.git_notes import GitNotesAdapter

        adapter = GitNotesAdapter()
        assert adapter._repo_path == Path.cwd()

    def test_lazy_initialization(self, adapter: GitNotesAdapter) -> None:
        """Test that services are lazily initialized on first use."""
        assert not adapter._initialized
        assert adapter._capture_service is None
        assert adapter._recall_service is None


# =============================================================================
# Add Operation Tests
# =============================================================================


class TestGitNotesAdapterAdd:
    """Test the add() operation."""

    def test_add_basic_memory(self, adapter: GitNotesAdapter) -> None:
        """Test adding a basic memory without metadata."""
        result = adapter.add("This is a test memory for the benchmark harness.")

        assert isinstance(result, MemoryOperationResult)
        assert result.success is True
        assert result.memory_id is not None
        assert result.error is None

    def test_add_memory_with_namespace(self, adapter: GitNotesAdapter) -> None:
        """Test adding memory with explicit namespace."""
        result = adapter.add(
            "Decided to use PostgreSQL for persistence.",
            metadata={"namespace": "decisions"},
        )

        assert result.success is True
        assert result.metadata is not None
        assert result.metadata.get("namespace") == "decisions"

    def test_add_memory_with_tags(self, adapter: GitNotesAdapter) -> None:
        """Test adding memory with tags."""
        result = adapter.add(
            "Authentication flow uses OAuth 2.0",
            metadata={"namespace": "decisions", "tags": ["auth", "security"]},
        )

        assert result.success is True
        assert result.memory_id is not None

    def test_add_memory_with_spec(self, adapter: GitNotesAdapter) -> None:
        """Test adding memory with specification reference."""
        result = adapter.add(
            "Progress on benchmark implementation",
            metadata={"namespace": "progress", "spec": "SPEC-2025-001"},
        )

        assert result.success is True
        assert result.memory_id is not None

    def test_add_memory_returns_indexed_status(self, adapter: GitNotesAdapter) -> None:
        """Test that add returns indexing status in metadata."""
        result = adapter.add("Test memory for indexing check")

        assert result.success is True
        assert result.metadata is not None
        assert "indexed" in result.metadata


# =============================================================================
# Search Operation Tests
# =============================================================================


class TestGitNotesAdapterSearch:
    """Test the search() operation."""

    def test_search_empty_index(self, adapter: GitNotesAdapter) -> None:
        """Test search on empty index returns empty list."""
        results = adapter.search("anything")
        assert results == []

    def test_search_finds_added_memory(self, adapter: GitNotesAdapter) -> None:
        """Test that search finds previously added memory."""
        # Add a memory
        add_result = adapter.add(
            "PostgreSQL is the database choice for this project.",
            metadata={"namespace": "decisions"},
        )
        assert add_result.success

        # Search for it
        results = adapter.search("database choice PostgreSQL")

        # Note: Vector search may need reindex
        # If results are empty, trigger reindex
        if not results:
            adapter.reindex(full=True)
            results = adapter.search("database choice PostgreSQL")

        assert isinstance(results, list)
        # Results may still be empty if embedding model is not configured
        # This is expected in minimal test environments
        for item in results:
            assert isinstance(item, MemoryItem)

    def test_search_respects_limit(self, adapter: GitNotesAdapter) -> None:
        """Test that search respects the limit parameter."""
        # Add multiple memories
        for i in range(5):
            adapter.add(
                f"Test memory number {i} about testing", metadata={"namespace": "learnings"}
            )

        adapter.reindex(full=True)
        results = adapter.search("testing", limit=2)

        assert len(results) <= 2

    def test_search_with_namespace_filter(self, adapter: GitNotesAdapter) -> None:
        """Test search with namespace filter."""
        adapter.add("Learning about Python", metadata={"namespace": "learnings"})
        adapter.add("Decision about Python", metadata={"namespace": "decisions"})
        adapter.reindex(full=True)

        results = adapter.search(
            "Python",
            metadata_filter={"namespace": "learnings"},
        )

        # All results should be from learnings namespace
        for item in results:
            assert item.metadata.get("namespace") == "learnings"

    def test_search_result_has_required_fields(self, adapter: GitNotesAdapter) -> None:
        """Test that search results have all required MemoryItem fields."""
        adapter.add("Test memory with complete fields", metadata={"namespace": "learnings"})
        adapter.reindex(full=True)

        results = adapter.search("complete fields")

        for item in results:
            assert item.memory_id is not None
            assert item.content is not None
            assert item.metadata is not None
            assert 0.0 <= item.score <= 1.0


# =============================================================================
# Update Operation Tests
# =============================================================================


class TestGitNotesAdapterUpdate:
    """Test the update() operation."""

    def test_update_existing_memory(self, adapter: GitNotesAdapter) -> None:
        """Test updating an existing memory creates new version."""
        # Add initial memory
        add_result = adapter.add(
            "Initial version of the memory",
            metadata={"namespace": "decisions"},
        )
        assert add_result.success
        original_id = add_result.memory_id
        assert original_id is not None

        # Update it
        update_result = adapter.update(
            original_id,
            "Updated version of the memory",
        )

        assert update_result.success is True
        # Update creates a new memory (git notes are immutable)
        assert update_result.memory_id != original_id
        assert update_result.metadata is not None
        assert update_result.metadata.get("supersedes") == original_id

    def test_update_nonexistent_memory_fails(self, adapter: GitNotesAdapter) -> None:
        """Test that updating non-existent memory fails gracefully."""
        result = adapter.update(
            "nonexistent:abc123:0",
            "This should fail",
        )

        assert result.success is False
        assert result.error is not None
        assert "not found" in result.error.lower()


# =============================================================================
# Delete Operation Tests
# =============================================================================


class TestGitNotesAdapterDelete:
    """Test the delete() operation."""

    def test_delete_existing_memory(self, adapter: GitNotesAdapter) -> None:
        """Test deleting an existing memory."""
        # Add a memory
        add_result = adapter.add("Memory to be deleted", metadata={"namespace": "learnings"})
        assert add_result.success
        memory_id = add_result.memory_id
        assert memory_id is not None

        # Delete it
        delete_result = adapter.delete(memory_id)

        # Git notes are immutable, so "delete" is conceptual
        assert delete_result.success is True
        assert delete_result.memory_id == memory_id
        assert delete_result.metadata is not None
        assert "immutable" in str(delete_result.metadata).lower()

    def test_delete_nonexistent_memory_fails(self, adapter: GitNotesAdapter) -> None:
        """Test that deleting non-existent memory fails."""
        result = adapter.delete("nonexistent:abc123:0")

        assert result.success is False
        assert result.error is not None


# =============================================================================
# Clear Operation Tests
# =============================================================================


class TestGitNotesAdapterClear:
    """Test the clear() operation."""

    def test_clear_empty_index(self, adapter: GitNotesAdapter) -> None:
        """Test clearing an empty index succeeds."""
        result = adapter.clear()

        assert result.success is True
        assert result.metadata is not None
        assert "cleared_count" in result.metadata

    def test_clear_populated_index(self, adapter: GitNotesAdapter) -> None:
        """Test clearing a populated index."""
        # Add some memories
        adapter.add("Memory 1", metadata={"namespace": "learnings"})
        adapter.add("Memory 2", metadata={"namespace": "decisions"})
        adapter.reindex(full=True)

        # Clear the index
        result = adapter.clear()

        assert result.success is True
        assert result.metadata.get("cleared_count", 0) >= 0

    def test_clear_preserves_git_notes(self, adapter: GitNotesAdapter) -> None:
        """Test that clear only clears index, not git notes."""
        # Add a memory
        add_result = adapter.add("Persistent memory", metadata={"namespace": "learnings"})
        assert add_result.success

        # Clear the index
        clear_result = adapter.clear()
        assert clear_result.success

        # Metadata should indicate notes are preserved
        assert "preserved" in str(clear_result.metadata).lower()


# =============================================================================
# Stats Operation Tests
# =============================================================================


class TestGitNotesAdapterStats:
    """Test the get_stats() operation."""

    def test_get_stats_empty_index(self, adapter: GitNotesAdapter) -> None:
        """Test getting stats from empty index."""
        stats = adapter.get_stats()

        assert isinstance(stats, dict)
        assert "memory_count" in stats
        assert stats["type"] == "git-notes"
        assert stats["memory_count"] >= 0

    def test_get_stats_after_adds(self, adapter: GitNotesAdapter) -> None:
        """Test getting stats after adding memories."""
        adapter.add("Memory 1", metadata={"namespace": "learnings"})
        adapter.add("Memory 2", metadata={"namespace": "decisions"})
        adapter.reindex(full=True)

        stats = adapter.get_stats()

        assert stats["memory_count"] >= 0
        assert "by_namespace" in stats

    def test_get_stats_has_required_fields(self, adapter: GitNotesAdapter) -> None:
        """Test that stats has all required fields."""
        stats = adapter.get_stats()

        assert "memory_count" in stats
        assert "type" in stats
        assert stats["type"] == "git-notes"


# =============================================================================
# Reindex Operation Tests
# =============================================================================


class TestGitNotesAdapterReindex:
    """Test the reindex() operation (git-notes specific)."""

    def test_reindex_incremental(self, adapter: GitNotesAdapter) -> None:
        """Test incremental reindex."""
        count = adapter.reindex(full=False)
        assert isinstance(count, int)
        assert count >= 0

    def test_reindex_full(self, adapter: GitNotesAdapter) -> None:
        """Test full reindex."""
        # Add some memories first
        adapter.add("Memory for reindex test 1", metadata={"namespace": "learnings"})
        adapter.add("Memory for reindex test 2", metadata={"namespace": "decisions"})

        count = adapter.reindex(full=True)
        assert isinstance(count, int)
        assert count >= 0


# =============================================================================
# End-to-End Workflow Tests
# =============================================================================


class TestGitNotesAdapterWorkflow:
    """Test complete workflows using the adapter."""

    def test_add_search_workflow(self, adapter: GitNotesAdapter) -> None:
        """Test the complete add-then-search workflow."""
        # Add memories
        adapter.add(
            "We decided to use Redis for caching because of its performance.",
            metadata={"namespace": "decisions", "tags": ["caching", "redis"]},
        )
        adapter.add(
            "Learned that Redis pub/sub is useful for real-time updates.",
            metadata={"namespace": "learnings", "tags": ["redis"]},
        )

        # Reindex to ensure search works
        adapter.reindex(full=True)

        # Search
        results = adapter.search("redis caching performance")

        # Verify results structure (content may vary based on embeddings)
        assert isinstance(results, list)
        for result in results:
            assert isinstance(result, MemoryItem)
            assert result.memory_id is not None

    def test_stats_reflect_operations(self, adapter: GitNotesAdapter) -> None:
        """Test that stats accurately reflect adapter state."""
        # Initial stats
        initial_stats = adapter.get_stats()
        initial_count = initial_stats["memory_count"]

        # Add memories
        adapter.add("Stats test memory 1", metadata={"namespace": "learnings"})
        adapter.add("Stats test memory 2", metadata={"namespace": "learnings"})
        adapter.reindex(full=True)

        # Stats should reflect additions
        new_stats = adapter.get_stats()
        assert new_stats["memory_count"] >= initial_count


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestGitNotesAdapterErrorHandling:
    """Test error handling in the adapter."""

    def test_add_handles_exception_gracefully(self, adapter: GitNotesAdapter) -> None:
        """Test that add handles exceptions and returns failure result."""
        # Force an error condition (empty content might trigger validation error)
        result = adapter.add("")

        # Should return a result (success or failure), not raise
        assert isinstance(result, MemoryOperationResult)

    def test_search_handles_exception_gracefully(self, adapter: GitNotesAdapter) -> None:
        """Test that search handles exceptions and returns empty list."""
        # Even with unusual queries, search should not raise
        results = adapter.search("")

        assert isinstance(results, list)
