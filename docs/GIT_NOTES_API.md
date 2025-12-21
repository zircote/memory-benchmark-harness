# Git Notes Memory Manager API Reference

This document details the git-notes-memory-manager API for integration with the benchmark harness.

**Version**: Compatible with git-notes-memory-manager v0.1.0+
**Last Updated**: 2025-12-19

## Overview

The memory system uses a service-oriented architecture:

| Service | Purpose | Factory Function |
|---------|---------|------------------|
| `CaptureService` | Write memories to git notes | `get_capture_service()` |
| `RecallService` | Read/search memories | `get_recall_service()` |
| `SyncService` | Synchronize index with git notes | `get_sync_service()` |

```python
from git_notes_memory import (
    get_capture_service,
    get_recall_service,
    get_sync_service,
)
```

---

## CaptureService

Stores memories to git notes with optional vector indexing.

### Factory

```python
def get_capture_service(
    repo_path: Path | None = None,
) -> CaptureService
```

### Core Method: `capture()`

```python
def capture(
    self,
    namespace: str,
    summary: str,
    content: str,
    *,
    spec: str | None = None,
    tags: list[str] | tuple[str, ...] | None = None,
    phase: str | None = None,
    status: str = "active",
    relates_to: list[str] | tuple[str, ...] | None = None,
    commit: str = "HEAD",
    skip_lock: bool = False,
) -> CaptureResult
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `namespace` | `str` | Yes | Memory type (see valid namespaces below) |
| `summary` | `str` | Yes | One-line summary (max 100 chars) |
| `content` | `str` | Yes | Full markdown content (max 1MB) |
| `spec` | `str` | No | Specification slug for grouping |
| `tags` | `list[str]` | No | Categorization tags |
| `phase` | `str` | No | Lifecycle phase |
| `status` | `str` | No | "active", "resolved", or "archived" |
| `relates_to` | `list[str]` | No | Related memory IDs |
| `commit` | `str` | No | Git commit SHA (default: "HEAD") |
| `skip_lock` | `bool` | No | Skip file locking |

**Valid Namespaces:**
- `inception`, `elicitation`, `research`, `decisions`
- `progress`, `blockers`, `reviews`, `learnings`
- `retrospective`, `patterns`

**Returns:**

```python
@dataclass(frozen=True)
class CaptureResult:
    success: bool
    memory: Memory | None = None
    indexed: bool = False
    warning: str | None = None
```

### Convenience Methods

```python
def capture_decision(summary: str, content: str, *, spec: str | None = None, tags: list[str] | None = None) -> CaptureResult
def capture_blocker(summary: str, content: str, *, spec: str | None = None) -> CaptureResult
def capture_learning(summary: str, content: str, *, spec: str | None = None, tags: list[str] | None = None) -> CaptureResult
def capture_progress(summary: str, content: str, *, spec: str | None = None) -> CaptureResult
def capture_pattern(name: str, pattern_type: str, description: str, *, evidence: list[str] | None = None, confidence: float = 0.5) -> CaptureResult
def capture_review(summary: str, findings: str, *, spec: str | None = None, severity: str = "medium", category: str | None = None) -> CaptureResult
def capture_retrospective(summary: str, outcome: str, content: str, *, spec: str | None = None) -> CaptureResult
def resolve_blocker(blocker_id: str, resolution: str) -> CaptureResult
```

---

## RecallService

Retrieves and searches memories using vector similarity.

### Factory

```python
def get_recall_service(
    index_path: Path | None = None,
) -> RecallService
```

### Search Methods

#### `search()` - Semantic Vector Search

```python
def search(
    self,
    query: str,
    k: int = 10,
    *,
    namespace: str | None = None,
    spec: str | None = None,
    min_similarity: float | None = None,
) -> list[MemoryResult]
```

**Returns:**

```python
@dataclass(frozen=True)
class MemoryResult:
    memory: Memory
    distance: float  # Euclidean distance (lower = more similar)
```

**Example:**
```python
recall = get_recall_service()
results = recall.search("authentication decision", k=5)
for result in results:
    print(f"{result.memory.id}: {result.memory.summary} (distance: {result.distance:.3f})")
```

#### `search_text()` - Full-Text Search

```python
def search_text(
    self,
    query: str,
    limit: int = 10,
    *,
    namespace: str | None = None,
    spec: str | None = None,
) -> list[Memory]
```

### Retrieval Methods

```python
def get(memory_id: str) -> Memory | None
def get_batch(memory_ids: Sequence[str]) -> list[Memory]
def get_by_namespace(namespace: str, *, spec: str | None = None, limit: int | None = None) -> list[Memory]
def get_by_spec(spec: str, *, namespace: str | None = None, limit: int | None = None) -> list[Memory]
def list_recent(limit: int = 10, *, namespace: str | None = None, spec: str | None = None) -> list[MemoryResult]
```

### Hydration

Load additional context for memories:

```python
def hydrate(
    self,
    memory_or_result: Memory | MemoryResult,
    level: HydrationLevel = HydrationLevel.SUMMARY,
) -> HydratedMemory
```

**HydrationLevel:**

| Level | Description |
|-------|-------------|
| `SUMMARY` | Metadata and summary only (fast) |
| `FULL` | Complete note content from git |
| `FILES` | Content plus file snapshots at commit |

**Returns:**

```python
@dataclass(frozen=True)
class HydratedMemory:
    result: MemoryResult
    full_content: str | None = None
    files: tuple[tuple[str, str], ...] = ()  # (path, content) tuples
    commit_info: CommitInfo | None = None
```

---

## SyncService

Manages synchronization between git notes and the vector index.

### Factory

```python
def get_sync_service(
    repo_path: Path | None = None,
) -> SyncService
```

### Methods

```python
def reindex(*, full: bool = False) -> int  # Returns count of indexed memories
def verify_consistency() -> VerificationResult
def collect_notes() -> list[NoteRecord]
```

**VerificationResult:**

```python
@dataclass(frozen=True)
class VerificationResult:
    is_consistent: bool
    missing_in_index: tuple[str, ...] = ()
    orphaned_in_index: tuple[str, ...] = ()
    mismatched: tuple[str, ...] = ()

    @property
    def total_issues(self) -> int
```

---

## IndexService (Low-Level)

Direct access to the vector index:

```python
def get_stats() -> IndexStats
def clear() -> int  # Returns count of deleted memories
def count(namespace: str | None = None, spec: str | None = None) -> int
```

**IndexStats:**

```python
@dataclass(frozen=True)
class IndexStats:
    total_memories: int
    by_namespace: tuple[tuple[str, int], ...] = ()
    by_spec: tuple[tuple[str, int], ...] = ()
    last_sync: datetime | None = None
    index_size_bytes: int = 0
```

---

## Memory Data Model

```python
@dataclass(frozen=True)
class Memory:
    id: str                    # Format: "namespace:commit_sha:index"
    commit_sha: str
    namespace: str
    summary: str               # Max 100 chars
    content: str
    timestamp: datetime
    repo_path: str | None = None
    spec: str | None = None
    phase: str | None = None
    tags: tuple[str, ...] = ()
    status: str = "active"     # "active", "resolved", "archived"
    relates_to: tuple[str, ...] = ()
```

---

## Configuration Constants

```python
from git_notes_memory.config import (
    NAMESPACES,              # frozenset of valid namespaces
    MAX_CONTENT_BYTES,       # 1,000,000
    MAX_SUMMARY_CHARS,       # 100
    EMBEDDING_DIMENSIONS,    # 384
    DEFAULT_RECALL_LIMIT,    # 10
    SEARCH_TIMEOUT_MS,       # 5000
    CAPTURE_TIMEOUT_MS,      # 10000
)
```

---

## Exception Hierarchy

```python
from git_notes_memory.exceptions import (
    MemoryError,        # Base exception
    StorageError,       # Git storage issues
    MemoryIndexError,   # Index operations failed
    EmbeddingError,     # Vector embedding issues
    ParseError,         # Note parsing issues
    CaptureError,       # Capture operation failed
    ValidationError,    # Input validation failed
    RecallError,        # Recall operation failed
)
```

---

## Adapter Mapping

The benchmark harness `MemorySystemAdapter` interface maps to git-notes-memory as follows:

| Adapter Method | Git Notes Service | Git Notes Method |
|----------------|-------------------|------------------|
| `add(content, metadata)` | CaptureService | `capture(namespace, summary, content, **metadata)` |
| `search(query, limit, min_score, filter)` | RecallService | `search(query, k, namespace, min_similarity)` |
| `update(memory_id, content, metadata)` | Not directly supported | Re-capture with same content |
| `delete(memory_id)` | Not directly supported | Status change to "archived" |
| `clear()` | IndexService | `clear()` |
| `get_stats()` | IndexService | `get_stats()` |

**Notes:**
- Git notes are append-only; "update" and "delete" are status-based
- `min_score` requires converting from Euclidean distance to similarity
- `metadata_filter` maps to `namespace` and `spec` parameters
