# GitHub Copilot Instructions for memory-benchmark-harness

## Project Overview

This is a Python benchmark harness for validating git-native semantic memory benefits for AI coding agents. The project evaluates memory systems against academic benchmarks including Context-Bench, LongMemEval, MemoryAgentBench, and TerminalBench.

## Code Style and Standards

### Python Version
- Target Python 3.11+ (uses modern typing syntax, dataclasses with slots)
- Use `from __future__ import annotations` for forward references

### Type Hints
- All functions must have complete type annotations
- Use `dict[str, Any]` not `Dict[str, Any]` (PEP 585)
- Use `list[T]` not `List[T]`
- Use `T | None` not `Optional[T]` (PEP 604)
- Use `Callable[[Args], Return]` from collections.abc

### Code Formatting
- Line length: 100 characters
- Formatter: ruff format
- Linter: ruff check
- Quote style: double quotes
- Indent: 4 spaces

### Docstrings
- Use Google-style docstrings
- Include Args, Returns, Raises sections
- Add Examples for public APIs
- Document all public classes and methods

### Example Code Pattern
```python
"""Module docstring describing purpose."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ExampleResult:
    """Result container with clear documentation.

    Attributes:
        value: The computed value
        metadata: Additional result context
    """

    value: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON export."""
        return {"value": self.value, "metadata": self.metadata}


def process_data(
    items: list[str],
    *,
    limit: int = 10,
    filter_fn: Callable[[str], bool] | None = None,
) -> ExampleResult:
    """Process items with optional filtering.

    Args:
        items: Input items to process
        limit: Maximum items to return
        filter_fn: Optional filter function

    Returns:
        ExampleResult containing processed data

    Raises:
        ValueError: If items is empty

    Example:
        >>> result = process_data(["a", "b", "c"], limit=2)
        >>> result.value
        'a,b'
    """
    if not items:
        raise ValueError("items cannot be empty")

    if filter_fn:
        items = [x for x in items if filter_fn(x)]

    return ExampleResult(value=",".join(items[:limit]))
```

## Project Structure

```
src/
  adapters/          # Memory system adapters (GitNotes, Mock, NoMemory)
  benchmarks/        # Benchmark implementations
    contextbench/    # Context-Bench (file navigation, multi-hop)
    longmemeval/     # LongMemEval (long-term memory)
    memoryagentbench/# MemoryAgentBench (agent memory)
    terminalbench/   # TerminalBench (terminal interaction)
  cli/               # Typer-based CLI interface
  evaluation/        # LLM judges and metrics
  experiments/       # Experiment runner and configuration

tests/
  unit/              # Fast, isolated unit tests
  integration/       # Slower tests with real dependencies
```

## Testing Patterns

### Unit Tests
- Use pytest with pytest-asyncio for async tests
- Fixtures go in conftest.py
- Mock external services (LLM APIs, file systems)
- Target >80% coverage

### Test Example
```python
import pytest
from src.adapters import MockAdapter, MemoryItem


@pytest.fixture
def adapter() -> MockAdapter:
    """Create a fresh mock adapter for each test."""
    return MockAdapter()


def test_add_memory(adapter: MockAdapter) -> None:
    """Test adding a memory item."""
    result = adapter.add("test content", {"tag": "test"})

    assert result.success
    assert result.memory_id is not None


def test_search_returns_ordered_results(adapter: MockAdapter) -> None:
    """Test search returns results in score order."""
    adapter.add("relevant content")
    adapter.add("less relevant")

    results = adapter.search("relevant", limit=2)

    assert len(results) <= 2
    assert all(isinstance(r, MemoryItem) for r in results)
```

## Common Patterns

### Adapter Interface
All memory system adapters implement `MemorySystemAdapter`:
- `add(content, metadata) -> MemoryOperationResult`
- `search(query, limit, min_score, metadata_filter) -> list[MemoryItem]`
- `update(memory_id, content, metadata) -> MemoryOperationResult`
- `delete(memory_id) -> MemoryOperationResult`
- `clear() -> MemoryOperationResult`
- `get_stats() -> dict[str, Any]`

### Pipeline Pattern
Benchmarks use a pipeline pattern:
1. Load dataset
2. Initialize adapter/agent
3. Evaluate questions/tasks
4. Aggregate metrics
5. Export results

### Error Handling
- Use specific exception types
- Log errors with context: `logger.error(f"Failed: {e}", exc_info=True)`
- Return structured error results, don't raise in pipelines

## Commit Messages

Use conventional commits:
- `feat:` new features
- `fix:` bug fixes
- `docs:` documentation
- `test:` test additions/changes
- `refactor:` code restructuring
- `perf:` performance improvements
- `chore:` maintenance tasks
- `ci:` CI/CD changes

## Development Commands

```bash
# Setup
make dev                 # Install all dependencies + pre-commit hooks

# Quality checks
make lint               # Run ruff linter
make format             # Format with ruff
make typecheck          # Run mypy
make check              # All checks (lint, format, typecheck)

# Testing
make test               # All tests with coverage
make test-unit          # Unit tests only
make test-quick         # Fast tests without coverage

# Maintenance
make clean              # Remove build artifacts
```

## Important Notes

1. **Dataclasses**: Always use `@dataclass(slots=True)` for memory efficiency
2. **Async**: Many operations are async; use `async def` and `await`
3. **Pydantic**: Used for configuration validation, not data classes
4. **Rich**: Used for CLI output formatting
5. **Logging**: Use `logger = logging.getLogger(__name__)` per module
