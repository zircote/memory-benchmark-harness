# CLAUDE.md - memory-benchmark-harness

This file provides guidance to Claude Code when working with this codebase.

## Project Overview

Memory Benchmark Harness is a Python framework for validating git-native semantic memory benefits for AI coding agents. It evaluates memory systems against academic benchmarks:

- **Context-Bench**: Multi-hop file navigation and entity relationship tracing
- **LongMemEval**: Long-term memory retention and retrieval
- **MemoryAgentBench**: Agent memory consistency across sessions
- **TerminalBench**: Terminal interaction and command memory

## Quick Start

```bash
# Install dependencies (uses uv if available)
make dev

# Run quality checks
make check              # lint + format-check + typecheck
make test               # all tests with coverage
make all                # everything (lint, typecheck, test)
```

## Code Standards

### Python Style
- Python 3.11+ with modern typing (`dict[str, T]`, `T | None`)
- `from __future__ import annotations` in all modules
- `@dataclass(slots=True)` for all data classes
- Line length: 100 characters
- Formatter/Linter: ruff

### Type Annotations
```python
# Correct patterns
def process(items: list[str], config: dict[str, Any] | None = None) -> Result: ...

# Avoid (old style)
def process(items: List[str], config: Optional[Dict[str, Any]] = None) -> Result: ...
```

### Docstrings
Google-style with Args, Returns, Raises, Example sections:
```python
def search(query: str, limit: int = 10) -> list[MemoryItem]:
    """Search memories by semantic similarity.

    Args:
        query: The search query
        limit: Maximum number of results

    Returns:
        List of MemoryItem ordered by relevance

    Example:
        >>> results = adapter.search("deployment config", limit=5)
    """
```

## Project Structure

```
src/
  adapters/           # Memory system adapters (abstract base + implementations)
    base.py           # MemorySystemAdapter ABC, MemoryItem, MemoryOperationResult
    git_notes.py      # GitNotesAdapter - wraps git-notes-memory-manager
    mock.py           # MockAdapter - for testing
    no_memory.py      # NoMemoryAdapter - baseline (stores but never retrieves)

  benchmarks/         # Benchmark implementations (each follows same pattern)
    contextbench/     # dataset.py, wrapper.py, pipeline.py, metrics.py
    longmemeval/
    memoryagentbench/
    terminalbench/

  cli/                # Typer CLI application
  evaluation/         # LLM judges, scoring, metrics
  experiments/        # Experiment runner, configuration, results

tests/
  unit/               # Fast isolated tests (mocked dependencies)
  integration/        # Slower tests with real systems
```

## Key Abstractions

### MemorySystemAdapter (src/adapters/base.py)
All memory systems implement this interface:
```python
class MemorySystemAdapter(ABC):
    def add(self, content: str, metadata: dict | None = None) -> MemoryOperationResult: ...
    def search(self, query: str, limit: int = 10, ...) -> list[MemoryItem]: ...
    def update(self, memory_id: str, content: str, ...) -> MemoryOperationResult: ...
    def delete(self, memory_id: str) -> MemoryOperationResult: ...
    def clear(self) -> MemoryOperationResult: ...
    def get_stats(self) -> dict[str, Any]: ...
```

### Benchmark Pipeline Pattern
Each benchmark follows: Dataset -> Agent/Wrapper -> Pipeline -> Metrics
```python
# Load data
dataset = load_contextbench()

# Create agent with memory adapter
agent = ContextBenchAgent(adapter=GitNotesAdapter(), dataset=dataset)

# Run evaluation
pipeline = ContextBenchPipeline(agent=agent, judge=LLMJudge())
results = pipeline.evaluate(dataset)

# Calculate metrics
metrics = MetricsCalculator.calculate(results)
```

## Development Workflow

### Before Committing
```bash
make check              # Quick validation
make test-unit          # Run unit tests
```

### Full CI Simulation
```bash
make all                # lint + typecheck + all tests
```

### Adding a New Benchmark
1. Create directory under `src/benchmarks/<name>/`
2. Implement: `dataset.py`, `wrapper.py`, `pipeline.py`, `metrics.py`, `__init__.py`
3. Follow existing benchmark patterns (see contextbench as reference)
4. Add unit tests in `tests/unit/benchmarks/test_<name>.py`
5. Update CLI to register new benchmark

### Adding a New Adapter
1. Create `src/adapters/<name>.py`
2. Inherit from `MemorySystemAdapter`
3. Implement all abstract methods
4. Export in `src/adapters/__init__.py`
5. Add comprehensive unit tests

## Testing

### Run Tests
```bash
make test               # All tests with coverage
make test-unit          # Unit tests only (fast)
make test-integration   # Integration tests (requires real systems)
make test-quick         # No coverage (fastest)
```

### Test Patterns
```python
import pytest
from src.adapters import MockAdapter

@pytest.fixture
def adapter() -> MockAdapter:
    return MockAdapter()

def test_add_returns_success(adapter: MockAdapter) -> None:
    result = adapter.add("content", {"key": "value"})
    assert result.success
    assert result.memory_id is not None
```

## CI/CD

### GitHub Actions Workflow
- **lint**: ruff check + format check
- **typecheck**: mypy strict mode
- **test**: pytest on Python 3.11 + 3.12
- **pre-commit**: all hooks validation
- **integration-test**: runs on main branch only
- **docker-build**: validates Docker image builds

### Pre-commit Hooks
```bash
make pre-commit         # Install/update hooks
```

Hooks include:
- trailing-whitespace, end-of-file-fixer
- ruff (lint + format)
- mypy (type checking)
- conventional-pre-commit (commit message format)
- detect-secrets

## Common Tasks

### Format Code
```bash
make format             # Auto-fix formatting and simple lint issues
```

### Check Types
```bash
make typecheck          # Run mypy on src/
```

### Generate Coverage Report
```bash
make coverage           # HTML report in htmlcov/
```

### Clean Build Artifacts
```bash
make clean              # Remove caches, build dirs
make clean-all          # Also remove .venv, uv cache
```

## Commit Message Format

Use conventional commits:
```
feat: add new benchmark support
fix: correct memory search scoring
docs: update API documentation
test: add adapter edge case tests
refactor: simplify pipeline aggregation
perf: optimize batch memory operations
chore: update dependencies
ci: add Python 3.12 to test matrix
```

## Error Handling

- Return structured results (MemoryOperationResult) instead of raising
- Log errors with context: `logger.error(f"Operation failed: {e}", exc_info=True)`
- Pipeline evaluations should continue on individual failures
- Aggregate errors in result metadata for analysis

## Performance Considerations

- Use `@dataclass(slots=True)` for memory efficiency
- Batch operations where possible
- Cache expensive computations (embeddings, API calls)
- Profile with `make test` coverage to identify hot paths
