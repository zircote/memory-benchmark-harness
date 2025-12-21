"""Pytest configuration and shared fixtures for memory-benchmark-harness tests."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

# Ensure src is importable
_root = Path(__file__).parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

if TYPE_CHECKING:
    from src.adapters.base import MemorySystemAdapter


@pytest.fixture
def mock_adapter() -> MemorySystemAdapter:
    """Create a fresh MockAdapter for each test."""
    from src.adapters.mock import MockAdapter

    return MockAdapter()


@pytest.fixture
def no_memory_adapter() -> MemorySystemAdapter:
    """Create a NoMemoryAdapter for baseline testing."""
    from src.adapters.no_memory import NoMemoryAdapter

    return NoMemoryAdapter()


@pytest.fixture
def sample_memories() -> list[tuple[str, dict[str, str]]]:
    """Sample memory content and metadata for testing."""
    return [
        ("User prefers dark mode", {"category": "preferences", "session": "s1"}),
        ("Project uses Python 3.11", {"category": "technical", "session": "s1"}),
        ("Deploy to production on Fridays", {"category": "workflow", "session": "s2"}),
        ("API key is stored in environment", {"category": "technical", "session": "s2"}),
        ("User is working on benchmark harness", {"category": "context", "session": "s3"}),
    ]


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for test artifacts."""
    test_dir = tmp_path / "test_artifacts"
    test_dir.mkdir(exist_ok=True)
    return test_dir
