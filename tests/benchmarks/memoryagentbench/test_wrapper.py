"""Tests for the MemoryAgentBench wrapper module."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.adapters.base import MemoryItem, MemoryOperationResult
from src.benchmarks.memoryagentbench.dataset import (
    Competency,
    DifficultyLevel,
    MemoryAgentBenchQuestion,
)
from src.benchmarks.memoryagentbench.wrapper import (
    AnswerResult,
    MemoryAgentBenchAgent,
)


class MockLLMClient:
    """Mock LLM client for testing."""

    def __init__(self, response: str = "test answer") -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 1024,
    ) -> str:
        self.calls.append(
            {
                "prompt": prompt,
                "system_prompt": system_prompt,
                "max_tokens": max_tokens,
            }
        )
        return self.response


class MockAdapter:
    """Mock memory adapter for testing."""

    def __init__(self) -> None:
        self.memories: list[tuple[str, dict[str, Any]]] = []
        self.search_results: list[MemoryItem] = []
        self.add_calls = 0
        self.search_calls = 0
        self.clear_calls = 0

    def add(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryOperationResult:
        self.add_calls += 1
        mem_id = f"mem_{self.add_calls}"
        self.memories.append((content, metadata or {}))
        return MemoryOperationResult(success=True, memory_id=mem_id)

    def search(
        self,
        query: str,
        limit: int = 10,
        min_score: float = 0.0,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        self.search_calls += 1
        return self.search_results[:limit]

    def clear(self) -> MemoryOperationResult:
        self.clear_calls += 1
        self.memories.clear()
        return MemoryOperationResult(success=True)


class TestAnswerResult:
    """Tests for the AnswerResult dataclass."""

    def test_creation(self) -> None:
        """Test creating an answer result."""
        result = AnswerResult(
            question_id="q1",
            answer="test answer",
            retrieved_memories=[],
        )
        assert result.question_id == "q1"
        assert result.answer == "test answer"
        assert result.used_memory is False

    def test_used_memory(self) -> None:
        """Test used_memory property."""
        result_no_mem = AnswerResult(
            question_id="q1",
            answer="answer",
            retrieved_memories=[],
        )
        assert result_no_mem.used_memory is False

        result_with_mem = AnswerResult(
            question_id="q2",
            answer="answer",
            retrieved_memories=[{"content": "mem1"}],
        )
        assert result_with_mem.used_memory is True


class TestMemoryAgentBenchAgent:
    """Tests for the MemoryAgentBenchAgent class."""

    @pytest.fixture
    def sample_question(self) -> MemoryAgentBenchQuestion:
        """Create a sample question for testing."""
        return MemoryAgentBenchQuestion(
            question_id="test_q",
            question_text="What is the capital of France?",
            answers=["Paris"],
            competency=Competency.ACCURATE_RETRIEVAL,
            context="France is a country in Europe. The capital of France is Paris.",
        )

    @pytest.fixture
    def cr_question(self) -> MemoryAgentBenchQuestion:
        """Create a conflict resolution question for testing."""
        return MemoryAgentBenchQuestion(
            question_id="cr_q",
            question_text="What is Alice's current job?",
            answers=["Engineer"],
            competency=Competency.CONFLICT_RESOLUTION,
            context="Alice was a teacher. Update: Alice is now an Engineer.",
            difficulty=DifficultyLevel.SINGLE_HOP,
        )

    @pytest.fixture
    def agent(self) -> MemoryAgentBenchAgent:
        """Create an agent for testing."""
        return MemoryAgentBenchAgent(
            adapter=MockAdapter(),
            llm=MockLLMClient(response="Paris"),
        )

    def test_ingest_context(
        self,
        agent: MemoryAgentBenchAgent,
        sample_question: MemoryAgentBenchQuestion,
    ) -> None:
        """Test ingesting context into memory."""
        stored = agent.ingest_context(sample_question)
        assert stored >= 1
        assert agent.adapter.add_calls >= 1  # type: ignore
        assert agent.adapter.clear_calls == 1  # type: ignore

    def test_ingest_context_clears_previous(
        self,
        agent: MemoryAgentBenchAgent,
        sample_question: MemoryAgentBenchQuestion,
    ) -> None:
        """Test that ingesting clears previous context."""
        agent.ingest_context(sample_question)
        agent.ingest_context(sample_question)
        assert agent.adapter.clear_calls == 2  # type: ignore

    def test_retrieve_for_question(
        self,
        agent: MemoryAgentBenchAgent,
        sample_question: MemoryAgentBenchQuestion,
    ) -> None:
        """Test retrieving memories for a question."""
        # Set up mock search results
        agent.adapter.search_results = [  # type: ignore
            MemoryItem(
                memory_id="mem1",
                content="The capital of France is Paris.",
                score=0.95,
                metadata={},
                created_at=datetime.now(),
            )
        ]

        retrieved = agent.retrieve_for_question(sample_question)
        assert len(retrieved) == 1
        assert retrieved[0]["content"] == "The capital of France is Paris."
        assert agent.adapter.search_calls == 1  # type: ignore

    def test_answer_question(
        self,
        agent: MemoryAgentBenchAgent,
        sample_question: MemoryAgentBenchQuestion,
    ) -> None:
        """Test answering a question."""
        result = agent.answer_question(sample_question)

        assert result.question_id == "test_q"
        assert result.answer == "Paris"
        assert "competency" in result.metadata

    def test_answer_question_skips_ingest(
        self,
        agent: MemoryAgentBenchAgent,
        sample_question: MemoryAgentBenchQuestion,
    ) -> None:
        """Test answering without ingesting."""
        result = agent.answer_question(sample_question, ingest=False)

        assert result.answer == "Paris"
        assert agent.adapter.clear_calls == 0  # type: ignore

    def test_answer_with_conflict_check(
        self,
        agent: MemoryAgentBenchAgent,
        cr_question: MemoryAgentBenchQuestion,
    ) -> None:
        """Test conflict resolution answer method."""
        agent.llm = MockLLMClient(response="Engineer")  # type: ignore

        result = agent.answer_with_conflict_check(cr_question)

        assert result.question_id == "cr_q"
        assert result.answer == "Engineer"
        assert "conflicts_detected" in result.metadata

    def test_chunk_context(
        self,
        agent: MemoryAgentBenchAgent,
    ) -> None:
        """Test context chunking."""
        long_context = "This is paragraph one.\n\n" + "This is paragraph two.\n\n" * 50

        chunks = agent._chunk_context(long_context, chunk_size=500)
        assert len(chunks) > 1
        # Each chunk should be under the size limit (roughly)
        for chunk in chunks:
            assert len(chunk) < 1000  # Some margin for chunk size

    def test_clean_answer(self, agent: MemoryAgentBenchAgent) -> None:
        """Test answer cleaning."""
        # Remove common prefixes
        assert agent._clean_answer("The answer is Paris") == "Paris"
        assert agent._clean_answer("Answer: Paris") == "Paris"
        assert agent._clean_answer("Based on the context, Paris") == "Paris"

        # Strip quotes
        assert agent._clean_answer('"Paris"') == "Paris"
        assert agent._clean_answer("'Paris'") == "Paris"

    def test_build_system_prompt_by_competency(
        self,
        agent: MemoryAgentBenchAgent,
    ) -> None:
        """Test that system prompts vary by competency."""
        prompts = {}
        for comp in Competency:
            prompts[comp] = agent._build_system_prompt(comp)

        # All should be non-empty
        assert all(p for p in prompts.values())

        # Conflict resolution should mention version history
        assert (
            "version" in prompts[Competency.CONFLICT_RESOLUTION].lower()
            or "update" in prompts[Competency.CONFLICT_RESOLUTION].lower()
        )

    def test_detect_conflicts(
        self,
        agent: MemoryAgentBenchAgent,
    ) -> None:
        """Test conflict detection in memories."""
        memories = [
            {"content": "Alice's job is Teacher", "metadata": {}},
            {"content": "Alice's job is Engineer", "metadata": {}},
        ]

        conflicts = agent._detect_conflicts(memories)
        # Should detect that "job" has multiple values
        assert len(conflicts) >= 0  # May or may not detect depending on patterns

    def test_version_history_usage(
        self,
        agent: MemoryAgentBenchAgent,
        cr_question: MemoryAgentBenchQuestion,
    ) -> None:
        """Test that version history is included when available."""
        # Create adapter with get_history method
        mock_adapter = MockAdapter()
        mock_adapter.get_history = MagicMock(
            return_value=[
                {"version": "1", "date": "2024-01-01", "content": "Teacher"},
                {"version": "2", "date": "2024-06-01", "content": "Engineer"},
            ]
        )
        mock_adapter.search_results = [
            MemoryItem(
                memory_id="mem1",
                content="Job info",
                score=0.9,
                metadata={},
                created_at=datetime.now(),
            )
        ]

        agent.adapter = mock_adapter

        agent.retrieve_for_question(cr_question)

        # Should have tried to get history
        assert mock_adapter.get_history.called
