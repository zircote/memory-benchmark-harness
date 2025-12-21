"""Tests for LoCoMo agent wrapper module."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from src.benchmarks.locomo.dataset import (
    LoCoMoConversation,
    LoCoMoQuestion,
    LoCoMoSession,
    LoCoMoTurn,
    QACategory,
)
from src.benchmarks.locomo.wrapper import (
    IngestionResult,
    LLMResponse,
    LoCoMoAgent,
    LoCoMoAnswer,
)

# ============================================================================
# Mock Implementations
# ============================================================================


@dataclass
class MockMemoryEntry:
    """Mock memory entry for search results."""

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float = 1.0


@dataclass
class MockOperationResult:
    """Mock result for adapter operations."""

    success: bool = True
    error: str | None = None


class MockMemoryAdapter:
    """Mock memory adapter for testing."""

    def __init__(self) -> None:
        self.memories: list[MockMemoryEntry] = []
        self.add_calls: list[tuple[str, dict[str, Any]]] = []
        self.search_calls: list[dict[str, Any]] = []
        self.should_fail_add: bool = False

    def add(self, content: str, metadata: dict[str, Any]) -> MockOperationResult:
        """Add a memory entry."""
        self.add_calls.append((content, metadata))
        if self.should_fail_add:
            return MockOperationResult(success=False, error="Mock failure")
        self.memories.append(MockMemoryEntry(content=content, metadata=metadata))
        return MockOperationResult(success=True)

    def search(
        self,
        query: str,
        limit: int = 10,
        min_score: float = 0.0,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[MockMemoryEntry]:
        """Search for memories."""
        self.search_calls.append(
            {
                "query": query,
                "limit": limit,
                "min_score": min_score,
                "metadata_filter": metadata_filter,
            }
        )
        # Return all memories (simple mock behavior)
        return self.memories[:limit]

    def clear(self) -> MockOperationResult:
        """Clear all memories."""
        self.memories.clear()
        return MockOperationResult(success=True)

    def get_stats(self) -> dict[str, Any]:
        """Get adapter statistics."""
        return {"memory_count": len(self.memories)}


class MockLLMClient:
    """Mock LLM client for testing."""

    def __init__(self, response: str = "Test answer") -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        system: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
    ) -> LLMResponse:
        """Generate a mock completion."""
        self.calls.append(
            {
                "system": system,
                "messages": messages,
                "temperature": temperature,
            }
        )
        return LLMResponse(
            content=self.response,
            model="mock-model",
            usage={"input_tokens": 100, "output_tokens": 50},
        )


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_adapter() -> MockMemoryAdapter:
    """Create a mock memory adapter."""
    return MockMemoryAdapter()


@pytest.fixture
def mock_llm() -> MockLLMClient:
    """Create a mock LLM client."""
    return MockLLMClient()


@pytest.fixture
def agent(mock_adapter: MockMemoryAdapter, mock_llm: MockLLMClient) -> LoCoMoAgent:
    """Create a LoCoMo agent with mocks."""
    return LoCoMoAgent(mock_adapter, mock_llm)


@pytest.fixture
def sample_turn() -> LoCoMoTurn:
    """Create a sample dialogue turn."""
    return LoCoMoTurn(
        speaker="Alice",
        dia_id="D1:5",
        text="I'm going to Paris next week for a conference.",
        session_num=1,
    )


@pytest.fixture
def sample_turn_with_image() -> LoCoMoTurn:
    """Create a sample turn with an image."""
    return LoCoMoTurn(
        speaker="Bob",
        dia_id="D1:6",
        text="That sounds exciting!",
        session_num=1,
        img_url="https://example.com/paris.jpg",
        img_caption="A photo of the Eiffel Tower",
    )


@pytest.fixture
def sample_session() -> LoCoMoSession:
    """Create a sample session."""
    turns = [
        LoCoMoTurn(
            speaker="Alice",
            dia_id="D1:1",
            text="Hi Bob! How are you?",
            session_num=1,
        ),
        LoCoMoTurn(
            speaker="Bob",
            dia_id="D1:2",
            text="I'm great! Just got back from vacation.",
            session_num=1,
        ),
        LoCoMoTurn(
            speaker="Alice",
            dia_id="D1:3",
            text="Oh nice! Where did you go?",
            session_num=1,
        ),
    ]
    return LoCoMoSession(
        session_num=1,
        timestamp="2023-01-15 10:30:00",
        turns=turns,
        speaker_a="Alice",
        speaker_b="Bob",
    )


@pytest.fixture
def sample_conversation() -> LoCoMoConversation:
    """Create a sample conversation with multiple sessions."""
    session1_turns = [
        LoCoMoTurn(
            speaker="Alice",
            dia_id="D1:1",
            text="Hi Bob!",
            session_num=1,
        ),
        LoCoMoTurn(
            speaker="Bob",
            dia_id="D1:2",
            text="Hey Alice!",
            session_num=1,
        ),
    ]
    session2_turns = [
        LoCoMoTurn(
            speaker="Alice",
            dia_id="D2:1",
            text="Remember last time?",
            session_num=2,
        ),
        LoCoMoTurn(
            speaker="Bob",
            dia_id="D2:2",
            text="Yes, of course!",
            session_num=2,
        ),
    ]

    session1 = LoCoMoSession(
        session_num=1,
        timestamp="2023-01-15",
        turns=session1_turns,
        speaker_a="Alice",
        speaker_b="Bob",
    )
    session2 = LoCoMoSession(
        session_num=2,
        timestamp="2023-01-20",
        turns=session2_turns,
        speaker_a="Alice",
        speaker_b="Bob",
    )

    questions = [
        LoCoMoQuestion(
            question_id="conv1_q0",
            conversation_id="conv1",
            question="What did Alice say in the first session?",
            answer="Hi Bob!",
            category=QACategory.IDENTITY,
            evidence=["D1:1"],
        ),
    ]

    return LoCoMoConversation(
        sample_id="conv1",
        sessions=[session1, session2],
        questions=questions,
    )


@pytest.fixture
def sample_question() -> LoCoMoQuestion:
    """Create a sample question."""
    return LoCoMoQuestion(
        question_id="q1",
        conversation_id="conv1",
        question="Where is Alice going next week?",
        answer="Paris for a conference",
        category=QACategory.CONTEXTUAL,
        evidence=["D1:5"],
    )


@pytest.fixture
def adversarial_question() -> LoCoMoQuestion:
    """Create an adversarial question."""
    return LoCoMoQuestion(
        question_id="q2",
        conversation_id="conv1",
        question="Why is Alice going to Tokyo for the wedding?",
        answer="Alice is not going to Tokyo, she is going to Paris for a conference",
        category=QACategory.ADVERSARIAL,
        evidence=["D1:5"],
        adversarial_answer="To attend her sister's wedding",
    )


# ============================================================================
# Test LoCoMoAnswer
# ============================================================================


class TestLoCoMoAnswer:
    """Tests for LoCoMoAnswer dataclass."""

    def test_creation(self) -> None:
        """Test answer creation with required fields."""
        answer = LoCoMoAnswer(
            question_id="q1",
            answer="Test answer",
            retrieved_memories=5,
        )
        assert answer.question_id == "q1"
        assert answer.answer == "Test answer"
        assert answer.retrieved_memories == 5
        assert answer.is_abstention is False
        assert answer.category is None
        assert answer.latency_ms == 0.0
        assert answer.metadata == {}

    def test_creation_full(self) -> None:
        """Test answer creation with all fields."""
        answer = LoCoMoAnswer(
            question_id="q1",
            answer="I don't know",
            retrieved_memories=0,
            is_abstention=True,
            category=QACategory.ADVERSARIAL,
            latency_ms=150.5,
            metadata={"model": "test"},
        )
        assert answer.is_abstention is True
        assert answer.category == QACategory.ADVERSARIAL
        assert answer.latency_ms == 150.5
        assert answer.metadata == {"model": "test"}


# ============================================================================
# Test IngestionResult
# ============================================================================


class TestIngestionResult:
    """Tests for IngestionResult dataclass."""

    def test_creation(self) -> None:
        """Test result creation."""
        result = IngestionResult(
            conversation_id="conv1",
            sessions_ingested=5,
            turns_ingested=100,
            total_turns=100,
        )
        assert result.conversation_id == "conv1"
        assert result.sessions_ingested == 5
        assert result.turns_ingested == 100
        assert result.total_turns == 100
        assert result.errors == []

    def test_success_rate_full(self) -> None:
        """Test success rate when all turns ingested."""
        result = IngestionResult(
            conversation_id="conv1",
            sessions_ingested=5,
            turns_ingested=100,
            total_turns=100,
        )
        assert result.success_rate == 1.0

    def test_success_rate_partial(self) -> None:
        """Test success rate when partial ingestion."""
        result = IngestionResult(
            conversation_id="conv1",
            sessions_ingested=5,
            turns_ingested=75,
            total_turns=100,
        )
        assert result.success_rate == 0.75

    def test_success_rate_empty(self) -> None:
        """Test success rate when no turns."""
        result = IngestionResult(
            conversation_id="conv1",
            sessions_ingested=0,
            turns_ingested=0,
            total_turns=0,
        )
        assert result.success_rate == 1.0  # Convention: empty is successful


# ============================================================================
# Test LoCoMoAgent Initialization
# ============================================================================


class TestLoCoMoAgentInit:
    """Tests for LoCoMoAgent initialization."""

    def test_default_initialization(
        self, mock_adapter: MockMemoryAdapter, mock_llm: MockLLMClient
    ) -> None:
        """Test agent initializes with defaults."""
        agent = LoCoMoAgent(mock_adapter, mock_llm)
        assert agent.adapter is mock_adapter
        assert agent.ingested_conversation_count == 0
        assert agent.total_turns_ingested == 0

    def test_custom_configuration(
        self, mock_adapter: MockMemoryAdapter, mock_llm: MockLLMClient
    ) -> None:
        """Test agent with custom configuration."""
        agent = LoCoMoAgent(
            mock_adapter,
            mock_llm,
            memory_search_limit=20,
            min_relevance_score=0.5,
            system_prompt="Custom prompt",
            use_category_prompts=False,
        )
        stats = agent.get_stats()
        assert stats["memory_search_limit"] == 20
        assert stats["min_relevance_score"] == 0.5
        assert stats["use_category_prompts"] is False


# ============================================================================
# Test Turn Ingestion
# ============================================================================


class TestTurnIngestion:
    """Tests for individual turn ingestion."""

    def test_ingest_simple_turn(self, agent: LoCoMoAgent, sample_turn: LoCoMoTurn) -> None:
        """Test ingesting a simple turn."""
        result = agent.ingest_turn(sample_turn, "conv1", "2023-01-15")
        assert result is True
        assert agent.total_turns_ingested == 1

    def test_ingest_turn_content_format(
        self,
        mock_adapter: MockMemoryAdapter,
        mock_llm: MockLLMClient,
        sample_turn: LoCoMoTurn,
    ) -> None:
        """Test that turn content is formatted correctly."""
        agent = LoCoMoAgent(mock_adapter, mock_llm)
        agent.ingest_turn(sample_turn, "conv1", "2023-01-15")

        assert len(mock_adapter.add_calls) == 1
        content, metadata = mock_adapter.add_calls[0]

        # Content should include speaker prefix
        assert content.startswith("Alice:")
        assert "Paris" in content

    def test_ingest_turn_with_image(
        self,
        mock_adapter: MockMemoryAdapter,
        mock_llm: MockLLMClient,
        sample_turn_with_image: LoCoMoTurn,
    ) -> None:
        """Test ingesting a turn with image caption."""
        agent = LoCoMoAgent(mock_adapter, mock_llm)
        agent.ingest_turn(sample_turn_with_image, "conv1", "2023-01-15")

        content, metadata = mock_adapter.add_calls[0]
        assert "[Image:" in content
        assert "Eiffel Tower" in content
        assert metadata["has_image"] is True

    def test_ingest_turn_metadata(
        self,
        mock_adapter: MockMemoryAdapter,
        mock_llm: MockLLMClient,
        sample_turn: LoCoMoTurn,
    ) -> None:
        """Test that turn metadata is set correctly."""
        agent = LoCoMoAgent(mock_adapter, mock_llm)
        agent.ingest_turn(sample_turn, "conv1", "2023-01-15")

        _, metadata = mock_adapter.add_calls[0]
        assert metadata["conversation_id"] == "conv1"
        assert metadata["session_num"] == 1
        assert metadata["dia_id"] == "D1:5"
        assert metadata["speaker"] == "Alice"
        assert metadata["turn_num"] == 5
        assert metadata["timestamp"] == "2023-01-15"

    def test_ingest_turn_failure(
        self,
        mock_adapter: MockMemoryAdapter,
        mock_llm: MockLLMClient,
        sample_turn: LoCoMoTurn,
    ) -> None:
        """Test handling of ingestion failure."""
        mock_adapter.should_fail_add = True
        agent = LoCoMoAgent(mock_adapter, mock_llm)

        result = agent.ingest_turn(sample_turn, "conv1", "2023-01-15")
        assert result is False
        assert agent.total_turns_ingested == 0


# ============================================================================
# Test Session Ingestion
# ============================================================================


class TestSessionIngestion:
    """Tests for session ingestion."""

    def test_ingest_session(self, agent: LoCoMoAgent, sample_session: LoCoMoSession) -> None:
        """Test ingesting a full session."""
        count = agent.ingest_session(sample_session, "conv1")
        assert count == 3
        assert agent.total_turns_ingested == 3

    def test_session_tracked(self, agent: LoCoMoAgent, sample_session: LoCoMoSession) -> None:
        """Test that sessions are tracked properly."""
        agent.ingest_session(sample_session, "conv1")
        stats = agent.get_stats()
        assert "conv1" in stats["sessions_per_conversation"]
        assert 1 in agent._ingested_sessions["conv1"]


# ============================================================================
# Test Conversation Ingestion
# ============================================================================


class TestConversationIngestion:
    """Tests for conversation ingestion."""

    def test_ingest_conversation(
        self, agent: LoCoMoAgent, sample_conversation: LoCoMoConversation
    ) -> None:
        """Test ingesting a full conversation."""
        result = agent.ingest_conversation(sample_conversation)

        assert result.conversation_id == "conv1"
        assert result.sessions_ingested == 2
        assert result.turns_ingested == 4
        assert result.total_turns == 4
        assert result.success_rate == 1.0
        assert result.errors == []

    def test_conversation_count_tracked(
        self, agent: LoCoMoAgent, sample_conversation: LoCoMoConversation
    ) -> None:
        """Test that conversations are tracked."""
        agent.ingest_conversation(sample_conversation)
        assert agent.ingested_conversation_count == 1
        assert "conv1" in agent._ingested_conversations

    def test_ingest_all_conversations(
        self, agent: LoCoMoAgent, sample_conversation: LoCoMoConversation
    ) -> None:
        """Test ingesting multiple conversations."""
        # Create a second conversation
        conv2 = LoCoMoConversation(
            sample_id="conv2",
            sessions=[sample_conversation.sessions[0]],
            questions=[],
        )

        results = agent.ingest_all_conversations([sample_conversation, conv2])

        assert len(results) == 2
        assert "conv1" in results
        assert "conv2" in results
        assert agent.ingested_conversation_count == 2


# ============================================================================
# Test Question Answering
# ============================================================================


class TestQuestionAnswering:
    """Tests for question answering."""

    def test_answer_question_basic(
        self, agent: LoCoMoAgent, sample_question: LoCoMoQuestion
    ) -> None:
        """Test answering a basic question."""
        answer = agent.answer_question(sample_question)

        assert answer.question_id == "q1"
        assert answer.answer == "Test answer"
        assert answer.category == QACategory.CONTEXTUAL
        assert answer.is_abstention is False
        assert answer.latency_ms > 0

    def test_answer_includes_metadata(
        self, agent: LoCoMoAgent, sample_question: LoCoMoQuestion
    ) -> None:
        """Test that answer includes proper metadata."""
        answer = agent.answer_question(sample_question)

        assert "model" in answer.metadata
        assert "usage" in answer.metadata
        assert answer.metadata["conversation_id"] == "conv1"
        assert answer.metadata["evidence_dia_ids"] == ["D1:5"]
        assert answer.metadata["is_adversarial"] is False

    def test_answer_adversarial_question(
        self, agent: LoCoMoAgent, adversarial_question: LoCoMoQuestion
    ) -> None:
        """Test answering an adversarial question."""
        answer = agent.answer_question(adversarial_question)

        assert answer.category == QACategory.ADVERSARIAL
        assert answer.metadata["is_adversarial"] is True
        assert answer.metadata["adversarial_answer"] == "To attend her sister's wedding"

    def test_adversarial_prompt_modification(
        self,
        mock_adapter: MockMemoryAdapter,
        mock_llm: MockLLMClient,
        adversarial_question: LoCoMoQuestion,
    ) -> None:
        """Test that adversarial questions modify the prompt."""
        agent = LoCoMoAgent(mock_adapter, mock_llm)
        agent.answer_question(adversarial_question)

        # Check the system prompt was modified
        call = mock_llm.calls[0]
        assert "premise" in call["system"].lower()

    def test_abstention_detection(
        self,
        mock_adapter: MockMemoryAdapter,
        sample_question: LoCoMoQuestion,
    ) -> None:
        """Test abstention detection in answers."""
        llm = MockLLMClient(response="I don't know the answer to this question.")
        agent = LoCoMoAgent(mock_adapter, llm)

        answer = agent.answer_question(sample_question)
        assert answer.is_abstention is True

    def test_abstention_detection_variants(
        self,
        mock_adapter: MockMemoryAdapter,
        sample_question: LoCoMoQuestion,
    ) -> None:
        """Test various abstention phrases are detected."""
        abstention_phrases = [
            "I cannot find this information",
            "This was not mentioned in the conversation",
            "The premise is incorrect",
            "Unable to determine from the context",
        ]

        for phrase in abstention_phrases:
            llm = MockLLMClient(response=phrase)
            agent = LoCoMoAgent(mock_adapter, llm)
            answer = agent.answer_question(sample_question)
            assert answer.is_abstention is True, f"Failed for: {phrase}"

    def test_answer_uses_metadata_filter(
        self,
        mock_adapter: MockMemoryAdapter,
        mock_llm: MockLLMClient,
        sample_question: LoCoMoQuestion,
    ) -> None:
        """Test that search uses metadata filter for conversation."""
        agent = LoCoMoAgent(mock_adapter, mock_llm)
        agent.answer_question(sample_question)

        assert len(mock_adapter.search_calls) == 1
        search_call = mock_adapter.search_calls[0]
        assert search_call["metadata_filter"]["conversation_id"] == "conv1"

    def test_answer_with_evidence_sessions(
        self,
        mock_adapter: MockMemoryAdapter,
        mock_llm: MockLLMClient,
        sample_question: LoCoMoQuestion,
    ) -> None:
        """Test oracle mode using evidence sessions."""
        agent = LoCoMoAgent(mock_adapter, mock_llm)
        agent.answer_question(sample_question, use_evidence_sessions=True)

        search_call = mock_adapter.search_calls[0]
        assert "session_num" in search_call["metadata_filter"]

    def test_answer_all_questions(
        self, agent: LoCoMoAgent, sample_question: LoCoMoQuestion
    ) -> None:
        """Test answering multiple questions."""
        questions = [sample_question, sample_question]  # Same question twice
        answers = agent.answer_all_questions(questions)

        assert len(answers) == 2
        assert all(a.question_id == "q1" for a in answers)


# ============================================================================
# Test Category-Specific Behavior
# ============================================================================


class TestCategoryBehavior:
    """Tests for category-specific prompt handling."""

    @pytest.mark.parametrize(
        "category,expected_keyword",
        [
            (QACategory.IDENTITY, "identity"),
            (QACategory.TEMPORAL, "when"),
            (QACategory.INFERENCE, "infer"),
            (QACategory.CONTEXTUAL, "detail"),
            (QACategory.ADVERSARIAL, "premise"),
        ],
    )
    def test_category_prompts_applied(
        self,
        mock_adapter: MockMemoryAdapter,
        mock_llm: MockLLMClient,
        category: QACategory,
        expected_keyword: str,
    ) -> None:
        """Test that category-specific prompts are applied."""
        agent = LoCoMoAgent(mock_adapter, mock_llm, use_category_prompts=True)

        question = LoCoMoQuestion(
            question_id="q1",
            conversation_id="conv1",
            question="Test question",
            answer="Test answer",
            category=category,
        )

        agent.answer_question(question)

        system_prompt = mock_llm.calls[0]["system"].lower()
        assert expected_keyword in system_prompt

    def test_category_prompts_disabled(
        self,
        mock_adapter: MockMemoryAdapter,
        mock_llm: MockLLMClient,
    ) -> None:
        """Test that category prompts can be disabled."""
        agent = LoCoMoAgent(mock_adapter, mock_llm, use_category_prompts=False)

        question = LoCoMoQuestion(
            question_id="q1",
            conversation_id="conv1",
            question="Test question",
            answer="Test answer",
            category=QACategory.TEMPORAL,
        )

        agent.answer_question(question)

        system_prompt = mock_llm.calls[0]["system"].lower()
        # Should not have temporal-specific text when disabled
        assert "when events occurred" not in system_prompt


# ============================================================================
# Test Memory Management
# ============================================================================


class TestMemoryManagement:
    """Tests for memory management operations."""

    def test_clear_memory(
        self, agent: LoCoMoAgent, sample_conversation: LoCoMoConversation
    ) -> None:
        """Test clearing all memories."""
        agent.ingest_conversation(sample_conversation)
        assert agent.ingested_conversation_count == 1

        result = agent.clear_memory()
        assert result is True
        assert agent.ingested_conversation_count == 0
        assert agent.total_turns_ingested == 0

    def test_get_stats(self, agent: LoCoMoAgent, sample_conversation: LoCoMoConversation) -> None:
        """Test getting agent statistics."""
        agent.ingest_conversation(sample_conversation)
        stats = agent.get_stats()

        assert stats["ingested_conversations"] == 1
        assert stats["total_sessions"] == 2
        assert stats["total_turns"] == 4
        assert "memory_count" in stats  # From adapter


# ============================================================================
# Test Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_conversation(self, agent: LoCoMoAgent) -> None:
        """Test ingesting an empty conversation."""
        conv = LoCoMoConversation(
            sample_id="empty",
            sessions=[],
            questions=[],
        )
        result = agent.ingest_conversation(conv)

        assert result.sessions_ingested == 0
        assert result.turns_ingested == 0
        assert result.success_rate == 1.0

    def test_question_without_conversation(self, agent: LoCoMoAgent) -> None:
        """Test answering question with no conversation_id."""
        question = LoCoMoQuestion(
            question_id="q1",
            conversation_id="",
            question="Test?",
            answer="Test",
            category=QACategory.IDENTITY,
        )

        answer = agent.answer_question(question)
        assert answer.answer == "Test answer"  # From mock LLM

    def test_no_memories_found(
        self,
        mock_llm: MockLLMClient,
        sample_question: LoCoMoQuestion,
    ) -> None:
        """Test answering when no memories are found."""
        # Create adapter that returns no memories
        adapter = MockMemoryAdapter()
        agent = LoCoMoAgent(adapter, mock_llm)

        answer = agent.answer_question(sample_question)

        # Should still generate an answer
        assert answer.retrieved_memories == 0
        assert answer.answer == "Test answer"
