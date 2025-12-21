"""Tests for LongMemEval agent wrapper."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from src.adapters.mock import MockAdapter
from src.benchmarks.longmemeval.dataset import (
    LongMemEvalQuestion,
    LongMemEvalSession,
    Message,
    QuestionType,
)
from src.benchmarks.longmemeval.wrapper import (
    AgentAnswer,
    LLMResponse,
    LongMemEvalAgent,
)


@dataclass
class MockLLMClient:
    """Mock LLM client for testing."""

    default_response: str = "Test response"
    responses: dict[str, str] = field(default_factory=dict)
    call_count: int = 0
    last_system: str = ""
    last_messages: list[dict[str, str]] = field(default_factory=list)

    def complete(
        self,
        system: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
    ) -> LLMResponse:
        """Return a mock response."""
        self.call_count += 1
        self.last_system = system
        self.last_messages = messages

        # Check for custom response based on question content
        user_content = messages[0]["content"] if messages else ""
        for key, response in self.responses.items():
            if key in user_content:
                return LLMResponse(
                    content=response,
                    model="mock-model",
                    usage={"prompt_tokens": 100, "completion_tokens": 50},
                )

        return LLMResponse(
            content=self.default_response,
            model="mock-model",
            usage={"prompt_tokens": 100, "completion_tokens": 50},
        )


class TestLLMResponse:
    """Tests for LLMResponse data class."""

    def test_create_response(self) -> None:
        """Test creating an LLM response."""
        response = LLMResponse(
            content="Hello!",
            model="gpt-4o",
            usage={"prompt_tokens": 10, "completion_tokens": 5},
        )
        assert response.content == "Hello!"
        assert response.model == "gpt-4o"
        assert response.usage["prompt_tokens"] == 10

    def test_default_values(self) -> None:
        """Test default values for optional fields."""
        response = LLMResponse(content="Hello!")
        assert response.model == ""
        assert response.usage == {}


class TestAgentAnswer:
    """Tests for AgentAnswer data class."""

    def test_create_answer(self) -> None:
        """Test creating an agent answer."""
        answer = AgentAnswer(
            question_id="q1",
            answer="The user's name is Alice.",
            retrieved_memories=5,
            is_abstention=False,
        )
        assert answer.question_id == "q1"
        assert answer.answer == "The user's name is Alice."
        assert answer.retrieved_memories == 5
        assert answer.is_abstention is False
        assert answer.metadata == {}

    def test_with_metadata(self) -> None:
        """Test answer with metadata."""
        answer = AgentAnswer(
            question_id="q1",
            answer="I don't know.",
            retrieved_memories=0,
            is_abstention=True,
            metadata={"model": "gpt-4o", "question_type": "multi-session"},
        )
        assert answer.is_abstention is True
        assert answer.metadata["model"] == "gpt-4o"


class TestLongMemEvalAgentInit:
    """Tests for LongMemEvalAgent initialization."""

    def test_init_basic(self) -> None:
        """Test basic initialization."""
        adapter = MockAdapter()
        llm = MockLLMClient()
        agent = LongMemEvalAgent(adapter, llm)

        assert agent.adapter is adapter
        assert agent.ingested_session_count == 0

    def test_init_with_custom_config(self) -> None:
        """Test initialization with custom configuration."""
        adapter = MockAdapter()
        llm = MockLLMClient()
        agent = LongMemEvalAgent(
            adapter,
            llm,
            memory_search_limit=20,
            min_relevance_score=0.5,
            system_prompt="Custom prompt",
        )

        stats = agent.get_stats()
        assert stats["memory_search_limit"] == 20
        assert stats["min_relevance_score"] == 0.5


class TestLongMemEvalAgentIngest:
    """Tests for session ingestion."""

    @pytest.fixture
    def agent(self) -> LongMemEvalAgent:
        """Create an agent with mock dependencies."""
        adapter = MockAdapter()
        llm = MockLLMClient()
        return LongMemEvalAgent(adapter, llm)

    @pytest.fixture
    def sample_session(self) -> LongMemEvalSession:
        """Create a sample session for testing."""
        return LongMemEvalSession(
            session_id="s1",
            messages=[
                Message(role="user", content="My name is Alice"),
                Message(role="assistant", content="Nice to meet you, Alice!"),
                Message(role="user", content="I love reading books"),
            ],
            timestamp="2024-01-01",
        )

    def test_ingest_session(
        self, agent: LongMemEvalAgent, sample_session: LongMemEvalSession
    ) -> None:
        """Test ingesting a single session."""
        count = agent.ingest_session(sample_session)

        assert count == 3
        assert agent.ingested_session_count == 1
        assert "s1" in agent._ingested_sessions

    def test_ingest_session_stores_metadata(
        self, agent: LongMemEvalAgent, sample_session: LongMemEvalSession
    ) -> None:
        """Test that ingested messages have correct metadata."""
        agent.ingest_session(sample_session)

        # Search for the ingested content
        memories = agent.adapter.search("Alice", limit=10)
        assert len(memories) > 0

        # Check metadata on first memory
        first_memory = memories[0]
        assert first_memory.metadata["session_id"] == "s1"
        assert first_memory.metadata["role"] in ("user", "assistant")

    def test_ingest_all_sessions(self, agent: LongMemEvalAgent) -> None:
        """Test ingesting multiple sessions."""
        sessions = [
            LongMemEvalSession(
                session_id="s1",
                messages=[Message(role="user", content="Hello")],
            ),
            LongMemEvalSession(
                session_id="s2",
                messages=[
                    Message(role="user", content="Hi"),
                    Message(role="assistant", content="Hello!"),
                ],
            ),
        ]

        results = agent.ingest_all_sessions(sessions)

        assert results["s1"] == 1
        assert results["s2"] == 2
        assert agent.ingested_session_count == 2

    def test_ingest_empty_session(self, agent: LongMemEvalAgent) -> None:
        """Test ingesting an empty session."""
        session = LongMemEvalSession(session_id="empty", messages=[])
        count = agent.ingest_session(session)

        assert count == 0
        assert agent.ingested_session_count == 1


class TestLongMemEvalAgentAnswer:
    """Tests for question answering."""

    @pytest.fixture
    def agent_with_data(self) -> LongMemEvalAgent:
        """Create an agent with ingested data."""
        adapter = MockAdapter()
        llm = MockLLMClient(
            default_response="The user's name is Alice.",
            responses={
                "hobby": "The user enjoys reading books.",
                "unknown": "I don't know based on the conversation history.",
            },
        )
        agent = LongMemEvalAgent(adapter, llm)

        # Ingest sample session
        session = LongMemEvalSession(
            session_id="s1",
            messages=[
                Message(role="user", content="My name is Alice"),
                Message(role="assistant", content="Nice to meet you!"),
                Message(role="user", content="I enjoy reading books"),
            ],
        )
        agent.ingest_session(session)

        return agent

    def test_answer_question(self, agent_with_data: LongMemEvalAgent) -> None:
        """Test answering a basic question."""
        # Note: MockAdapter checks if query is SUBSTRING of stored content.
        # Ingested content: "user: My name is Alice"
        # Query must be a substring, so we use "name" which appears in the content.
        question = LongMemEvalQuestion(
            question_id="q1",
            question_text="name",  # Simple substring that matches stored content
            ground_truth=["Alice"],
            question_type=QuestionType.SINGLE_SESSION_USER,
            relevant_session_ids=["s1"],
        )

        # Disable session filtering for MockAdapter (it doesn't support list-in checks)
        answer = agent_with_data.answer_question(question, relevant_sessions_only=False)

        assert answer.question_id == "q1"
        assert "Alice" in answer.answer
        assert answer.retrieved_memories > 0
        assert answer.is_abstention is False

    def test_answer_question_detects_abstention(self, agent_with_data: LongMemEvalAgent) -> None:
        """Test that abstention is detected in answers."""
        question = LongMemEvalQuestion(
            question_id="q_abs",
            question_text="What is the user's unknown attribute?",
            ground_truth=["unknown"],
            question_type=QuestionType.MULTI_SESSION,
            relevant_session_ids=[],
            is_abstention=True,
        )

        answer = agent_with_data.answer_question(question)

        assert answer.is_abstention is True
        assert answer.metadata["expected_abstention"] is True

    def test_answer_question_metadata(self, agent_with_data: LongMemEvalAgent) -> None:
        """Test answer metadata includes question context."""
        question = LongMemEvalQuestion(
            question_id="q1",
            question_text="What hobby does the user have?",
            ground_truth=["reading"],
            question_type=QuestionType.SINGLE_SESSION_USER,
            relevant_session_ids=["s1"],
        )

        answer = agent_with_data.answer_question(question)

        assert answer.metadata["model"] == "mock-model"
        assert answer.metadata["question_type"] == "single-session-user"
        assert "relevant_sessions" in answer.metadata

    def test_answer_all_questions(self, agent_with_data: LongMemEvalAgent) -> None:
        """Test answering multiple questions."""
        questions = [
            LongMemEvalQuestion(
                question_id="q1",
                question_text="What is the user's name?",
                ground_truth=["Alice"],
                question_type=QuestionType.SINGLE_SESSION_USER,
                relevant_session_ids=["s1"],
            ),
            LongMemEvalQuestion(
                question_id="q2",
                question_text="What hobby does the user have?",
                ground_truth=["reading"],
                question_type=QuestionType.SINGLE_SESSION_USER,
                relevant_session_ids=["s1"],
            ),
        ]

        answers = agent_with_data.answer_all_questions(questions)

        assert len(answers) == 2
        assert answers[0].question_id == "q1"
        assert answers[1].question_id == "q2"


class TestAbstentionDetection:
    """Tests for abstention detection logic."""

    @pytest.fixture
    def agent(self) -> LongMemEvalAgent:
        """Create an agent for testing."""
        return LongMemEvalAgent(MockAdapter(), MockLLMClient())

    @pytest.mark.parametrize(
        "answer,expected",
        [
            ("The user's name is Alice.", False),
            ("I don't know based on the conversation.", True),
            ("This was not mentioned in our conversations.", True),
            ("I cannot find this information.", True),
            ("The information is not available in the context.", True),
            ("Based on session 1, the answer is yes.", False),
            ("I'm unable to find any reference to that.", True),
            ("It wasn't discussed in the conversations.", True),
            ("The user clearly stated their preference.", False),
            ("There is no information about that topic.", True),
            ("Not specified in the conversation history.", True),
        ],
    )
    def test_abstention_detection(
        self, agent: LongMemEvalAgent, answer: str, expected: bool
    ) -> None:
        """Test various abstention patterns are detected."""
        result = agent._detect_abstention(answer)
        assert result == expected, f"Expected {expected} for: {answer}"


class TestAgentClearAndStats:
    """Tests for memory clearing and statistics."""

    def test_clear_memory(self) -> None:
        """Test clearing all memories."""
        adapter = MockAdapter()
        llm = MockLLMClient()
        agent = LongMemEvalAgent(adapter, llm)

        # Ingest some data
        session = LongMemEvalSession(
            session_id="s1",
            messages=[Message(role="user", content="Hello")],
        )
        agent.ingest_session(session)

        assert agent.ingested_session_count == 1

        # Clear
        success = agent.clear_memory()

        assert success is True
        assert agent.ingested_session_count == 0

    def test_get_stats(self) -> None:
        """Test getting agent statistics."""
        adapter = MockAdapter()
        llm = MockLLMClient()
        agent = LongMemEvalAgent(
            adapter,
            llm,
            memory_search_limit=15,
            min_relevance_score=0.3,
        )

        # Ingest a session
        session = LongMemEvalSession(
            session_id="s1",
            messages=[Message(role="user", content="Hello")],
        )
        agent.ingest_session(session)

        stats = agent.get_stats()

        assert stats["ingested_sessions"] == 1
        assert stats["memory_search_limit"] == 15
        assert stats["min_relevance_score"] == 0.3
        assert "type" in stats  # From adapter stats
        assert stats["memory_count"] == 1


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_answer_with_no_memories(self) -> None:
        """Test answering when no memories are retrieved."""
        adapter = MockAdapter()
        llm = MockLLMClient(default_response="I don't know.")
        agent = LongMemEvalAgent(adapter, llm)

        question = LongMemEvalQuestion(
            question_id="q1",
            question_text="What is anything?",
            ground_truth=["unknown"],
            question_type=QuestionType.MULTI_SESSION,
            relevant_session_ids=[],
        )

        answer = agent.answer_question(question)

        assert answer.retrieved_memories == 0
        assert answer.is_abstention is True

    def test_answer_with_empty_relevant_sessions(self) -> None:
        """Test answering with empty relevant_session_ids."""
        adapter = MockAdapter()
        llm = MockLLMClient(default_response="Answer from all sessions")
        agent = LongMemEvalAgent(adapter, llm)

        # Ingest data
        session = LongMemEvalSession(
            session_id="s1",
            messages=[Message(role="user", content="Important info")],
        )
        agent.ingest_session(session)

        question = LongMemEvalQuestion(
            question_id="q1",
            question_text="What info?",
            ground_truth=["Important info"],
            question_type=QuestionType.MULTI_SESSION,
            relevant_session_ids=[],  # Empty - search all
        )

        answer = agent.answer_question(question, relevant_sessions_only=False)

        # Should still search across all sessions
        assert answer.answer == "Answer from all sessions"

    def test_session_with_timestamps(self) -> None:
        """Test ingesting session with message timestamps."""
        adapter = MockAdapter()
        llm = MockLLMClient()
        agent = LongMemEvalAgent(adapter, llm)

        session = LongMemEvalSession(
            session_id="s1",
            messages=[
                Message(
                    role="user",
                    content="Hello",
                    timestamp="2024-01-01T10:00:00",
                ),
            ],
            timestamp="2024-01-01",
        )

        count = agent.ingest_session(session)
        assert count == 1

        # Check timestamp in metadata
        memories = agent.adapter.search("Hello", limit=1)
        assert len(memories) == 1
        assert memories[0].metadata["timestamp"] == "2024-01-01T10:00:00"
        assert memories[0].metadata["session_timestamp"] == "2024-01-01"
