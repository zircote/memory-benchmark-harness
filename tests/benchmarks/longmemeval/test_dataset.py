"""Tests for LongMemEval dataset loader and data classes."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.benchmarks.longmemeval.dataset import (
    LongMemEvalDataset,
    LongMemEvalQuestion,
    LongMemEvalSession,
    Message,
    QuestionType,
    load_longmemeval_from_file,
)


class TestQuestionType:
    """Tests for QuestionType enum."""

    def test_from_string_hyphen_format(self) -> None:
        """Test parsing hyphenated format."""
        assert QuestionType.from_string("single-session-user") == QuestionType.SINGLE_SESSION_USER
        assert QuestionType.from_string("temporal-reasoning") == QuestionType.TEMPORAL_REASONING
        assert QuestionType.from_string("knowledge-update") == QuestionType.KNOWLEDGE_UPDATE

    def test_from_string_underscore_format(self) -> None:
        """Test parsing underscore format."""
        assert QuestionType.from_string("single_session_user") == QuestionType.SINGLE_SESSION_USER
        assert QuestionType.from_string("multi_session") == QuestionType.MULTI_SESSION

    def test_from_string_case_insensitive(self) -> None:
        """Test case insensitivity."""
        assert QuestionType.from_string("SINGLE-SESSION-USER") == QuestionType.SINGLE_SESSION_USER
        assert QuestionType.from_string("Multi_Session") == QuestionType.MULTI_SESSION

    def test_from_string_unknown_raises(self) -> None:
        """Test that unknown type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown question type"):
            QuestionType.from_string("unknown-type")

    def test_all_types_have_values(self) -> None:
        """Test all question types have valid string values."""
        for qtype in QuestionType:
            assert isinstance(qtype.value, str)
            assert "-" in qtype.value  # All use hyphenated format


class TestMessage:
    """Tests for Message data class."""

    def test_create_user_message(self) -> None:
        """Test creating a user message."""
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.timestamp is None

    def test_create_assistant_message(self) -> None:
        """Test creating an assistant message."""
        msg = Message(role="assistant", content="Hi there!", timestamp="2024-01-01T10:00:00")
        assert msg.role == "assistant"
        assert msg.content == "Hi there!"
        assert msg.timestamp == "2024-01-01T10:00:00"

    def test_invalid_role_raises(self) -> None:
        """Test that invalid role raises ValueError."""
        with pytest.raises(ValueError, match="role must be 'user' or 'assistant'"):
            Message(role="system", content="test")

    def test_message_is_frozen(self) -> None:
        """Test that Message is immutable."""
        msg = Message(role="user", content="Hello")
        with pytest.raises(AttributeError):
            msg.content = "Modified"  # type: ignore[misc]


class TestLongMemEvalQuestion:
    """Tests for LongMemEvalQuestion data class."""

    def test_create_basic_question(self) -> None:
        """Test creating a basic question."""
        q = LongMemEvalQuestion(
            question_id="q1",
            question_text="What is the user's name?",
            ground_truth=["Alice"],
            question_type=QuestionType.SINGLE_SESSION_USER,
            relevant_session_ids=["s1"],
        )
        assert q.question_id == "q1"
        assert q.question_text == "What is the user's name?"
        assert q.ground_truth == ["Alice"]
        assert q.question_type == QuestionType.SINGLE_SESSION_USER
        assert q.relevant_session_ids == ["s1"]
        assert q.is_abstention is False

    def test_abstention_question(self) -> None:
        """Test creating an abstention question."""
        q = LongMemEvalQuestion(
            question_id="q1_abs",
            question_text="What is unknown?",
            ground_truth=["I don't know"],
            question_type=QuestionType.MULTI_SESSION,
            relevant_session_ids=[],
            is_abstention=True,
        )
        assert q.is_abstention is True

    def test_multiple_ground_truths(self) -> None:
        """Test question with multiple valid answers."""
        q = LongMemEvalQuestion(
            question_id="q2",
            question_text="What are the user's hobbies?",
            ground_truth=["reading", "coding", "gaming"],
            question_type=QuestionType.SINGLE_SESSION_USER,
            relevant_session_ids=["s1", "s2"],
        )
        assert len(q.ground_truth) == 3

    def test_empty_ground_truth_raises(self) -> None:
        """Test that empty ground truth raises ValueError."""
        with pytest.raises(ValueError, match="ground_truth must have at least one answer"):
            LongMemEvalQuestion(
                question_id="q1",
                question_text="Test?",
                ground_truth=[],
                question_type=QuestionType.MULTI_SESSION,
                relevant_session_ids=[],
            )

    def test_question_is_frozen(self) -> None:
        """Test that Question is immutable."""
        q = LongMemEvalQuestion(
            question_id="q1",
            question_text="Test?",
            ground_truth=["answer"],
            question_type=QuestionType.MULTI_SESSION,
            relevant_session_ids=[],
        )
        with pytest.raises(AttributeError):
            q.question_id = "q2"  # type: ignore[misc]


class TestLongMemEvalSession:
    """Tests for LongMemEvalSession data class."""

    def test_create_session(self) -> None:
        """Test creating a session with messages."""
        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there!"),
        ]
        session = LongMemEvalSession(
            session_id="s1",
            messages=messages,
            timestamp="2024-01-01",
        )
        assert session.session_id == "s1"
        assert len(session.messages) == 2
        assert session.timestamp == "2024-01-01"

    def test_message_count_property(self) -> None:
        """Test message_count property."""
        session = LongMemEvalSession(
            session_id="s1",
            messages=[
                Message(role="user", content="Hello"),
                Message(role="assistant", content="Hi!"),
                Message(role="user", content="How are you?"),
            ],
        )
        assert session.message_count == 3

    def test_token_estimate_property(self) -> None:
        """Test token_estimate property."""
        session = LongMemEvalSession(
            session_id="s1",
            messages=[
                Message(role="user", content="Hello world"),  # 11 chars
                Message(role="assistant", content="Hi there friend"),  # 15 chars
            ],
        )
        # Total: 26 chars, estimate: 26 // 4 = 6
        assert session.token_estimate == 6

    def test_empty_session(self) -> None:
        """Test session with no messages."""
        session = LongMemEvalSession(session_id="empty", messages=[])
        assert session.message_count == 0
        assert session.token_estimate == 0


class TestLongMemEvalDataset:
    """Tests for LongMemEvalDataset data class."""

    @pytest.fixture
    def sample_dataset(self) -> LongMemEvalDataset:
        """Create a sample dataset for testing."""
        sessions = [
            LongMemEvalSession(
                session_id="s1",
                messages=[
                    Message(role="user", content="My name is Alice"),
                    Message(role="assistant", content="Nice to meet you, Alice!"),
                ],
            ),
            LongMemEvalSession(
                session_id="s2",
                messages=[
                    Message(role="user", content="I like reading books"),
                    Message(role="assistant", content="That's a great hobby!"),
                ],
            ),
        ]
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
                ground_truth=["reading", "reading books"],
                question_type=QuestionType.SINGLE_SESSION_USER,
                relevant_session_ids=["s2"],
            ),
            LongMemEvalQuestion(
                question_id="q3_abs",
                question_text="What is the user's age?",
                ground_truth=["unknown"],
                question_type=QuestionType.MULTI_SESSION,
                relevant_session_ids=[],
                is_abstention=True,
            ),
        ]
        return LongMemEvalDataset(
            subset="S",
            sessions=sessions,
            questions=questions,
            metadata={"source": "test"},
        )

    def test_session_count(self, sample_dataset: LongMemEvalDataset) -> None:
        """Test session_count property."""
        assert sample_dataset.session_count == 2

    def test_question_count(self, sample_dataset: LongMemEvalDataset) -> None:
        """Test question_count property."""
        assert sample_dataset.question_count == 3

    def test_total_messages(self, sample_dataset: LongMemEvalDataset) -> None:
        """Test total_messages property."""
        assert sample_dataset.total_messages == 4  # 2 per session

    def test_get_session_found(self, sample_dataset: LongMemEvalDataset) -> None:
        """Test get_session with existing session."""
        session = sample_dataset.get_session("s1")
        assert session is not None
        assert session.session_id == "s1"

    def test_get_session_not_found(self, sample_dataset: LongMemEvalDataset) -> None:
        """Test get_session with non-existing session."""
        session = sample_dataset.get_session("nonexistent")
        assert session is None

    def test_questions_by_type(self, sample_dataset: LongMemEvalDataset) -> None:
        """Test filtering questions by type."""
        user_questions = sample_dataset.questions_by_type(QuestionType.SINGLE_SESSION_USER)
        assert len(user_questions) == 2

        multi_questions = sample_dataset.questions_by_type(QuestionType.MULTI_SESSION)
        assert len(multi_questions) == 1

    def test_get_stats(self, sample_dataset: LongMemEvalDataset) -> None:
        """Test get_stats method."""
        stats = sample_dataset.get_stats()

        assert stats["subset"] == "S"
        assert stats["session_count"] == 2
        assert stats["question_count"] == 3
        assert stats["abstention_questions"] == 1
        assert "single-session-user" in stats["questions_by_type"]
        assert stats["questions_by_type"]["single-session-user"] == 2


class TestLoadFromFile:
    """Tests for load_longmemeval_from_file function."""

    def test_load_list_format(self) -> None:
        """Test loading dataset in list format."""
        data = [
            {
                "session_id": "s1",
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi"},
                ],
            },
            {
                "question_id": "q1",
                "question": "Test question?",
                "answer": "Test answer",
                "question_type": "single-session-user",
                "relevant_sessions": ["s1"],
            },
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()

            dataset = load_longmemeval_from_file(f.name, subset="S")

        assert dataset.session_count == 1
        assert dataset.question_count == 1
        assert dataset.sessions[0].session_id == "s1"
        assert dataset.questions[0].question_text == "Test question?"

        Path(f.name).unlink()

    def test_load_dict_format(self) -> None:
        """Test loading dataset in dict format with sessions and questions keys."""
        data = {
            "sessions": [
                {
                    "session_id": "s1",
                    "messages": [{"role": "user", "content": "Hello"}],
                }
            ],
            "questions": [
                {
                    "question_id": "q1",
                    "question": "Test?",
                    "answer": ["answer1", "answer2"],
                    "question_type": "multi-session",
                    "relevant_sessions": [],
                }
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()

            dataset = load_longmemeval_from_file(f.name, subset="M")

        assert dataset.subset == "M"
        assert dataset.session_count == 1
        assert dataset.question_count == 1
        assert len(dataset.questions[0].ground_truth) == 2

        Path(f.name).unlink()

    def test_load_nonexistent_file_raises(self) -> None:
        """Test that loading nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Dataset file not found"):
            load_longmemeval_from_file("/nonexistent/path.json")

    def test_auto_generates_question_ids(self) -> None:
        """Test that missing question_ids are auto-generated."""
        data = [
            {
                "question": "Test?",
                "answer": "answer",
                "question_type": "multi-session",
                "relevant_sessions": [],
            }
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()

            dataset = load_longmemeval_from_file(f.name)

        assert dataset.questions[0].question_id == "q_0"

        Path(f.name).unlink()

    def test_handles_alternative_field_names(self) -> None:
        """Test handling of alternative field names (query vs question)."""
        data = [
            {
                "question_id": "q1",
                "query": "Alternative field name?",  # 'query' instead of 'question'
                "ground_truth": ["answer"],  # 'ground_truth' instead of 'answer'
                "type": "temporal-reasoning",  # 'type' instead of 'question_type'
                "session_ids": ["s1"],  # 'session_ids' instead of 'relevant_sessions'
            }
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()

            dataset = load_longmemeval_from_file(f.name)

        assert dataset.questions[0].question_text == "Alternative field name?"
        assert dataset.questions[0].ground_truth == ["answer"]
        assert dataset.questions[0].question_type == QuestionType.TEMPORAL_REASONING
        assert dataset.questions[0].relevant_session_ids == ["s1"]

        Path(f.name).unlink()


class TestDatasetMetadata:
    """Tests for dataset metadata handling."""

    def test_metadata_preserved(self) -> None:
        """Test that metadata is preserved in dataset."""
        dataset = LongMemEvalDataset(
            subset="S",
            sessions=[],
            questions=[],
            metadata={"source": "test", "version": "1.0"},
        )
        assert dataset.metadata["source"] == "test"
        assert dataset.metadata["version"] == "1.0"

    def test_session_metadata(self) -> None:
        """Test session metadata handling."""
        session = LongMemEvalSession(
            session_id="s1",
            messages=[],
            metadata={"custom": "value"},
        )
        assert session.metadata["custom"] == "value"

    def test_question_metadata(self) -> None:
        """Test question metadata handling."""
        question = LongMemEvalQuestion(
            question_id="q1",
            question_text="Test?",
            ground_truth=["answer"],
            question_type=QuestionType.MULTI_SESSION,
            relevant_session_ids=[],
            metadata={"difficulty": "hard"},
        )
        assert question.metadata["difficulty"] == "hard"
