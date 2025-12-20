"""Tests for LoCoMo dataset loader and data classes."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.benchmarks.locomo.dataset import (
    LoCoMoConversation,
    LoCoMoDataset,
    LoCoMoQuestion,
    LoCoMoSession,
    LoCoMoTurn,
    QACategory,
    load_locomo_from_file,
)


class TestQACategory:
    """Tests for QACategory enum."""

    def test_all_categories_have_values(self) -> None:
        """Test all 5 categories have distinct values 1-5."""
        values = {cat.value for cat in QACategory}
        assert values == {1, 2, 3, 4, 5}

    def test_from_int_valid_values(self) -> None:
        """Test from_int works for all valid categories."""
        assert QACategory.from_int(1) == QACategory.IDENTITY
        assert QACategory.from_int(2) == QACategory.TEMPORAL
        assert QACategory.from_int(3) == QACategory.INFERENCE
        assert QACategory.from_int(4) == QACategory.CONTEXTUAL
        assert QACategory.from_int(5) == QACategory.ADVERSARIAL

    def test_from_int_invalid_value_raises(self) -> None:
        """Test from_int raises ValueError for invalid values."""
        with pytest.raises(ValueError, match="Unknown QA category: 0"):
            QACategory.from_int(0)
        with pytest.raises(ValueError, match="Unknown QA category: 6"):
            QACategory.from_int(6)
        with pytest.raises(ValueError, match="Unknown QA category: -1"):
            QACategory.from_int(-1)

    def test_description_returns_string(self) -> None:
        """Test description returns meaningful text for all categories."""
        assert "identity" in QACategory.IDENTITY.description().lower()
        assert "temporal" in QACategory.TEMPORAL.description().lower()
        assert "inference" in QACategory.INFERENCE.description().lower()
        assert "contextual" in QACategory.CONTEXTUAL.description().lower()
        assert "adversarial" in QACategory.ADVERSARIAL.description().lower()

    def test_all_categories_have_descriptions(self) -> None:
        """Test all categories have non-empty descriptions."""
        for cat in QACategory:
            desc = cat.description()
            assert isinstance(desc, str)
            assert len(desc) > 0


class TestLoCoMoTurn:
    """Tests for LoCoMoTurn data class."""

    def test_create_basic_turn(self) -> None:
        """Test creating a basic turn."""
        turn = LoCoMoTurn(
            speaker="Alice",
            dia_id="D1:5",
            text="Hello, how are you?",
            session_num=1,
        )
        assert turn.speaker == "Alice"
        assert turn.dia_id == "D1:5"
        assert turn.text == "Hello, how are you?"
        assert turn.session_num == 1
        assert turn.img_url is None
        assert turn.img_caption is None

    def test_create_turn_with_image(self) -> None:
        """Test creating a turn with image data."""
        turn = LoCoMoTurn(
            speaker="Bob",
            dia_id="D2:10",
            text="Check out this photo!",
            session_num=2,
            img_url="https://example.com/image.jpg",
            img_caption="A sunset over the ocean",
        )
        assert turn.img_url == "https://example.com/image.jpg"
        assert turn.img_caption == "A sunset over the ocean"

    def test_turn_num_property(self) -> None:
        """Test extracting turn number from dia_id."""
        turn = LoCoMoTurn(speaker="A", dia_id="D1:5", text="Hi", session_num=1)
        assert turn.turn_num == 5

        turn2 = LoCoMoTurn(speaker="A", dia_id="D25:123", text="Hi", session_num=25)
        assert turn2.turn_num == 123

    def test_turn_num_without_colon(self) -> None:
        """Test turn_num returns 0 for malformed dia_id."""
        turn = LoCoMoTurn(speaker="A", dia_id="invalid", text="Hi", session_num=1)
        assert turn.turn_num == 0

    def test_turn_is_frozen(self) -> None:
        """Test that LoCoMoTurn is immutable."""
        turn = LoCoMoTurn(speaker="A", dia_id="D1:1", text="Hi", session_num=1)
        with pytest.raises(AttributeError):
            turn.text = "Modified"  # type: ignore[misc]


class TestLoCoMoSession:
    """Tests for LoCoMoSession data class."""

    def test_create_session(self) -> None:
        """Test creating a session."""
        turns = [
            LoCoMoTurn(speaker="Alice", dia_id="D1:1", text="Hello", session_num=1),
            LoCoMoTurn(speaker="Bob", dia_id="D1:2", text="Hi there!", session_num=1),
        ]
        session = LoCoMoSession(
            session_num=1,
            timestamp="2023-01-15 10:00:00",
            turns=turns,
            speaker_a="Alice",
            speaker_b="Bob",
        )
        assert session.session_num == 1
        assert session.timestamp == "2023-01-15 10:00:00"
        assert len(session.turns) == 2
        assert session.speaker_a == "Alice"
        assert session.speaker_b == "Bob"

    def test_turn_count_property(self) -> None:
        """Test turn_count property."""
        turns = [
            LoCoMoTurn(speaker="A", dia_id="D1:1", text="One", session_num=1),
            LoCoMoTurn(speaker="B", dia_id="D1:2", text="Two", session_num=1),
            LoCoMoTurn(speaker="A", dia_id="D1:3", text="Three", session_num=1),
        ]
        session = LoCoMoSession(
            session_num=1,
            timestamp="",
            turns=turns,
            speaker_a="A",
            speaker_b="B",
        )
        assert session.turn_count == 3

    def test_token_estimate_property(self) -> None:
        """Test token_estimate property."""
        turns = [
            LoCoMoTurn(speaker="A", dia_id="D1:1", text="word " * 75, session_num=1),
        ]
        session = LoCoMoSession(
            session_num=1,
            timestamp="",
            turns=turns,
            speaker_a="A",
            speaker_b="B",
        )
        # 75 words / 0.75 = 100 tokens
        assert session.token_estimate == 100

    def test_session_is_frozen(self) -> None:
        """Test that LoCoMoSession is immutable."""
        session = LoCoMoSession(
            session_num=1,
            timestamp="",
            turns=[],
            speaker_a="A",
            speaker_b="B",
        )
        with pytest.raises(AttributeError):
            session.session_num = 2  # type: ignore[misc]


class TestLoCoMoQuestion:
    """Tests for LoCoMoQuestion data class."""

    def test_create_basic_question(self) -> None:
        """Test creating a basic question."""
        q = LoCoMoQuestion(
            question_id="conv1_q0",
            conversation_id="conv1",
            question="What is Alice's job?",
            answer="She is a software engineer.",
            category=QACategory.IDENTITY,
            evidence=["D1:5", "D2:10"],
        )
        assert q.question_id == "conv1_q0"
        assert q.conversation_id == "conv1"
        assert q.question == "What is Alice's job?"
        assert q.answer == "She is a software engineer."
        assert q.category == QACategory.IDENTITY
        assert q.evidence == ["D1:5", "D2:10"]
        assert q.adversarial_answer is None
        assert q.metadata == {}

    def test_create_adversarial_question(self) -> None:
        """Test creating an adversarial question."""
        q = LoCoMoQuestion(
            question_id="conv1_q5",
            conversation_id="conv1",
            question="When did Alice visit Paris?",
            answer="Alice never mentioned visiting Paris.",
            category=QACategory.ADVERSARIAL,
            evidence=[],
            adversarial_answer="Alice visited Paris last summer.",
        )
        assert q.is_adversarial is True
        assert q.adversarial_answer == "Alice visited Paris last summer."

    def test_is_adversarial_property(self) -> None:
        """Test is_adversarial property."""
        q_adv = LoCoMoQuestion(
            question_id="q1",
            conversation_id="c1",
            question="Q",
            answer="A",
            category=QACategory.ADVERSARIAL,
        )
        assert q_adv.is_adversarial is True

        q_normal = LoCoMoQuestion(
            question_id="q2",
            conversation_id="c1",
            question="Q",
            answer="A",
            category=QACategory.IDENTITY,
        )
        assert q_normal.is_adversarial is False

    def test_evidence_session_nums_property(self) -> None:
        """Test extracting session numbers from evidence."""
        q = LoCoMoQuestion(
            question_id="q1",
            conversation_id="c1",
            question="Q",
            answer="A",
            category=QACategory.CONTEXTUAL,
            evidence=["D1:5", "D1:10", "D3:7", "D5:1"],
        )
        assert q.evidence_session_nums == {1, 3, 5}

    def test_evidence_session_nums_handles_malformed(self) -> None:
        """Test evidence_session_nums handles malformed dia_ids."""
        q = LoCoMoQuestion(
            question_id="q1",
            conversation_id="c1",
            question="Q",
            answer="A",
            category=QACategory.CONTEXTUAL,
            evidence=["D1:5", "invalid", "Dxyz:10"],
        )
        # Only D1:5 is valid
        assert q.evidence_session_nums == {1}

    def test_question_is_frozen(self) -> None:
        """Test that LoCoMoQuestion is immutable."""
        q = LoCoMoQuestion(
            question_id="q1",
            conversation_id="c1",
            question="Q",
            answer="A",
            category=QACategory.IDENTITY,
        )
        with pytest.raises(AttributeError):
            q.answer = "Modified"  # type: ignore[misc]


class TestLoCoMoConversation:
    """Tests for LoCoMoConversation data class."""

    @pytest.fixture
    def sample_conversation(self) -> LoCoMoConversation:
        """Create a sample conversation for testing."""
        session1_turns = [
            LoCoMoTurn(speaker="Alice", dia_id="D1:1", text="Hi Bob!", session_num=1),
            LoCoMoTurn(speaker="Bob", dia_id="D1:2", text="Hey Alice!", session_num=1),
        ]
        session2_turns = [
            LoCoMoTurn(speaker="Alice", dia_id="D2:1", text="Hello again", session_num=2),
        ]
        sessions = [
            LoCoMoSession(
                session_num=1,
                timestamp="2023-01-01 10:00:00",
                turns=session1_turns,
                speaker_a="Alice",
                speaker_b="Bob",
            ),
            LoCoMoSession(
                session_num=2,
                timestamp="2023-01-02 10:00:00",
                turns=session2_turns,
                speaker_a="Alice",
                speaker_b="Bob",
            ),
        ]
        questions = [
            LoCoMoQuestion(
                question_id="c1_q0",
                conversation_id="conv1",
                question="Who greeted first?",
                answer="Alice",
                category=QACategory.IDENTITY,
            ),
            LoCoMoQuestion(
                question_id="c1_q1",
                conversation_id="conv1",
                question="When was session 2?",
                answer="January 2nd",
                category=QACategory.TEMPORAL,
            ),
        ]
        return LoCoMoConversation(
            sample_id="conv1",
            sessions=sessions,
            questions=questions,
            event_summary={"Alice": ["Greeted Bob"], "Bob": ["Responded to Alice"]},
            session_summary={"1": "Initial greeting", "2": "Follow-up"},
        )

    def test_total_turns(self, sample_conversation: LoCoMoConversation) -> None:
        """Test total_turns property."""
        assert sample_conversation.total_turns == 3

    def test_total_sessions(self, sample_conversation: LoCoMoConversation) -> None:
        """Test total_sessions property."""
        assert sample_conversation.total_sessions == 2

    def test_speakers(self, sample_conversation: LoCoMoConversation) -> None:
        """Test speakers property."""
        assert sample_conversation.speakers == ("Alice", "Bob")

    def test_speakers_empty_sessions(self) -> None:
        """Test speakers returns empty tuple for no sessions."""
        conv = LoCoMoConversation(sample_id="empty", sessions=[], questions=[])
        assert conv.speakers == ("", "")

    def test_token_estimate(self, sample_conversation: LoCoMoConversation) -> None:
        """Test token_estimate property."""
        # Each turn has ~2 words: 3 turns * 2 words / 0.75 ≈ 8 tokens
        estimate = sample_conversation.token_estimate
        assert isinstance(estimate, int)
        assert estimate > 0

    def test_get_session(self, sample_conversation: LoCoMoConversation) -> None:
        """Test get_session method."""
        session1 = sample_conversation.get_session(1)
        assert session1 is not None
        assert session1.session_num == 1

        session2 = sample_conversation.get_session(2)
        assert session2 is not None
        assert session2.session_num == 2

        session3 = sample_conversation.get_session(3)
        assert session3 is None

    def test_questions_by_category(self, sample_conversation: LoCoMoConversation) -> None:
        """Test questions_by_category method."""
        identity_qs = sample_conversation.questions_by_category(QACategory.IDENTITY)
        assert len(identity_qs) == 1
        assert identity_qs[0].question_id == "c1_q0"

        temporal_qs = sample_conversation.questions_by_category(QACategory.TEMPORAL)
        assert len(temporal_qs) == 1

        inference_qs = sample_conversation.questions_by_category(QACategory.INFERENCE)
        assert len(inference_qs) == 0

    def test_get_all_turns(self, sample_conversation: LoCoMoConversation) -> None:
        """Test get_all_turns method."""
        all_turns = sample_conversation.get_all_turns()
        assert len(all_turns) == 3
        assert all_turns[0].dia_id == "D1:1"
        assert all_turns[1].dia_id == "D1:2"
        assert all_turns[2].dia_id == "D2:1"


class TestLoCoMoDataset:
    """Tests for LoCoMoDataset data class."""

    @pytest.fixture
    def sample_dataset(self) -> LoCoMoDataset:
        """Create a sample dataset for testing."""
        conv1 = LoCoMoConversation(
            sample_id="conv1",
            sessions=[
                LoCoMoSession(
                    session_num=1,
                    timestamp="",
                    turns=[
                        LoCoMoTurn(speaker="A", dia_id="D1:1", text="Hi", session_num=1),
                        LoCoMoTurn(speaker="B", dia_id="D1:2", text="Hello", session_num=1),
                    ],
                    speaker_a="A",
                    speaker_b="B",
                )
            ],
            questions=[
                LoCoMoQuestion(
                    question_id="c1_q0",
                    conversation_id="conv1",
                    question="Q1",
                    answer="A1",
                    category=QACategory.IDENTITY,
                ),
            ],
        )
        conv2 = LoCoMoConversation(
            sample_id="conv2",
            sessions=[
                LoCoMoSession(
                    session_num=1,
                    timestamp="",
                    turns=[
                        LoCoMoTurn(speaker="C", dia_id="D1:1", text="Hey", session_num=1),
                    ],
                    speaker_a="C",
                    speaker_b="D",
                )
            ],
            questions=[
                LoCoMoQuestion(
                    question_id="c2_q0",
                    conversation_id="conv2",
                    question="Q2",
                    answer="A2",
                    category=QACategory.TEMPORAL,
                ),
                LoCoMoQuestion(
                    question_id="c2_q1",
                    conversation_id="conv2",
                    question="Q3",
                    answer="A3",
                    category=QACategory.IDENTITY,
                ),
            ],
        )
        return LoCoMoDataset(conversations=[conv1, conv2], source="test")

    def test_conversation_count(self, sample_dataset: LoCoMoDataset) -> None:
        """Test conversation_count property."""
        assert sample_dataset.conversation_count == 2

    def test_total_questions(self, sample_dataset: LoCoMoDataset) -> None:
        """Test total_questions property."""
        assert sample_dataset.total_questions == 3

    def test_total_turns(self, sample_dataset: LoCoMoDataset) -> None:
        """Test total_turns property."""
        assert sample_dataset.total_turns == 3

    def test_get_conversation(self, sample_dataset: LoCoMoDataset) -> None:
        """Test get_conversation method."""
        conv1 = sample_dataset.get_conversation("conv1")
        assert conv1 is not None
        assert conv1.sample_id == "conv1"

        conv_none = sample_dataset.get_conversation("nonexistent")
        assert conv_none is None

    def test_all_questions(self, sample_dataset: LoCoMoDataset) -> None:
        """Test all_questions method."""
        all_qs = sample_dataset.all_questions()
        assert len(all_qs) == 3
        question_ids = {q.question_id for q in all_qs}
        assert question_ids == {"c1_q0", "c2_q0", "c2_q1"}

    def test_questions_by_category(self, sample_dataset: LoCoMoDataset) -> None:
        """Test questions_by_category method."""
        identity_qs = sample_dataset.questions_by_category(QACategory.IDENTITY)
        assert len(identity_qs) == 2

        temporal_qs = sample_dataset.questions_by_category(QACategory.TEMPORAL)
        assert len(temporal_qs) == 1

    def test_get_stats(self, sample_dataset: LoCoMoDataset) -> None:
        """Test get_stats method."""
        stats = sample_dataset.get_stats()
        assert stats["conversations"] == 2
        assert stats["total_questions"] == 3
        assert stats["total_turns"] == 3
        assert stats["questions_by_category"]["IDENTITY"] == 2
        assert stats["questions_by_category"]["TEMPORAL"] == 1
        assert stats["avg_turns_per_conversation"] == 1.5
        assert stats["avg_questions_per_conversation"] == 1.5


class TestLoadLocomoFromFile:
    """Tests for load_locomo_from_file function."""

    @pytest.fixture
    def sample_json(self) -> list[dict]:
        """Create sample JSON data matching LoCoMo format."""
        return [
            {
                "sample_id": "test_conv_1",
                "conversation": {
                    "speaker_a": "Caroline",
                    "speaker_b": "Janet",
                    "session_1": [
                        {"speaker": "Caroline", "dia_id": "D1:1", "text": "Hey Janet!"},
                        {"speaker": "Janet", "dia_id": "D1:2", "text": "Hi Caroline!"},
                    ],
                    "session_1_date_time": "2023-01-15 10:00:00",
                    "session_2": [
                        {"speaker": "Caroline", "dia_id": "D2:1", "text": "How are you?"},
                    ],
                    "session_2_date_time": "2023-01-16 11:00:00",
                },
                "qa": [
                    {
                        "question": "Who is Janet's friend?",
                        "answer": "Caroline",
                        "category": 1,
                        "evidence": ["D1:1", "D1:2"],
                    },
                    {
                        "question": "When did they first talk?",
                        "answer": "January 15th",
                        "category": 2,
                        "evidence": ["D1:1"],
                    },
                ],
                "event_summary": {
                    "Caroline": ["Greeted Janet"],
                    "Janet": ["Greeted Caroline back"],
                },
                "session_summary": {"1": "Initial greeting", "2": "Follow-up"},
            }
        ]

    def test_load_from_array_format(self, sample_json: list[dict]) -> None:
        """Test loading from array of conversations."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(sample_json, f)
            f.flush()
            path = Path(f.name)

        try:
            dataset = load_locomo_from_file(path)
            assert dataset.conversation_count == 1
            conv = dataset.conversations[0]
            assert conv.sample_id == "test_conv_1"
            assert conv.total_sessions == 2
            assert conv.total_turns == 3
            assert len(conv.questions) == 2
        finally:
            path.unlink()

    def test_load_from_single_object_format(self, sample_json: list[dict]) -> None:
        """Test loading from single conversation object."""
        single_conv = sample_json[0]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(single_conv, f)
            f.flush()
            path = Path(f.name)

        try:
            dataset = load_locomo_from_file(path)
            assert dataset.conversation_count == 1
        finally:
            path.unlink()

    def test_parses_sessions_correctly(self, sample_json: list[dict]) -> None:
        """Test that sessions are parsed with correct data."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(sample_json, f)
            f.flush()
            path = Path(f.name)

        try:
            dataset = load_locomo_from_file(path)
            conv = dataset.conversations[0]

            session1 = conv.get_session(1)
            assert session1 is not None
            assert session1.timestamp == "2023-01-15 10:00:00"
            assert session1.turn_count == 2
            assert session1.speaker_a == "Caroline"
            assert session1.speaker_b == "Janet"

            session2 = conv.get_session(2)
            assert session2 is not None
            assert session2.timestamp == "2023-01-16 11:00:00"
            assert session2.turn_count == 1
        finally:
            path.unlink()

    def test_parses_questions_correctly(self, sample_json: list[dict]) -> None:
        """Test that questions are parsed with correct data."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(sample_json, f)
            f.flush()
            path = Path(f.name)

        try:
            dataset = load_locomo_from_file(path)
            conv = dataset.conversations[0]

            assert len(conv.questions) == 2
            q1 = conv.questions[0]
            assert q1.question == "Who is Janet's friend?"
            assert q1.answer == "Caroline"
            assert q1.category == QACategory.IDENTITY
            assert q1.evidence == ["D1:1", "D1:2"]

            q2 = conv.questions[1]
            assert q2.category == QACategory.TEMPORAL
        finally:
            path.unlink()

    def test_parses_turn_with_image(self) -> None:
        """Test parsing turn with image URL and caption."""
        data = [
            {
                "sample_id": "img_test",
                "conversation": {
                    "speaker_a": "A",
                    "speaker_b": "B",
                    "session_1": [
                        {
                            "speaker": "A",
                            "dia_id": "D1:1",
                            "text": "Look at this!",
                            "img_url": "https://example.com/img.jpg",
                            "blip_caption": "A beautiful landscape",
                        },
                    ],
                    "session_1_date_time": "",
                },
                "qa": [],
            }
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            path = Path(f.name)

        try:
            dataset = load_locomo_from_file(path)
            turn = dataset.conversations[0].sessions[0].turns[0]
            assert turn.img_url == "https://example.com/img.jpg"
            assert turn.img_caption == "A beautiful landscape"
        finally:
            path.unlink()

    def test_file_not_found_raises(self) -> None:
        """Test that missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="LoCoMo dataset file not found"):
            load_locomo_from_file("/nonexistent/path/locomo.json")

    def test_invalid_format_raises(self) -> None:
        """Test that invalid format raises ValueError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"invalid": "data"}, f)
            f.flush()
            path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="Invalid LoCoMo format"):
                load_locomo_from_file(path)
        finally:
            path.unlink()

    def test_string_type_raises(self) -> None:
        """Test that non-dict/list JSON raises ValueError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump("just a string", f)
            f.flush()
            path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="expected dict or list"):
                load_locomo_from_file(path)
        finally:
            path.unlink()

    def test_handles_missing_optional_fields(self) -> None:
        """Test that missing optional fields use defaults."""
        data = [
            {
                "conversation": {
                    "session_1": [
                        {"speaker": "A", "dia_id": "D1:1", "text": "Hi"},
                    ],
                },
                "qa": [],
            }
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            path = Path(f.name)

        try:
            dataset = load_locomo_from_file(path)
            conv = dataset.conversations[0]
            # sample_id defaults to conv_0
            assert conv.sample_id == "conv_0"
            # speaker defaults
            session = conv.sessions[0]
            assert session.speaker_a == "Speaker A"
            assert session.speaker_b == "Speaker B"
            assert session.timestamp == ""
        finally:
            path.unlink()

    def test_handles_adversarial_questions(self) -> None:
        """Test parsing adversarial questions with adversarial_answer."""
        data = [
            {
                "sample_id": "adv_test",
                "conversation": {
                    "session_1": [
                        {"speaker": "A", "dia_id": "D1:1", "text": "I work in tech."},
                    ],
                },
                "qa": [
                    {
                        "question": "What medical condition does A have?",
                        "answer": "A never mentioned any medical condition.",
                        "category": 5,
                        "evidence": [],
                        "adversarial_answer": "A has diabetes.",
                    },
                ],
            }
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            path = Path(f.name)

        try:
            dataset = load_locomo_from_file(path)
            q = dataset.conversations[0].questions[0]
            assert q.category == QACategory.ADVERSARIAL
            assert q.is_adversarial is True
            assert q.adversarial_answer == "A has diabetes."
        finally:
            path.unlink()

    def test_metadata_stored(self, sample_json: list[dict]) -> None:
        """Test that dataset metadata is properly stored."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(sample_json, f)
            f.flush()
            path = Path(f.name)

        try:
            dataset = load_locomo_from_file(path)
            assert dataset.source == str(path)
            assert "file_path" in dataset.metadata
            assert dataset.metadata["conversation_count"] == 1
        finally:
            path.unlink()
