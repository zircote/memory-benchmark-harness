"""LoCoMo dataset loader and data structures.

This module provides data structures and loaders for the LoCoMo benchmark,
which evaluates long-term conversational memory in LLM agents.

The dataset structure follows the LoCoMo format from snap-research/locomo:
- 10 conversations with ~300 turns each
- Multi-session dialogues spanning up to 35 sessions
- 5 QA categories testing different memory aspects
- Evidence annotations linking questions to source turns

Typical usage:
    ```python
    from src.benchmarks.locomo import load_locomo

    # Load from file
    dataset = load_locomo_from_file("data/locomo10.json")

    # Access conversations
    for conv in dataset.conversations:
        print(f"Conversation {conv.sample_id}: {conv.total_turns} turns")

    # Access questions by category
    cat1_questions = dataset.questions_by_category(QACategory.IDENTITY)
    ```
"""

from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class QACategory(Enum):
    """Question-answer category types in LoCoMo.

    The 5 categories test different aspects of long-term memory:
    - IDENTITY: Basic identity/background facts about speakers
    - TEMPORAL: When events occurred (temporal reasoning)
    - INFERENCE: Prediction/inference based on conversation context
    - CONTEXTUAL: Detailed contextual questions about specific events
    - ADVERSARIAL: Questions with intentionally incorrect premises
    """

    IDENTITY = 1
    TEMPORAL = 2
    INFERENCE = 3
    CONTEXTUAL = 4
    ADVERSARIAL = 5

    @classmethod
    def from_int(cls, value: int) -> QACategory:
        """Convert integer category to enum.

        Args:
            value: Category integer (1-5)

        Returns:
            Corresponding QACategory enum value

        Raises:
            ValueError: If value is not 1-5
        """
        for cat in cls:
            if cat.value == value:
                return cat
        raise ValueError(f"Unknown QA category: {value}. Expected 1-5.")

    def description(self) -> str:
        """Get human-readable description of the category."""
        descriptions = {
            QACategory.IDENTITY: "Identity/background facts",
            QACategory.TEMPORAL: "Temporal event reasoning",
            QACategory.INFERENCE: "Inference/prediction questions",
            QACategory.CONTEXTUAL: "Detailed contextual questions",
            QACategory.ADVERSARIAL: "Adversarial questions",
        }
        return descriptions[self]


@dataclass(frozen=True, slots=True)
class LoCoMoTurn:
    """A single dialogue turn in a LoCoMo conversation.

    Attributes:
        speaker: Speaker name (e.g., "Caroline", "Janet")
        dia_id: Dialogue identifier (e.g., "D1:5" for session 1, turn 5)
        text: The dialogue content
        img_url: Optional URL to an associated image
        img_caption: Optional auto-generated image caption
        session_num: Session number this turn belongs to
    """

    speaker: str
    dia_id: str
    text: str
    session_num: int
    img_url: str | None = None
    img_caption: str | None = None

    @property
    def turn_num(self) -> int:
        """Extract turn number from dia_id."""
        # dia_id format: "D1:5" -> session 1, turn 5
        if ":" in self.dia_id:
            return int(self.dia_id.split(":")[1])
        return 0


@dataclass(frozen=True, slots=True)
class LoCoMoSession:
    """A single session within a LoCoMo conversation.

    Sessions represent separate conversation instances over time,
    each with a timestamp indicating when the conversation occurred.

    Attributes:
        session_num: Session number (1-indexed)
        timestamp: Date/time string for when the session occurred
        turns: List of dialogue turns in this session
        speaker_a: First speaker's name
        speaker_b: Second speaker's name
    """

    session_num: int
    timestamp: str
    turns: list[LoCoMoTurn]
    speaker_a: str
    speaker_b: str

    @property
    def turn_count(self) -> int:
        """Number of turns in this session."""
        return len(self.turns)

    @property
    def token_estimate(self) -> int:
        """Rough token estimate (words / 0.75)."""
        total_words = sum(len(turn.text.split()) for turn in self.turns)
        return int(total_words / 0.75)


@dataclass(frozen=True, slots=True)
class LoCoMoQuestion:
    """A question-answer pair from the LoCoMo benchmark.

    Attributes:
        question_id: Unique identifier for this question
        conversation_id: ID of the source conversation
        question: The question text
        answer: The correct answer
        category: QA category (1-5)
        evidence: List of dialogue IDs containing the answer
        adversarial_answer: For category 5, the incorrect premise
        metadata: Additional question metadata
    """

    question_id: str
    conversation_id: str
    question: str
    answer: str
    category: QACategory
    evidence: list[str] = field(default_factory=list)
    adversarial_answer: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_adversarial(self) -> bool:
        """Whether this is an adversarial question."""
        return self.category == QACategory.ADVERSARIAL

    @property
    def evidence_session_nums(self) -> set[int]:
        """Extract session numbers from evidence dia_ids."""
        sessions = set()
        for eid in self.evidence:
            if ":" in eid:
                # Format: "D1:5" -> session 1
                session_str = eid.split(":")[0].lstrip("D")
                with contextlib.suppress(ValueError):
                    sessions.add(int(session_str))
        return sessions


@dataclass(slots=True)
class LoCoMoConversation:
    """A complete LoCoMo conversation with all sessions and questions.

    Attributes:
        sample_id: Unique conversation identifier
        sessions: Ordered list of conversation sessions
        questions: Questions associated with this conversation
        event_summary: Annotated significant events per speaker
        session_summary: Auto-generated per-session summaries
        observations: Generated observations (for RAG)
        metadata: Additional conversation metadata
    """

    sample_id: str
    sessions: list[LoCoMoSession]
    questions: list[LoCoMoQuestion]
    event_summary: dict[str, list[str]] = field(default_factory=dict)
    session_summary: dict[str, str] = field(default_factory=dict)
    observations: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_turns(self) -> int:
        """Total number of turns across all sessions."""
        return sum(s.turn_count for s in self.sessions)

    @property
    def total_sessions(self) -> int:
        """Number of sessions in this conversation."""
        return len(self.sessions)

    @property
    def speakers(self) -> tuple[str, str]:
        """Get the two speaker names."""
        if self.sessions:
            return (self.sessions[0].speaker_a, self.sessions[0].speaker_b)
        return ("", "")

    @property
    def token_estimate(self) -> int:
        """Rough token estimate for the entire conversation."""
        return sum(s.token_estimate for s in self.sessions)

    def get_session(self, session_num: int) -> LoCoMoSession | None:
        """Get a specific session by number."""
        for session in self.sessions:
            if session.session_num == session_num:
                return session
        return None

    def questions_by_category(self, category: QACategory) -> list[LoCoMoQuestion]:
        """Get questions filtered by category."""
        return [q for q in self.questions if q.category == category]

    def get_all_turns(self) -> list[LoCoMoTurn]:
        """Get all turns from all sessions in order."""
        turns: list[LoCoMoTurn] = []
        for session in self.sessions:
            turns.extend(session.turns)
        return turns


@dataclass(slots=True)
class LoCoMoDataset:
    """Complete LoCoMo benchmark dataset.

    Contains all conversations with their questions, organized
    for easy access and filtering.

    Attributes:
        conversations: List of all conversations
        source: Origin of the dataset (file path or URL)
        metadata: Dataset-level metadata
    """

    conversations: list[LoCoMoConversation]
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def conversation_count(self) -> int:
        """Number of conversations in the dataset."""
        return len(self.conversations)

    @property
    def total_questions(self) -> int:
        """Total number of questions across all conversations."""
        return sum(len(c.questions) for c in self.conversations)

    @property
    def total_turns(self) -> int:
        """Total number of dialogue turns across all conversations."""
        return sum(c.total_turns for c in self.conversations)

    def get_conversation(self, sample_id: str) -> LoCoMoConversation | None:
        """Get a specific conversation by ID."""
        for conv in self.conversations:
            if conv.sample_id == sample_id:
                return conv
        return None

    def all_questions(self) -> list[LoCoMoQuestion]:
        """Get all questions from all conversations."""
        questions: list[LoCoMoQuestion] = []
        for conv in self.conversations:
            questions.extend(conv.questions)
        return questions

    def questions_by_category(self, category: QACategory) -> list[LoCoMoQuestion]:
        """Get all questions of a specific category."""
        return [q for q in self.all_questions() if q.category == category]

    def get_stats(self) -> dict[str, Any]:
        """Get dataset statistics."""
        all_questions = self.all_questions()
        category_counts = {}
        for cat in QACategory:
            category_counts[cat.name] = len([q for q in all_questions if q.category == cat])

        return {
            "conversations": self.conversation_count,
            "total_questions": self.total_questions,
            "total_turns": self.total_turns,
            "questions_by_category": category_counts,
            "avg_turns_per_conversation": (
                self.total_turns / self.conversation_count if self.conversation_count > 0 else 0
            ),
            "avg_questions_per_conversation": (
                self.total_questions / self.conversation_count if self.conversation_count > 0 else 0
            ),
        }


def _parse_turns(session_data: list[dict[str, Any]], session_num: int) -> list[LoCoMoTurn]:
    """Parse dialogue turns from session data."""
    turns: list[LoCoMoTurn] = []
    for turn_data in session_data:
        turn = LoCoMoTurn(
            speaker=turn_data.get("speaker", ""),
            dia_id=turn_data.get("dia_id", ""),
            text=turn_data.get("text", ""),
            session_num=session_num,
            img_url=turn_data.get("img_url"),
            img_caption=turn_data.get("blip_caption"),
        )
        turns.append(turn)
    return turns


def _parse_sessions(conversation_data: dict[str, Any]) -> list[LoCoMoSession]:
    """Parse all sessions from conversation data."""
    sessions: list[LoCoMoSession] = []
    session_num = 1

    # Find speaker names from first session
    speaker_a = conversation_data.get("speaker_a", "Speaker A")
    speaker_b = conversation_data.get("speaker_b", "Speaker B")

    while True:
        session_key = f"session_{session_num}"
        timestamp_key = f"session_{session_num}_date_time"

        if session_key not in conversation_data:
            break

        session_turns = conversation_data[session_key]
        timestamp = conversation_data.get(timestamp_key, "")

        session = LoCoMoSession(
            session_num=session_num,
            timestamp=timestamp,
            turns=_parse_turns(session_turns, session_num),
            speaker_a=speaker_a,
            speaker_b=speaker_b,
        )
        sessions.append(session)
        session_num += 1

    return sessions


def _parse_questions(qa_data: list[dict[str, Any]], conversation_id: str) -> list[LoCoMoQuestion]:
    """Parse QA pairs from question data."""
    questions: list[LoCoMoQuestion] = []

    for idx, qa in enumerate(qa_data):
        category_int = qa.get("category", 1)
        try:
            category = QACategory.from_int(category_int)
        except ValueError:
            category = QACategory.IDENTITY  # Default fallback

        question = LoCoMoQuestion(
            question_id=f"{conversation_id}_q{idx}",
            conversation_id=conversation_id,
            question=qa.get("question", ""),
            answer=qa.get("answer", ""),
            category=category,
            evidence=qa.get("evidence", []),
            adversarial_answer=qa.get("adversarial_answer"),
            metadata={
                k: v
                for k, v in qa.items()
                if k not in ("question", "answer", "category", "evidence", "adversarial_answer")
            },
        )
        questions.append(question)

    return questions


def _parse_conversation(data: dict[str, Any], sample_id: str) -> LoCoMoConversation:
    """Parse a single conversation entry."""
    conversation_data = data.get("conversation", {})
    qa_data = data.get("qa", [])

    sessions = _parse_sessions(conversation_data)
    questions = _parse_questions(qa_data, sample_id)

    return LoCoMoConversation(
        sample_id=sample_id,
        sessions=sessions,
        questions=questions,
        event_summary=data.get("event_summary", {}),
        session_summary=data.get("session_summary", {}),
        observations=data.get("observation", {}),
        metadata={
            k: v
            for k, v in data.items()
            if k not in ("conversation", "qa", "event_summary", "session_summary", "observation")
        },
    )


def load_locomo_from_file(file_path: str | Path) -> LoCoMoDataset:
    """Load LoCoMo dataset from a local JSON file.

    Args:
        file_path: Path to the locomo10.json file

    Returns:
        LoCoMoDataset containing all conversations and questions

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file format is invalid
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"LoCoMo dataset file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Handle both array and single-object formats
    if isinstance(data, list):
        conversations_data = data
    elif isinstance(data, dict):
        # Single conversation or wrapped format
        if "conversation" in data:
            conversations_data = [data]
        else:
            raise ValueError("Invalid LoCoMo format: expected 'conversation' key or array")
    else:
        raise ValueError("Invalid LoCoMo format: expected dict or list")

    conversations: list[LoCoMoConversation] = []
    for idx, conv_data in enumerate(conversations_data):
        sample_id = conv_data.get("sample_id", f"conv_{idx}")
        conversation = _parse_conversation(conv_data, sample_id)
        conversations.append(conversation)

    return LoCoMoDataset(
        conversations=conversations,
        source=str(path),
        metadata={"file_path": str(path), "conversation_count": len(conversations)},
    )


def load_locomo(
    cache_dir: str | Path | None = None,
    force_download: bool = False,
) -> LoCoMoDataset:
    """Load LoCoMo dataset, downloading if necessary.

    Downloads the dataset from the snap-research/locomo GitHub repository
    if not already cached locally.

    Args:
        cache_dir: Directory to cache the dataset. Defaults to data/locomo/
        force_download: If True, re-download even if cached

    Returns:
        LoCoMoDataset containing all conversations and questions

    Raises:
        RuntimeError: If download fails
    """
    import urllib.error
    import urllib.request

    cache_dir = Path("data/locomo") if cache_dir is None else Path(cache_dir)

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "locomo10.json"

    if not cache_file.exists() or force_download:
        url = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
        try:
            urllib.request.urlretrieve(url, cache_file)
        except urllib.error.URLError as e:
            raise RuntimeError(f"Failed to download LoCoMo dataset: {e}") from e

    return load_locomo_from_file(cache_file)
