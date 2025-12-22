"""LongMemEval dataset loader and data classes.

This module handles loading and parsing the LongMemEval benchmark dataset
from HuggingFace. It provides typed data classes for sessions, messages,
and evaluation questions.

Dataset source: xiaowu0162/longmemeval-cleaned
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class QuestionType(Enum):
    """Types of questions in the benchmark.

    Each type tests different memory capabilities:
    - SINGLE_SESSION_USER: User-stated preferences/facts in one session
    - SINGLE_SESSION_ASSISTANT: Assistant-shared information in one session
    - SINGLE_SESSION_PREFERENCE: User preference expressed in conversation
    - TEMPORAL_REASONING: Questions about temporal order of events
    - KNOWLEDGE_UPDATE: Facts that change over time (tests update tracking)
    - MULTI_SESSION: Information spanning multiple sessions
    """

    SINGLE_SESSION_USER = "single-session-user"
    SINGLE_SESSION_ASSISTANT = "single-session-assistant"
    SINGLE_SESSION_PREFERENCE = "single-session-preference"
    TEMPORAL_REASONING = "temporal-reasoning"
    KNOWLEDGE_UPDATE = "knowledge-update"
    MULTI_SESSION = "multi-session"

    @classmethod
    def from_string(cls, s: str) -> QuestionType:
        """Parse question type from string, handling both formats."""
        # Handle both underscore and hyphen formats
        normalized = s.replace("_", "-").lower()
        for qt in cls:
            if qt.value == normalized:
                return qt
        raise ValueError(f"Unknown question type: {s}")


@dataclass(slots=True, frozen=True)
class Message:
    """A single message in a conversation session.

    Attributes:
        role: Either 'user' or 'assistant'
        content: The message text content
        timestamp: Optional timestamp for temporal ordering
    """

    role: str
    content: str
    timestamp: str | None = None

    def __post_init__(self) -> None:
        """Validate role value."""
        if self.role not in ("user", "assistant"):
            raise ValueError(f"role must be 'user' or 'assistant', got {self.role}")


@dataclass(slots=True, frozen=True)
class LongMemEvalQuestion:
    """An assessment question from the benchmark.

    Attributes:
        question_id: Unique identifier for this question
        question_text: The question to answer
        ground_truth: The correct answer(s) - may be list for multi-answer
        question_type: Category of question (tests different memory aspects)
        relevant_session_ids: Session IDs containing relevant information
        is_abstention: Whether correct answer is "I don't know" (ends with _abs)
        metadata: Additional question metadata
    """

    question_id: str
    question_text: str
    ground_truth: list[str]
    question_type: QuestionType
    relevant_session_ids: list[str]
    is_abstention: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate ground truth is not empty."""
        if not self.ground_truth:
            raise ValueError("ground_truth must have at least one answer")


@dataclass(slots=True)
class LongMemEvalSession:
    """A conversation session containing messages.

    Attributes:
        session_id: Unique identifier for this session
        messages: List of messages in chronological order
        timestamp: When this session occurred (for temporal ordering)
        metadata: Additional session metadata
    """

    session_id: str
    messages: list[Message]
    timestamp: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def message_count(self) -> int:
        """Return number of messages in session."""
        return len(self.messages)

    @property
    def token_estimate(self) -> int:
        """Rough token estimate (chars / 4)."""
        total_chars = sum(len(m.content) for m in self.messages)
        return total_chars // 4


@dataclass(slots=True)
class LongMemEvalDataset:
    """Complete dataset with sessions and questions.

    Attributes:
        subset: Which subset this is ('S' or 'M')
        sessions: All conversation sessions
        questions: All assessment questions
        metadata: Dataset metadata (source, version, etc.)
    """

    subset: str
    sessions: list[LongMemEvalSession]
    questions: list[LongMemEvalQuestion]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def session_count(self) -> int:
        """Return number of sessions."""
        return len(self.sessions)

    @property
    def question_count(self) -> int:
        """Return number of questions."""
        return len(self.questions)

    @property
    def total_messages(self) -> int:
        """Return total messages across all sessions."""
        return sum(s.message_count for s in self.sessions)

    @property
    def estimated_tokens(self) -> int:
        """Return estimated total tokens."""
        return sum(s.token_estimate for s in self.sessions)

    def get_session(self, session_id: str) -> LongMemEvalSession | None:
        """Get a session by ID."""
        for session in self.sessions:
            if session.session_id == session_id:
                return session
        return None

    def questions_by_type(self, qtype: QuestionType) -> list[LongMemEvalQuestion]:
        """Get all questions of a specific type."""
        return [q for q in self.questions if q.question_type == qtype]

    def get_stats(self) -> dict[str, Any]:
        """Get dataset statistics."""
        type_counts: dict[str, int] = {}
        abstention_count = 0

        for q in self.questions:
            type_name = q.question_type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
            if q.is_abstention:
                abstention_count += 1

        return {
            "subset": self.subset,
            "session_count": self.session_count,
            "question_count": self.question_count,
            "total_messages": self.total_messages,
            "estimated_tokens": self.estimated_tokens,
            "questions_by_type": type_counts,
            "abstention_questions": abstention_count,
        }


def _parse_message(raw: dict[str, Any]) -> Message:
    """Parse a message from raw dataset format."""
    return Message(
        role=raw["role"],
        content=raw["content"],
        timestamp=raw.get("timestamp"),
    )


def _parse_session(raw: dict[str, Any]) -> LongMemEvalSession:
    """Parse a session from raw dataset format."""
    messages = [_parse_message(m) for m in raw.get("messages", [])]
    return LongMemEvalSession(
        session_id=raw["session_id"],
        messages=messages,
        timestamp=raw.get("timestamp"),
        metadata=raw.get("metadata", {}),
    )


def _parse_question(raw: dict[str, Any]) -> LongMemEvalQuestion:
    """Parse a question from raw dataset format."""
    question_id = raw["question_id"]
    is_abstention = question_id.endswith("_abs")

    # Ground truth can be string or list
    gt = raw.get("answer", raw.get("ground_truth", []))
    if isinstance(gt, str):
        gt = [gt]

    # Parse question type
    qtype_str = raw.get("question_type", raw.get("type", ""))
    try:
        qtype = QuestionType.from_string(qtype_str)
    except ValueError:
        logger.warning(
            f"Unknown question type '{qtype_str}' for {question_id}, using MULTI_SESSION"
        )
        qtype = QuestionType.MULTI_SESSION

    # Relevant sessions can be string or list
    rel_sessions = raw.get("relevant_sessions", raw.get("session_ids", []))
    if isinstance(rel_sessions, str):
        rel_sessions = [rel_sessions]

    return LongMemEvalQuestion(
        question_id=question_id,
        question_text=raw.get("question", raw.get("query", "")),
        ground_truth=gt,
        question_type=qtype,
        relevant_session_ids=rel_sessions,
        is_abstention=is_abstention,
        metadata=raw.get("metadata", {}),
    )


def load_longmemeval(
    subset: str = "S",
    cache_dir: Path | str | None = None,
    force_download: bool = False,
) -> LongMemEvalDataset:
    """Load the benchmark dataset, downloading if necessary.

    Downloads the dataset from HuggingFace if not already cached locally.
    This avoids PyArrow parsing issues with the HuggingFace datasets library.

    Args:
        subset: Which subset to load - 'S' (small, ~40 sessions) or 'M' (medium, ~500 sessions)
        cache_dir: Directory to cache the dataset. Defaults to data/longmemeval/
        force_download: If True, re-download even if cached

    Returns:
        LongMemEvalDataset with parsed sessions and questions

    Raises:
        ValueError: If subset is not 'S' or 'M'
        RuntimeError: If download fails
    """
    import urllib.error
    import urllib.request

    if subset not in ("S", "M"):
        raise ValueError(f"subset must be 'S' or 'M', got {subset}")

    cache_path = Path("data/longmemeval") if cache_dir is None else Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # Map subset to filename
    filename = f"longmemeval_{subset.lower()}_cleaned.json"
    cache_file = cache_path / filename

    if not cache_file.exists() or force_download:
        # Download from HuggingFace direct URL (bypasses datasets library PyArrow issues)
        base_url = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned"
        url = f"{base_url}/resolve/main/{filename}"
        logger.info(f"Downloading {subset} subset from {url}...")

        try:
            urllib.request.urlretrieve(url, cache_file)
            logger.info(f"Downloaded to {cache_file}")
        except urllib.error.URLError as e:
            raise RuntimeError(f"Failed to download dataset: {e}") from e

    return load_longmemeval_from_file(cache_file, subset=subset)


def load_longmemeval_from_file(
    filepath: Path | str,
    subset: str = "S",
) -> LongMemEvalDataset:
    """Load the benchmark dataset from a local JSON file.

    This is useful for offline testing or when HuggingFace is not accessible.
    Supports multiple formats:
    - Oracle format: list of questions with embedded haystack_sessions
    - Standard format: separate sessions and questions lists

    Args:
        filepath: Path to JSON file with dataset
        subset: Label for this subset ('S' or 'M')

    Returns:
        LongMemEvalDataset with parsed sessions and questions
    """
    import json

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset file not found: {filepath}")

    with open(filepath) as f:
        data = json.load(f)

    sessions: list[LongMemEvalSession] = []
    questions: list[LongMemEvalQuestion] = []
    seen_session_ids: set[str] = set()

    # Handle different possible structures
    if isinstance(data, list):
        # List of items - could be oracle format or standard
        for idx, item in enumerate(data):
            # Oracle format: questions with embedded haystack_sessions
            if "haystack_sessions" in item:
                session_ids = item.get(
                    "haystack_session_ids",
                    [f"s_{idx}_{i}" for i in range(len(item["haystack_sessions"]))],
                )
                for sid, session_messages in zip(
                    session_ids, item["haystack_sessions"], strict=False
                ):
                    if sid not in seen_session_ids:
                        seen_session_ids.add(sid)
                        messages = [
                            Message(role=m["role"], content=m["content"]) for m in session_messages
                        ]
                        sessions.append(
                            LongMemEvalSession(
                                session_id=sid,
                                messages=messages,
                                timestamp=item.get("haystack_dates", [None])[0]
                                if item.get("haystack_dates")
                                else None,
                            )
                        )
                # Parse question with oracle-specific field mapping
                if "question_id" not in item:
                    item["question_id"] = f"q_{idx}"
                questions.append(_parse_question_oracle(item))
            # Standard format
            elif "messages" in item or "session_id" in item:
                sessions.append(_parse_session(item))
            if "question" in item or "query" in item:
                if "question_id" not in item:
                    item["question_id"] = f"q_{idx}"
                if "haystack_sessions" not in item:  # Avoid double-parsing
                    questions.append(_parse_question(item))
    elif isinstance(data, dict):
        # Dict with sessions and questions keys
        if "sessions" in data:
            for item in data["sessions"]:
                sessions.append(_parse_session(item))
        if "questions" in data:
            for idx, item in enumerate(data["questions"]):
                if "question_id" not in item:
                    item["question_id"] = f"q_{idx}"
                questions.append(_parse_question(item))

    return LongMemEvalDataset(
        subset=subset,
        sessions=sessions,
        questions=questions,
        metadata={
            "source": str(filepath),
            "loaded_at": datetime.now().isoformat(),
        },
    )


def _parse_question_oracle(raw: dict[str, Any]) -> LongMemEvalQuestion:
    """Parse a question from oracle format with haystack_sessions."""
    question_id = raw["question_id"]
    is_abstention = question_id.endswith("_abs")

    # Answer can be string or list
    answer = raw.get("answer", "")
    if isinstance(answer, str):
        gt = [answer]
    elif isinstance(answer, int):
        gt = [str(answer)]
    else:
        gt = [str(a) for a in answer]

    # Parse question type
    qtype_str = raw.get("question_type", "multi-session")
    try:
        qtype = QuestionType.from_string(qtype_str)
    except ValueError:
        logger.warning(
            f"Unknown question type '{qtype_str}' for {question_id}, using MULTI_SESSION"
        )
        qtype = QuestionType.MULTI_SESSION

    # Relevant sessions from answer_session_ids
    rel_sessions = raw.get("answer_session_ids", raw.get("haystack_session_ids", []))
    if isinstance(rel_sessions, str):
        rel_sessions = [rel_sessions]

    return LongMemEvalQuestion(
        question_id=question_id,
        question_text=raw.get("question", ""),
        ground_truth=gt,
        question_type=qtype,
        relevant_session_ids=rel_sessions,
        is_abstention=is_abstention,
        metadata={
            "question_date": raw.get("question_date"),
            "haystack_dates": raw.get("haystack_dates"),
        },
    )
