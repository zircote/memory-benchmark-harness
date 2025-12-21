"""MemoryAgentBench dataset loader and data classes.

This module handles loading and parsing the MemoryAgentBench benchmark dataset
from HuggingFace. It provides typed data classes for the four competencies
and their associated questions.

Dataset source: ai-hyz/MemoryAgentBench
Paper: https://arxiv.org/abs/2507.05257

Competencies:
- Accurate_Retrieval: 22 examples
- Test_Time_Learning: 6 examples
- Long_Range_Understanding: 110 examples
- Conflict_Resolution: 8 examples
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class Competency(Enum):
    """Core competencies evaluated by MemoryAgentBench.

    Each competency tests a different aspect of memory capabilities:
    - ACCURATE_RETRIEVAL: Precise information location from dialogue histories
    - TEST_TIME_LEARNING: Learning new skills during interactions
    - LONG_RANGE_UNDERSTANDING: Forming global cognition from long conversations
    - CONFLICT_RESOLUTION: Identifying and updating outdated information
    """

    ACCURATE_RETRIEVAL = "Accurate_Retrieval"
    TEST_TIME_LEARNING = "Test_Time_Learning"
    LONG_RANGE_UNDERSTANDING = "Long_Range_Understanding"
    CONFLICT_RESOLUTION = "Conflict_Resolution"

    @classmethod
    def from_string(cls, s: str) -> Competency:
        """Parse competency from string, handling various formats."""
        # Normalize: replace spaces/hyphens with underscores, title case
        normalized = s.replace(" ", "_").replace("-", "_")
        # Handle case variations
        for comp in cls:
            if comp.value.lower() == normalized.lower():
                return comp
            if comp.name.lower() == normalized.lower():
                return comp
        raise ValueError(f"Unknown competency: {s}")

    @property
    def short_name(self) -> str:
        """Return short abbreviation for competency."""
        abbreviations = {
            Competency.ACCURATE_RETRIEVAL: "AR",
            Competency.TEST_TIME_LEARNING: "TTL",
            Competency.LONG_RANGE_UNDERSTANDING: "LRU",
            Competency.CONFLICT_RESOLUTION: "CR",
        }
        return abbreviations[self]


class DifficultyLevel(Enum):
    """Difficulty levels for questions, particularly in Conflict Resolution."""

    SINGLE_HOP = "single_hop"
    MULTI_HOP = "multi_hop"
    UNKNOWN = "unknown"


@dataclass(slots=True, frozen=True)
class HaystackSession:
    """A session from the haystack metadata.

    Attributes:
        content: Session text content
        has_answer: Whether this session contains the answer
        role: Role identifier for the session
    """

    content: str
    has_answer: bool
    role: str


@dataclass(slots=True, frozen=True)
class MemoryAgentBenchQuestion:
    """A question from the MemoryAgentBench dataset.

    Attributes:
        question_id: Unique identifier for this question
        question_text: The question to answer
        answers: List of acceptable answers (multiple correct answers possible)
        competency: Which competency this question tests
        context: The full context/document for this question
        difficulty: Difficulty level (single_hop, multi_hop, unknown)
        keypoints: Key information points for this question
        previous_events: Events that occurred before the query
        question_date: Date associated with the question
        question_type: Specific type within the competency
        source: Origin dataset (e.g., EventQA, FactConsolidation)
        demo: Optional demonstration example
        haystack_sessions: Sessions from haystack metadata
        metadata: Additional metadata
    """

    question_id: str
    question_text: str
    answers: list[str]
    competency: Competency
    context: str
    difficulty: DifficultyLevel = DifficultyLevel.UNKNOWN
    keypoints: list[str] = field(default_factory=list)
    previous_events: list[str] = field(default_factory=list)
    question_date: str | None = None
    question_type: str | None = None
    source: str | None = None
    demo: str | None = None
    haystack_sessions: list[HaystackSession] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate answers is not empty."""
        if not self.answers:
            raise ValueError("answers must have at least one acceptable answer")

    @property
    def is_conflict_resolution(self) -> bool:
        """Check if this is a conflict resolution question."""
        return self.competency == Competency.CONFLICT_RESOLUTION

    @property
    def context_length(self) -> int:
        """Return character length of context."""
        return len(self.context)

    @property
    def token_estimate(self) -> int:
        """Rough token estimate (chars / 4)."""
        return len(self.context) // 4


@dataclass(slots=True)
class MemoryAgentBenchSplit:
    """A split (competency) from the dataset.

    Attributes:
        competency: Which competency this split represents
        questions: All questions in this split
        metadata: Split-level metadata
    """

    competency: Competency
    questions: list[MemoryAgentBenchQuestion]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def question_count(self) -> int:
        """Return number of questions."""
        return len(self.questions)

    @property
    def total_context_chars(self) -> int:
        """Return total context characters."""
        return sum(q.context_length for q in self.questions)

    @property
    def estimated_tokens(self) -> int:
        """Return estimated total tokens."""
        return self.total_context_chars // 4

    def get_stats(self) -> dict[str, Any]:
        """Get split statistics."""
        difficulty_counts: dict[str, int] = {}
        source_counts: dict[str, int] = {}

        for q in self.questions:
            diff = q.difficulty.value
            difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1

            src = q.source or "unknown"
            source_counts[src] = source_counts.get(src, 0) + 1

        return {
            "competency": self.competency.value,
            "question_count": self.question_count,
            "total_context_chars": self.total_context_chars,
            "estimated_tokens": self.estimated_tokens,
            "difficulty_distribution": difficulty_counts,
            "source_distribution": source_counts,
        }


@dataclass(slots=True)
class MemoryAgentBenchDataset:
    """Complete MemoryAgentBench dataset with all competencies.

    Attributes:
        splits: Dictionary mapping competency to its split
        metadata: Dataset-level metadata
    """

    splits: dict[Competency, MemoryAgentBenchSplit]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_questions(self) -> int:
        """Return total questions across all splits."""
        return sum(split.question_count for split in self.splits.values())

    @property
    def competencies(self) -> list[Competency]:
        """Return list of available competencies."""
        return list(self.splits.keys())

    def get_split(self, competency: Competency) -> MemoryAgentBenchSplit | None:
        """Get a split by competency."""
        return self.splits.get(competency)

    def get_conflict_resolution_split(self) -> MemoryAgentBenchSplit | None:
        """Get the conflict resolution split (primary focus for this benchmark)."""
        return self.splits.get(Competency.CONFLICT_RESOLUTION)

    def all_questions(self) -> list[MemoryAgentBenchQuestion]:
        """Get all questions across all splits."""
        questions: list[MemoryAgentBenchQuestion] = []
        for split in self.splits.values():
            questions.extend(split.questions)
        return questions

    def questions_by_difficulty(
        self, difficulty: DifficultyLevel
    ) -> list[MemoryAgentBenchQuestion]:
        """Get all questions of a specific difficulty."""
        return [q for q in self.all_questions() if q.difficulty == difficulty]

    def get_stats(self) -> dict[str, Any]:
        """Get dataset statistics."""
        split_stats = {comp.short_name: split.get_stats() for comp, split in self.splits.items()}

        return {
            "total_questions": self.total_questions,
            "competencies": [c.value for c in self.competencies],
            "splits": split_stats,
        }


def _parse_haystack_session(raw: dict[str, Any]) -> HaystackSession:
    """Parse a haystack session from raw format."""
    return HaystackSession(
        content=raw.get("content", ""),
        has_answer=raw.get("has_answer", False),
        role=raw.get("role", "unknown"),
    )


def _infer_difficulty(question: dict[str, Any], competency: Competency) -> DifficultyLevel:
    """Infer difficulty level from question metadata."""
    # Check explicit difficulty field
    if "difficulty" in question:
        diff = question["difficulty"].lower()
        if "single" in diff:
            return DifficultyLevel.SINGLE_HOP
        if "multi" in diff:
            return DifficultyLevel.MULTI_HOP

    # For conflict resolution, check source
    if competency == Competency.CONFLICT_RESOLUTION:
        source = question.get("metadata", {}).get("source", "")
        if "single" in source.lower():
            return DifficultyLevel.SINGLE_HOP
        if "multi" in source.lower():
            return DifficultyLevel.MULTI_HOP

    # Check question types for hints
    qtypes = question.get("metadata", {}).get("question_types", [])
    if isinstance(qtypes, list):
        for qt in qtypes:
            if "single" in str(qt).lower():
                return DifficultyLevel.SINGLE_HOP
            if "multi" in str(qt).lower():
                return DifficultyLevel.MULTI_HOP

    return DifficultyLevel.UNKNOWN


def _parse_question(
    raw: dict[str, Any],
    competency: Competency,
    question_idx: int,
) -> list[MemoryAgentBenchQuestion]:
    """Parse questions from a raw dataset row.

    Each row can contain multiple questions, so this returns a list.
    """
    questions: list[MemoryAgentBenchQuestion] = []

    # Get base fields
    context = raw.get("context", "")
    metadata = raw.get("metadata", {})

    # Get question lists
    question_texts = raw.get("questions", [])
    answers_list = raw.get("answers", [])
    question_ids = metadata.get("question_ids", [])
    question_types = metadata.get("question_types", [])
    question_dates = metadata.get("question_dates", [])
    qa_pair_ids = metadata.get("qa_pair_ids", [])

    # Parse haystack sessions if present
    haystack_raw = metadata.get("haystack_sessions", [])
    haystack_sessions: list[HaystackSession] = []
    if isinstance(haystack_raw, list):
        for session_group in haystack_raw:
            if isinstance(session_group, list):
                for session in session_group:
                    if isinstance(session, dict):
                        haystack_sessions.append(_parse_haystack_session(session))

    # Get other metadata fields
    keypoints = metadata.get("keypoints", [])
    previous_events = metadata.get("previous_events", [])
    source = metadata.get("source", "")
    demo = metadata.get("demo")

    # Create a question for each question text
    for i, question_text in enumerate(question_texts):
        # Get corresponding answers
        answers = answers_list[i] if i < len(answers_list) else []
        if isinstance(answers, str):
            answers = [answers]
        elif not isinstance(answers, list):
            answers = [str(answers)]

        # Generate question ID
        if i < len(question_ids):
            qid = question_ids[i]
        elif i < len(qa_pair_ids):
            qid = qa_pair_ids[i]
        else:
            qid = f"{competency.short_name}_{question_idx}_{i}"

        # Get question type
        qtype = question_types[i] if i < len(question_types) else None

        # Get question date
        qdate = question_dates[i] if i < len(question_dates) else None

        # Infer difficulty
        difficulty = _infer_difficulty(raw, competency)

        try:
            question = MemoryAgentBenchQuestion(
                question_id=qid,
                question_text=question_text,
                answers=answers,
                competency=competency,
                context=context,
                difficulty=difficulty,
                keypoints=keypoints if isinstance(keypoints, list) else [],
                previous_events=previous_events if isinstance(previous_events, list) else [],
                question_date=qdate,
                question_type=qtype,
                source=source,
                demo=demo,
                haystack_sessions=haystack_sessions,
                metadata={
                    "row_index": question_idx,
                    "question_index": i,
                },
            )
            questions.append(question)
        except ValueError as e:
            logger.warning(f"Skipping invalid question {qid}: {e}")

    return questions


def load_memoryagentbench(
    competencies: list[Competency] | None = None,
    cache_dir: Path | str | None = None,
) -> MemoryAgentBenchDataset:
    """Load the MemoryAgentBench dataset from HuggingFace.

    Args:
        competencies: List of competencies to load. If None, loads all.
        cache_dir: Optional cache directory for HuggingFace datasets

    Returns:
        MemoryAgentBenchDataset with parsed splits and questions

    Raises:
        ImportError: If datasets library is not installed
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "The 'datasets' library is required. Install it with: pip install datasets"
        ) from e

    if competencies is None:
        competencies = list(Competency)

    logger.info(
        f"Loading MemoryAgentBench from HuggingFace "
        f"(competencies: {[c.short_name for c in competencies]})..."
    )

    dataset_name = "ai-hyz/MemoryAgentBench"
    kwargs: dict[str, Any] = {}
    if cache_dir:
        kwargs["cache_dir"] = str(cache_dir)

    try:
        ds = load_dataset(dataset_name, **kwargs)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise

    splits: dict[Competency, MemoryAgentBenchSplit] = {}

    for competency in competencies:
        split_name = competency.value

        if split_name not in ds:
            logger.warning(f"Split '{split_name}' not found in dataset")
            continue

        data = ds[split_name]
        questions: list[MemoryAgentBenchQuestion] = []

        for idx, row in enumerate(data):
            row_dict = dict(row)
            parsed = _parse_question(row_dict, competency, idx)
            questions.extend(parsed)

        splits[competency] = MemoryAgentBenchSplit(
            competency=competency,
            questions=questions,
            metadata={
                "source": dataset_name,
                "split": split_name,
            },
        )

        logger.info(
            f"Loaded {competency.short_name}: {len(questions)} questions from {len(data)} rows"
        )

    logger.info(
        f"Loaded MemoryAgentBench: {sum(s.question_count for s in splits.values())} "
        f"total questions across {len(splits)} competencies"
    )

    return MemoryAgentBenchDataset(
        splits=splits,
        metadata={
            "source": dataset_name,
            "loaded_at": datetime.now().isoformat(),
            "cache_dir": str(cache_dir) if cache_dir else None,
        },
    )


def load_memoryagentbench_from_file(
    filepath: Path | str,
    competency: Competency,
) -> MemoryAgentBenchSplit:
    """Load a MemoryAgentBench split from a local JSON file.

    This is useful for offline testing or when HuggingFace is not accessible.

    Args:
        filepath: Path to JSON file with split data
        competency: Which competency this file represents

    Returns:
        MemoryAgentBenchSplit with parsed questions
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset file not found: {filepath}")

    with open(filepath) as f:
        data = json.load(f)

    questions: list[MemoryAgentBenchQuestion] = []

    if isinstance(data, list):
        for idx, row in enumerate(data):
            parsed = _parse_question(row, competency, idx)
            questions.extend(parsed)
    elif isinstance(data, dict):
        if "questions" in data:
            for idx, row in enumerate(data["questions"]):
                parsed = _parse_question(row, competency, idx)
                questions.extend(parsed)
        else:
            # Single row format
            parsed = _parse_question(data, competency, 0)
            questions.extend(parsed)

    return MemoryAgentBenchSplit(
        competency=competency,
        questions=questions,
        metadata={
            "source": str(filepath),
            "loaded_at": datetime.now().isoformat(),
        },
    )
