"""Tests for the MemoryAgentBench dataset module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.benchmarks.memoryagentbench.dataset import (
    Competency,
    DifficultyLevel,
    HaystackSession,
    MemoryAgentBenchDataset,
    MemoryAgentBenchQuestion,
    MemoryAgentBenchSplit,
    load_memoryagentbench_from_file,
)


class TestCompetency:
    """Tests for the Competency enum."""

    def test_all_competencies_exist(self) -> None:
        """Test that all expected competencies exist."""
        expected = [
            "Accurate_Retrieval",
            "Test_Time_Learning",
            "Long_Range_Understanding",
            "Conflict_Resolution",
        ]
        actual = [c.value for c in Competency]
        assert set(actual) == set(expected)

    def test_from_string_exact_match(self) -> None:
        """Test parsing competency from exact string."""
        assert Competency.from_string("Accurate_Retrieval") == Competency.ACCURATE_RETRIEVAL
        assert Competency.from_string("Conflict_Resolution") == Competency.CONFLICT_RESOLUTION

    def test_from_string_case_insensitive(self) -> None:
        """Test parsing is case-insensitive."""
        assert Competency.from_string("accurate_retrieval") == Competency.ACCURATE_RETRIEVAL
        assert Competency.from_string("CONFLICT_RESOLUTION") == Competency.CONFLICT_RESOLUTION

    def test_from_string_with_hyphens(self) -> None:
        """Test parsing with hyphens instead of underscores."""
        assert Competency.from_string("Accurate-Retrieval") == Competency.ACCURATE_RETRIEVAL

    def test_from_string_invalid(self) -> None:
        """Test parsing invalid competency raises error."""
        with pytest.raises(ValueError, match="Unknown competency"):
            Competency.from_string("invalid_competency")

    def test_short_names(self) -> None:
        """Test short name abbreviations."""
        assert Competency.ACCURATE_RETRIEVAL.short_name == "AR"
        assert Competency.TEST_TIME_LEARNING.short_name == "TTL"
        assert Competency.LONG_RANGE_UNDERSTANDING.short_name == "LRU"
        assert Competency.CONFLICT_RESOLUTION.short_name == "CR"


class TestDifficultyLevel:
    """Tests for the DifficultyLevel enum."""

    def test_all_levels_exist(self) -> None:
        """Test that all expected difficulty levels exist."""
        expected = ["single_hop", "multi_hop", "unknown"]
        actual = [d.value for d in DifficultyLevel]
        assert set(actual) == set(expected)


class TestHaystackSession:
    """Tests for the HaystackSession dataclass."""

    def test_creation(self) -> None:
        """Test creating a haystack session."""
        session = HaystackSession(
            content="test content",
            has_answer=True,
            role="user",
        )
        assert session.content == "test content"
        assert session.has_answer is True
        assert session.role == "user"

    def test_frozen(self) -> None:
        """Test that haystack session is frozen."""
        session = HaystackSession(content="test", has_answer=False, role="assistant")
        with pytest.raises(AttributeError):
            session.content = "modified"  # type: ignore


class TestMemoryAgentBenchQuestion:
    """Tests for the MemoryAgentBenchQuestion dataclass."""

    def test_creation(self) -> None:
        """Test creating a question."""
        question = MemoryAgentBenchQuestion(
            question_id="q1",
            question_text="What is the answer?",
            answers=["42"],
            competency=Competency.ACCURATE_RETRIEVAL,
            context="The answer to everything is 42.",
        )
        assert question.question_id == "q1"
        assert question.question_text == "What is the answer?"
        assert question.answers == ["42"]
        assert question.competency == Competency.ACCURATE_RETRIEVAL
        assert question.context_length == len("The answer to everything is 42.")

    def test_empty_answers_raises_error(self) -> None:
        """Test that empty answers raises ValueError."""
        with pytest.raises(ValueError, match="answers must have at least one"):
            MemoryAgentBenchQuestion(
                question_id="q1",
                question_text="What?",
                answers=[],
                competency=Competency.ACCURATE_RETRIEVAL,
                context="context",
            )

    def test_is_conflict_resolution(self) -> None:
        """Test conflict resolution detection."""
        cr_question = MemoryAgentBenchQuestion(
            question_id="cr1",
            question_text="What is X now?",
            answers=["updated value"],
            competency=Competency.CONFLICT_RESOLUTION,
            context="context",
        )
        assert cr_question.is_conflict_resolution is True

        ar_question = MemoryAgentBenchQuestion(
            question_id="ar1",
            question_text="What is X?",
            answers=["value"],
            competency=Competency.ACCURATE_RETRIEVAL,
            context="context",
        )
        assert ar_question.is_conflict_resolution is False

    def test_token_estimate(self) -> None:
        """Test token estimation."""
        question = MemoryAgentBenchQuestion(
            question_id="q1",
            question_text="What?",
            answers=["answer"],
            competency=Competency.ACCURATE_RETRIEVAL,
            context="a" * 400,  # 400 chars -> ~100 tokens
        )
        assert question.token_estimate == 100


class TestMemoryAgentBenchSplit:
    """Tests for the MemoryAgentBenchSplit dataclass."""

    @pytest.fixture
    def sample_split(self) -> MemoryAgentBenchSplit:
        """Create a sample split for testing."""
        questions = [
            MemoryAgentBenchQuestion(
                question_id=f"q{i}",
                question_text=f"Question {i}?",
                answers=[f"answer{i}"],
                competency=Competency.CONFLICT_RESOLUTION,
                context="a" * 100,
                difficulty=DifficultyLevel.SINGLE_HOP if i % 2 == 0 else DifficultyLevel.MULTI_HOP,
            )
            for i in range(5)
        ]
        return MemoryAgentBenchSplit(
            competency=Competency.CONFLICT_RESOLUTION,
            questions=questions,
        )

    def test_question_count(self, sample_split: MemoryAgentBenchSplit) -> None:
        """Test question count property."""
        assert sample_split.question_count == 5

    def test_total_context_chars(self, sample_split: MemoryAgentBenchSplit) -> None:
        """Test total context characters."""
        assert sample_split.total_context_chars == 500  # 5 questions * 100 chars

    def test_estimated_tokens(self, sample_split: MemoryAgentBenchSplit) -> None:
        """Test token estimation."""
        assert sample_split.estimated_tokens == 125  # 500 / 4

    def test_get_stats(self, sample_split: MemoryAgentBenchSplit) -> None:
        """Test get_stats method."""
        stats = sample_split.get_stats()
        assert stats["competency"] == "Conflict_Resolution"
        assert stats["question_count"] == 5
        assert "single_hop" in stats["difficulty_distribution"]
        assert "multi_hop" in stats["difficulty_distribution"]


class TestMemoryAgentBenchDataset:
    """Tests for the MemoryAgentBenchDataset dataclass."""

    @pytest.fixture
    def sample_dataset(self) -> MemoryAgentBenchDataset:
        """Create a sample dataset for testing."""
        splits = {}
        for comp in [Competency.ACCURATE_RETRIEVAL, Competency.CONFLICT_RESOLUTION]:
            questions = [
                MemoryAgentBenchQuestion(
                    question_id=f"{comp.short_name}_{i}",
                    question_text=f"Question {i} for {comp.short_name}?",
                    answers=[f"answer{i}"],
                    competency=comp,
                    context="context",
                )
                for i in range(3)
            ]
            splits[comp] = MemoryAgentBenchSplit(
                competency=comp,
                questions=questions,
            )
        return MemoryAgentBenchDataset(splits=splits)

    def test_total_questions(self, sample_dataset: MemoryAgentBenchDataset) -> None:
        """Test total questions count."""
        assert sample_dataset.total_questions == 6  # 3 per competency * 2

    def test_competencies(self, sample_dataset: MemoryAgentBenchDataset) -> None:
        """Test competencies list."""
        comps = sample_dataset.competencies
        assert Competency.ACCURATE_RETRIEVAL in comps
        assert Competency.CONFLICT_RESOLUTION in comps

    def test_get_split(self, sample_dataset: MemoryAgentBenchDataset) -> None:
        """Test getting a split by competency."""
        split = sample_dataset.get_split(Competency.CONFLICT_RESOLUTION)
        assert split is not None
        assert split.competency == Competency.CONFLICT_RESOLUTION

    def test_get_conflict_resolution_split(self, sample_dataset: MemoryAgentBenchDataset) -> None:
        """Test getting conflict resolution split specifically."""
        cr_split = sample_dataset.get_conflict_resolution_split()
        assert cr_split is not None
        assert cr_split.competency == Competency.CONFLICT_RESOLUTION

    def test_all_questions(self, sample_dataset: MemoryAgentBenchDataset) -> None:
        """Test getting all questions."""
        all_q = sample_dataset.all_questions()
        assert len(all_q) == 6

    def test_get_stats(self, sample_dataset: MemoryAgentBenchDataset) -> None:
        """Test get_stats method."""
        stats = sample_dataset.get_stats()
        assert stats["total_questions"] == 6
        assert "AR" in stats["splits"]
        assert "CR" in stats["splits"]


class TestLoadFromFile:
    """Tests for loading datasets from files."""

    def test_load_from_json_list(self, tmp_path: Path) -> None:
        """Test loading from a JSON file with list format."""
        data = [
            {
                "context": "The sky is blue.",
                "questions": ["What color is the sky?"],
                "answers": [["blue"]],
                "metadata": {
                    "question_ids": ["q1"],
                    "source": "test",
                },
            }
        ]

        filepath = tmp_path / "test_data.json"
        with open(filepath, "w") as f:
            json.dump(data, f)

        split = load_memoryagentbench_from_file(filepath, Competency.ACCURATE_RETRIEVAL)
        assert split.question_count == 1
        assert split.questions[0].question_text == "What color is the sky?"

    def test_load_from_json_dict(self, tmp_path: Path) -> None:
        """Test loading from a JSON file with dict format."""
        data = {
            "questions": [
                {
                    "context": "Alice works at Acme.",
                    "questions": ["Where does Alice work?"],
                    "answers": [["Acme"]],
                }
            ]
        }

        filepath = tmp_path / "test_data.json"
        with open(filepath, "w") as f:
            json.dump(data, f)

        split = load_memoryagentbench_from_file(filepath, Competency.ACCURATE_RETRIEVAL)
        assert split.question_count == 1

    def test_load_file_not_found(self, tmp_path: Path) -> None:
        """Test loading from non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_memoryagentbench_from_file(
                tmp_path / "nonexistent.json", Competency.ACCURATE_RETRIEVAL
            )
