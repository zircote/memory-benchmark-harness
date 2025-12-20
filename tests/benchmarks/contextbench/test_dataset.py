"""Tests for the Context-Bench dataset module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.benchmarks.contextbench.dataset import (
    ContextBenchDataset,
    ContextBenchFile,
    ContextBenchQuestion,
    QuestionCategory,
    generate_synthetic_dataset,
    load_contextbench_from_file,
)


class TestQuestionCategory:
    """Tests for the QuestionCategory enum."""

    def test_all_categories_exist(self) -> None:
        """Test that all expected categories exist."""
        expected = ["direct", "relationship", "multi_hop", "aggregation", "temporal", "unknown"]
        actual = [c.value for c in QuestionCategory]
        assert set(actual) == set(expected)

    def test_from_string_exact(self) -> None:
        """Test parsing category from exact string."""
        assert QuestionCategory.from_string("direct") == QuestionCategory.DIRECT
        assert QuestionCategory.from_string("multi_hop") == QuestionCategory.MULTI_HOP

    def test_from_string_normalized(self) -> None:
        """Test parsing with various formats."""
        assert QuestionCategory.from_string("multi-hop") == QuestionCategory.MULTI_HOP
        assert QuestionCategory.from_string("DIRECT") == QuestionCategory.DIRECT

    def test_from_string_unknown(self) -> None:
        """Test parsing unknown category returns UNKNOWN."""
        assert QuestionCategory.from_string("invalid") == QuestionCategory.UNKNOWN


class TestContextBenchFile:
    """Tests for the ContextBenchFile dataclass."""

    def test_creation(self) -> None:
        """Test creating a file."""
        file = ContextBenchFile(
            path="people/alice.txt",
            content="Name: Alice\nDepartment: Engineering",
            entity_type="person",
        )
        assert file.path == "people/alice.txt"
        assert file.name == "alice.txt"
        assert file.directory == "people"
        assert file.size == len("Name: Alice\nDepartment: Engineering")

    def test_frozen(self) -> None:
        """Test that file is frozen."""
        file = ContextBenchFile(path="test.txt", content="content")
        with pytest.raises(AttributeError):
            file.content = "new content"  # type: ignore


class TestContextBenchQuestion:
    """Tests for the ContextBenchQuestion dataclass."""

    def test_creation(self) -> None:
        """Test creating a question."""
        question = ContextBenchQuestion(
            question_id="q1",
            question_text="What department does Alice work in?",
            answer="Engineering",
            category=QuestionCategory.DIRECT,
            hop_count=1,
        )
        assert question.question_id == "q1"
        assert question.is_multi_hop is False

    def test_is_multi_hop(self) -> None:
        """Test multi-hop detection."""
        single_hop = ContextBenchQuestion(
            question_id="q1",
            question_text="?",
            answer="A",
            category=QuestionCategory.DIRECT,
            hop_count=1,
        )
        assert single_hop.is_multi_hop is False

        multi_hop = ContextBenchQuestion(
            question_id="q2",
            question_text="?",
            answer="B",
            category=QuestionCategory.MULTI_HOP,
            hop_count=3,
        )
        assert multi_hop.is_multi_hop is True


class TestContextBenchDataset:
    """Tests for the ContextBenchDataset dataclass."""

    @pytest.fixture
    def sample_dataset(self) -> ContextBenchDataset:
        """Create a sample dataset for testing."""
        files = [
            ContextBenchFile(
                path="people/alice.txt",
                content="Name: Alice\nDepartment: Engineering\nProject: Alpha",
                entity_type="person",
            ),
            ContextBenchFile(
                path="people/bob.txt",
                content="Name: Bob\nDepartment: Marketing\nProject: Beta",
                entity_type="person",
            ),
            ContextBenchFile(
                path="projects/alpha.txt",
                content="Project: Alpha\nTeam: Alice, Charlie",
                entity_type="project",
            ),
        ]
        questions = [
            ContextBenchQuestion(
                question_id="q1",
                question_text="What department does Alice work in?",
                answer="Engineering",
                category=QuestionCategory.DIRECT,
                hop_count=1,
            ),
            ContextBenchQuestion(
                question_id="q2",
                question_text="Who works on Project Alpha with Alice?",
                answer="Charlie",
                category=QuestionCategory.MULTI_HOP,
                hop_count=2,
            ),
        ]
        return ContextBenchDataset(files=files, questions=questions)

    def test_file_count(self, sample_dataset: ContextBenchDataset) -> None:
        """Test file count."""
        assert sample_dataset.file_count == 3

    def test_question_count(self, sample_dataset: ContextBenchDataset) -> None:
        """Test question count."""
        assert sample_dataset.question_count == 2

    def test_get_file(self, sample_dataset: ContextBenchDataset) -> None:
        """Test getting a file by path."""
        alice = sample_dataset.get_file("people/alice.txt")
        assert alice is not None
        assert "Alice" in alice.content

        missing = sample_dataset.get_file("nonexistent.txt")
        assert missing is None

    def test_grep_files(self, sample_dataset: ContextBenchDataset) -> None:
        """Test grep functionality."""
        results = sample_dataset.grep_files("Alice")
        assert len(results) >= 1  # Should find in at least alice.txt

        # Check that results contain matching lines
        for file, lines in results:
            assert any("Alice" in line for line in lines)

    def test_grep_case_insensitive(self, sample_dataset: ContextBenchDataset) -> None:
        """Test case-insensitive grep."""
        results = sample_dataset.grep_files("alice", case_sensitive=False)
        assert len(results) >= 1

    def test_questions_by_category(self, sample_dataset: ContextBenchDataset) -> None:
        """Test filtering by category."""
        direct = sample_dataset.questions_by_category(QuestionCategory.DIRECT)
        assert len(direct) == 1
        assert direct[0].question_id == "q1"

    def test_multi_hop_questions(self, sample_dataset: ContextBenchDataset) -> None:
        """Test getting multi-hop questions."""
        multi = sample_dataset.multi_hop_questions()
        assert len(multi) == 1
        assert multi[0].question_id == "q2"

    def test_get_stats(self, sample_dataset: ContextBenchDataset) -> None:
        """Test get_stats method."""
        stats = sample_dataset.get_stats()
        assert stats["file_count"] == 3
        assert stats["question_count"] == 2
        assert "direct" in stats["category_distribution"]
        assert "multi_hop" in stats["category_distribution"]


class TestSyntheticDataset:
    """Tests for synthetic dataset generation."""

    def test_generate_synthetic(self) -> None:
        """Test generating synthetic dataset."""
        dataset = generate_synthetic_dataset(n_files=10, n_questions=20, seed=42)

        assert dataset.file_count > 0
        assert dataset.question_count > 0
        assert dataset.metadata.get("synthetic") is True

    def test_reproducibility(self) -> None:
        """Test that same seed produces same dataset."""
        dataset1 = generate_synthetic_dataset(seed=123)
        dataset2 = generate_synthetic_dataset(seed=123)

        assert dataset1.file_count == dataset2.file_count
        assert dataset1.question_count == dataset2.question_count

    def test_different_seeds(self) -> None:
        """Test that different seeds produce different datasets."""
        dataset1 = generate_synthetic_dataset(seed=1)
        dataset2 = generate_synthetic_dataset(seed=2)

        # Should have some differences (questions may differ)
        q1_ids = {q.question_id for q in dataset1.questions}
        q2_ids = {q.question_id for q in dataset2.questions}
        # IDs will be the same but content may differ


class TestLoadFromFile:
    """Tests for loading datasets from files."""

    def test_load_from_json(self, tmp_path: Path) -> None:
        """Test loading from a JSON file."""
        data = {
            "files": [{"path": "test.txt", "content": "Test content", "entity_type": "test"}],
            "questions": [
                {
                    "question_id": "q1",
                    "question": "What is this?",
                    "answer": "Test",
                    "category": "direct",
                    "hop_count": 1,
                }
            ],
        }

        filepath = tmp_path / "benchmark.json"
        with open(filepath, "w") as f:
            json.dump(data, f)

        dataset = load_contextbench_from_file(filepath)
        assert dataset.file_count == 1
        assert dataset.question_count == 1
        assert dataset.questions[0].question_text == "What is this?"

    def test_load_file_not_found(self, tmp_path: Path) -> None:
        """Test loading from non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_contextbench_from_file(tmp_path / "nonexistent.json")
