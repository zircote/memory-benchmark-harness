"""Tests for annotation guidelines and rubrics."""

from __future__ import annotations

import pytest

from src.validation.annotation import (
    AnnotationGuidelines,
    AnnotationRubric,
    RubricLevel,
    create_default_guidelines,
)


class TestRubricLevel:
    """Tests for RubricLevel enum."""

    def test_rubric_level_values(self) -> None:
        """Test all rubric levels exist with correct values."""
        assert RubricLevel.CORRECT.value == "correct"
        assert RubricLevel.PARTIALLY_CORRECT.value == "partially_correct"
        assert RubricLevel.INCORRECT.value == "incorrect"
        assert RubricLevel.CANNOT_JUDGE.value == "cannot_judge"

    def test_from_score_correct(self) -> None:
        """Test score >= 0.8 maps to CORRECT."""
        assert RubricLevel.from_score(1.0) == RubricLevel.CORRECT
        assert RubricLevel.from_score(0.9) == RubricLevel.CORRECT
        assert RubricLevel.from_score(0.8) == RubricLevel.CORRECT

    def test_from_score_partially_correct(self) -> None:
        """Test score in [0.4, 0.8) maps to PARTIALLY_CORRECT."""
        assert RubricLevel.from_score(0.79) == RubricLevel.PARTIALLY_CORRECT
        assert RubricLevel.from_score(0.5) == RubricLevel.PARTIALLY_CORRECT
        assert RubricLevel.from_score(0.4) == RubricLevel.PARTIALLY_CORRECT

    def test_from_score_incorrect(self) -> None:
        """Test score < 0.4 maps to INCORRECT."""
        assert RubricLevel.from_score(0.39) == RubricLevel.INCORRECT
        assert RubricLevel.from_score(0.2) == RubricLevel.INCORRECT
        assert RubricLevel.from_score(0.0) == RubricLevel.INCORRECT

    def test_from_score_edge_cases(self) -> None:
        """Test boundary conditions."""
        # Exactly at thresholds
        assert RubricLevel.from_score(0.8) == RubricLevel.CORRECT
        assert RubricLevel.from_score(0.4) == RubricLevel.PARTIALLY_CORRECT


class TestAnnotationRubric:
    """Tests for AnnotationRubric dataclass."""

    def test_rubric_creation(self) -> None:
        """Test creating a rubric."""
        rubric = AnnotationRubric(
            level=RubricLevel.CORRECT,
            description="Fully correct answer",
            examples=("Example 1", "Example 2"),
            criteria=("Criterion 1", "Criterion 2"),
        )

        assert rubric.level == RubricLevel.CORRECT
        assert rubric.description == "Fully correct answer"
        assert len(rubric.examples) == 2
        assert len(rubric.criteria) == 2

    def test_rubric_defaults(self) -> None:
        """Test rubric with default values."""
        rubric = AnnotationRubric(
            level=RubricLevel.INCORRECT,
            description="Wrong answer",
        )

        assert rubric.examples == ()
        assert rubric.criteria == ()

    def test_rubric_is_frozen(self) -> None:
        """Test that rubric is immutable."""
        rubric = AnnotationRubric(
            level=RubricLevel.CORRECT,
            description="Test",
        )

        with pytest.raises(AttributeError):
            rubric.description = "Changed"  # type: ignore


class TestAnnotationGuidelines:
    """Tests for AnnotationGuidelines dataclass."""

    @pytest.fixture
    def sample_guidelines(self) -> AnnotationGuidelines:
        """Create sample guidelines for testing."""
        rubrics = [
            AnnotationRubric(
                level=RubricLevel.CORRECT,
                description="Correct answer",
            ),
            AnnotationRubric(
                level=RubricLevel.INCORRECT,
                description="Wrong answer",
            ),
        ]
        return AnnotationGuidelines(
            title="Test Guidelines",
            version="1.0",
            overview="Overview text",
            task_description="Task description",
            rubrics=rubrics,
        )

    def test_guidelines_creation(self, sample_guidelines: AnnotationGuidelines) -> None:
        """Test creating guidelines."""
        assert sample_guidelines.title == "Test Guidelines"
        assert sample_guidelines.version == "1.0"
        assert len(sample_guidelines.rubrics) == 2

    def test_get_rubric_found(self, sample_guidelines: AnnotationGuidelines) -> None:
        """Test getting existing rubric."""
        rubric = sample_guidelines.get_rubric(RubricLevel.CORRECT)
        assert rubric is not None
        assert rubric.level == RubricLevel.CORRECT
        assert rubric.description == "Correct answer"

    def test_get_rubric_not_found(self, sample_guidelines: AnnotationGuidelines) -> None:
        """Test getting non-existent rubric."""
        rubric = sample_guidelines.get_rubric(RubricLevel.CANNOT_JUDGE)
        assert rubric is None

    def test_format_markdown(self, sample_guidelines: AnnotationGuidelines) -> None:
        """Test markdown formatting."""
        markdown = sample_guidelines.format_markdown()

        assert "# Test Guidelines" in markdown
        assert "Version: 1.0" in markdown
        assert "## Overview" in markdown
        assert "## Task Description" in markdown
        assert "## Annotation Rubric" in markdown
        assert "### Correct" in markdown
        assert "### Incorrect" in markdown

    def test_format_markdown_with_edge_cases(self) -> None:
        """Test markdown with edge cases."""
        guidelines = AnnotationGuidelines(
            title="Test",
            version="1.0",
            overview="Overview",
            task_description="Task",
            rubrics=[],
            edge_cases=["Edge case 1", "Edge case 2"],
            disagreement_protocol="Protocol text",
        )

        markdown = guidelines.format_markdown()

        assert "## Edge Cases" in markdown
        assert "Edge case 1" in markdown
        assert "## Disagreement Protocol" in markdown
        assert "Protocol text" in markdown

    def test_format_markdown_with_criteria_and_examples(self) -> None:
        """Test markdown includes criteria and examples."""
        guidelines = AnnotationGuidelines(
            title="Test",
            version="1.0",
            overview="Overview",
            task_description="Task",
            rubrics=[
                AnnotationRubric(
                    level=RubricLevel.CORRECT,
                    description="Desc",
                    criteria=("Criterion A",),
                    examples=("Example X",),
                ),
            ],
        )

        markdown = guidelines.format_markdown()

        assert "**Criteria:**" in markdown
        assert "Criterion A" in markdown
        assert "**Examples:**" in markdown
        assert "Example X" in markdown

    def test_save_markdown(self, sample_guidelines: AnnotationGuidelines, tmp_path) -> None:
        """Test saving guidelines to file."""
        filepath = tmp_path / "guidelines.md"
        sample_guidelines.save_markdown(str(filepath))

        assert filepath.exists()
        content = filepath.read_text()
        assert "# Test Guidelines" in content


class TestDefaultGuidelines:
    """Tests for default guidelines creation."""

    def test_create_default_guidelines(self) -> None:
        """Test creating default guidelines."""
        guidelines = create_default_guidelines()

        assert guidelines.title == "Memory System Evaluation Annotation Guidelines"
        assert guidelines.version == "1.0.0"
        assert len(guidelines.rubrics) == 4

    def test_default_rubrics_complete(self) -> None:
        """Test all rubric levels are present."""
        guidelines = create_default_guidelines()

        for level in RubricLevel:
            rubric = guidelines.get_rubric(level)
            assert rubric is not None, f"Missing rubric for {level}"
            assert rubric.description, f"Empty description for {level}"
            assert rubric.criteria, f"No criteria for {level}"

    def test_default_edge_cases(self) -> None:
        """Test edge cases are defined."""
        guidelines = create_default_guidelines()

        assert len(guidelines.edge_cases) > 0
        # Check for key edge cases
        edge_text = " ".join(guidelines.edge_cases)
        assert "synonym" in edge_text.lower()
        assert "date" in edge_text.lower()

    def test_default_disagreement_protocol(self) -> None:
        """Test disagreement protocol is defined."""
        guidelines = create_default_guidelines()

        assert guidelines.disagreement_protocol
        assert "disagree" in guidelines.disagreement_protocol.lower()

    def test_default_metadata(self) -> None:
        """Test metadata is included."""
        guidelines = create_default_guidelines()

        assert "created" in guidelines.metadata
        assert "purpose" in guidelines.metadata
