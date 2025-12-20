"""Annotation guidelines and rubrics for human validation.

This module defines the annotation guidelines, rubrics, and quality
criteria for human validation of benchmark answers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class RubricLevel(Enum):
    """Levels for annotation rubrics."""

    CORRECT = "correct"
    PARTIALLY_CORRECT = "partially_correct"
    INCORRECT = "incorrect"
    CANNOT_JUDGE = "cannot_judge"

    @classmethod
    def from_score(cls, score: float) -> RubricLevel:
        """Convert a numeric score to rubric level.

        Args:
            score: Score between 0 and 1

        Returns:
            Corresponding RubricLevel
        """
        if score >= 0.8:
            return cls.CORRECT
        if score >= 0.4:
            return cls.PARTIALLY_CORRECT
        return cls.INCORRECT


@dataclass(frozen=True, slots=True)
class AnnotationRubric:
    """A rubric item for annotation.

    Attributes:
        level: The rubric level
        description: What this level means
        examples: Example answers at this level
        criteria: Specific criteria for this level
    """

    level: RubricLevel
    description: str
    examples: tuple[str, ...] = ()
    criteria: tuple[str, ...] = ()


@dataclass
class AnnotationGuidelines:
    """Complete annotation guidelines for a validation task.

    Attributes:
        title: Guidelines title
        version: Version identifier
        overview: High-level description
        task_description: What annotators should do
        rubrics: List of rubric items
        edge_cases: Edge case handling guidance
        disagreement_protocol: How to handle disagreements
        metadata: Additional metadata
    """

    title: str
    version: str
    overview: str
    task_description: str
    rubrics: list[AnnotationRubric]
    edge_cases: list[str] = field(default_factory=list)
    disagreement_protocol: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_rubric(self, level: RubricLevel) -> AnnotationRubric | None:
        """Get rubric for a specific level.

        Args:
            level: Rubric level to find

        Returns:
            Matching AnnotationRubric or None
        """
        for rubric in self.rubrics:
            if rubric.level == level:
                return rubric
        return None

    def format_markdown(self) -> str:
        """Format guidelines as markdown.

        Returns:
            Markdown string
        """
        lines = [
            f"# {self.title}",
            "",
            f"*Version: {self.version}*",
            "",
            "## Overview",
            "",
            self.overview,
            "",
            "## Task Description",
            "",
            self.task_description,
            "",
            "## Annotation Rubric",
            "",
        ]

        for rubric in self.rubrics:
            lines.extend(
                [
                    f"### {rubric.level.value.replace('_', ' ').title()}",
                    "",
                    rubric.description,
                    "",
                ]
            )

            if rubric.criteria:
                lines.append("**Criteria:**")
                for criterion in rubric.criteria:
                    lines.append(f"- {criterion}")
                lines.append("")

            if rubric.examples:
                lines.append("**Examples:**")
                for example in rubric.examples:
                    lines.append(f"- {example}")
                lines.append("")

        if self.edge_cases:
            lines.extend(
                [
                    "## Edge Cases",
                    "",
                ]
            )
            for case in self.edge_cases:
                lines.append(f"- {case}")
            lines.append("")

        if self.disagreement_protocol:
            lines.extend(
                [
                    "## Disagreement Protocol",
                    "",
                    self.disagreement_protocol,
                    "",
                ]
            )

        return "\n".join(lines)

    def save_markdown(self, path: str) -> None:
        """Save guidelines to a markdown file.

        Args:
            path: Path to save to
        """
        with open(path, "w") as f:
            f.write(self.format_markdown())


def create_default_guidelines() -> AnnotationGuidelines:
    """Create the default annotation guidelines for memory system evaluation.

    Returns:
        Default AnnotationGuidelines
    """
    rubrics = [
        AnnotationRubric(
            level=RubricLevel.CORRECT,
            description="The answer is completely correct and directly addresses the question.",
            criteria=(
                "Answer matches the expected answer exactly or semantically",
                "All key facts are accurately stated",
                "No significant information is missing",
                "No incorrect information is included",
            ),
            examples=(
                "Q: 'What is Alice's job?' A: 'Engineer' (expected: 'Engineer')",
                "Q: 'When did the event occur?' A: 'January 15, 2024' (expected: 'Jan 15, 2024')",
            ),
        ),
        AnnotationRubric(
            level=RubricLevel.PARTIALLY_CORRECT,
            description="The answer is partially correct but missing key information or contains minor errors.",
            criteria=(
                "Core answer is correct but incomplete",
                "Contains the right concept but wrong details",
                "Answer is related but not precisely what was asked",
                "Some but not all key facts are correct",
            ),
            examples=(
                "Q: 'What are Alice's hobbies?' A: 'hiking' (expected: 'hiking, reading, chess')",
                "Q: 'When was the project completed?' A: '2024' (expected: 'March 2024')",
            ),
        ),
        AnnotationRubric(
            level=RubricLevel.INCORRECT,
            description="The answer is wrong or unrelated to the question.",
            criteria=(
                "Answer contradicts the correct answer",
                "Answer is completely off-topic",
                "Critical facts are wrong",
                "Answer is a hallucination not supported by context",
            ),
            examples=(
                "Q: 'What is Alice's job?' A: 'Teacher' (expected: 'Engineer')",
                "Q: 'When did the meeting occur?' A: 'I don't have that information' (when it's in context)",
            ),
        ),
        AnnotationRubric(
            level=RubricLevel.CANNOT_JUDGE,
            description="The question or answer is ambiguous and cannot be reliably judged.",
            criteria=(
                "Question is ambiguous or unclear",
                "Expected answer has multiple valid interpretations",
                "Context is insufficient to determine correctness",
                "Technical domain knowledge is required beyond guidelines",
            ),
            examples=(
                "Q: 'How did they feel?' (subjective, multiple valid answers)",
                "Q: 'What happened next?' (context cutoff, incomplete information)",
            ),
        ),
    ]

    edge_cases = [
        "Synonym equivalence: 'Engineer' and 'Software Engineer' may be equivalent depending on context",
        "Date formats: 'Jan 1' and 'January 1st' are equivalent",
        "Numerical precision: '3.14' and '3.1' may or may not be equivalent depending on context",
        "Case sensitivity: Generally ignore case unless specifically relevant",
        "Abstention: 'I don't know' should be INCORRECT if the answer exists in context",
        "Extra information: Correct answer with additional correct details is still CORRECT",
        "Ordering: For list answers, order matters only if explicitly stated",
    ]

    disagreement_protocol = """
When annotators disagree:
1. Both annotators review the disputed sample together
2. Discuss specific criteria that led to different judgments
3. Reference edge case guidance if applicable
4. If still disagreeing, escalate to a third annotator
5. Record the disagreement and final resolution in metadata
"""

    return AnnotationGuidelines(
        title="Memory System Evaluation Annotation Guidelines",
        version="1.0.0",
        overview="""
These guidelines define how to evaluate answers generated by memory-augmented
language models on benchmark questions. The goal is to assess whether the
model's answer is correct given the context and question.

Annotators should focus on semantic correctness rather than exact string
matching. An answer that conveys the same meaning as the expected answer
should be marked as correct.
""".strip(),
        task_description="""
For each sample, you will see:
1. The question asked
2. The expected answer(s)
3. The model's generated answer
4. The context/memories used (if any)

Your task is to rate the model's answer using the rubric below. Focus on
whether the answer is factually correct and addresses the question, not on
style or verbosity.
""".strip(),
        rubrics=rubrics,
        edge_cases=edge_cases,
        disagreement_protocol=disagreement_protocol,
        metadata={
            "created": "2025-12-20",
            "purpose": "Memory system benchmark validation",
        },
    )
