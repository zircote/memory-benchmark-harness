"""MemoryAgentBench evaluation pipeline.

This module orchestrates the evaluation process for MemoryAgentBench,
handling question answering, judgment collection, and result aggregation.

The pipeline supports:
1. Per-competency evaluation
2. Focus on conflict resolution tasks
3. Difficulty-stratified analysis
4. Progress tracking via callbacks
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from src.benchmarks.memoryagentbench.dataset import (
    Competency,
    DifficultyLevel,
    MemoryAgentBenchDataset,
    MemoryAgentBenchQuestion,
    MemoryAgentBenchSplit,
)
from src.benchmarks.memoryagentbench.wrapper import (
    MemoryAgentBenchAgent,
)
from src.evaluation.judge import JudgmentResult, LLMJudge

logger = logging.getLogger(__name__)

# Callback type for progress reporting
ProgressCallback = Callable[[str, int, int], None]


@dataclass(slots=True)
class QuestionResult:
    """Result for a single question evaluation.

    Attributes:
        question_id: ID of the evaluated question
        question_text: The question text
        ground_truth: Expected answer(s)
        generated_answer: Agent's answer
        judgment: LLM judge result
        correct: Whether the answer was correct
        competency: Question's competency category
        difficulty: Question's difficulty level
        retrieved_count: Number of memories retrieved
        metadata: Additional result metadata
    """

    question_id: str
    question_text: str
    ground_truth: list[str]
    generated_answer: str
    judgment: JudgmentResult | None
    correct: bool
    competency: Competency
    difficulty: DifficultyLevel
    retrieved_count: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "question_id": self.question_id,
            "question_text": self.question_text,
            "ground_truth": self.ground_truth,
            "generated_answer": self.generated_answer,
            "judgment": self.judgment.to_dict() if self.judgment else None,
            "correct": self.correct,
            "competency": self.competency.value,
            "difficulty": self.difficulty.value,
            "retrieved_count": self.retrieved_count,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class CompetencyResult:
    """Aggregated results for a competency.

    Attributes:
        competency: The evaluated competency
        question_results: Individual question results
        total_questions: Total questions evaluated
        correct_count: Number of correct answers
        accuracy: Accuracy percentage
        difficulty_breakdown: Accuracy by difficulty level
        metadata: Additional result metadata
    """

    competency: Competency
    question_results: list[QuestionResult]
    total_questions: int
    correct_count: int
    accuracy: float
    difficulty_breakdown: dict[str, dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "competency": self.competency.value,
            "question_results": [r.to_dict() for r in self.question_results],
            "total_questions": self.total_questions,
            "correct_count": self.correct_count,
            "accuracy": self.accuracy,
            "difficulty_breakdown": self.difficulty_breakdown,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class SplitResult:
    """Complete evaluation result for a dataset split.

    Attributes:
        competency_results: Results by competency
        total_questions: Total questions across all competencies
        overall_accuracy: Overall accuracy across all competencies
        conflict_resolution_accuracy: Specific CR accuracy (primary metric)
        started_at: Evaluation start time
        completed_at: Evaluation completion time
        metadata: Additional result metadata
    """

    competency_results: dict[Competency, CompetencyResult]
    total_questions: int
    overall_accuracy: float
    conflict_resolution_accuracy: float | None
    started_at: str
    completed_at: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "competency_results": {
                c.value: r.to_dict() for c, r in self.competency_results.items()
            },
            "total_questions": self.total_questions,
            "overall_accuracy": self.overall_accuracy,
            "conflict_resolution_accuracy": self.conflict_resolution_accuracy,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class MemoryAgentBenchPipeline:
    """Orchestrates MemoryAgentBench evaluation.

    This pipeline manages the complete evaluation flow:
    1. Context ingestion for each question
    2. Answer generation using memory-augmented agent
    3. Judgment collection using LLM-as-Judge
    4. Result aggregation by competency and difficulty

    Attributes:
        agent: The memory-augmented agent
        judge: LLM judge for evaluation
        progress_callback: Optional callback for progress updates
    """

    agent: MemoryAgentBenchAgent
    judge: LLMJudge
    progress_callback: ProgressCallback | None = None

    def _report_progress(self, phase: str, current: int, total: int) -> None:
        """Report progress if callback is set."""
        if self.progress_callback:
            self.progress_callback(phase, current, total)

    def _evaluate_question(
        self,
        question: MemoryAgentBenchQuestion,
    ) -> QuestionResult:
        """Evaluate a single question.

        Args:
            question: The question to evaluate

        Returns:
            QuestionResult with answer and judgment
        """
        # Generate answer using appropriate method
        if question.competency == Competency.CONFLICT_RESOLUTION:
            answer_result = self.agent.answer_with_conflict_check(question)
        else:
            answer_result = self.agent.answer_question(question)

        # Get judgment from LLM judge
        try:
            judgment = self.judge.judge(
                question=question.question_text,
                reference_answer=question.answers[0],  # Use first answer as reference
                generated_answer=answer_result.answer,
                additional_references=question.answers[1:] if len(question.answers) > 1 else None,
            )
            correct = judgment.is_correct
        except Exception as e:
            logger.error(f"Judgment failed for {question.question_id}: {e}")
            judgment = None
            correct = False

        return QuestionResult(
            question_id=question.question_id,
            question_text=question.question_text,
            ground_truth=question.answers,
            generated_answer=answer_result.answer,
            judgment=judgment,
            correct=correct,
            competency=question.competency,
            difficulty=question.difficulty,
            retrieved_count=len(answer_result.retrieved_memories),
            metadata={
                **answer_result.metadata,
                "source": question.source,
            },
        )

    def _aggregate_by_difficulty(
        self,
        results: list[QuestionResult],
    ) -> dict[str, dict[str, Any]]:
        """Aggregate results by difficulty level."""
        breakdown: dict[str, dict[str, Any]] = {}

        for level in DifficultyLevel:
            level_results = [r for r in results if r.difficulty == level]
            if level_results:
                correct = sum(1 for r in level_results if r.correct)
                breakdown[level.value] = {
                    "total": len(level_results),
                    "correct": correct,
                    "accuracy": correct / len(level_results) if level_results else 0,
                }

        return breakdown

    def evaluate_split(
        self,
        split: MemoryAgentBenchSplit,
    ) -> CompetencyResult:
        """Evaluate a single competency split.

        Args:
            split: The split to evaluate

        Returns:
            CompetencyResult with all question results
        """
        logger.info(f"Evaluating {split.competency.short_name}: {split.question_count} questions")

        results: list[QuestionResult] = []
        total = split.question_count

        for i, question in enumerate(split.questions):
            self._report_progress(
                f"Evaluating {split.competency.short_name}",
                i + 1,
                total,
            )

            try:
                result = self._evaluate_question(question)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to evaluate {question.question_id}: {e}")
                # Create failed result
                results.append(
                    QuestionResult(
                        question_id=question.question_id,
                        question_text=question.question_text,
                        ground_truth=question.answers,
                        generated_answer="",
                        judgment=None,
                        correct=False,
                        competency=question.competency,
                        difficulty=question.difficulty,
                        retrieved_count=0,
                        metadata={"error": str(e)},
                    )
                )

        correct_count = sum(1 for r in results if r.correct)
        accuracy = correct_count / total if total > 0 else 0

        return CompetencyResult(
            competency=split.competency,
            question_results=results,
            total_questions=total,
            correct_count=correct_count,
            accuracy=accuracy,
            difficulty_breakdown=self._aggregate_by_difficulty(results),
            metadata={
                "evaluated_at": datetime.now().isoformat(),
            },
        )

    def evaluate_dataset(
        self,
        dataset: MemoryAgentBenchDataset,
        competencies: list[Competency] | None = None,
    ) -> SplitResult:
        """Evaluate the full dataset or selected competencies.

        Args:
            dataset: The dataset to evaluate
            competencies: Specific competencies to evaluate (None = all)

        Returns:
            SplitResult with all competency results
        """
        started_at = datetime.now().isoformat()

        if competencies is None:
            competencies = dataset.competencies

        logger.info(
            f"Starting MemoryAgentBench evaluation: " f"{[c.short_name for c in competencies]}"
        )

        competency_results: dict[Competency, CompetencyResult] = {}

        for competency in competencies:
            split = dataset.get_split(competency)
            if split is None:
                logger.warning(f"Split not found for {competency.value}")
                continue

            result = self.evaluate_split(split)
            competency_results[competency] = result

        # Calculate overall metrics
        total_questions = sum(r.total_questions for r in competency_results.values())
        total_correct = sum(r.correct_count for r in competency_results.values())
        overall_accuracy = total_correct / total_questions if total_questions > 0 else 0

        # Get conflict resolution accuracy specifically
        cr_result = competency_results.get(Competency.CONFLICT_RESOLUTION)
        cr_accuracy = cr_result.accuracy if cr_result else None

        completed_at = datetime.now().isoformat()

        return SplitResult(
            competency_results=competency_results,
            total_questions=total_questions,
            overall_accuracy=overall_accuracy,
            conflict_resolution_accuracy=cr_accuracy,
            started_at=started_at,
            completed_at=completed_at,
            metadata={
                "competencies_evaluated": [c.value for c in competencies],
            },
        )

    def evaluate_conflict_resolution(
        self,
        dataset: MemoryAgentBenchDataset,
    ) -> CompetencyResult:
        """Focused evaluation of conflict resolution competency.

        This is the primary metric of interest for this benchmark,
        testing the advantage of git version history.

        Args:
            dataset: The dataset to evaluate

        Returns:
            CompetencyResult for conflict resolution
        """
        split = dataset.get_conflict_resolution_split()
        if split is None:
            raise ValueError("Conflict Resolution split not found in dataset")

        logger.info(f"Evaluating Conflict Resolution: {split.question_count} questions")

        return self.evaluate_split(split)

    def evaluate_single_question(
        self,
        question: MemoryAgentBenchQuestion,
    ) -> QuestionResult:
        """Evaluate a single question (for debugging/testing).

        Args:
            question: The question to evaluate

        Returns:
            QuestionResult with answer and judgment
        """
        return self._evaluate_question(question)
