"""Context-Bench evaluation pipeline.

This module orchestrates the evaluation process for Context-Bench,
handling file navigation, answer generation, and result aggregation.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from src.benchmarks.contextbench.dataset import (
    ContextBenchDataset,
    ContextBenchQuestion,
    QuestionCategory,
)
from src.benchmarks.contextbench.wrapper import (
    ContextBenchAgent,
)
from src.evaluation.judge import Judgment, JudgmentResult, LLMJudge

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, int, int], None]


@dataclass(slots=True)
class QuestionResult:
    """Result for a single question evaluation.

    Attributes:
        question_id: ID of the question
        question_text: The question
        ground_truth: Expected answer
        generated_answer: Agent's answer
        judgment: LLM judge result
        correct: Whether answer was correct
        category: Question category
        hop_count: Number of hops required
        operations_count: Number of file operations
        tokens_read: Total tokens read from files
        cost_estimate: Estimated cost in tokens
        metadata: Additional result metadata
    """

    question_id: str
    question_text: str
    ground_truth: str
    generated_answer: str
    judgment: Judgment | None
    correct: bool
    category: QuestionCategory
    hop_count: int
    operations_count: int
    tokens_read: int
    cost_estimate: float
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
            "category": self.category.value,
            "hop_count": self.hop_count,
            "operations_count": self.operations_count,
            "tokens_read": self.tokens_read,
            "cost_estimate": self.cost_estimate,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class EvaluationResult:
    """Complete evaluation result for Context-Bench.

    Attributes:
        question_results: Individual question results
        total_questions: Total questions evaluated
        correct_count: Number of correct answers
        accuracy: Overall accuracy
        total_cost: Total estimated cost
        avg_operations: Average operations per question
        category_breakdown: Accuracy by category
        hop_breakdown: Accuracy by hop count
        started_at: Start time
        completed_at: Completion time
        metadata: Additional metadata
    """

    question_results: list[QuestionResult]
    total_questions: int
    correct_count: int
    accuracy: float
    total_cost: float
    avg_operations: float
    category_breakdown: dict[str, dict[str, Any]]
    hop_breakdown: dict[int, dict[str, Any]]
    started_at: str
    completed_at: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "question_results": [r.to_dict() for r in self.question_results],
            "total_questions": self.total_questions,
            "correct_count": self.correct_count,
            "accuracy": self.accuracy,
            "total_cost": self.total_cost,
            "avg_operations": self.avg_operations,
            "category_breakdown": self.category_breakdown,
            "hop_breakdown": self.hop_breakdown,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class ContextBenchPipeline:
    """Orchestrates Context-Bench evaluation.

    Attributes:
        agent: The Context-Bench agent
        judge: LLM judge for evaluation
        cost_per_1k_tokens: Cost per 1000 tokens for estimation
        progress_callback: Optional progress callback
    """

    agent: ContextBenchAgent
    judge: LLMJudge
    cost_per_1k_tokens: float = 0.01  # Default cost estimate
    progress_callback: ProgressCallback | None = None

    def _report_progress(self, phase: str, current: int, total: int) -> None:
        """Report progress if callback is set."""
        if self.progress_callback:
            self.progress_callback(phase, current, total)

    def _estimate_cost(self, tokens: int) -> float:
        """Estimate cost based on token count."""
        return (tokens / 1000) * self.cost_per_1k_tokens

    def _evaluate_question(
        self,
        question: ContextBenchQuestion,
        use_simple_mode: bool = False,
    ) -> QuestionResult:
        """Evaluate a single question.

        Args:
            question: The question to evaluate
            use_simple_mode: Use simple single-shot mode

        Returns:
            QuestionResult with answer and judgment
        """
        # Generate answer
        if use_simple_mode:
            op_result = self.agent.answer_question_simple(question)
        else:
            op_result = self.agent.answer_question(question)

        # Get judgment
        try:
            judgment = self.judge.judge(
                question=question.question_text,
                reference_answer=question.answer,
                model_answer=op_result.answer,
            )
            correct = judgment.result == JudgmentResult.CORRECT
        except Exception as e:
            logger.error(f"Judgment failed for {question.question_id}: {e}")
            judgment = None
            correct = False

        cost = self._estimate_cost(op_result.total_tokens_read)

        return QuestionResult(
            question_id=question.question_id,
            question_text=question.question_text,
            ground_truth=question.answer,
            generated_answer=op_result.answer,
            judgment=judgment,
            correct=correct,
            category=question.category,
            hop_count=question.hop_count,
            operations_count=op_result.operation_count,
            tokens_read=op_result.total_tokens_read,
            cost_estimate=cost,
            metadata=op_result.metadata,
        )

    def _aggregate_by_category(
        self,
        results: list[QuestionResult],
    ) -> dict[str, dict[str, Any]]:
        """Aggregate results by question category."""
        breakdown: dict[str, dict[str, Any]] = {}

        for cat in QuestionCategory:
            cat_results = [r for r in results if r.category == cat]
            if cat_results:
                correct = sum(1 for r in cat_results if r.correct)
                total_cost = sum(r.cost_estimate for r in cat_results)
                breakdown[cat.value] = {
                    "total": len(cat_results),
                    "correct": correct,
                    "accuracy": correct / len(cat_results),
                    "total_cost": total_cost,
                    "avg_operations": sum(r.operations_count for r in cat_results)
                    / len(cat_results),
                }

        return breakdown

    def _aggregate_by_hops(
        self,
        results: list[QuestionResult],
    ) -> dict[int, dict[str, Any]]:
        """Aggregate results by hop count."""
        breakdown: dict[int, dict[str, Any]] = {}

        hop_counts = {r.hop_count for r in results}
        for hop in sorted(hop_counts):
            hop_results = [r for r in results if r.hop_count == hop]
            if hop_results:
                correct = sum(1 for r in hop_results if r.correct)
                breakdown[hop] = {
                    "total": len(hop_results),
                    "correct": correct,
                    "accuracy": correct / len(hop_results),
                    "avg_operations": sum(r.operations_count for r in hop_results)
                    / len(hop_results),
                }

        return breakdown

    def evaluate(
        self,
        dataset: ContextBenchDataset | None = None,
        questions: list[ContextBenchQuestion] | None = None,
        use_simple_mode: bool = False,
        index_files: bool = True,
    ) -> EvaluationResult:
        """Evaluate the Context-Bench dataset.

        Args:
            dataset: Dataset to evaluate (uses agent's dataset if None)
            questions: Specific questions to evaluate (all if None)
            use_simple_mode: Use simple single-shot answering
            index_files: Whether to index files into memory first

        Returns:
            EvaluationResult with all results and aggregations
        """
        started_at = datetime.now().isoformat()

        if dataset is None:
            dataset = self.agent.dataset

        if questions is None:
            questions = dataset.questions

        # Index files if requested
        if index_files:
            logger.info("Indexing files into memory...")
            self.agent.index_files()

        logger.info(f"Evaluating {len(questions)} questions...")

        results: list[QuestionResult] = []
        total = len(questions)

        for i, question in enumerate(questions):
            self._report_progress("Evaluating", i + 1, total)

            try:
                result = self._evaluate_question(question, use_simple_mode)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to evaluate {question.question_id}: {e}")
                results.append(
                    QuestionResult(
                        question_id=question.question_id,
                        question_text=question.question_text,
                        ground_truth=question.answer,
                        generated_answer="",
                        judgment=None,
                        correct=False,
                        category=question.category,
                        hop_count=question.hop_count,
                        operations_count=0,
                        tokens_read=0,
                        cost_estimate=0,
                        metadata={"error": str(e)},
                    )
                )

        # Calculate aggregates
        correct_count = sum(1 for r in results if r.correct)
        accuracy = correct_count / total if total > 0 else 0
        total_cost = sum(r.cost_estimate for r in results)
        avg_ops = sum(r.operations_count for r in results) / total if total > 0 else 0

        completed_at = datetime.now().isoformat()

        return EvaluationResult(
            question_results=results,
            total_questions=total,
            correct_count=correct_count,
            accuracy=accuracy,
            total_cost=total_cost,
            avg_operations=avg_ops,
            category_breakdown=self._aggregate_by_category(results),
            hop_breakdown=self._aggregate_by_hops(results),
            started_at=started_at,
            completed_at=completed_at,
            metadata={
                "use_simple_mode": use_simple_mode,
                "indexed_files": index_files,
            },
        )

    def evaluate_category(
        self,
        category: QuestionCategory,
        dataset: ContextBenchDataset | None = None,
    ) -> EvaluationResult:
        """Evaluate questions of a specific category.

        Args:
            category: Category to evaluate
            dataset: Dataset to use

        Returns:
            EvaluationResult for the category
        """
        if dataset is None:
            dataset = self.agent.dataset

        questions = dataset.questions_by_category(category)
        return self.evaluate(dataset=dataset, questions=questions)

    def evaluate_multi_hop(
        self,
        dataset: ContextBenchDataset | None = None,
    ) -> EvaluationResult:
        """Evaluate only multi-hop questions.

        Args:
            dataset: Dataset to use

        Returns:
            EvaluationResult for multi-hop questions
        """
        if dataset is None:
            dataset = self.agent.dataset

        questions = dataset.multi_hop_questions()
        return self.evaluate(dataset=dataset, questions=questions)
