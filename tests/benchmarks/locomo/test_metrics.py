"""Tests for LoCoMo metrics calculation."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from src.benchmarks.locomo.dataset import QACategory
from src.benchmarks.locomo.metrics import (
    AdversarialMetrics,
    CategoryMetricsReport,
    ConversationDifficultyMetrics,
    LoCoMoMetrics,
    LoCoMoMetricsCalculator,
    calculate_metrics,
    compare_metrics,
)
from src.benchmarks.locomo.pipeline import (
    AssessmentResult,
    CategoryMetrics,
    ConversationMetrics,
    QuestionResult,
)
from src.evaluation.judge import Judgment, JudgmentResult


class TestCategoryMetricsReport:
    """Tests for CategoryMetricsReport dataclass."""

    def test_creation(self) -> None:
        """Test creating a CategoryMetricsReport."""
        report = CategoryMetricsReport(
            category_name="IDENTITY",
            category_number=1,
            total_questions=10,
            correct_count=8,
            partial_count=1,
            accuracy=0.8,
            mean_score=0.85,
            mean_latency_ms=150.0,
            confidence_interval=(0.75, 0.90),
        )

        assert report.category_name == "IDENTITY"
        assert report.category_number == 1
        assert report.total_questions == 10
        assert report.correct_count == 8
        assert report.partial_count == 1
        assert report.accuracy == 0.8
        assert report.mean_score == 0.85
        assert report.mean_latency_ms == 150.0
        assert report.confidence_interval == (0.75, 0.90)

    def test_without_confidence_interval(self) -> None:
        """Test creating without CI."""
        report = CategoryMetricsReport(
            category_name="TEMPORAL",
            category_number=2,
            total_questions=5,
            correct_count=3,
            partial_count=1,
            accuracy=0.6,
            mean_score=0.65,
            mean_latency_ms=200.0,
        )

        assert report.confidence_interval is None


class TestAdversarialMetrics:
    """Tests for AdversarialMetrics dataclass."""

    def test_creation(self) -> None:
        """Test creating AdversarialMetrics."""
        metrics = AdversarialMetrics(
            total_adversarial=10,
            correctly_identified=7,
            incorrectly_answered=3,
            identification_rate=0.7,
            false_acceptance_rate=0.3,
        )

        assert metrics.total_adversarial == 10
        assert metrics.correctly_identified == 7
        assert metrics.incorrectly_answered == 3
        assert metrics.identification_rate == 0.7
        assert metrics.false_acceptance_rate == 0.3


class TestConversationDifficultyMetrics:
    """Tests for ConversationDifficultyMetrics dataclass."""

    def test_creation(self) -> None:
        """Test creating ConversationDifficultyMetrics."""
        metrics = ConversationDifficultyMetrics(
            conversation_id="conv_1",
            num_sessions=10,
            num_turns=150,
            token_count_estimate=7500,
            questions_assessed=20,
            accuracy=0.75,
            mean_score=0.80,
            difficulty_score=0.45,
        )

        assert metrics.conversation_id == "conv_1"
        assert metrics.num_sessions == 10
        assert metrics.num_turns == 150
        assert metrics.token_count_estimate == 7500
        assert metrics.questions_assessed == 20
        assert metrics.accuracy == 0.75
        assert metrics.mean_score == 0.80
        assert metrics.difficulty_score == 0.45


class TestLoCoMoMetrics:
    """Tests for LoCoMoMetrics dataclass."""

    @pytest.fixture
    def sample_metrics(self) -> LoCoMoMetrics:
        """Create sample metrics for testing."""
        return LoCoMoMetrics(
            total_questions=50,
            total_conversations=5,
            aggregate_accuracy=0.75,
            aggregate_mean_score=0.80,
            category_metrics={
                "IDENTITY": CategoryMetricsReport(
                    category_name="IDENTITY",
                    category_number=1,
                    total_questions=10,
                    correct_count=8,
                    partial_count=1,
                    accuracy=0.8,
                    mean_score=0.85,
                    mean_latency_ms=100.0,
                ),
                "ADVERSARIAL": CategoryMetricsReport(
                    category_name="ADVERSARIAL",
                    category_number=5,
                    total_questions=10,
                    correct_count=6,
                    partial_count=0,
                    accuracy=0.6,
                    mean_score=0.6,
                    mean_latency_ms=200.0,
                ),
            },
            adversarial_metrics=AdversarialMetrics(
                total_adversarial=10,
                correctly_identified=6,
                incorrectly_answered=4,
                identification_rate=0.6,
                false_acceptance_rate=0.4,
            ),
            conversation_difficulty=[
                ConversationDifficultyMetrics(
                    conversation_id="conv_1",
                    num_sessions=10,
                    num_turns=150,
                    token_count_estimate=7500,
                    questions_assessed=10,
                    accuracy=0.8,
                    mean_score=0.85,
                    difficulty_score=0.35,
                ),
            ],
            aggregate_ci=(0.70, 0.85),
            latency_stats={
                "mean_ms": 150.0,
                "median_ms": 140.0,
                "p95_ms": 250.0,
            },
            metadata={"model": "test-model"},
        )

    def test_to_dict(self, sample_metrics: LoCoMoMetrics) -> None:
        """Test conversion to dictionary."""
        result = sample_metrics.to_dict()

        assert result["total_questions"] == 50
        assert result["total_conversations"] == 5
        assert result["aggregate_accuracy"] == 0.75
        assert result["aggregate_mean_score"] == 0.80
        assert result["aggregate_ci"] == (0.70, 0.85)
        assert "IDENTITY" in result["category_metrics"]
        assert "ADVERSARIAL" in result["category_metrics"]
        assert result["adversarial_metrics"]["total_adversarial"] == 10
        assert len(result["conversation_difficulty"]) == 1
        assert result["latency_stats"]["mean_ms"] == 150.0
        assert result["metadata"]["model"] == "test-model"

    def test_get_summary(self, sample_metrics: LoCoMoMetrics) -> None:
        """Test human-readable summary generation."""
        summary = sample_metrics.get_summary()

        assert "LoCoMo Benchmark Results" in summary
        assert "Total Questions: 50" in summary
        assert "Aggregate Accuracy: 75.00%" in summary
        assert "IDENTITY" in summary
        assert "ADVERSARIAL" in summary
        assert "Correctly Identified: 6" in summary
        assert "conv_1" in summary


def create_mock_question_result(
    question_id: str,
    category: QACategory,
    is_correct: bool,
    is_adversarial: bool = False,
    adversarial_handled: bool = False,
    score: float | None = None,
    latency_ms: float = 100.0,
) -> QuestionResult:
    """Create a mock QuestionResult for testing."""
    if score is None:
        score = 1.0 if is_correct else 0.0

    result_enum = JudgmentResult.CORRECT if is_correct else JudgmentResult.INCORRECT

    judgment = Judgment(
        result=result_enum,
        score=score,
        reasoning="Test reasoning",
        question="Test question",
        reference_answer="Test answer",
        model_answer="Model answer",
        metadata={},
        cached=False,
        timestamp=datetime.now(),
    )

    return QuestionResult(
        question_id=question_id,
        conversation_id="conv_1",
        question_text="Test question?",
        category=category,
        ground_truth="Test answer",
        agent_answer="Model answer",
        judgment=judgment,
        is_adversarial=is_adversarial,
        is_abstention_actual=False,
        adversarial_handled=adversarial_handled,
        latency_ms=latency_ms,
        evidence_sessions=[1, 2],
    )


def create_mock_assessment_result(
    question_results: list[QuestionResult],
) -> AssessmentResult:
    """Create a mock AssessmentResult for testing."""
    # Create category metrics
    from collections import defaultdict

    by_category: dict[str, list[QuestionResult]] = defaultdict(list)
    for result in question_results:
        by_category[result.category_name].append(result)

    category_metrics = {}
    for cat_name, results in by_category.items():
        correct = sum(1 for r in results if r.is_correct)
        partial = sum(1 for r in results if r.is_partial)
        scores = [r.score for r in results]
        latencies = [r.latency_ms for r in results]

        category_metrics[cat_name] = CategoryMetrics(
            category=results[0].category,
            total_questions=len(results),
            correct_count=correct,
            partial_count=partial,
            mean_score=float(np.mean(scores)) if scores else 0.0,
            mean_latency_ms=float(np.mean(latencies)) if latencies else 0.0,
        )

    # Create conversation metrics
    conversation_metrics = {
        "conv_1": ConversationMetrics(
            conversation_id="conv_1",
            sessions_ingested=5,
            turns_ingested=50,
            questions_assessed=len(question_results),
            correct_count=sum(1 for r in question_results if r.is_correct),
            mean_score=float(np.mean([r.score for r in question_results]))
            if question_results
            else 0.0,
        ),
    }

    return AssessmentResult(
        question_results=question_results,
        category_metrics=category_metrics,
        conversation_metrics=conversation_metrics,
        total_questions=len(question_results),
        ingestion_time_ms=1000.0,
        assessment_time_ms=5000.0,
        started_at=datetime.now(),
        completed_at=datetime.now(),
    )


class TestLoCoMoMetricsCalculator:
    """Tests for LoCoMoMetricsCalculator."""

    def test_calculate_basic(self) -> None:
        """Test basic metrics calculation."""
        results = [
            create_mock_question_result("q1", QACategory.IDENTITY, True),
            create_mock_question_result("q2", QACategory.IDENTITY, True),
            create_mock_question_result("q3", QACategory.IDENTITY, False),
            create_mock_question_result("q4", QACategory.TEMPORAL, True),
            create_mock_question_result("q5", QACategory.TEMPORAL, False),
        ]

        assessment = create_mock_assessment_result(results)
        calculator = LoCoMoMetricsCalculator()
        metrics = calculator.calculate(assessment)

        assert metrics.total_questions == 5
        assert metrics.aggregate_accuracy == 0.6  # 3/5
        assert "IDENTITY" in metrics.category_metrics
        assert "TEMPORAL" in metrics.category_metrics
        assert metrics.category_metrics["IDENTITY"].accuracy == pytest.approx(2 / 3, rel=0.01)
        assert metrics.category_metrics["TEMPORAL"].accuracy == pytest.approx(0.5, rel=0.01)

    def test_calculate_adversarial_metrics(self) -> None:
        """Test adversarial metrics calculation."""
        results = [
            create_mock_question_result(
                "q1", QACategory.ADVERSARIAL, True, is_adversarial=True, adversarial_handled=True
            ),
            create_mock_question_result(
                "q2", QACategory.ADVERSARIAL, True, is_adversarial=True, adversarial_handled=True
            ),
            create_mock_question_result(
                "q3", QACategory.ADVERSARIAL, False, is_adversarial=True, adversarial_handled=False
            ),
        ]

        assessment = create_mock_assessment_result(results)
        calculator = LoCoMoMetricsCalculator()
        metrics = calculator.calculate(assessment)

        assert metrics.adversarial_metrics.total_adversarial == 3
        assert metrics.adversarial_metrics.correctly_identified == 2
        assert metrics.adversarial_metrics.incorrectly_answered == 1
        assert metrics.adversarial_metrics.identification_rate == pytest.approx(2 / 3, rel=0.01)

    def test_calculate_no_adversarial(self) -> None:
        """Test with no adversarial questions."""
        results = [
            create_mock_question_result("q1", QACategory.IDENTITY, True),
            create_mock_question_result("q2", QACategory.TEMPORAL, True),
        ]

        assessment = create_mock_assessment_result(results)
        calculator = LoCoMoMetricsCalculator()
        metrics = calculator.calculate(assessment)

        assert metrics.adversarial_metrics.total_adversarial == 0
        assert metrics.adversarial_metrics.identification_rate == 1.0  # Default perfect

    def test_calculate_latency_stats(self) -> None:
        """Test latency statistics calculation."""
        results = [
            create_mock_question_result("q1", QACategory.IDENTITY, True, latency_ms=100.0),
            create_mock_question_result("q2", QACategory.IDENTITY, True, latency_ms=150.0),
            create_mock_question_result("q3", QACategory.IDENTITY, True, latency_ms=200.0),
        ]

        assessment = create_mock_assessment_result(results)
        calculator = LoCoMoMetricsCalculator()
        metrics = calculator.calculate(assessment)

        assert metrics.latency_stats["mean_ms"] == 150.0
        assert metrics.latency_stats["median_ms"] == 150.0
        assert metrics.latency_stats["min_ms"] == 100.0
        assert metrics.latency_stats["max_ms"] == 200.0

    def test_calculate_conversation_difficulty(self) -> None:
        """Test conversation difficulty calculation."""
        results = [
            create_mock_question_result("q1", QACategory.IDENTITY, True),
            create_mock_question_result("q2", QACategory.IDENTITY, False),
        ]

        assessment = create_mock_assessment_result(results)
        calculator = LoCoMoMetricsCalculator()
        metrics = calculator.calculate(assessment)

        assert len(metrics.conversation_difficulty) == 1
        diff = metrics.conversation_difficulty[0]
        assert diff.conversation_id == "conv_1"
        assert diff.num_sessions == 5
        assert diff.num_turns == 50
        assert diff.questions_assessed == 2
        assert 0.0 <= diff.difficulty_score <= 1.0

    def test_calculate_empty_results(self) -> None:
        """Test with empty results."""
        assessment = create_mock_assessment_result([])
        calculator = LoCoMoMetricsCalculator()
        metrics = calculator.calculate(assessment)

        assert metrics.total_questions == 0
        assert metrics.aggregate_accuracy == 0.0
        assert len(metrics.category_metrics) == 0
        assert metrics.latency_stats["mean_ms"] == 0.0


class TestCalculateMetricsFunction:
    """Tests for calculate_metrics convenience function."""

    def test_calculate_metrics_basic(self) -> None:
        """Test the convenience function."""
        results = [
            create_mock_question_result("q1", QACategory.IDENTITY, True),
            create_mock_question_result("q2", QACategory.TEMPORAL, False),
        ]

        assessment = create_mock_assessment_result(results)
        metrics = calculate_metrics(assessment)

        assert isinstance(metrics, LoCoMoMetrics)
        assert metrics.total_questions == 2


class TestCompareMetrics:
    """Tests for compare_metrics function."""

    def test_compare_metrics(self) -> None:
        """Test comparing two sets of metrics."""
        results_a = [
            create_mock_question_result("q1", QACategory.IDENTITY, True),
            create_mock_question_result("q2", QACategory.IDENTITY, True),
        ]
        results_b = [
            create_mock_question_result("q1", QACategory.IDENTITY, True),
            create_mock_question_result("q2", QACategory.IDENTITY, False),
        ]

        assessment_a = create_mock_assessment_result(results_a)
        assessment_b = create_mock_assessment_result(results_b)

        metrics_a = calculate_metrics(assessment_a)
        metrics_b = calculate_metrics(assessment_b)

        comparison = compare_metrics(metrics_a, metrics_b, label_a="Git-Notes", label_b="No-Memory")

        assert comparison["labels"]["a"] == "Git-Notes"
        assert comparison["labels"]["b"] == "No-Memory"
        assert comparison["aggregate"]["accuracy_a"] == 1.0
        assert comparison["aggregate"]["accuracy_b"] == 0.5
        assert comparison["aggregate"]["accuracy_diff"] == 0.5

    def test_compare_different_categories(self) -> None:
        """Test comparing with different category coverage."""
        results_a = [
            create_mock_question_result("q1", QACategory.IDENTITY, True),
        ]
        results_b = [
            create_mock_question_result("q1", QACategory.TEMPORAL, True),
        ]

        assessment_a = create_mock_assessment_result(results_a)
        assessment_b = create_mock_assessment_result(results_b)

        metrics_a = calculate_metrics(assessment_a)
        metrics_b = calculate_metrics(assessment_b)

        comparison = compare_metrics(metrics_a, metrics_b)

        # Should have both categories
        assert "IDENTITY" in comparison["by_category"]
        assert "TEMPORAL" in comparison["by_category"]

        # IDENTITY only in A
        assert comparison["by_category"]["IDENTITY"]["accuracy_a"] == 1.0
        assert comparison["by_category"]["IDENTITY"]["accuracy_b"] is None

        # TEMPORAL only in B
        assert comparison["by_category"]["TEMPORAL"]["accuracy_a"] is None
        assert comparison["by_category"]["TEMPORAL"]["accuracy_b"] == 1.0
