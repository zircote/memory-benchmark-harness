"""Tests for LongMemEval metrics calculation."""

from __future__ import annotations

from datetime import datetime

import pytest

from src.benchmarks.longmemeval.metrics import (
    AbilityMetrics,
    AbstentionMetrics,
    LongMemEvalMetrics,
    MetricsCalculator,
    compare_results,
)
from src.benchmarks.longmemeval.pipeline import (
    AssessmentResult,
    QuestionResult,
)
from src.evaluation.judge import Judgment, JudgmentResult
from src.evaluation.statistics import StatisticalAnalyzer

# Test fixtures


@pytest.fixture
def sample_judgment_correct() -> Judgment:
    """Create a correct judgment."""
    return Judgment(
        result=JudgmentResult.CORRECT,
        score=1.0,
        reasoning="Correct answer",
        question="What is the name?",
        reference_answer="Alice",
        model_answer="Alice",
        metadata={},
        cached=False,
        timestamp=datetime.now(),
    )


@pytest.fixture
def sample_judgment_partial() -> Judgment:
    """Create a partial judgment."""
    return Judgment(
        result=JudgmentResult.PARTIAL,
        score=0.5,
        reasoning="Partially correct",
        question="What are the hobbies?",
        reference_answer="reading and hiking",
        model_answer="reading",
        metadata={},
        cached=False,
        timestamp=datetime.now(),
    )


@pytest.fixture
def sample_judgment_incorrect() -> Judgment:
    """Create an incorrect judgment."""
    return Judgment(
        result=JudgmentResult.INCORRECT,
        score=0.0,
        reasoning="Incorrect answer",
        question="What is the color?",
        reference_answer="blue",
        model_answer="red",
        metadata={},
        cached=False,
        timestamp=datetime.now(),
    )


@pytest.fixture
def sample_question_results(
    sample_judgment_correct: Judgment,
    sample_judgment_partial: Judgment,
    sample_judgment_incorrect: Judgment,
) -> list[QuestionResult]:
    """Create sample question results covering different types."""
    return [
        # Single-session-user: 2 correct, 1 partial
        QuestionResult(
            question_id="q1",
            question_text="What is the name?",
            question_type="single-session-user",
            ground_truth=["Alice"],
            agent_answer="Alice",
            judgment=sample_judgment_correct,
            is_abstention_expected=False,
            is_abstention_actual=False,
            latency_ms=100.0,
        ),
        QuestionResult(
            question_id="q2",
            question_text="What is the age?",
            question_type="single-session-user",
            ground_truth=["25"],
            agent_answer="25",
            judgment=sample_judgment_correct,
            is_abstention_expected=False,
            is_abstention_actual=False,
            latency_ms=120.0,
        ),
        QuestionResult(
            question_id="q3",
            question_text="What are the hobbies?",
            question_type="single-session-user",
            ground_truth=["reading and hiking"],
            agent_answer="reading",
            judgment=sample_judgment_partial,
            is_abstention_expected=False,
            is_abstention_actual=False,
            latency_ms=150.0,
        ),
        # Multi-session: 1 correct, 1 incorrect
        QuestionResult(
            question_id="q4",
            question_text="What city?",
            question_type="multi-session",
            ground_truth=["NYC"],
            agent_answer="NYC",
            judgment=sample_judgment_correct,
            is_abstention_expected=False,
            is_abstention_actual=False,
            latency_ms=200.0,
        ),
        QuestionResult(
            question_id="q5",
            question_text="What color?",
            question_type="multi-session",
            ground_truth=["blue"],
            agent_answer="red",
            judgment=sample_judgment_incorrect,
            is_abstention_expected=False,
            is_abstention_actual=False,
            latency_ms=180.0,
        ),
        # Abstention: 1 correct abstention, 1 missed abstention
        QuestionResult(
            question_id="q6",
            question_text="Unknown question",
            question_type="abstention",
            ground_truth=["abstain"],
            agent_answer="I don't know",
            judgment=sample_judgment_correct,
            is_abstention_expected=True,
            is_abstention_actual=True,
            latency_ms=90.0,
        ),
        QuestionResult(
            question_id="q7",
            question_text="Another unknown",
            question_type="abstention",
            ground_truth=["abstain"],
            agent_answer="Some guess",
            judgment=sample_judgment_incorrect,
            is_abstention_expected=True,
            is_abstention_actual=False,
            latency_ms=110.0,
        ),
    ]


@pytest.fixture
def sample_assessment_result(
    sample_question_results: list[QuestionResult],
) -> AssessmentResult:
    """Create a sample assessment result."""
    return AssessmentResult(
        dataset_subset="S",
        question_results=sample_question_results,
        total_questions=len(sample_question_results),
        ingestion_time_ms=1000.0,
        assessment_time_ms=5000.0,
        started_at=datetime(2024, 1, 1, 10, 0, 0),
        completed_at=datetime(2024, 1, 1, 10, 5, 0),
        metadata={"adapter": "git-notes"},
    )


# Test AbilityMetrics dataclass


class TestAbilityMetrics:
    """Tests for AbilityMetrics dataclass."""

    def test_create_ability_metrics(self) -> None:
        """Test creating ability metrics."""
        metrics = AbilityMetrics(
            ability_name="single-session-user",
            total_questions=10,
            correct_count=8,
            partial_count=1,
            accuracy=0.8,
            mean_score=0.85,
        )
        assert metrics.ability_name == "single-session-user"
        assert metrics.total_questions == 10
        assert metrics.accuracy == 0.8
        assert metrics.confidence_interval is None

    def test_with_confidence_interval(self) -> None:
        """Test ability metrics with CI."""
        metrics = AbilityMetrics(
            ability_name="multi-session",
            total_questions=20,
            correct_count=15,
            partial_count=2,
            accuracy=0.75,
            mean_score=0.80,
            confidence_interval=(0.65, 0.90),
        )
        assert metrics.confidence_interval == (0.65, 0.90)


# Test AbstentionMetrics dataclass


class TestAbstentionMetrics:
    """Tests for AbstentionMetrics dataclass."""

    def test_create_abstention_metrics(self) -> None:
        """Test creating abstention metrics."""
        metrics = AbstentionMetrics(
            total_abstention_expected=10,
            correct_abstentions=7,
            false_abstentions=2,
            missed_abstentions=3,
            abstention_precision=0.78,
            abstention_recall=0.70,
            abstention_f1=0.74,
        )
        assert metrics.total_abstention_expected == 10
        assert metrics.correct_abstentions == 7
        assert metrics.abstention_f1 == 0.74

    def test_perfect_abstention(self) -> None:
        """Test perfect abstention metrics."""
        metrics = AbstentionMetrics(
            total_abstention_expected=5,
            correct_abstentions=5,
            false_abstentions=0,
            missed_abstentions=0,
            abstention_precision=1.0,
            abstention_recall=1.0,
            abstention_f1=1.0,
        )
        assert metrics.abstention_f1 == 1.0


# Test LongMemEvalMetrics dataclass


class TestLongMemEvalMetrics:
    """Tests for LongMemEvalMetrics dataclass."""

    @pytest.fixture
    def sample_metrics(self) -> LongMemEvalMetrics:
        """Create sample LongMemEvalMetrics."""
        return LongMemEvalMetrics(
            subset="S",
            total_questions=100,
            aggregate_accuracy=0.75,
            aggregate_mean_score=0.80,
            ability_metrics={
                "single-session-user": AbilityMetrics(
                    ability_name="single-session-user",
                    total_questions=40,
                    correct_count=32,
                    partial_count=4,
                    accuracy=0.80,
                    mean_score=0.85,
                ),
                "multi-session": AbilityMetrics(
                    ability_name="multi-session",
                    total_questions=30,
                    correct_count=21,
                    partial_count=3,
                    accuracy=0.70,
                    mean_score=0.75,
                ),
            },
            abstention_metrics=AbstentionMetrics(
                total_abstention_expected=10,
                correct_abstentions=8,
                false_abstentions=1,
                missed_abstentions=2,
                abstention_precision=0.89,
                abstention_recall=0.80,
                abstention_f1=0.84,
            ),
        )

    def test_to_dict(self, sample_metrics: LongMemEvalMetrics) -> None:
        """Test serialization to dictionary."""
        d = sample_metrics.to_dict()

        assert d["subset"] == "S"
        assert d["total_questions"] == 100
        assert d["aggregate_accuracy"] == 0.75
        assert "single-session-user" in d["ability_metrics"]
        assert d["abstention_metrics"]["abstention_f1"] == 0.84

    def test_format_report(self, sample_metrics: LongMemEvalMetrics) -> None:
        """Test human-readable report generation."""
        report = sample_metrics.format_report()

        assert "LongMemEval Metrics Report" in report
        assert "Subset S" in report
        assert "Total Questions: 100" in report
        assert "Accuracy: 75.00%" in report
        assert "single-session-user" in report
        assert "Abstention Analysis" in report

    def test_format_report_with_ci(self) -> None:
        """Test report includes CI when available."""
        metrics = LongMemEvalMetrics(
            subset="M",
            total_questions=50,
            aggregate_accuracy=0.80,
            aggregate_mean_score=0.85,
            ability_metrics={},
            abstention_metrics=AbstentionMetrics(
                total_abstention_expected=0,
                correct_abstentions=0,
                false_abstentions=0,
                missed_abstentions=0,
                abstention_precision=0.0,
                abstention_recall=1.0,
                abstention_f1=0.0,
            ),
            aggregate_ci=(0.72, 0.88),
        )
        report = metrics.format_report()

        assert "95% CI: [0.720, 0.880]" in report


# Test MetricsCalculator


class TestMetricsCalculator:
    """Tests for MetricsCalculator class."""

    def test_init_without_analyzer(self) -> None:
        """Test initialization without analyzer."""
        calculator = MetricsCalculator()
        assert calculator._analyzer is None

    def test_init_with_analyzer(self) -> None:
        """Test initialization with analyzer."""
        analyzer = StatisticalAnalyzer(n_bootstrap=100, random_seed=42)
        calculator = MetricsCalculator(analyzer)
        assert calculator._analyzer is analyzer

    def test_calculate_basic(
        self,
        sample_assessment_result: AssessmentResult,
    ) -> None:
        """Test basic metrics calculation without CI."""
        calculator = MetricsCalculator()
        metrics = calculator.calculate(
            sample_assessment_result,
            compute_confidence_intervals=False,
        )

        assert metrics.subset == "S"
        assert metrics.total_questions == 7
        assert metrics.aggregate_ci is None

    def test_calculate_ability_breakdown(
        self,
        sample_assessment_result: AssessmentResult,
    ) -> None:
        """Test ability breakdown calculation."""
        calculator = MetricsCalculator()
        metrics = calculator.calculate(
            sample_assessment_result,
            compute_confidence_intervals=False,
        )

        # Check ability breakdown
        assert "single-session-user" in metrics.ability_metrics
        assert "multi-session" in metrics.ability_metrics
        assert "abstention" in metrics.ability_metrics

        ssu = metrics.ability_metrics["single-session-user"]
        assert ssu.total_questions == 3
        assert ssu.correct_count == 2  # q1, q2
        assert ssu.partial_count == 1  # q3

    def test_calculate_abstention_metrics(
        self,
        sample_assessment_result: AssessmentResult,
    ) -> None:
        """Test abstention metrics calculation."""
        calculator = MetricsCalculator()
        metrics = calculator.calculate(
            sample_assessment_result,
            compute_confidence_intervals=False,
        )

        abst = metrics.abstention_metrics
        assert abst.total_abstention_expected == 2  # q6, q7
        assert abst.correct_abstentions == 1  # q6 (correctly abstained)
        assert abst.missed_abstentions == 1  # q7 (should have abstained)
        assert abst.false_abstentions == 0  # No false abstentions

    def test_calculate_latency_stats(
        self,
        sample_assessment_result: AssessmentResult,
    ) -> None:
        """Test latency statistics calculation."""
        calculator = MetricsCalculator()
        metrics = calculator.calculate(
            sample_assessment_result,
            compute_confidence_intervals=False,
        )

        assert "min" in metrics.latency_stats
        assert "max" in metrics.latency_stats
        assert "mean" in metrics.latency_stats
        assert "median" in metrics.latency_stats
        assert metrics.latency_stats["min"] == 90.0  # q6
        assert metrics.latency_stats["max"] == 200.0  # q4

    def test_calculate_with_confidence_intervals(
        self,
        sample_assessment_result: AssessmentResult,
    ) -> None:
        """Test CI calculation when analyzer is provided."""
        analyzer = StatisticalAnalyzer(n_bootstrap=100, random_seed=42)
        calculator = MetricsCalculator(analyzer)

        metrics = calculator.calculate(
            sample_assessment_result,
            compute_confidence_intervals=True,
        )

        # Check aggregate CI
        assert metrics.aggregate_ci is not None
        lower, upper = metrics.aggregate_ci
        assert 0.0 <= lower <= upper <= 1.0

        # Check ability CIs
        for ability in metrics.ability_metrics.values():
            assert ability.confidence_interval is not None

    def test_calculate_metadata(
        self,
        sample_assessment_result: AssessmentResult,
    ) -> None:
        """Test metadata propagation."""
        calculator = MetricsCalculator()
        metrics = calculator.calculate(
            sample_assessment_result,
            compute_confidence_intervals=False,
        )

        assert "ingestion_time_ms" in metrics.metadata
        assert "assessment_time_ms" in metrics.metadata
        assert "adapter" in metrics.metadata
        assert metrics.metadata["adapter"] == "git-notes"


class TestMetricsCalculatorEdgeCases:
    """Edge case tests for MetricsCalculator."""

    def test_empty_results(self) -> None:
        """Test with no question results."""
        calculator = MetricsCalculator()
        result = AssessmentResult(
            dataset_subset="S",
            question_results=[],
            total_questions=0,
            ingestion_time_ms=0.0,
            assessment_time_ms=0.0,
            started_at=datetime.now(),
            completed_at=datetime.now(),
        )

        metrics = calculator.calculate(result, compute_confidence_intervals=False)

        assert metrics.total_questions == 0
        assert metrics.aggregate_accuracy == 0.0
        assert metrics.aggregate_mean_score == 0.0
        assert len(metrics.ability_metrics) == 0

    def test_no_abstention_questions(
        self,
        sample_judgment_correct: Judgment,
    ) -> None:
        """Test when no abstention questions exist."""
        results = [
            QuestionResult(
                question_id="q1",
                question_text="Simple Q",
                question_type="single-session-user",
                ground_truth=["answer"],
                agent_answer="answer",
                judgment=sample_judgment_correct,
                is_abstention_expected=False,
                is_abstention_actual=False,
                latency_ms=100.0,
            )
        ]

        calculator = MetricsCalculator()
        assessment = AssessmentResult(
            dataset_subset="S",
            question_results=results,
            total_questions=1,
            ingestion_time_ms=100.0,
            assessment_time_ms=200.0,
            started_at=datetime.now(),
            completed_at=datetime.now(),
        )

        metrics = calculator.calculate(assessment, compute_confidence_intervals=False)

        # No expected abstentions = recall is 1.0 by convention
        assert metrics.abstention_metrics.total_abstention_expected == 0
        assert metrics.abstention_metrics.abstention_recall == 1.0

    def test_all_correct(
        self,
        sample_judgment_correct: Judgment,
    ) -> None:
        """Test with all correct answers."""
        results = [
            QuestionResult(
                question_id=f"q{i}",
                question_text=f"Question {i}",
                question_type="single-session-user",
                ground_truth=["answer"],
                agent_answer="answer",
                judgment=sample_judgment_correct,
                is_abstention_expected=False,
                is_abstention_actual=False,
                latency_ms=100.0,
            )
            for i in range(5)
        ]

        calculator = MetricsCalculator()
        assessment = AssessmentResult(
            dataset_subset="S",
            question_results=results,
            total_questions=5,
            ingestion_time_ms=100.0,
            assessment_time_ms=500.0,
            started_at=datetime.now(),
            completed_at=datetime.now(),
        )

        metrics = calculator.calculate(assessment, compute_confidence_intervals=False)

        assert metrics.aggregate_accuracy == 1.0
        assert metrics.aggregate_mean_score == 1.0


# Test compare_results function


class TestCompareResults:
    """Tests for compare_results function."""

    @pytest.fixture
    def memory_result(
        self,
        sample_judgment_correct: Judgment,
    ) -> AssessmentResult:
        """Create assessment result for memory condition."""
        results = [
            QuestionResult(
                question_id=f"q{i}",
                question_text=f"Question {i}",
                question_type="single-session-user",
                ground_truth=["answer"],
                agent_answer="answer",
                judgment=sample_judgment_correct,
                is_abstention_expected=False,
                is_abstention_actual=False,
                latency_ms=100.0,
            )
            for i in range(10)
        ]

        return AssessmentResult(
            dataset_subset="S",
            question_results=results,
            total_questions=10,
            ingestion_time_ms=100.0,
            assessment_time_ms=1000.0,
            started_at=datetime.now(),
            completed_at=datetime.now(),
        )

    @pytest.fixture
    def baseline_result(
        self,
        sample_judgment_correct: Judgment,
        sample_judgment_incorrect: Judgment,
    ) -> AssessmentResult:
        """Create assessment result for baseline condition."""
        results = []
        for i in range(10):
            # 60% correct for baseline
            judgment = sample_judgment_correct if i < 6 else sample_judgment_incorrect
            results.append(
                QuestionResult(
                    question_id=f"q{i}",
                    question_text=f"Question {i}",
                    question_type="single-session-user",
                    ground_truth=["answer"],
                    agent_answer="answer" if i < 6 else "wrong",
                    judgment=judgment,
                    is_abstention_expected=False,
                    is_abstention_actual=False,
                    latency_ms=100.0,
                )
            )

        return AssessmentResult(
            dataset_subset="S",
            question_results=results,
            total_questions=10,
            ingestion_time_ms=100.0,
            assessment_time_ms=1000.0,
            started_at=datetime.now(),
            completed_at=datetime.now(),
        )

    def test_compare_two_conditions(
        self,
        memory_result: AssessmentResult,
        baseline_result: AssessmentResult,
    ) -> None:
        """Test comparing two conditions."""
        comparison = compare_results(
            [memory_result, baseline_result],
            ["git-notes", "no-memory"],
        )

        assert "conditions" in comparison
        assert "git-notes" in comparison["conditions"]
        assert "no-memory" in comparison["conditions"]

        # Memory should have 100% accuracy, baseline 60%
        assert comparison["conditions"]["git-notes"]["accuracy"] == 1.0
        assert comparison["conditions"]["no-memory"]["accuracy"] == 0.6

    def test_compare_computes_difference(
        self,
        memory_result: AssessmentResult,
        baseline_result: AssessmentResult,
    ) -> None:
        """Test that differences are computed."""
        comparison = compare_results(
            [memory_result, baseline_result],
            ["git-notes", "no-memory"],
        )

        diff_key = "git-notes_vs_no-memory"
        assert diff_key in comparison["differences"]

        diff = comparison["differences"][diff_key]
        assert diff["accuracy_diff"] == pytest.approx(0.4, abs=0.01)

    def test_compare_with_statistical_analysis(
        self,
        memory_result: AssessmentResult,
        baseline_result: AssessmentResult,
    ) -> None:
        """Test comparison with statistical analysis."""
        analyzer = StatisticalAnalyzer(n_bootstrap=100, random_seed=42)

        comparison = compare_results(
            [memory_result, baseline_result],
            ["git-notes", "no-memory"],
            analyzer=analyzer,
        )

        diff = comparison["differences"]["git-notes_vs_no-memory"]
        assert "statistical_test" in diff
        assert "p_value" in diff["statistical_test"]
        assert "effect_size" in diff["statistical_test"]
        assert "is_significant" in diff["statistical_test"]

    def test_compare_mismatched_lengths_raises(
        self,
        memory_result: AssessmentResult,
    ) -> None:
        """Test that mismatched lengths raise error."""
        with pytest.raises(ValueError, match="must match"):
            compare_results(
                [memory_result],
                ["git-notes", "no-memory"],
            )

    def test_compare_single_result(
        self,
        memory_result: AssessmentResult,
    ) -> None:
        """Test with a single result (no comparison)."""
        comparison = compare_results(
            [memory_result],
            ["git-notes"],
        )

        assert "git-notes" in comparison["conditions"]
        assert len(comparison["differences"]) == 0


# Test module exports


class TestModuleExports:
    """Test that module exports are correct."""

    def test_import_from_metrics(self) -> None:
        """Test importing from metrics module."""
        from src.benchmarks.longmemeval.metrics import (
            AbilityMetrics,
            AbstentionMetrics,
            LongMemEvalMetrics,
            MetricsCalculator,
            compare_results,
        )

        assert AbilityMetrics is not None
        assert AbstentionMetrics is not None
        assert LongMemEvalMetrics is not None
        assert MetricsCalculator is not None
        assert compare_results is not None

    def test_import_from_package(self) -> None:
        """Test importing from package __init__."""
        from src.benchmarks.longmemeval import (
            AbilityMetrics,
            AbstentionMetrics,
            LongMemEvalMetrics,
            MetricsCalculator,
            compare_results,
        )

        assert AbilityMetrics is not None
        assert AbstentionMetrics is not None
        assert LongMemEvalMetrics is not None
        assert MetricsCalculator is not None
        assert compare_results is not None
