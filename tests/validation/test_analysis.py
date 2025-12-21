"""Tests for validation analysis."""

from __future__ import annotations

from datetime import datetime

import pytest

from src.validation.analysis import (
    AgreementMetrics,
    ValidationAnalyzer,
    ValidationReport,
)
from src.validation.annotation import RubricLevel
from src.validation.collector import AnnotatedSample
from src.validation.compiler import CompiledSample, SourceBenchmark


class TestAgreementMetrics:
    """Tests for AgreementMetrics dataclass."""

    def test_metrics_creation(self) -> None:
        """Test creating metrics."""
        metrics = AgreementMetrics(
            agreement_rate=0.85,
            total_samples=100,
            agreed_samples=85,
            kappa=0.75,
            weighted_kappa=0.80,
        )

        assert metrics.agreement_rate == 0.85
        assert metrics.total_samples == 100
        assert metrics.agreed_samples == 85
        assert metrics.kappa == 0.75
        assert metrics.weighted_kappa == 0.80

    def test_metrics_defaults(self) -> None:
        """Test default values."""
        metrics = AgreementMetrics(
            agreement_rate=0.5,
            total_samples=10,
            agreed_samples=5,
        )

        assert metrics.kappa == 0.0
        assert metrics.weighted_kappa == 0.0
        assert metrics.confusion_matrix == {}


class TestValidationAnalyzer:
    """Tests for ValidationAnalyzer class."""

    @pytest.fixture
    def make_sample(self):
        """Factory for creating samples."""

        def _make(
            sample_id: str,
            llm_judgment: str = "correct",
            benchmark: SourceBenchmark = SourceBenchmark.LONGMEMEVAL,
            adapter: str = "test_adapter",
        ) -> CompiledSample:
            return CompiledSample(
                sample_id=sample_id,
                source_benchmark=benchmark,
                question_id=f"q_{sample_id}",
                question=f"Question for {sample_id}",
                expected_answers=("Answer",),
                model_answer="Model answer",
                adapter_name=adapter,
                llm_judgment=llm_judgment,
                llm_confidence=0.9,
            )

        return _make

    @pytest.fixture
    def make_annotation(self, make_sample):
        """Factory for creating annotations."""

        def _make(
            sample_id: str,
            human_judgment: RubricLevel,
            annotator_id: str = "ann_001",
            llm_judgment: str = "correct",
            flagged: bool = False,
            confidence: int = 4,
            time_seconds: float = 20.0,
            benchmark: SourceBenchmark = SourceBenchmark.LONGMEMEVAL,
            adapter: str = "test_adapter",
        ) -> AnnotatedSample:
            sample = make_sample(sample_id, llm_judgment, benchmark, adapter)
            return AnnotatedSample(
                sample=sample,
                human_judgment=human_judgment,
                annotator_id=annotator_id,
                annotation_time_seconds=time_seconds,
                confidence=confidence,
                flagged=flagged,
                timestamp=datetime.now(),
            )

        return _make

    def test_analyzer_empty(self) -> None:
        """Test analyzer with no annotations."""
        analyzer = ValidationAnalyzer([])

        metrics = analyzer.compute_human_llm_agreement()
        assert metrics.total_samples == 0
        assert metrics.agreement_rate == 0.0

    def test_human_llm_agreement_perfect(self, make_annotation) -> None:
        """Test perfect agreement between human and LLM."""
        annotations = [
            make_annotation("s1", RubricLevel.CORRECT, llm_judgment="correct"),
            make_annotation("s2", RubricLevel.INCORRECT, llm_judgment="incorrect"),
            make_annotation("s3", RubricLevel.PARTIALLY_CORRECT, llm_judgment="partial"),
        ]

        analyzer = ValidationAnalyzer(annotations)
        metrics = analyzer.compute_human_llm_agreement()

        assert metrics.total_samples == 3
        assert metrics.agreed_samples == 3
        assert metrics.agreement_rate == 1.0
        assert metrics.kappa == 1.0

    def test_human_llm_agreement_none(self, make_annotation) -> None:
        """Test no agreement between human and LLM."""
        annotations = [
            make_annotation("s1", RubricLevel.CORRECT, llm_judgment="incorrect"),
            make_annotation("s2", RubricLevel.INCORRECT, llm_judgment="correct"),
        ]

        analyzer = ValidationAnalyzer(annotations)
        metrics = analyzer.compute_human_llm_agreement()

        assert metrics.agreement_rate == 0.0

    def test_human_llm_agreement_partial(self, make_annotation) -> None:
        """Test partial agreement."""
        annotations = [
            make_annotation("s1", RubricLevel.CORRECT, llm_judgment="correct"),  # Agree
            make_annotation("s2", RubricLevel.CORRECT, llm_judgment="incorrect"),  # Disagree
            make_annotation("s3", RubricLevel.INCORRECT, llm_judgment="incorrect"),  # Agree
            make_annotation("s4", RubricLevel.INCORRECT, llm_judgment="correct"),  # Disagree
        ]

        analyzer = ValidationAnalyzer(annotations)
        metrics = analyzer.compute_human_llm_agreement()

        assert metrics.agreement_rate == 0.5
        assert metrics.agreed_samples == 2
        assert metrics.total_samples == 4

    def test_human_llm_agreement_unknown_llm(self, make_annotation) -> None:
        """Test handling unknown LLM judgments."""
        annotations = [
            make_annotation("s1", RubricLevel.CORRECT, llm_judgment="correct"),
            make_annotation("s2", RubricLevel.CORRECT, llm_judgment=""),  # Unknown
            make_annotation("s3", RubricLevel.CORRECT, llm_judgment="unknown"),  # Unknown
        ]

        analyzer = ValidationAnalyzer(annotations)
        metrics = analyzer.compute_human_llm_agreement()

        # Only s1 should be counted
        assert metrics.total_samples == 1
        assert metrics.agreement_rate == 1.0

    def test_inter_annotator_no_multi(self, make_annotation) -> None:
        """Test inter-annotator when no samples have multiple annotations."""
        annotations = [
            make_annotation("s1", RubricLevel.CORRECT, annotator_id="ann_001"),
            make_annotation("s2", RubricLevel.CORRECT, annotator_id="ann_002"),
        ]

        analyzer = ValidationAnalyzer(annotations)
        metrics = analyzer.compute_inter_annotator_agreement()

        assert metrics is None

    def test_inter_annotator_agreement(self, make_annotation) -> None:
        """Test inter-annotator agreement calculation."""
        annotations = [
            # Sample 1: Both agree
            make_annotation("s1", RubricLevel.CORRECT, annotator_id="ann_001"),
            make_annotation("s1", RubricLevel.CORRECT, annotator_id="ann_002"),
            # Sample 2: Disagree
            make_annotation("s2", RubricLevel.CORRECT, annotator_id="ann_001"),
            make_annotation("s2", RubricLevel.INCORRECT, annotator_id="ann_002"),
        ]

        analyzer = ValidationAnalyzer(annotations)
        metrics = analyzer.compute_inter_annotator_agreement()

        assert metrics is not None
        assert metrics.total_samples == 2
        assert metrics.agreed_samples == 1
        assert metrics.agreement_rate == 0.5

    def test_normalize_judgment_string(self, make_annotation) -> None:
        """Test normalizing string judgments."""
        analyzer = ValidationAnalyzer([])

        assert analyzer._normalize_judgment("correct") == RubricLevel.CORRECT
        assert analyzer._normalize_judgment("CORRECT") == RubricLevel.CORRECT
        assert analyzer._normalize_judgment("incorrect") == RubricLevel.INCORRECT
        assert analyzer._normalize_judgment("partial") == RubricLevel.PARTIALLY_CORRECT
        assert analyzer._normalize_judgment("partially_correct") == RubricLevel.PARTIALLY_CORRECT
        assert analyzer._normalize_judgment("cannot_judge") == RubricLevel.CANNOT_JUDGE
        assert analyzer._normalize_judgment("unknown") == RubricLevel.CANNOT_JUDGE

    def test_normalize_judgment_enum(self, make_annotation) -> None:
        """Test normalizing enum judgments."""
        analyzer = ValidationAnalyzer([])

        for level in RubricLevel:
            assert analyzer._normalize_judgment(level) == level

    def test_normalize_judgment_unknown(self, make_annotation) -> None:
        """Test normalizing unknown judgments."""
        analyzer = ValidationAnalyzer([])

        assert analyzer._normalize_judgment("gibberish") is None
        assert analyzer._normalize_judgment("") is None

    def test_compute_kappa_perfect(self, make_annotation) -> None:
        """Test Kappa = 1 for perfect agreement."""
        pairs = [
            (RubricLevel.CORRECT, RubricLevel.CORRECT),
            (RubricLevel.INCORRECT, RubricLevel.INCORRECT),
        ]

        analyzer = ValidationAnalyzer([])
        kappa = analyzer._compute_kappa(pairs)

        assert kappa == 1.0

    def test_compute_kappa_random(self, make_annotation) -> None:
        """Test Kappa ~ 0 for random agreement."""
        # This pattern approximates random agreement
        pairs = [
            (RubricLevel.CORRECT, RubricLevel.CORRECT),
            (RubricLevel.CORRECT, RubricLevel.INCORRECT),
            (RubricLevel.INCORRECT, RubricLevel.CORRECT),
            (RubricLevel.INCORRECT, RubricLevel.INCORRECT),
        ]

        analyzer = ValidationAnalyzer([])
        kappa = analyzer._compute_kappa(pairs)

        # Kappa should be close to 0 for random
        assert -0.5 < kappa < 0.5

    def test_compute_weighted_kappa(self, make_annotation) -> None:
        """Test weighted Kappa computation."""
        # More realistic set of pairs with variety
        pairs = [
            (RubricLevel.CORRECT, RubricLevel.CORRECT),
            (RubricLevel.CORRECT, RubricLevel.CORRECT),
            (RubricLevel.INCORRECT, RubricLevel.INCORRECT),
            (RubricLevel.PARTIALLY_CORRECT, RubricLevel.CORRECT),  # Close disagreement
        ]

        analyzer = ValidationAnalyzer([])
        weighted = analyzer._compute_weighted_kappa(pairs)

        # Weighted kappa should be high with mostly agreements and close disagreements
        assert weighted >= 0.5

    def test_confusion_matrix(self, make_annotation) -> None:
        """Test confusion matrix generation."""
        annotations = [
            make_annotation("s1", RubricLevel.CORRECT, annotator_id="ann_001"),
            make_annotation("s1", RubricLevel.CORRECT, annotator_id="ann_002"),
            make_annotation("s2", RubricLevel.CORRECT, annotator_id="ann_001"),
            make_annotation("s2", RubricLevel.INCORRECT, annotator_id="ann_002"),
        ]

        analyzer = ValidationAnalyzer(annotations)
        metrics = analyzer.compute_inter_annotator_agreement()

        assert metrics is not None
        assert "correct" in metrics.confusion_matrix
        assert metrics.confusion_matrix["correct"]["correct"] == 1
        assert metrics.confusion_matrix["correct"]["incorrect"] == 1

    def test_generate_report(self, make_annotation) -> None:
        """Test report generation."""
        annotations = [
            make_annotation(
                "s1",
                RubricLevel.CORRECT,
                llm_judgment="correct",
                flagged=True,
                confidence=5,
                time_seconds=30.0,
                benchmark=SourceBenchmark.LONGMEMEVAL,
                adapter="adapter_a",
            ),
            make_annotation(
                "s2",
                RubricLevel.INCORRECT,
                llm_judgment="incorrect",
                flagged=False,
                confidence=3,
                time_seconds=20.0,
                benchmark=SourceBenchmark.LOCOMO,
                adapter="adapter_b",
            ),
        ]

        analyzer = ValidationAnalyzer(annotations)
        report = analyzer.generate_report()

        assert report.total_annotations == 2
        assert report.unique_samples == 2
        assert report.annotator_count == 1
        assert report.flagged_samples == 1
        assert report.avg_confidence == 4.0
        assert report.avg_time_per_sample == 25.0

        # Check distributions
        assert "correct" in report.judgment_distribution
        assert "incorrect" in report.judgment_distribution

        # Check by benchmark
        assert "longmemeval" in report.by_benchmark
        assert "locomo" in report.by_benchmark
        assert report.by_benchmark["longmemeval"]["correct"] == 1

        # Check by adapter
        assert "adapter_a" in report.by_adapter
        assert "adapter_b" in report.by_adapter

    def test_format_report(self, make_annotation) -> None:
        """Test report markdown formatting."""
        annotations = [
            make_annotation("s1", RubricLevel.CORRECT, llm_judgment="correct"),
            make_annotation("s2", RubricLevel.INCORRECT, llm_judgment="incorrect"),
        ]

        analyzer = ValidationAnalyzer(annotations)
        markdown = analyzer.format_report()

        assert "# Human Validation Report" in markdown
        assert "## Summary" in markdown
        assert "**Total Annotations**: 2" in markdown
        assert "## Judgment Distribution" in markdown
        assert "## Human-LLM Agreement" in markdown
        assert "## By Benchmark" in markdown
        assert "## By Adapter" in markdown

    def test_format_report_with_inter_annotator(self, make_annotation) -> None:
        """Test report includes inter-annotator when available."""
        annotations = [
            make_annotation("s1", RubricLevel.CORRECT, annotator_id="ann_001"),
            make_annotation("s1", RubricLevel.CORRECT, annotator_id="ann_002"),
        ]

        analyzer = ValidationAnalyzer(annotations)
        markdown = analyzer.format_report()

        assert "## Inter-Annotator Agreement" in markdown
        assert "**Agreement Rate**" in markdown
        assert "**Weighted Kappa**" in markdown


class TestValidationReport:
    """Tests for ValidationReport dataclass."""

    def test_report_creation(self) -> None:
        """Test creating a report."""
        report = ValidationReport(
            total_annotations=100,
            unique_samples=80,
            annotator_count=3,
            human_llm_agreement=AgreementMetrics(
                agreement_rate=0.85,
                total_samples=80,
                agreed_samples=68,
            ),
            inter_annotator_agreement=AgreementMetrics(
                agreement_rate=0.90,
                total_samples=20,
                agreed_samples=18,
            ),
            judgment_distribution={"correct": 60, "incorrect": 20, "partial": 20},
            by_benchmark={"longmemeval": {"total": 50}},
            by_adapter={"semantic": {"total": 30}},
            flagged_samples=5,
            avg_confidence=4.2,
            avg_time_per_sample=25.5,
        )

        assert report.total_annotations == 100
        assert report.unique_samples == 80
        assert report.annotator_count == 3
        assert report.human_llm_agreement.agreement_rate == 0.85
        assert report.inter_annotator_agreement is not None
        assert report.inter_annotator_agreement.agreement_rate == 0.90
        assert report.flagged_samples == 5

    def test_report_without_inter_annotator(self) -> None:
        """Test report without inter-annotator agreement."""
        report = ValidationReport(
            total_annotations=50,
            unique_samples=50,
            annotator_count=1,
            human_llm_agreement=AgreementMetrics(
                agreement_rate=0.80,
                total_samples=50,
                agreed_samples=40,
            ),
            inter_annotator_agreement=None,  # Single annotator
            judgment_distribution={"correct": 40, "incorrect": 10},
            by_benchmark={},
            by_adapter={},
            flagged_samples=0,
            avg_confidence=3.5,
            avg_time_per_sample=18.0,
        )

        assert report.inter_annotator_agreement is None
