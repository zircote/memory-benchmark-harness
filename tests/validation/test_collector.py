"""Tests for validation data collection."""

from __future__ import annotations

import json

import pytest

from src.validation.annotation import RubricLevel
from src.validation.collector import (
    AnnotatedSample,
    AnnotationSession,
    ValidationCollector,
)
from src.validation.compiler import CompiledSample, SourceBenchmark


class TestAnnotatedSample:
    """Tests for AnnotatedSample dataclass."""

    @pytest.fixture
    def sample(self) -> CompiledSample:
        """Create a sample for testing."""
        return CompiledSample(
            sample_id="test_001",
            source_benchmark=SourceBenchmark.LONGMEMEVAL,
            question_id="q1",
            question="What is Alice's job?",
            expected_answers=("Engineer",),
            model_answer="Software Engineer",
            adapter_name="semantic_search",
            llm_judgment="correct",
            llm_confidence=0.95,
        )

    def test_annotated_sample_creation(self, sample: CompiledSample) -> None:
        """Test creating an annotated sample."""
        annotated = AnnotatedSample(
            sample=sample,
            human_judgment=RubricLevel.CORRECT,
            annotator_id="ann_001",
            annotation_time_seconds=30.5,
            notes="Clear match",
            confidence=5,
        )

        assert annotated.sample.sample_id == "test_001"
        assert annotated.human_judgment == RubricLevel.CORRECT
        assert annotated.annotator_id == "ann_001"
        assert annotated.annotation_time_seconds == 30.5
        assert annotated.confidence == 5
        assert not annotated.flagged

    def test_agrees_with_llm_correct(self, sample: CompiledSample) -> None:
        """Test agreement when human agrees with LLM."""
        annotated = AnnotatedSample(
            sample=sample,
            human_judgment=RubricLevel.CORRECT,
            annotator_id="ann_001",
        )

        assert annotated.agrees_with_llm() is True

    def test_agrees_with_llm_incorrect(self, sample: CompiledSample) -> None:
        """Test disagreement detection."""
        annotated = AnnotatedSample(
            sample=sample,
            human_judgment=RubricLevel.INCORRECT,
            annotator_id="ann_001",
        )

        assert annotated.agrees_with_llm() is False

    def test_agrees_with_llm_unknown(self) -> None:
        """Test when LLM judgment is unknown."""
        sample = CompiledSample(
            sample_id="test_002",
            source_benchmark=SourceBenchmark.LOCOMO,
            question_id="q2",
            question="Test",
            expected_answers=("Answer",),
            model_answer="Model",
            adapter_name="test",
            llm_judgment="",  # Unknown
        )

        annotated = AnnotatedSample(
            sample=sample,
            human_judgment=RubricLevel.CORRECT,
            annotator_id="ann_001",
        )

        assert annotated.agrees_with_llm() is None

    def test_agrees_with_llm_partial(self) -> None:
        """Test partial agreement mapping."""
        sample = CompiledSample(
            sample_id="test_003",
            source_benchmark=SourceBenchmark.LOCOMO,
            question_id="q3",
            question="Test",
            expected_answers=("Answer",),
            model_answer="Model",
            adapter_name="test",
            llm_judgment="partial",
        )

        annotated = AnnotatedSample(
            sample=sample,
            human_judgment=RubricLevel.PARTIALLY_CORRECT,
            annotator_id="ann_001",
        )

        assert annotated.agrees_with_llm() is True

    def test_to_dict(self, sample: CompiledSample) -> None:
        """Test converting to dictionary."""
        annotated = AnnotatedSample(
            sample=sample,
            human_judgment=RubricLevel.CORRECT,
            annotator_id="ann_001",
            annotation_time_seconds=25.0,
            notes="Test notes",
            confidence=4,
            flagged=True,
        )

        d = annotated.to_dict()

        assert d["sample_id"] == "test_001"
        assert d["human_judgment"] == "correct"
        assert d["annotator_id"] == "ann_001"
        assert d["annotation_time_seconds"] == 25.0
        assert d["notes"] == "Test notes"
        assert d["confidence"] == 4
        assert d["flagged"] is True
        assert "timestamp" in d
        assert d["agrees_with_llm"] is True


class TestAnnotationSession:
    """Tests for AnnotationSession dataclass."""

    @pytest.fixture
    def session(self) -> AnnotationSession:
        """Create a session for testing."""
        return AnnotationSession(
            session_id="sess_001",
            annotator_id="ann_001",
        )

    @pytest.fixture
    def sample(self) -> CompiledSample:
        """Create a sample for testing."""
        return CompiledSample(
            sample_id="test_001",
            source_benchmark=SourceBenchmark.LONGMEMEVAL,
            question_id="q1",
            question="What is Alice's job?",
            expected_answers=("Engineer",),
            model_answer="Software Engineer",
            adapter_name="test",
        )

    def test_session_creation(self, session: AnnotationSession) -> None:
        """Test session creation."""
        assert session.session_id == "sess_001"
        assert session.annotator_id == "ann_001"
        assert not session.is_complete
        assert session.annotation_count == 0

    def test_add_annotation(self, session: AnnotationSession, sample: CompiledSample) -> None:
        """Test adding annotations to session."""
        annotation = AnnotatedSample(
            sample=sample,
            human_judgment=RubricLevel.CORRECT,
            annotator_id="ann_001",
        )

        session.add_annotation(annotation)

        assert session.annotation_count == 1
        assert session.annotations[0].sample.sample_id == "test_001"

    def test_complete_session(self, session: AnnotationSession) -> None:
        """Test completing a session."""
        assert session.end_time is None
        assert not session.is_complete

        session.complete()

        assert session.end_time is not None
        assert session.is_complete

    def test_duration_seconds(self, session: AnnotationSession) -> None:
        """Test duration calculation."""
        # Should return positive duration
        duration = session.duration_seconds
        assert duration >= 0

    def test_avg_time_per_sample_empty(self, session: AnnotationSession) -> None:
        """Test average time with no annotations."""
        assert session.avg_time_per_sample == 0.0

    def test_avg_time_per_sample(self, session: AnnotationSession, sample: CompiledSample) -> None:
        """Test average time calculation."""
        for t in [10.0, 20.0, 30.0]:
            annotation = AnnotatedSample(
                sample=sample,
                human_judgment=RubricLevel.CORRECT,
                annotator_id="ann_001",
                annotation_time_seconds=t,
            )
            session.add_annotation(annotation)

        assert session.avg_time_per_sample == 20.0

    def test_to_dict(self, session: AnnotationSession, sample: CompiledSample) -> None:
        """Test converting session to dictionary."""
        annotation = AnnotatedSample(
            sample=sample,
            human_judgment=RubricLevel.CORRECT,
            annotator_id="ann_001",
        )
        session.add_annotation(annotation)
        session.complete()

        d = session.to_dict()

        assert d["session_id"] == "sess_001"
        assert d["annotator_id"] == "ann_001"
        assert d["annotation_count"] == 1
        assert "start_time" in d
        assert "end_time" in d
        assert len(d["annotations"]) == 1


class TestValidationCollector:
    """Tests for ValidationCollector class."""

    @pytest.fixture
    def collector(self, tmp_path) -> ValidationCollector:
        """Create a collector for testing."""
        return ValidationCollector(data_dir=tmp_path)

    @pytest.fixture
    def sample(self) -> CompiledSample:
        """Create a sample for testing."""
        return CompiledSample(
            sample_id="test_001",
            source_benchmark=SourceBenchmark.LONGMEMEVAL,
            question_id="q1",
            question="What is Alice's job?",
            expected_answers=("Engineer",),
            model_answer="Software Engineer",
            adapter_name="test",
            llm_judgment="correct",
        )

    def test_collector_initialization(self, collector: ValidationCollector, tmp_path) -> None:
        """Test collector initialization."""
        assert collector.data_dir == tmp_path
        assert len(collector.sessions) == 0
        assert len(collector.all_annotations) == 0

    def test_start_session(self, collector: ValidationCollector) -> None:
        """Test starting a new session."""
        session = collector.start_session("ann_001")

        assert session.annotator_id == "ann_001"
        assert session.session_id in collector.sessions

    def test_start_session_with_metadata(self, collector: ValidationCollector) -> None:
        """Test starting session with metadata."""
        session = collector.start_session(
            "ann_001",
            metadata={"environment": "test", "version": "1.0"},
        )

        assert session.metadata["environment"] == "test"
        assert session.metadata["version"] == "1.0"

    def test_add_annotation(self, collector: ValidationCollector, sample: CompiledSample) -> None:
        """Test adding annotation to session."""
        session = collector.start_session("ann_001")

        annotation = collector.add_annotation(
            session_id=session.session_id,
            sample=sample,
            judgment=RubricLevel.CORRECT,
            time_seconds=25.0,
            notes="Clear match",
            confidence=5,
        )

        assert annotation.human_judgment == RubricLevel.CORRECT
        assert annotation.annotator_id == "ann_001"
        assert session.annotation_count == 1

    def test_add_annotation_string_judgment(
        self, collector: ValidationCollector, sample: CompiledSample
    ) -> None:
        """Test adding annotation with string judgment."""
        session = collector.start_session("ann_001")

        annotation = collector.add_annotation(
            session_id=session.session_id,
            sample=sample,
            judgment="correct",  # String instead of enum
        )

        assert annotation.human_judgment == RubricLevel.CORRECT

    def test_add_annotation_confidence_bounds(
        self, collector: ValidationCollector, sample: CompiledSample
    ) -> None:
        """Test confidence is bounded 1-5."""
        session = collector.start_session("ann_001")

        annotation1 = collector.add_annotation(
            session_id=session.session_id,
            sample=sample,
            judgment=RubricLevel.CORRECT,
            confidence=0,  # Below minimum
        )
        assert annotation1.confidence == 1

        annotation2 = collector.add_annotation(
            session_id=session.session_id,
            sample=sample,
            judgment=RubricLevel.CORRECT,
            confidence=10,  # Above maximum
        )
        assert annotation2.confidence == 5

    def test_add_annotation_unknown_session(
        self, collector: ValidationCollector, sample: CompiledSample
    ) -> None:
        """Test error on unknown session."""
        with pytest.raises(ValueError, match="Unknown session"):
            collector.add_annotation(
                session_id="nonexistent",
                sample=sample,
                judgment=RubricLevel.CORRECT,
            )

    def test_complete_session(self, collector: ValidationCollector, sample: CompiledSample) -> None:
        """Test completing a session."""
        session = collector.start_session("ann_001")
        collector.add_annotation(
            session_id=session.session_id,
            sample=sample,
            judgment=RubricLevel.CORRECT,
        )

        collector.complete_session(session.session_id)

        assert session.is_complete
        assert len(collector.all_annotations) == 1

    def test_complete_session_saves_file(
        self, collector: ValidationCollector, sample: CompiledSample, tmp_path
    ) -> None:
        """Test session is saved on completion."""
        session = collector.start_session("ann_001")
        collector.add_annotation(
            session_id=session.session_id,
            sample=sample,
            judgment=RubricLevel.CORRECT,
        )
        collector.complete_session(session.session_id)

        # Check file was saved
        saved_files = list(tmp_path.glob("*.json"))
        assert len(saved_files) == 1

        # Check content
        with open(saved_files[0]) as f:
            data = json.load(f)
        assert data["session_id"] == session.session_id
        assert len(data["annotations"]) == 1

    def test_complete_unknown_session(self, collector: ValidationCollector) -> None:
        """Test error completing unknown session."""
        with pytest.raises(ValueError, match="Unknown session"):
            collector.complete_session("nonexistent")

    def test_load_all_sessions(
        self, collector: ValidationCollector, sample: CompiledSample, tmp_path
    ) -> None:
        """Test loading saved sessions."""
        # Create and save a session
        session = collector.start_session("ann_001")
        collector.add_annotation(
            session_id=session.session_id,
            sample=sample,
            judgment=RubricLevel.CORRECT,
        )
        collector.complete_session(session.session_id)

        # Create new collector and load
        new_collector = ValidationCollector(data_dir=tmp_path)
        new_collector.load_all_sessions()

        assert len(new_collector.sessions) == 1
        assert len(new_collector.all_annotations) == 1

    def test_get_progress(self, collector: ValidationCollector, sample: CompiledSample) -> None:
        """Test getting progress statistics."""
        session = collector.start_session("ann_001")
        collector.add_annotation(
            session_id=session.session_id,
            sample=sample,
            judgment=RubricLevel.CORRECT,
        )
        collector.add_annotation(
            session_id=session.session_id,
            sample=sample,
            judgment=RubricLevel.INCORRECT,
        )
        collector.complete_session(session.session_id)

        progress = collector.get_progress()

        assert progress["total_annotations"] == 2
        assert progress["total_sessions"] == 1
        assert progress["completed_sessions"] == 1
        assert progress["by_annotator"]["ann_001"] == 2
        assert "correct" in progress["by_judgment"]
        assert "incorrect" in progress["by_judgment"]

    def test_get_annotations_for_sample(
        self, collector: ValidationCollector, sample: CompiledSample
    ) -> None:
        """Test getting annotations for a specific sample."""
        # Multiple annotators
        session1 = collector.start_session("ann_001")
        collector.add_annotation(
            session_id=session1.session_id,
            sample=sample,
            judgment=RubricLevel.CORRECT,
        )
        collector.complete_session(session1.session_id)

        session2 = collector.start_session("ann_002")
        collector.add_annotation(
            session_id=session2.session_id,
            sample=sample,
            judgment=RubricLevel.PARTIALLY_CORRECT,
        )
        collector.complete_session(session2.session_id)

        annotations = collector.get_annotations_for_sample("test_001")

        assert len(annotations) == 2
        assert {a.annotator_id for a in annotations} == {"ann_001", "ann_002"}

    def test_export_all(
        self, collector: ValidationCollector, sample: CompiledSample, tmp_path
    ) -> None:
        """Test exporting all annotations."""
        session = collector.start_session("ann_001")
        collector.add_annotation(
            session_id=session.session_id,
            sample=sample,
            judgment=RubricLevel.CORRECT,
        )
        collector.complete_session(session.session_id)

        export_path = tmp_path / "export" / "all_annotations.json"
        collector.export_all(export_path)

        assert export_path.exists()
        with open(export_path) as f:
            data = json.load(f)

        assert "metadata" in data
        assert "annotations" in data
        assert data["metadata"]["total_annotations"] == 1
