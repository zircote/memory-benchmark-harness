"""Validation data collection and session management.

This module handles collecting human annotations and managing
annotation sessions.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from src.validation.annotation import RubricLevel
from src.validation.compiler import CompiledSample

logger = logging.getLogger(__name__)


@dataclass
class AnnotatedSample:
    """A sample with human annotation.

    Attributes:
        sample: The original compiled sample
        human_judgment: Human annotator's judgment
        annotator_id: ID of the annotator
        annotation_time_seconds: Time spent on annotation
        notes: Annotator's notes or comments
        confidence: Annotator's confidence (1-5)
        flagged: Whether sample was flagged for review
        timestamp: When annotation was made
    """

    sample: CompiledSample
    human_judgment: RubricLevel
    annotator_id: str
    annotation_time_seconds: float = 0.0
    notes: str = ""
    confidence: int = 3
    flagged: bool = False
    timestamp: datetime = field(default_factory=datetime.now)

    def agrees_with_llm(self) -> bool | None:
        """Check if human judgment agrees with LLM judgment.

        Returns:
            True if agree, False if disagree, None if LLM judgment unknown
        """
        llm = self.sample.llm_judgment.lower()

        if not llm or llm == "unknown":
            return None

        # Map LLM judgments to rubric levels
        llm_mapping = {
            "correct": RubricLevel.CORRECT,
            "incorrect": RubricLevel.INCORRECT,
            "partial": RubricLevel.PARTIALLY_CORRECT,
            "partially_correct": RubricLevel.PARTIALLY_CORRECT,
        }

        llm_level = llm_mapping.get(llm)
        if llm_level is None:
            return None

        return llm_level == self.human_judgment

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation
        """
        return {
            "sample_id": self.sample.sample_id,
            "source_benchmark": self.sample.source_benchmark.value,
            "question_id": self.sample.question_id,
            "question": self.sample.question,
            "expected_answers": list(self.sample.expected_answers),
            "model_answer": self.sample.model_answer,
            "adapter_name": self.sample.adapter_name,
            "llm_judgment": self.sample.llm_judgment,
            "llm_confidence": self.sample.llm_confidence,
            "human_judgment": self.human_judgment.value,
            "annotator_id": self.annotator_id,
            "annotation_time_seconds": self.annotation_time_seconds,
            "notes": self.notes,
            "confidence": self.confidence,
            "flagged": self.flagged,
            "timestamp": self.timestamp.isoformat(),
            "agrees_with_llm": self.agrees_with_llm(),
        }


@dataclass
class AnnotationSession:
    """A session of human annotation work.

    Attributes:
        session_id: Unique session identifier
        annotator_id: ID of the annotator
        start_time: Session start time
        end_time: Session end time (None if ongoing)
        annotations: List of annotations made
        metadata: Additional session metadata
    """

    session_id: str
    annotator_id: str
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime | None = None
    annotations: list[AnnotatedSample] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_annotation(self, annotation: AnnotatedSample) -> None:
        """Add an annotation to the session.

        Args:
            annotation: The annotation to add
        """
        self.annotations.append(annotation)

    def complete(self) -> None:
        """Mark the session as complete."""
        self.end_time = datetime.now()

    @property
    def is_complete(self) -> bool:
        """Check if session is complete."""
        return self.end_time is not None

    @property
    def duration_seconds(self) -> float:
        """Calculate session duration."""
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()

    @property
    def annotation_count(self) -> int:
        """Get number of annotations."""
        return len(self.annotations)

    @property
    def avg_time_per_sample(self) -> float:
        """Calculate average time per sample."""
        if not self.annotations:
            return 0.0
        total_time = sum(a.annotation_time_seconds for a in self.annotations)
        return total_time / len(self.annotations)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation
        """
        return {
            "session_id": self.session_id,
            "annotator_id": self.annotator_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "annotation_count": self.annotation_count,
            "avg_time_per_sample": self.avg_time_per_sample,
            "annotations": [a.to_dict() for a in self.annotations],
            "metadata": self.metadata,
        }


class ValidationCollector:
    """Collects and manages human validation data.

    This class handles:
    - Creating and managing annotation sessions
    - Storing and loading validation data
    - Tracking progress across annotators
    - Computing inter-annotator statistics
    """

    def __init__(
        self,
        data_dir: Path | str,
    ) -> None:
        """Initialize the collector.

        Args:
            data_dir: Directory to store validation data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.sessions: dict[str, AnnotationSession] = {}
        self.all_annotations: list[AnnotatedSample] = []

    def start_session(
        self,
        annotator_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> AnnotationSession:
        """Start a new annotation session.

        Args:
            annotator_id: ID of the annotator
            metadata: Optional session metadata

        Returns:
            New AnnotationSession
        """
        session_id = f"{annotator_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        session = AnnotationSession(
            session_id=session_id,
            annotator_id=annotator_id,
            metadata=metadata or {},
        )
        self.sessions[session_id] = session
        logger.info(f"Started annotation session: {session_id}")
        return session

    def complete_session(self, session_id: str) -> None:
        """Complete an annotation session.

        Args:
            session_id: ID of the session to complete
        """
        if session_id not in self.sessions:
            raise ValueError(f"Unknown session: {session_id}")

        session = self.sessions[session_id]
        session.complete()

        # Add annotations to master list
        self.all_annotations.extend(session.annotations)

        # Save session
        self._save_session(session)

        logger.info(
            f"Completed session {session_id}: "
            f"{session.annotation_count} annotations, "
            f"{session.duration_seconds:.0f}s duration"
        )

    def add_annotation(
        self,
        session_id: str,
        sample: CompiledSample,
        judgment: RubricLevel | str,
        time_seconds: float = 0.0,
        notes: str = "",
        confidence: int = 3,
        flagged: bool = False,
    ) -> AnnotatedSample:
        """Add an annotation to a session.

        Args:
            session_id: Session to add to
            sample: The sample being annotated
            judgment: Human judgment
            time_seconds: Time spent on annotation
            notes: Annotator notes
            confidence: Annotator confidence (1-5)
            flagged: Whether to flag for review

        Returns:
            The created AnnotatedSample
        """
        if session_id not in self.sessions:
            raise ValueError(f"Unknown session: {session_id}")

        session = self.sessions[session_id]

        if isinstance(judgment, str):
            judgment = RubricLevel(judgment)

        annotation = AnnotatedSample(
            sample=sample,
            human_judgment=judgment,
            annotator_id=session.annotator_id,
            annotation_time_seconds=time_seconds,
            notes=notes,
            confidence=min(5, max(1, confidence)),
            flagged=flagged,
        )

        session.add_annotation(annotation)
        return annotation

    def _save_session(self, session: AnnotationSession) -> None:
        """Save a session to disk.

        Args:
            session: Session to save
        """
        output_path = self.data_dir / f"{session.session_id}.json"
        with open(output_path, "w") as f:
            json.dump(session.to_dict(), f, indent=2)

    def load_all_sessions(self) -> None:
        """Load all saved sessions from disk."""
        for filepath in self.data_dir.glob("*.json"):
            try:
                with open(filepath) as f:
                    data = json.load(f)

                session = self._parse_session(data)
                self.sessions[session.session_id] = session
                self.all_annotations.extend(session.annotations)

            except Exception as e:
                logger.warning(f"Failed to load session {filepath}: {e}")

    def _parse_session(self, data: dict[str, Any]) -> AnnotationSession:
        """Parse a session from JSON data.

        Args:
            data: JSON data

        Returns:
            AnnotationSession
        """
        from src.validation.compiler import SourceBenchmark

        annotations = []
        for a in data.get("annotations", []):
            sample = CompiledSample(
                sample_id=a["sample_id"],
                source_benchmark=SourceBenchmark(a["source_benchmark"]),
                question_id=a["question_id"],
                question=a["question"],
                expected_answers=tuple(a["expected_answers"]),
                model_answer=a["model_answer"],
                adapter_name=a["adapter_name"],
                llm_judgment=a.get("llm_judgment", ""),
                llm_confidence=a.get("llm_confidence", 0.0),
            )

            annotation = AnnotatedSample(
                sample=sample,
                human_judgment=RubricLevel(a["human_judgment"]),
                annotator_id=a["annotator_id"],
                annotation_time_seconds=a.get("annotation_time_seconds", 0.0),
                notes=a.get("notes", ""),
                confidence=a.get("confidence", 3),
                flagged=a.get("flagged", False),
                timestamp=datetime.fromisoformat(a["timestamp"]),
            )
            annotations.append(annotation)

        return AnnotationSession(
            session_id=data["session_id"],
            annotator_id=data["annotator_id"],
            start_time=datetime.fromisoformat(data["start_time"]),
            end_time=(datetime.fromisoformat(data["end_time"]) if data.get("end_time") else None),
            annotations=annotations,
            metadata=data.get("metadata", {}),
        )

    def get_progress(self) -> dict[str, Any]:
        """Get annotation progress statistics.

        Returns:
            Dictionary of progress statistics
        """
        by_annotator: dict[str, int] = {}
        by_benchmark: dict[str, int] = {}
        by_judgment: dict[str, int] = {}

        for annotation in self.all_annotations:
            # By annotator
            ann_id = annotation.annotator_id
            by_annotator[ann_id] = by_annotator.get(ann_id, 0) + 1

            # By benchmark
            bm = annotation.sample.source_benchmark.value
            by_benchmark[bm] = by_benchmark.get(bm, 0) + 1

            # By judgment
            jdg = annotation.human_judgment.value
            by_judgment[jdg] = by_judgment.get(jdg, 0) + 1

        return {
            "total_annotations": len(self.all_annotations),
            "total_sessions": len(self.sessions),
            "completed_sessions": sum(1 for s in self.sessions.values() if s.is_complete),
            "by_annotator": by_annotator,
            "by_benchmark": by_benchmark,
            "by_judgment": by_judgment,
        }

    def get_annotations_for_sample(self, sample_id: str) -> list[AnnotatedSample]:
        """Get all annotations for a specific sample.

        Args:
            sample_id: Sample ID to look up

        Returns:
            List of annotations for that sample
        """
        return [a for a in self.all_annotations if a.sample.sample_id == sample_id]

    def export_all(self, output_path: Path | str) -> None:
        """Export all annotations to a single file.

        Args:
            output_path: Path to output file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "metadata": {
                "exported_at": datetime.now().isoformat(),
                "total_annotations": len(self.all_annotations),
                "total_sessions": len(self.sessions),
            },
            "annotations": [a.to_dict() for a in self.all_annotations],
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported {len(self.all_annotations)} annotations to {output_path}")
