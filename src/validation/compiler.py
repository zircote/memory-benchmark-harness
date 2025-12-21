"""Sample compiler for human validation.

This module compiles validation samples from multiple benchmark sources
into a unified format for human annotation.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class SourceBenchmark(Enum):
    """Source benchmark identifiers."""

    LONGMEMEVAL = "longmemeval"
    LOCOMO = "locomo"
    MEMORYAGENTBENCH = "memoryagentbench"
    CONTEXTBENCH = "contextbench"
    TERMINALBENCH = "terminalbench"

    @classmethod
    def from_string(cls, value: str) -> SourceBenchmark:
        """Parse benchmark from string.

        Args:
            value: String identifier

        Returns:
            Matching SourceBenchmark
        """
        normalized = value.lower().strip().replace("-", "").replace("_", "")
        for benchmark in cls:
            if benchmark.value.replace("_", "") == normalized:
                return benchmark
        raise ValueError(f"Unknown benchmark: {value}")


@dataclass(frozen=True, slots=True)
class CompiledSample:
    """A sample compiled for human validation.

    Attributes:
        sample_id: Unique sample identifier
        source_benchmark: Origin benchmark
        question_id: Original question ID
        question: The question text
        expected_answers: List of expected correct answers
        model_answer: The model's generated answer
        adapter_name: Name of the adapter used
        context: Context/memories used (if any)
        llm_judgment: LLM judge's verdict
        llm_confidence: LLM judge's confidence
        metadata: Additional sample metadata
    """

    sample_id: str
    source_benchmark: SourceBenchmark
    question_id: str
    question: str
    expected_answers: tuple[str, ...]
    model_answer: str
    adapter_name: str
    context: str = ""
    llm_judgment: str = ""
    llm_confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SampleCompiler:
    """Compiles samples from multiple benchmarks for validation.

    This class:
    1. Loads samples from multiple benchmark result files
    2. Converts them to a unified format
    3. Applies stratified sampling if needed
    4. Exports to formats suitable for annotation
    """

    samples: list[CompiledSample] = field(default_factory=list)
    seed: int = 42
    target_total: int = 500

    def add_samples_from_file(
        self,
        filepath: Path | str,
        benchmark: SourceBenchmark | str,
    ) -> int:
        """Add samples from a result file.

        Args:
            filepath: Path to the result file
            benchmark: Source benchmark

        Returns:
            Number of samples added
        """
        filepath = Path(filepath)
        if not filepath.exists():
            logger.warning(f"File not found: {filepath}")
            return 0

        if isinstance(benchmark, str):
            benchmark = SourceBenchmark.from_string(benchmark)

        with open(filepath) as f:
            data = json.load(f)

        # Handle different file formats
        samples_data = data.get("samples", data.get("results", []))
        if isinstance(samples_data, dict):
            samples_data = list(samples_data.values())

        count = 0
        for item in samples_data:
            sample = self._parse_sample(item, benchmark)
            if sample:
                self.samples.append(sample)
                count += 1

        logger.info(f"Added {count} samples from {filepath}")
        return count

    def _parse_sample(
        self,
        data: dict[str, Any],
        benchmark: SourceBenchmark,
    ) -> CompiledSample | None:
        """Parse a sample from raw data.

        Args:
            data: Raw sample data
            benchmark: Source benchmark

        Returns:
            CompiledSample or None if parsing fails
        """
        try:
            # Handle different field names across benchmarks
            question_id = data.get(
                "question_id",
                data.get("id", data.get("sample_id", "")),
            )
            question = data.get(
                "question",
                data.get("question_text", data.get("query", "")),
            )
            expected = data.get(
                "expected_answers",
                data.get("answers", data.get("expected", [])),
            )
            if isinstance(expected, str):
                expected = [expected]

            model_answer = data.get(
                "model_answer",
                data.get("answer", data.get("generated", "")),
            )
            adapter_name = data.get(
                "adapter_name",
                data.get("adapter", data.get("condition", "")),
            )
            context = data.get(
                "context",
                data.get("memory_context", data.get("retrieved", "")),
            )
            llm_judgment = data.get(
                "llm_judgment",
                data.get("judgment", data.get("verdict", "")),
            )
            llm_confidence = data.get(
                "llm_confidence",
                data.get("confidence", data.get("score", 0.0)),
            )

            sample_id = f"{benchmark.value}_{question_id}_{len(self.samples)}"

            return CompiledSample(
                sample_id=sample_id,
                source_benchmark=benchmark,
                question_id=str(question_id),
                question=str(question),
                expected_answers=tuple(str(a) for a in expected),
                model_answer=str(model_answer),
                adapter_name=str(adapter_name),
                context=str(context) if context else "",
                llm_judgment=str(llm_judgment),
                llm_confidence=float(llm_confidence) if llm_confidence else 0.0,
                metadata=data.get("metadata", {}),
            )
        except Exception as e:
            logger.warning(f"Failed to parse sample: {e}")
            return None

    def stratified_sample(
        self,
        n: int | None = None,
        by_benchmark: bool = True,
        by_judgment: bool = True,
    ) -> list[CompiledSample]:
        """Get a stratified sample of compiled samples.

        Args:
            n: Number of samples to return (default: target_total)
            by_benchmark: Stratify by source benchmark
            by_judgment: Stratify by LLM judgment

        Returns:
            Stratified list of samples
        """
        if n is None:
            n = self.target_total

        if not self.samples:
            return []

        if n >= len(self.samples):
            return list(self.samples)

        rng = random.Random(self.seed)

        # Group samples by strata
        strata: dict[tuple[str, ...], list[CompiledSample]] = {}
        for sample in self.samples:
            key_parts = []
            if by_benchmark:
                key_parts.append(sample.source_benchmark.value)
            if by_judgment:
                key_parts.append(sample.llm_judgment or "unknown")
            key = tuple(key_parts) if key_parts else ("all",)

            if key not in strata:
                strata[key] = []
            strata[key].append(sample)

        # Calculate samples per stratum
        per_stratum = n // len(strata)
        remainder = n % len(strata)

        selected = []
        stratum_keys = sorted(strata.keys())
        rng.shuffle(stratum_keys)

        for i, key in enumerate(stratum_keys):
            stratum_samples = strata[key]
            rng.shuffle(stratum_samples)

            # Extra sample for first 'remainder' strata
            count = per_stratum + (1 if i < remainder else 0)
            count = min(count, len(stratum_samples))

            selected.extend(stratum_samples[:count])

        return selected

    def export_json(
        self,
        output_path: Path | str,
        samples: list[CompiledSample] | None = None,
    ) -> None:
        """Export samples to JSON format.

        Args:
            output_path: Path to output file
            samples: Samples to export (default: all samples)
        """
        if samples is None:
            samples = self.samples

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "metadata": {
                "exported_at": datetime.now().isoformat(),
                "total_samples": len(samples),
                "target_total": self.target_total,
                "seed": self.seed,
            },
            "samples": [
                {
                    "sample_id": s.sample_id,
                    "source_benchmark": s.source_benchmark.value,
                    "question_id": s.question_id,
                    "question": s.question,
                    "expected_answers": list(s.expected_answers),
                    "model_answer": s.model_answer,
                    "adapter_name": s.adapter_name,
                    "context": s.context,
                    "llm_judgment": s.llm_judgment,
                    "llm_confidence": s.llm_confidence,
                    "metadata": s.metadata,
                }
                for s in samples
            ],
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported {len(samples)} samples to {output_path}")

    def export_csv(
        self,
        output_path: Path | str,
        samples: list[CompiledSample] | None = None,
    ) -> None:
        """Export samples to CSV format for spreadsheet annotation.

        Args:
            output_path: Path to output file
            samples: Samples to export (default: all samples)
        """
        import csv

        if samples is None:
            samples = self.samples

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "sample_id",
            "source_benchmark",
            "question_id",
            "question",
            "expected_answers",
            "model_answer",
            "adapter_name",
            "llm_judgment",
            "llm_confidence",
            "human_judgment",  # To be filled by annotator
            "human_notes",  # To be filled by annotator
        ]

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for s in samples:
                writer.writerow(
                    {
                        "sample_id": s.sample_id,
                        "source_benchmark": s.source_benchmark.value,
                        "question_id": s.question_id,
                        "question": s.question,
                        "expected_answers": "; ".join(s.expected_answers),
                        "model_answer": s.model_answer,
                        "adapter_name": s.adapter_name,
                        "llm_judgment": s.llm_judgment,
                        "llm_confidence": f"{s.llm_confidence:.2f}",
                        "human_judgment": "",
                        "human_notes": "",
                    }
                )

        logger.info(f"Exported {len(samples)} samples to {output_path}")

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about compiled samples.

        Returns:
            Dictionary of statistics
        """
        by_benchmark: dict[str, int] = {}
        by_judgment: dict[str, int] = {}
        by_adapter: dict[str, int] = {}

        for sample in self.samples:
            # By benchmark
            bm = sample.source_benchmark.value
            by_benchmark[bm] = by_benchmark.get(bm, 0) + 1

            # By judgment
            jdg = sample.llm_judgment or "unknown"
            by_judgment[jdg] = by_judgment.get(jdg, 0) + 1

            # By adapter
            adp = sample.adapter_name or "unknown"
            by_adapter[adp] = by_adapter.get(adp, 0) + 1

        return {
            "total_samples": len(self.samples),
            "target_total": self.target_total,
            "by_benchmark": by_benchmark,
            "by_judgment": by_judgment,
            "by_adapter": by_adapter,
        }
