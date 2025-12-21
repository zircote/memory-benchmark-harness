"""Human validation sample exporter.

This module exports stratified samples from benchmark results for human
validation of LLM-as-Judge accuracy. Per spec requirements:
- 100 samples from LongMemEval
- 100 samples from LoCoMo
- Each sample includes: question, reference, response, judgment

The samples are exported in a format suitable for annotation tools.
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ValidationSample:
    """A single sample for human validation.

    Attributes:
        sample_id: Unique identifier for this sample
        benchmark: Source benchmark (longmemeval, locomo)
        category: Question category or type
        question: The original question text
        reference_answer: Ground truth or expected answer
        model_response: The model's generated response
        llm_judgment: The LLM judge's assessment
        llm_score: Numeric score from LLM judge (if applicable)
        condition: Memory adapter condition used
        metadata: Additional context (session_id, etc.)
    """

    sample_id: str
    benchmark: str
    category: str
    question: str
    reference_answer: str
    model_response: str
    llm_judgment: str
    llm_score: float | None
    condition: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "sample_id": self.sample_id,
            "benchmark": self.benchmark,
            "category": self.category,
            "question": self.question,
            "reference_answer": self.reference_answer,
            "model_response": self.model_response,
            "llm_judgment": self.llm_judgment,
            "llm_score": self.llm_score,
            "condition": self.condition,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class ValidationExporter:
    """Exports stratified samples for human validation.

    This exporter creates balanced sample sets from experiment results,
    ensuring representation across categories and conditions.

    Attributes:
        samples_per_benchmark: Number of samples per benchmark (default: 100)
        stratify_by_category: Whether to stratify by question category
        include_both_conditions: Include samples from both memory conditions
        seed: Random seed for reproducible sampling
    """

    samples_per_benchmark: int = 100
    stratify_by_category: bool = True
    include_both_conditions: bool = True
    seed: int = 42

    def extract_longmemeval_samples(
        self,
        experiment_results: dict[str, Any],
    ) -> list[ValidationSample]:
        """Extract validation samples from LongMemEval results.

        Args:
            experiment_results: Full experiment results dictionary

        Returns:
            List of ValidationSample objects
        """
        samples: list[ValidationSample] = []
        rng = random.Random(self.seed)

        trials = experiment_results.get("trials", {})

        for condition, trial_list in trials.items():
            for trial in trial_list:
                if not trial.get("success", True):
                    continue

                raw_results = trial.get("raw_results", {})
                question_results = raw_results.get("question_results", [])

                for qr in question_results:
                    sample = ValidationSample(
                        sample_id=f"lme_{condition}_{qr.get('question_id', 'unknown')}",
                        benchmark="longmemeval",
                        category=qr.get("question_type", "unknown"),
                        question=qr.get("question", ""),
                        reference_answer=qr.get("reference_answer", ""),
                        model_response=qr.get("predicted", ""),
                        llm_judgment=qr.get("judgment_text", ""),
                        llm_score=1.0 if qr.get("correct") else 0.0,
                        condition=condition,
                        metadata={
                            "session_id": qr.get("session_id"),
                            "abstained": qr.get("abstained", False),
                            "trial_id": trial.get("trial_id"),
                        },
                    )
                    samples.append(sample)

        # Stratified sampling
        return self._stratified_sample(
            samples,
            self.samples_per_benchmark,
            rng,
        )

    def extract_locomo_samples(
        self,
        experiment_results: dict[str, Any],
    ) -> list[ValidationSample]:
        """Extract validation samples from LoCoMo results.

        Args:
            experiment_results: Full experiment results dictionary

        Returns:
            List of ValidationSample objects
        """
        samples: list[ValidationSample] = []
        rng = random.Random(self.seed + 1)  # Different seed for LoCoMo

        trials = experiment_results.get("trials", {})

        for condition, trial_list in trials.items():
            for trial in trial_list:
                if not trial.get("success", True):
                    continue

                raw_results = trial.get("raw_results", {})
                conversation_results = raw_results.get("conversation_results", [])

                for cr in conversation_results:
                    question_results = cr.get("question_results", [])
                    for qr in question_results:
                        sample = ValidationSample(
                            sample_id=f"loco_{condition}_{cr.get('sample_id', 'unknown')}_{qr.get('question_id', 'unknown')}",
                            benchmark="locomo",
                            category=qr.get("category", "unknown"),
                            question=qr.get("question", ""),
                            reference_answer=qr.get("reference_answer", ""),
                            model_response=qr.get("predicted", ""),
                            llm_judgment=qr.get("judgment_text", ""),
                            llm_score=qr.get("score"),
                            condition=condition,
                            metadata={
                                "sample_id": cr.get("sample_id"),
                                "is_adversarial": qr.get("is_adversarial", False),
                                "difficulty": qr.get("difficulty"),
                                "trial_id": trial.get("trial_id"),
                            },
                        )
                        samples.append(sample)

        return self._stratified_sample(
            samples,
            self.samples_per_benchmark,
            rng,
        )

    def _stratified_sample(
        self,
        samples: list[ValidationSample],
        n: int,
        rng: random.Random,
    ) -> list[ValidationSample]:
        """Perform stratified sampling across categories and conditions.

        Args:
            samples: All available samples
            n: Target number of samples
            rng: Random number generator

        Returns:
            Stratified sample list
        """
        if len(samples) <= n:
            return samples

        if not self.stratify_by_category:
            return rng.sample(samples, n)

        # Group by category and condition
        groups: dict[tuple[str, str], list[ValidationSample]] = {}
        for sample in samples:
            key = (sample.category, sample.condition)
            if key not in groups:
                groups[key] = []
            groups[key].append(sample)

        # Calculate samples per group
        n_groups = len(groups)
        base_per_group = n // n_groups
        remainder = n % n_groups

        result: list[ValidationSample] = []
        sorted_keys = sorted(groups.keys())

        for i, key in enumerate(sorted_keys):
            group_samples = groups[key]
            # Distribute remainder across first groups
            target = base_per_group + (1 if i < remainder else 0)
            target = min(target, len(group_samples))

            selected = rng.sample(group_samples, target)
            result.extend(selected)

        return result

    def export_to_json(
        self,
        samples: list[ValidationSample],
        output_path: Path,
    ) -> None:
        """Export samples to JSON file.

        Args:
            samples: Samples to export
            output_path: Path to output file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "export_metadata": {
                "total_samples": len(samples),
                "samples_per_benchmark": self.samples_per_benchmark,
                "stratified": self.stratify_by_category,
                "seed": self.seed,
            },
            "samples": [s.to_dict() for s in samples],
        }

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def export_to_csv(
        self,
        samples: list[ValidationSample],
        output_path: Path,
    ) -> None:
        """Export samples to CSV file for annotation tools.

        Args:
            samples: Samples to export
            output_path: Path to output file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "sample_id",
            "benchmark",
            "category",
            "condition",
            "question",
            "reference_answer",
            "model_response",
            "llm_judgment",
            "llm_score",
            "human_score",  # Empty column for annotation
            "human_notes",  # Empty column for notes
        ]

        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for sample in samples:
                row = {
                    "sample_id": sample.sample_id,
                    "benchmark": sample.benchmark,
                    "category": sample.category,
                    "condition": sample.condition,
                    "question": sample.question,
                    "reference_answer": sample.reference_answer,
                    "model_response": sample.model_response,
                    "llm_judgment": sample.llm_judgment,
                    "llm_score": sample.llm_score if sample.llm_score is not None else "",
                    "human_score": "",
                    "human_notes": "",
                }
                writer.writerow(row)

    def export_combined(
        self,
        longmemeval_results: dict[str, Any] | None,
        locomo_results: dict[str, Any] | None,
        output_dir: Path,
    ) -> dict[str, Any]:
        """Export validation samples from both benchmarks.

        Args:
            longmemeval_results: LongMemEval experiment results
            locomo_results: LoCoMo experiment results
            output_dir: Directory to save exports

        Returns:
            Summary of exported samples
        """
        all_samples: list[ValidationSample] = []

        if longmemeval_results:
            lme_samples = self.extract_longmemeval_samples(longmemeval_results)
            all_samples.extend(lme_samples)

        if locomo_results:
            loco_samples = self.extract_locomo_samples(locomo_results)
            all_samples.extend(loco_samples)

        # Export to both formats
        self.export_to_json(all_samples, output_dir / "validation_samples.json")
        self.export_to_csv(all_samples, output_dir / "validation_samples.csv")

        # Summary by benchmark and category
        summary: dict[str, dict[str, int]] = {}
        for sample in all_samples:
            if sample.benchmark not in summary:
                summary[sample.benchmark] = {}
            if sample.category not in summary[sample.benchmark]:
                summary[sample.benchmark][sample.category] = 0
            summary[sample.benchmark][sample.category] += 1

        return {
            "total_samples": len(all_samples),
            "by_benchmark": summary,
            "output_files": [
                str(output_dir / "validation_samples.json"),
                str(output_dir / "validation_samples.csv"),
            ],
        }
