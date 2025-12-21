"""Analysis of human validation data.

This module provides tools for analyzing human validation data,
including inter-annotator agreement and comparison with LLM judgments.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from src.validation.annotation import RubricLevel
from src.validation.collector import AnnotatedSample

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AgreementMetrics:
    """Metrics for inter-annotator or human-LLM agreement.

    Attributes:
        agreement_rate: Proportion of samples with agreement
        total_samples: Total samples compared
        agreed_samples: Number of agreed samples
        kappa: Cohen's Kappa coefficient
        weighted_kappa: Weighted Kappa for ordinal data
        confusion_matrix: Confusion matrix of judgments
    """

    agreement_rate: float
    total_samples: int
    agreed_samples: int
    kappa: float = 0.0
    weighted_kappa: float = 0.0
    confusion_matrix: dict[str, dict[str, int]] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Complete validation analysis report.

    Attributes:
        total_annotations: Total number of annotations
        unique_samples: Number of unique samples annotated
        annotator_count: Number of annotators
        human_llm_agreement: Agreement between humans and LLM
        inter_annotator_agreement: Agreement between annotators
        judgment_distribution: Distribution of human judgments
        by_benchmark: Metrics by source benchmark
        by_adapter: Metrics by adapter condition
        flagged_samples: Number of samples flagged for review
        avg_confidence: Average annotator confidence
        avg_time_per_sample: Average annotation time
    """

    total_annotations: int
    unique_samples: int
    annotator_count: int
    human_llm_agreement: AgreementMetrics
    inter_annotator_agreement: AgreementMetrics | None
    judgment_distribution: dict[str, int]
    by_benchmark: dict[str, dict[str, Any]]
    by_adapter: dict[str, dict[str, Any]]
    flagged_samples: int
    avg_confidence: float
    avg_time_per_sample: float


class ValidationAnalyzer:
    """Analyzes human validation data.

    This class computes:
    - Agreement metrics between annotators
    - Agreement metrics between humans and LLM judges
    - Judgment distributions and statistics
    - Quality metrics per benchmark and adapter
    """

    def __init__(
        self,
        annotations: list[AnnotatedSample],
    ) -> None:
        """Initialize the analyzer.

        Args:
            annotations: List of annotated samples to analyze
        """
        self.annotations = annotations

        # Index by sample ID
        self._by_sample: dict[str, list[AnnotatedSample]] = {}
        for a in annotations:
            sid = a.sample.sample_id
            if sid not in self._by_sample:
                self._by_sample[sid] = []
            self._by_sample[sid].append(a)

    def compute_human_llm_agreement(self) -> AgreementMetrics:
        """Compute agreement between human annotations and LLM judgments.

        Returns:
            AgreementMetrics
        """
        # Use first human annotation per sample
        pairs = []
        for _sample_id, annotations in self._by_sample.items():
            human_judgment = annotations[0].human_judgment
            llm_judgment = annotations[0].sample.llm_judgment

            if llm_judgment and llm_judgment.lower() not in {"", "unknown"}:
                pairs.append((human_judgment, llm_judgment))

        if not pairs:
            return AgreementMetrics(
                agreement_rate=0.0,
                total_samples=0,
                agreed_samples=0,
            )

        return self._compute_agreement(pairs)

    def compute_inter_annotator_agreement(self) -> AgreementMetrics | None:
        """Compute agreement between human annotators.

        Returns:
            AgreementMetrics or None if not enough data
        """
        # Find samples with multiple annotations
        multi_annotated = {sid: anns for sid, anns in self._by_sample.items() if len(anns) >= 2}

        if not multi_annotated:
            return None

        pairs = []
        for _sample_id, annotations in multi_annotated.items():
            # Compare first two annotators
            j1 = annotations[0].human_judgment
            j2 = annotations[1].human_judgment
            pairs.append((j1, j2))

        return self._compute_agreement(pairs)

    def _compute_agreement(
        self,
        pairs: Sequence[tuple[RubricLevel | str, RubricLevel | str]],
    ) -> AgreementMetrics:
        """Compute agreement metrics from judgment pairs.

        Args:
            pairs: List of (judgment1, judgment2) pairs

        Returns:
            AgreementMetrics
        """
        # Normalize judgments
        normalized = []
        for j1, j2 in pairs:
            n1 = self._normalize_judgment(j1)
            n2 = self._normalize_judgment(j2)
            if n1 is not None and n2 is not None:
                normalized.append((n1, n2))

        if not normalized:
            return AgreementMetrics(
                agreement_rate=0.0,
                total_samples=0,
                agreed_samples=0,
            )

        # Simple agreement
        agreed = sum(1 for j1, j2 in normalized if j1 == j2)
        total = len(normalized)
        agreement_rate = agreed / total if total > 0 else 0.0

        # Build confusion matrix
        labels = [r.value for r in RubricLevel]
        confusion: dict[str, dict[str, int]] = {l1: dict.fromkeys(labels, 0) for l1 in labels}
        for j1, j2 in normalized:
            confusion[j1.value][j2.value] += 1

        # Compute Cohen's Kappa
        kappa = self._compute_kappa(normalized)

        # Compute weighted Kappa
        weighted_kappa = self._compute_weighted_kappa(normalized)

        return AgreementMetrics(
            agreement_rate=agreement_rate,
            total_samples=total,
            agreed_samples=agreed,
            kappa=kappa,
            weighted_kappa=weighted_kappa,
            confusion_matrix=confusion,
        )

    def _normalize_judgment(self, judgment: RubricLevel | str) -> RubricLevel | None:
        """Normalize a judgment to RubricLevel.

        Args:
            judgment: Judgment to normalize

        Returns:
            RubricLevel or None if unrecognized
        """
        if isinstance(judgment, RubricLevel):
            return judgment

        mapping = {
            "correct": RubricLevel.CORRECT,
            "incorrect": RubricLevel.INCORRECT,
            "partial": RubricLevel.PARTIALLY_CORRECT,
            "partially_correct": RubricLevel.PARTIALLY_CORRECT,
            "partially correct": RubricLevel.PARTIALLY_CORRECT,
            "cannot_judge": RubricLevel.CANNOT_JUDGE,
            "cannot judge": RubricLevel.CANNOT_JUDGE,
            "unknown": RubricLevel.CANNOT_JUDGE,
        }

        normalized = judgment.lower().strip()
        return mapping.get(normalized)

    def _compute_kappa(
        self,
        pairs: list[tuple[RubricLevel, RubricLevel]],
    ) -> float:
        """Compute Cohen's Kappa.

        Args:
            pairs: Judgment pairs

        Returns:
            Kappa coefficient
        """
        if not pairs:
            return 0.0

        n = len(pairs)

        # Count agreements and marginals
        labels = list(RubricLevel)
        counts_a: dict[RubricLevel, int] = dict.fromkeys(labels, 0)
        counts_b: dict[RubricLevel, int] = dict.fromkeys(labels, 0)
        agreements = 0

        for j1, j2 in pairs:
            counts_a[j1] += 1
            counts_b[j2] += 1
            if j1 == j2:
                agreements += 1

        # Observed agreement
        p_o = agreements / n

        # Expected agreement
        p_e = sum((counts_a[level] / n) * (counts_b[level] / n) for level in labels)

        # Kappa
        if p_e == 1.0:
            return 1.0
        return (p_o - p_e) / (1 - p_e)

    def _compute_weighted_kappa(
        self,
        pairs: list[tuple[RubricLevel, RubricLevel]],
    ) -> float:
        """Compute weighted Kappa using linear weights.

        Args:
            pairs: Judgment pairs

        Returns:
            Weighted Kappa coefficient
        """
        if not pairs:
            return 0.0

        # Define ordinal order
        order = {
            RubricLevel.CORRECT: 0,
            RubricLevel.PARTIALLY_CORRECT: 1,
            RubricLevel.INCORRECT: 2,
            RubricLevel.CANNOT_JUDGE: 3,
        }
        max_dist = max(order.values())

        # Weight function (linear)
        def weight(j1: RubricLevel, j2: RubricLevel) -> float:
            dist = abs(order.get(j1, 0) - order.get(j2, 0))
            return 1 - (dist / max_dist) if max_dist > 0 else 1.0

        # Observed weighted agreement
        n = len(pairs)
        observed = sum(weight(j1, j2) for j1, j2 in pairs) / n

        # Expected weighted agreement
        labels = list(RubricLevel)
        counts_a: dict[RubricLevel, int] = dict.fromkeys(labels, 0)
        counts_b: dict[RubricLevel, int] = dict.fromkeys(labels, 0)

        for j1, j2 in pairs:
            counts_a[j1] += 1
            counts_b[j2] += 1

        expected = 0.0
        for l1 in labels:
            for l2 in labels:
                expected += (counts_a[l1] / n) * (counts_b[l2] / n) * weight(l1, l2)

        # Weighted Kappa
        if expected == 1.0:
            return 1.0
        return (observed - expected) / (1 - expected)

    def generate_report(self) -> ValidationReport:
        """Generate a complete validation report.

        Returns:
            ValidationReport
        """
        # Basic counts
        unique_samples = len(self._by_sample)
        annotators = {a.annotator_id for a in self.annotations}

        # Judgment distribution
        judgment_dist: dict[str, int] = {}
        for a in self.annotations:
            j = a.human_judgment.value
            judgment_dist[j] = judgment_dist.get(j, 0) + 1

        # Flagged samples
        flagged = sum(1 for a in self.annotations if a.flagged)

        # Average confidence
        confidences = [a.confidence for a in self.annotations]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        # Average time
        times = [a.annotation_time_seconds for a in self.annotations]
        avg_time = sum(times) / len(times) if times else 0.0

        # By benchmark
        by_benchmark: dict[str, dict[str, Any]] = {}
        for a in self.annotations:
            bm = a.sample.source_benchmark.value
            if bm not in by_benchmark:
                by_benchmark[bm] = {
                    "total": 0,
                    "correct": 0,
                    "incorrect": 0,
                    "partial": 0,
                }
            by_benchmark[bm]["total"] += 1
            if a.human_judgment == RubricLevel.CORRECT:
                by_benchmark[bm]["correct"] += 1
            elif a.human_judgment == RubricLevel.INCORRECT:
                by_benchmark[bm]["incorrect"] += 1
            elif a.human_judgment == RubricLevel.PARTIALLY_CORRECT:
                by_benchmark[bm]["partial"] += 1

        # By adapter
        by_adapter: dict[str, dict[str, Any]] = {}
        for a in self.annotations:
            adapter = a.sample.adapter_name or "unknown"
            if adapter not in by_adapter:
                by_adapter[adapter] = {
                    "total": 0,
                    "correct": 0,
                    "incorrect": 0,
                    "partial": 0,
                }
            by_adapter[adapter]["total"] += 1
            if a.human_judgment == RubricLevel.CORRECT:
                by_adapter[adapter]["correct"] += 1
            elif a.human_judgment == RubricLevel.INCORRECT:
                by_adapter[adapter]["incorrect"] += 1
            elif a.human_judgment == RubricLevel.PARTIALLY_CORRECT:
                by_adapter[adapter]["partial"] += 1

        # Agreement metrics
        human_llm = self.compute_human_llm_agreement()
        inter_annotator = self.compute_inter_annotator_agreement()

        return ValidationReport(
            total_annotations=len(self.annotations),
            unique_samples=unique_samples,
            annotator_count=len(annotators),
            human_llm_agreement=human_llm,
            inter_annotator_agreement=inter_annotator,
            judgment_distribution=judgment_dist,
            by_benchmark=by_benchmark,
            by_adapter=by_adapter,
            flagged_samples=flagged,
            avg_confidence=avg_confidence,
            avg_time_per_sample=avg_time,
        )

    def format_report(self, report: ValidationReport | None = None) -> str:
        """Format a report as markdown.

        Args:
            report: Report to format (generates if None)

        Returns:
            Markdown string
        """
        if report is None:
            report = self.generate_report()

        lines = [
            "# Human Validation Report",
            "",
            "## Summary",
            "",
            f"- **Total Annotations**: {report.total_annotations}",
            f"- **Unique Samples**: {report.unique_samples}",
            f"- **Annotators**: {report.annotator_count}",
            f"- **Flagged Samples**: {report.flagged_samples}",
            f"- **Avg Confidence**: {report.avg_confidence:.1f}/5",
            f"- **Avg Time/Sample**: {report.avg_time_per_sample:.1f}s",
            "",
            "## Judgment Distribution",
            "",
            "| Judgment | Count | % |",
            "|----------|-------|---|",
        ]

        total = sum(report.judgment_distribution.values())
        for judgment, count in sorted(report.judgment_distribution.items()):
            pct = count / total * 100 if total > 0 else 0
            lines.append(f"| {judgment} | {count} | {pct:.1f}% |")

        lines.extend(
            [
                "",
                "## Human-LLM Agreement",
                "",
                f"- **Agreement Rate**: {report.human_llm_agreement.agreement_rate:.1%}",
                f"- **Cohen's Kappa**: {report.human_llm_agreement.kappa:.3f}",
                f"- **Samples Compared**: {report.human_llm_agreement.total_samples}",
            ]
        )

        if report.inter_annotator_agreement:
            lines.extend(
                [
                    "",
                    "## Inter-Annotator Agreement",
                    "",
                    f"- **Agreement Rate**: {report.inter_annotator_agreement.agreement_rate:.1%}",
                    f"- **Cohen's Kappa**: {report.inter_annotator_agreement.kappa:.3f}",
                    f"- **Weighted Kappa**: {report.inter_annotator_agreement.weighted_kappa:.3f}",
                    f"- **Samples Compared**: {report.inter_annotator_agreement.total_samples}",
                ]
            )

        lines.extend(
            [
                "",
                "## By Benchmark",
                "",
                "| Benchmark | Total | Correct | Incorrect | Partial |",
                "|-----------|-------|---------|-----------|---------|",
            ]
        )

        for bm, stats in sorted(report.by_benchmark.items()):
            lines.append(
                f"| {bm} | {stats['total']} | {stats['correct']} | "
                f"{stats['incorrect']} | {stats['partial']} |"
            )

        lines.extend(
            [
                "",
                "## By Adapter",
                "",
                "| Adapter | Total | Correct | Incorrect | Partial |",
                "|---------|-------|---------|-----------|---------|",
            ]
        )

        for adapter, stats in sorted(report.by_adapter.items()):
            lines.append(
                f"| {adapter} | {stats['total']} | {stats['correct']} | "
                f"{stats['incorrect']} | {stats['partial']} |"
            )

        return "\n".join(lines)
