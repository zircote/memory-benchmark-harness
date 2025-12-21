"""Preliminary results reporter for benchmark experiments.

This module generates summary reports from experiment results including:
- Summary statistics tables
- Accuracy comparisons with confidence intervals
- Raw data exports for further analysis

Per spec Task 1.5.4: Generate preliminary results.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.evaluation.statistics import (
    ComparisonResult,
    ConfidenceInterval,
    StatisticalAnalyzer,
)


@dataclass(frozen=True, slots=True)
class ConditionSummary:
    """Summary statistics for a single experimental condition.

    Attributes:
        condition: Name of the condition (e.g., 'git-notes', 'no-memory')
        n_trials: Number of trials completed
        n_successful: Number of successful trials
        accuracy_mean: Mean accuracy across trials
        accuracy_ci: Bootstrap confidence interval for accuracy
        metrics: Additional benchmark-specific metrics
    """

    condition: str
    n_trials: int
    n_successful: int
    accuracy_mean: float
    accuracy_ci: ConfidenceInterval | None
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "condition": self.condition,
            "n_trials": self.n_trials,
            "n_successful": self.n_successful,
            "accuracy_mean": self.accuracy_mean,
            "accuracy_ci": self.accuracy_ci.to_dict() if self.accuracy_ci else None,
            "metrics": self.metrics,
        }


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    """Complete report for a single benchmark.

    Attributes:
        benchmark: Benchmark name
        experiment_id: Unique experiment identifier
        condition_summaries: Summary for each condition
        comparison: Statistical comparison between conditions
        metadata: Additional experiment metadata
    """

    benchmark: str
    experiment_id: str
    condition_summaries: dict[str, ConditionSummary]
    comparison: ComparisonResult | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "benchmark": self.benchmark,
            "experiment_id": self.experiment_id,
            "condition_summaries": {k: v.to_dict() for k, v in self.condition_summaries.items()},
            "comparison": self.comparison.to_dict() if self.comparison else None,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class ResultsReporter:
    """Generates summary reports from experiment results.

    This reporter analyzes experiment results and produces:
    - Per-condition summary statistics
    - Statistical comparisons between conditions
    - Formatted output for publication

    Attributes:
        analyzer: Statistical analyzer for CI and comparisons
        confidence_level: Confidence level for intervals (default: 0.95)
        n_bootstrap: Number of bootstrap iterations (default: 2000)
    """

    analyzer: StatisticalAnalyzer = field(default_factory=StatisticalAnalyzer)
    confidence_level: float = 0.95
    n_bootstrap: int = 2000

    def _get_accuracy_key(self, benchmark: str) -> str:
        """Get the accuracy metric key for a benchmark."""
        if benchmark == "longmemeval":
            return "accuracy"
        elif benchmark == "locomo":
            return "overall_accuracy"
        return "accuracy"

    def _extract_accuracies(
        self,
        trials: list[dict[str, Any]],
        metric_key: str,
    ) -> list[float]:
        """Extract accuracy values from trial results."""
        accuracies: list[float] = []
        for trial in trials:
            if trial.get("success", True):
                metrics = trial.get("metrics", {})
                acc = metrics.get(metric_key)
                if acc is not None:
                    accuracies.append(float(acc))
        return accuracies

    def _compute_condition_summary(
        self,
        condition: str,
        trials: list[dict[str, Any]],
        benchmark: str,
    ) -> ConditionSummary:
        """Compute summary statistics for a condition."""
        metric_key = self._get_accuracy_key(benchmark)
        accuracies = self._extract_accuracies(trials, metric_key)

        n_trials = len(trials)
        n_successful = len([t for t in trials if t.get("success", True)])

        if not accuracies:
            return ConditionSummary(
                condition=condition,
                n_trials=n_trials,
                n_successful=n_successful,
                accuracy_mean=0.0,
                accuracy_ci=None,
                metrics={},
            )

        # Compute mean
        accuracy_mean = sum(accuracies) / len(accuracies)

        # Compute confidence interval if enough data
        accuracy_ci = None
        if len(accuracies) >= 2:
            try:
                accuracy_ci = self.analyzer.bootstrap_ci(np.array(accuracies))
            except Exception:
                pass  # Skip CI if computation fails

        # Collect additional metrics from first successful trial
        additional_metrics: dict[str, float] = {}
        for trial in trials:
            if trial.get("success", True):
                metrics = trial.get("metrics", {})
                for key, value in metrics.items():
                    if key != metric_key and isinstance(value, (int, float)):
                        if key not in additional_metrics:
                            additional_metrics[key] = value
                break

        return ConditionSummary(
            condition=condition,
            n_trials=n_trials,
            n_successful=n_successful,
            accuracy_mean=accuracy_mean,
            accuracy_ci=accuracy_ci,
            metrics=additional_metrics,
        )

    def generate_report(
        self,
        experiment_results: dict[str, Any],
    ) -> BenchmarkReport:
        """Generate a complete report from experiment results.

        Args:
            experiment_results: Full experiment results dictionary

        Returns:
            BenchmarkReport with summaries and comparisons
        """
        benchmark = experiment_results.get("benchmark", "unknown")
        experiment_id = experiment_results.get("experiment_id", "unknown")
        trials = experiment_results.get("trials", {})

        # Compute summaries for each condition
        condition_summaries: dict[str, ConditionSummary] = {}
        for condition, trial_list in trials.items():
            summary = self._compute_condition_summary(condition, trial_list, benchmark)
            condition_summaries[condition] = summary

        # Compute statistical comparison if we have two conditions
        comparison = None
        conditions = list(trials.keys())
        if len(conditions) == 2:
            metric_key = self._get_accuracy_key(benchmark)
            acc_a = self._extract_accuracies(trials[conditions[0]], metric_key)
            acc_b = self._extract_accuracies(trials[conditions[1]], metric_key)

            if len(acc_a) >= 2 and len(acc_b) >= 2:
                try:
                    comparison = self.analyzer.compare_conditions(
                        condition_a=acc_a,
                        condition_b=acc_b,
                        condition_a_name=conditions[0],
                        condition_b_name=conditions[1],
                    )
                except Exception:
                    pass  # Skip comparison if computation fails

        return BenchmarkReport(
            benchmark=benchmark,
            experiment_id=experiment_id,
            condition_summaries=condition_summaries,
            comparison=comparison,
            metadata={
                "config": experiment_results.get("config", {}),
                "started_at": experiment_results.get("started_at", ""),
                "completed_at": experiment_results.get("completed_at", ""),
            },
        )

    def format_summary_table(
        self,
        report: BenchmarkReport,
    ) -> str:
        """Format report as a markdown summary table.

        Args:
            report: Benchmark report to format

        Returns:
            Markdown-formatted table string
        """
        lines: list[str] = []

        lines.append(f"# {report.benchmark.upper()} Experiment Results")
        lines.append(f"\n**Experiment ID:** {report.experiment_id}")
        lines.append("")

        # Summary table
        lines.append("## Summary by Condition")
        lines.append("")
        lines.append("| Condition | Trials | Accuracy | 95% CI |")
        lines.append("|-----------|--------|----------|--------|")

        for condition, summary in sorted(report.condition_summaries.items()):
            ci_str = "N/A"
            if summary.accuracy_ci:
                ci_str = f"[{summary.accuracy_ci.lower:.1%}, {summary.accuracy_ci.upper:.1%}]"

            lines.append(
                f"| {condition} | {summary.n_successful}/{summary.n_trials} | "
                f"{summary.accuracy_mean:.1%} | {ci_str} |"
            )

        lines.append("")

        # Statistical comparison
        if report.comparison:
            lines.append("## Statistical Comparison")
            lines.append("")
            lines.append(f"- **Effect Size (Cohen's d):** {report.comparison.effect_size:.3f}")
            lines.append(f"- **p-value:** {report.comparison.p_value:.4f}")
            lines.append(
                f"- **Significant (α=0.05):** {'Yes' if report.comparison.is_significant else 'No'}"
            )
            lines.append("")

        return "\n".join(lines)

    def export_report(
        self,
        report: BenchmarkReport,
        output_dir: Path,
    ) -> dict[str, str]:
        """Export report to multiple formats.

        Args:
            report: Report to export
            output_dir: Directory for output files

        Returns:
            Dictionary mapping format to output path
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        outputs: dict[str, str] = {}

        # JSON export
        json_path = output_dir / f"{report.experiment_id}_report.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)
        outputs["json"] = str(json_path)

        # Markdown export
        md_path = output_dir / f"{report.experiment_id}_report.md"
        with md_path.open("w", encoding="utf-8") as f:
            f.write(self.format_summary_table(report))
        outputs["markdown"] = str(md_path)

        return outputs

    def generate_combined_report(
        self,
        experiment_results_list: list[dict[str, Any]],
        output_dir: Path,
    ) -> dict[str, Any]:
        """Generate combined report from multiple experiments.

        Args:
            experiment_results_list: List of experiment results
            output_dir: Directory for output files

        Returns:
            Summary of generated reports
        """
        reports: list[BenchmarkReport] = []
        all_outputs: dict[str, dict[str, str]] = {}

        for results in experiment_results_list:
            report = self.generate_report(results)
            reports.append(report)
            outputs = self.export_report(report, output_dir)
            all_outputs[report.experiment_id] = outputs

        # Generate combined summary
        combined_lines: list[str] = [
            "# Combined Benchmark Results",
            "",
            f"**Total Experiments:** {len(reports)}",
            "",
            "## Experiments Summary",
            "",
            "| Experiment | Benchmark | Conditions | Best Accuracy |",
            "|------------|-----------|------------|---------------|",
        ]

        for report in reports:
            conditions = ", ".join(sorted(report.condition_summaries.keys()))
            best_acc = max(s.accuracy_mean for s in report.condition_summaries.values())
            combined_lines.append(
                f"| {report.experiment_id} | {report.benchmark} | {conditions} | {best_acc:.1%} |"
            )

        combined_lines.append("")

        combined_path = output_dir / "combined_summary.md"
        with combined_path.open("w", encoding="utf-8") as f:
            f.write("\n".join(combined_lines))

        return {
            "reports_generated": len(reports),
            "output_files": all_outputs,
            "combined_summary": str(combined_path),
        }
