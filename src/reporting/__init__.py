"""Result reporting and publication artifact generation.

This module provides tools for generating reports from benchmark experiments:
- ValidationExporter: Export samples for human validation
- ResultsReporter: Generate summary statistics and comparison reports

Typical usage:
    ```python
    from src.reporting import ValidationExporter, ResultsReporter

    # Export validation samples
    exporter = ValidationExporter(samples_per_benchmark=100)
    summary = exporter.export_combined(
        longmemeval_results=lme_results,
        locomo_results=loco_results,
        output_dir=Path("results/validation"),
    )

    # Generate experiment report
    reporter = ResultsReporter()
    report = reporter.generate_report(experiment_results)
    reporter.export_report(report, Path("results/reports"))
    ```
"""

from src.reporting.results_reporter import (
    BenchmarkReport,
    ConditionSummary,
    ResultsReporter,
)
from src.reporting.validation_samples import (
    ValidationExporter,
    ValidationSample,
)

__all__ = [
    "BenchmarkReport",
    "ConditionSummary",
    "ResultsReporter",
    "ValidationExporter",
    "ValidationSample",
]
