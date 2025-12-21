"""CLI entry point for memory benchmark harness."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated, Any

import typer

from src.experiments.runner import (
    AdapterCondition,
    ExperimentConfig,
    ExperimentResults,
    ExperimentRunner,
)
from src.publication import (
    AblationHeatmap,
    AblationTable,
    CategoryBreakdownTable,
    CategoryRadarPlot,
    MainResultsTable,
    PerformanceBarChart,
    PublicationStatistics,
)
from src.publication.figures import ConfidenceIntervalPlot
from src.reporting import ResultsReporter, ValidationExporter

app = typer.Typer(
    name="benchmark",
    help="Benchmark harness for git-native semantic memory validation.",
    no_args_is_help=True,
)

# Publication sub-app for final analysis commands
publication_app = typer.Typer(
    name="publication",
    help="Generate publication artifacts (tables, figures, statistics).",
    no_args_is_help=True,
)
app.add_typer(publication_app, name="publication")


def _progress_callback(message: str, completed: int, total: int) -> None:
    """Print progress updates."""
    pct = (completed / total * 100) if total > 0 else 0
    typer.echo(f"[{completed}/{total}] ({pct:.0f}%) {message}")


@app.command()
def run(
    benchmark: Annotated[
        str,
        typer.Argument(
            help="Benchmark to run: 'longmemeval' or 'locomo'",
        ),
    ],
    adapter: Annotated[
        str,
        typer.Option(
            "--adapter",
            "-a",
            help="Memory adapter condition(s): 'git-notes', 'no-memory', 'mock'. "
            "Can specify multiple with comma separation.",
        ),
    ] = "no-memory",
    trials: Annotated[
        int,
        typer.Option(
            "--trials",
            "-n",
            help="Number of trials per condition (default: 5)",
        ),
    ] = 5,
    seed: Annotated[
        int,
        typer.Option(
            "--seed",
            "-s",
            help="Base random seed for reproducibility",
        ),
    ] = 42,
    output_dir: Annotated[
        str,
        typer.Option(
            "--output",
            "-o",
            help="Directory to save results",
        ),
    ] = "results",
    dataset: Annotated[
        str | None,
        typer.Option(
            "--dataset",
            "-d",
            help="Path to dataset file (uses default download if not specified)",
        ),
    ] = None,
    quiet: Annotated[
        bool,
        typer.Option(
            "--quiet",
            "-q",
            help="Suppress progress output",
        ),
    ] = False,
) -> None:
    """Run a benchmark experiment with the specified adapter(s).

    Example:
        benchmark run longmemeval --adapter no-memory,mock --trials 3
    """
    # Parse adapter conditions
    adapter_names = [a.strip() for a in adapter.split(",")]
    try:
        adapters = [AdapterCondition(name) for name in adapter_names]
    except ValueError as e:
        typer.echo(f"Error: Invalid adapter condition: {e}", err=True)
        raise typer.Exit(1) from e

    # Validate benchmark
    valid_benchmarks = {"longmemeval", "locomo"}
    if benchmark not in valid_benchmarks:
        typer.echo(
            f"Error: Invalid benchmark '{benchmark}'. "
            f"Must be one of: {', '.join(valid_benchmarks)}",
            err=True,
        )
        raise typer.Exit(1)

    # Create config
    config = ExperimentConfig(
        benchmark=benchmark,
        adapters=adapters,
        num_trials=trials,
        base_seed=seed,
        output_dir=output_dir,
        dataset_path=dataset,
        progress_callback=None if quiet else _progress_callback,
    )

    typer.echo(f"Running {benchmark} benchmark")
    typer.echo(f"  Adapters: {', '.join(a.value for a in adapters)}")
    typer.echo(f"  Trials per condition: {trials}")
    typer.echo(f"  Base seed: {seed}")
    typer.echo(f"  Output directory: {output_dir}")
    typer.echo()

    # Run experiment
    runner = ExperimentRunner(config)
    results: ExperimentResults = asyncio.run(runner.run())  # type: ignore[arg-type]

    # Report results
    typer.echo()
    typer.echo("=" * 60)
    typer.echo(f"Experiment complete: {results.experiment_id}")
    typer.echo(f"  Total trials: {results.total_trials}")
    typer.echo(f"  Successful: {results.successful_trials}")
    typer.echo(f"  Failed: {results.total_trials - results.successful_trials}")
    typer.echo()

    # Show per-condition summary
    for condition, trial_list in results.trials.items():
        successful = [t for t in trial_list if t.success]
        if successful:
            # Get average metrics
            if benchmark == "longmemeval":
                avg_acc = sum(t.metrics.get("accuracy", 0) for t in successful) / len(successful)
                typer.echo(f"  {condition}: avg accuracy = {avg_acc:.2%}")
            elif benchmark == "locomo":
                avg_acc = sum(t.metrics.get("overall_accuracy", 0) for t in successful) / len(
                    successful
                )
                typer.echo(f"  {condition}: avg overall accuracy = {avg_acc:.2%}")
        else:
            typer.echo(f"  {condition}: all trials failed")

    # Show output file
    output_path = Path(output_dir) / f"{results.experiment_id}.json"
    typer.echo()
    typer.echo(f"Results saved to: {output_path}")


@app.command()
def list_benchmarks() -> None:
    """List available benchmarks."""
    typer.echo("Available benchmarks:")
    typer.echo()
    typer.echo("  longmemeval  - LongMemEval: Long-term memory QA (HuggingFace)")
    typer.echo("                 Tests single-session and multi-session memory")
    typer.echo("                 Question types: factoid, reasoning, temporal")
    typer.echo()
    typer.echo("  locomo       - LoCoMo: Long Conversational Memory (Snap Research)")
    typer.echo("                 Tests multi-session conversational memory")
    typer.echo("                 5 QA categories: identity, temporal, inference,")
    typer.echo("                 contextual, adversarial")


@app.command()
def list_adapters() -> None:
    """List available memory adapter conditions."""
    typer.echo("Available adapter conditions:")
    typer.echo()
    typer.echo("  git-notes  - GitNotesAdapter using git-notes-memory-manager")
    typer.echo("               Requires git-notes-memory-manager package installed")
    typer.echo()
    typer.echo("  no-memory  - NoMemoryAdapter (baseline without memory)")
    typer.echo("               All memory operations return empty results")
    typer.echo()
    typer.echo("  mock       - MockAdapter for testing")
    typer.echo("               Configurable behavior for test scenarios")


@app.command()
def compare(
    results_a: Annotated[
        Path,
        typer.Argument(
            help="Path to first results JSON file",
        ),
    ],
    results_b: Annotated[
        Path,
        typer.Argument(
            help="Path to second results JSON file",
        ),
    ],
) -> None:
    """Compare results from two experiment runs.

    Uses statistical analysis to determine if differences are significant.
    """
    import json

    # Load results
    if not results_a.exists():
        typer.echo(f"Error: File not found: {results_a}")
        raise typer.Exit(1)
    if not results_b.exists():
        typer.echo(f"Error: File not found: {results_b}")
        raise typer.Exit(1)

    with results_a.open() as f:
        data_a = json.load(f)
    with results_b.open() as f:
        data_b = json.load(f)

    typer.echo("Comparing experiments:")
    typer.echo(f"  A: {data_a['experiment_id']} ({data_a['benchmark']})")
    typer.echo(f"  B: {data_b['experiment_id']} ({data_b['benchmark']})")
    typer.echo()

    # Check same benchmark
    if data_a["benchmark"] != data_b["benchmark"]:
        typer.echo(
            "Warning: Different benchmarks - comparison may not be meaningful",
            err=True,
        )

    # Compare conditions
    conditions_a = set(data_a["trials"].keys())
    conditions_b = set(data_b["trials"].keys())
    common = conditions_a & conditions_b

    if not common:
        typer.echo("Error: No common conditions to compare", err=True)
        raise typer.Exit(1)

    typer.echo(f"Common conditions: {', '.join(common)}")
    typer.echo()

    # For each common condition, compare metrics
    for condition in sorted(common):
        typer.echo(f"Condition: {condition}")

        trials_a = [t for t in data_a["trials"][condition] if t.get("success", True)]
        trials_b = [t for t in data_b["trials"][condition] if t.get("success", True)]

        if not trials_a or not trials_b:
            typer.echo("  Insufficient successful trials for comparison")
            continue

        # Get accuracy metric
        metric_key = "accuracy" if data_a["benchmark"] == "longmemeval" else "overall_accuracy"

        values_a = [t["metrics"].get(metric_key, 0) for t in trials_a]
        values_b = [t["metrics"].get(metric_key, 0) for t in trials_b]

        avg_a = sum(values_a) / len(values_a)
        avg_b = sum(values_b) / len(values_b)
        diff = avg_b - avg_a

        typer.echo(f"  A: {avg_a:.2%} (n={len(values_a)})")
        typer.echo(f"  B: {avg_b:.2%} (n={len(values_b)})")
        typer.echo(f"  Difference: {diff:+.2%}")
        typer.echo()


@app.command("export-samples")
def export_samples(
    results: Annotated[
        Path,
        typer.Argument(
            help="Path to experiment results JSON file",
        ),
    ],
    output_dir: Annotated[
        str,
        typer.Option(
            "--output",
            "-o",
            help="Directory to save validation samples",
        ),
    ] = "results/validation",
    samples: Annotated[
        int,
        typer.Option(
            "--samples",
            "-n",
            help="Number of samples per benchmark (default: 100)",
        ),
    ] = 100,
    seed: Annotated[
        int,
        typer.Option(
            "--seed",
            "-s",
            help="Random seed for reproducible sampling",
        ),
    ] = 42,
    no_stratify: Annotated[
        bool,
        typer.Option(
            "--no-stratify",
            help="Disable stratification by category",
        ),
    ] = False,
) -> None:
    """Export validation samples from experiment results for human review.

    This command extracts a stratified sample of question-answer pairs from
    experiment results for human validation of LLM-as-Judge accuracy.

    Example:
        benchmark export-samples results/exp_001.json -o validation/ -n 50
    """
    if not results.exists():
        typer.echo(f"Error: File not found: {results}")
        raise typer.Exit(1)

    with results.open() as f:
        data = json.load(f)

    benchmark = data.get("benchmark", "unknown")

    typer.echo(f"Exporting validation samples from: {results}")
    typer.echo(f"  Benchmark: {benchmark}")
    typer.echo(f"  Samples per benchmark: {samples}")
    typer.echo(f"  Stratified: {not no_stratify}")
    typer.echo()

    exporter = ValidationExporter(
        samples_per_benchmark=samples,
        stratify_by_category=not no_stratify,
        seed=seed,
    )

    output_path = Path(output_dir)

    # Determine benchmark type and extract samples
    if benchmark == "longmemeval":
        summary = exporter.export_combined(
            longmemeval_results=data,
            locomo_results=None,
            output_dir=output_path,
        )
    elif benchmark == "locomo":
        summary = exporter.export_combined(
            longmemeval_results=None,
            locomo_results=data,
            output_dir=output_path,
        )
    else:
        typer.echo(f"Warning: Unknown benchmark '{benchmark}', treating as generic")
        summary = exporter.export_combined(
            longmemeval_results=data,
            locomo_results=None,
            output_dir=output_path,
        )

    typer.echo(f"Exported {summary['total_samples']} samples")

    if summary.get("by_benchmark"):
        for bench, categories in summary["by_benchmark"].items():
            typer.echo(f"\n  {bench}:")
            for category, count in sorted(categories.items()):
                typer.echo(f"    {category}: {count}")

    typer.echo()
    typer.echo("Output files:")
    for f in summary.get("output_files", []):
        typer.echo(f"  {f}")


@app.command("export-samples-combined")
def export_samples_combined(
    longmemeval_results: Annotated[
        Path | None,
        typer.Option(
            "--longmemeval",
            "-l",
            help="Path to LongMemEval experiment results JSON",
        ),
    ] = None,
    locomo_results: Annotated[
        Path | None,
        typer.Option(
            "--locomo",
            "-c",
            help="Path to LoCoMo experiment results JSON",
        ),
    ] = None,
    output_dir: Annotated[
        str,
        typer.Option(
            "--output",
            "-o",
            help="Directory to save validation samples",
        ),
    ] = "results/validation",
    samples: Annotated[
        int,
        typer.Option(
            "--samples",
            "-n",
            help="Number of samples per benchmark (default: 100)",
        ),
    ] = 100,
    seed: Annotated[
        int,
        typer.Option(
            "--seed",
            "-s",
            help="Random seed for reproducible sampling",
        ),
    ] = 42,
) -> None:
    """Export combined validation samples from multiple benchmark results.

    This command combines samples from both LongMemEval and LoCoMo experiments
    into a single validation set.

    Example:
        benchmark export-samples-combined \\
            --longmemeval results/lme_exp.json \\
            --locomo results/loco_exp.json \\
            -o validation/
    """
    if not longmemeval_results and not locomo_results:
        typer.echo("Error: Must provide at least one results file")
        raise typer.Exit(1)

    lme_data = None
    loco_data = None

    if longmemeval_results:
        if not longmemeval_results.exists():
            typer.echo(f"Error: File not found: {longmemeval_results}")
            raise typer.Exit(1)
        with longmemeval_results.open() as f:
            lme_data = json.load(f)

    if locomo_results:
        if not locomo_results.exists():
            typer.echo(f"Error: File not found: {locomo_results}")
            raise typer.Exit(1)
        with locomo_results.open() as f:
            loco_data = json.load(f)

    typer.echo("Exporting combined validation samples")
    if longmemeval_results:
        typer.echo(f"  LongMemEval: {longmemeval_results}")
    if locomo_results:
        typer.echo(f"  LoCoMo: {locomo_results}")
    typer.echo(f"  Samples per benchmark: {samples}")
    typer.echo()

    exporter = ValidationExporter(
        samples_per_benchmark=samples,
        seed=seed,
    )

    output_path = Path(output_dir)
    summary = exporter.export_combined(
        longmemeval_results=lme_data,
        locomo_results=loco_data,
        output_dir=output_path,
    )

    typer.echo(f"Exported {summary['total_samples']} total samples")

    if summary.get("by_benchmark"):
        for bench, categories in summary["by_benchmark"].items():
            typer.echo(f"\n  {bench}:")
            for category, count in sorted(categories.items()):
                typer.echo(f"    {category}: {count}")

    typer.echo()
    typer.echo("Output files:")
    for f in summary.get("output_files", []):
        typer.echo(f"  {f}")


@app.command()
def report(
    results: Annotated[
        Path,
        typer.Argument(
            help="Path to experiment results JSON file",
        ),
    ],
    output_dir: Annotated[
        str,
        typer.Option(
            "--output",
            "-o",
            help="Directory to save reports",
        ),
    ] = "results/reports",
    confidence: Annotated[
        float,
        typer.Option(
            "--confidence",
            "-c",
            help="Confidence level for intervals (default: 0.95)",
        ),
    ] = 0.95,
    bootstrap: Annotated[
        int,
        typer.Option(
            "--bootstrap",
            "-b",
            help="Number of bootstrap iterations (default: 2000)",
        ),
    ] = 2000,
    markdown_only: Annotated[
        bool,
        typer.Option(
            "--markdown-only",
            "-m",
            help="Only output markdown summary (no JSON)",
        ),
    ] = False,
    stdout: Annotated[
        bool,
        typer.Option(
            "--stdout",
            help="Print markdown report to stdout instead of file",
        ),
    ] = False,
) -> None:
    """Generate a summary report from experiment results.

    This command analyzes experiment results and generates:
    - Summary statistics per condition
    - Confidence intervals for accuracy
    - Statistical comparisons between conditions
    - Formatted markdown report

    Example:
        benchmark report results/exp_001.json -o reports/
        benchmark report results/exp_001.json --stdout
    """
    if not results.exists():
        typer.echo(f"Error: File not found: {results}")
        raise typer.Exit(1)

    with results.open() as f:
        data = json.load(f)

    typer.echo(f"Generating report from: {results}")
    typer.echo(f"  Confidence level: {confidence:.0%}")
    typer.echo(f"  Bootstrap iterations: {bootstrap}")
    typer.echo()

    reporter = ResultsReporter(
        confidence_level=confidence,
        n_bootstrap=bootstrap,
    )

    report_obj = reporter.generate_report(data)

    if stdout:
        typer.echo(reporter.format_summary_table(report_obj))
    else:
        output_path = Path(output_dir)
        outputs = reporter.export_report(report_obj, output_path)

        typer.echo("Report generated:")
        for fmt, path in outputs.items():
            if markdown_only and fmt != "markdown":
                continue
            typer.echo(f"  {fmt}: {path}")


@app.command("report-combined")
def report_combined(
    results_dir: Annotated[
        Path,
        typer.Argument(
            help="Directory containing experiment result JSON files",
        ),
    ],
    output_dir: Annotated[
        str,
        typer.Option(
            "--output",
            "-o",
            help="Directory to save reports",
        ),
    ] = "results/reports",
    pattern: Annotated[
        str,
        typer.Option(
            "--pattern",
            "-p",
            help="Glob pattern for result files (default: *.json)",
        ),
    ] = "*.json",
) -> None:
    """Generate a combined report from multiple experiment results.

    This command finds all experiment result files in a directory and generates
    a combined summary report comparing across experiments.

    Example:
        benchmark report-combined results/ -o reports/
        benchmark report-combined results/ -p "longmemeval_*.json"
    """
    if not results_dir.exists():
        typer.echo(f"Error: Directory not found: {results_dir}")
        raise typer.Exit(1)

    result_files = list(results_dir.glob(pattern))
    if not result_files:
        typer.echo(f"Error: No files matching '{pattern}' in {results_dir}")
        raise typer.Exit(1)

    typer.echo(f"Found {len(result_files)} result files in {results_dir}")

    # Load all results
    all_results: list[dict[str, Any]] = []
    for rf in result_files:
        try:
            with rf.open() as f:
                data = json.load(f)
                # Validate it looks like experiment results
                if "experiment_id" in data and "benchmark" in data:
                    all_results.append(data)
                    typer.echo(f"  Loaded: {rf.name}")
                else:
                    typer.echo(f"  Skipped (not experiment results): {rf.name}")
        except json.JSONDecodeError:
            typer.echo(f"  Skipped (invalid JSON): {rf.name}")

    if not all_results:
        typer.echo("Error: No valid experiment results found")
        raise typer.Exit(1)

    typer.echo()

    reporter = ResultsReporter()
    output_path = Path(output_dir)
    summary = reporter.generate_combined_report(all_results, output_path)

    typer.echo(f"Generated {summary['reports_generated']} individual reports")
    typer.echo(f"Combined summary: {summary['combined_summary']}")
    typer.echo()
    typer.echo("Individual reports:")
    for exp_id, outputs in summary.get("output_files", {}).items():
        typer.echo(f"  {exp_id}:")
        for fmt, path in outputs.items():
            typer.echo(f"    {fmt}: {path}")


# =============================================================================
# Publication Commands (Tasks 2.4.1, 2.4.2, 2.4.3, 3.3.1, 3.3.2, 3.3.3)
# =============================================================================


def _load_results_files(
    results_dir: Path,
    pattern: str = "*.json",
) -> tuple[PublicationStatistics, list[Path]]:
    """Load all results files into PublicationStatistics.

    Args:
        results_dir: Directory containing result files
        pattern: Glob pattern for result files

    Returns:
        Tuple of (PublicationStatistics, list of loaded files)
    """
    stats = PublicationStatistics()
    loaded_files: list[Path] = []

    for result_file in sorted(results_dir.glob(pattern)):
        try:
            with result_file.open() as f:
                data = json.load(f)

            # Determine benchmark and adapter from file content
            benchmark = data.get("benchmark", result_file.stem.split("_")[0])
            adapter = data.get("adapter", "unknown")

            # If experiment results format, extract per-adapter summaries
            if "trials" in data:
                for adapter_name, trials in data["trials"].items():
                    successful = [t for t in trials if t.get("success", True)]
                    if successful:
                        stats.load_results_file(
                            filepath=result_file,
                            benchmark_name=benchmark,
                            adapter_name=adapter_name,
                        )
                        loaded_files.append(result_file)
            else:
                stats.load_results_file(
                    filepath=result_file,
                    benchmark_name=benchmark,
                    adapter_name=adapter,
                )
                loaded_files.append(result_file)

        except (json.JSONDecodeError, KeyError) as e:
            typer.echo(f"  Skipped {result_file.name}: {e}", err=True)

    return stats, loaded_files


@publication_app.command("stats")
def publication_stats(
    results_dir: Annotated[
        Path,
        typer.Argument(
            help="Directory containing experiment result JSON files",
        ),
    ],
    output_dir: Annotated[
        str,
        typer.Option(
            "--output",
            "-o",
            help="Directory to save statistics output",
        ),
    ] = "results/publication",
    pattern: Annotated[
        str,
        typer.Option(
            "--pattern",
            "-p",
            help="Glob pattern for result files",
        ),
    ] = "*.json",
    compare_adapters: Annotated[
        bool,
        typer.Option(
            "--compare",
            "-c",
            help="Run statistical comparison between adapters",
        ),
    ] = True,
) -> None:
    """Compute final statistics across all benchmarks (Tasks 2.4.1, 3.3.1).

    Aggregates results from all benchmark experiments and computes:
    - Per-adapter accuracy with 95% bootstrap confidence intervals
    - Statistical comparisons between git-notes and no-memory conditions
    - Holm-Bonferroni corrected p-values for multiple comparisons
    - Effect sizes (Cohen's d) for each benchmark

    Example:
        benchmark publication stats results/ -o publication/
        benchmark publication stats results/ --compare --pattern "locomo*.json"
    """
    if not results_dir.exists():
        typer.echo(f"Error: Directory not found: {results_dir}")
        raise typer.Exit(1)

    typer.echo(f"Loading results from: {results_dir}")

    stats, loaded_files = _load_results_files(results_dir, pattern)

    if not stats.summaries:
        typer.echo("Error: No valid results found")
        raise typer.Exit(1)

    typer.echo(f"  Loaded {len(loaded_files)} result files")
    typer.echo(f"  Found {len(stats.summaries)} benchmark summaries")
    typer.echo()

    # Display summary
    typer.echo("=" * 60)
    typer.echo("BENCHMARK SUMMARIES")
    typer.echo("=" * 60)

    adapters = sorted({s.adapter_name for s in stats.summaries})

    for adapter in adapters:
        typer.echo(f"\n{adapter}:")
        for s in stats.summaries:
            if s.adapter_name == adapter:
                ci_str = ""
                if s.metrics.accuracy_ci[0] > 0 or s.metrics.accuracy_ci[1] > 0:
                    ci_str = f" [{s.metrics.accuracy_ci[0]:.1%}, {s.metrics.accuracy_ci[1]:.1%}]"
                typer.echo(f"  {s.benchmark_name}: {s.metrics.accuracy:.1%}{ci_str}")

    # Adapter comparison
    if compare_adapters and len(adapters) >= 2:
        typer.echo()
        typer.echo("=" * 60)
        typer.echo("ADAPTER COMPARISON")
        typer.echo("=" * 60)

        # Compare first two adapters (typically git-notes vs no-memory)
        adapter_a, adapter_b = adapters[0], adapters[1]
        comparison = stats.compare_adapters(adapter_a, adapter_b)

        if comparison.get("valid"):
            typer.echo(f"\n{adapter_a} vs {adapter_b}:")
            typer.echo(f"  Mean A: {comparison['mean_a']:.1%}")
            typer.echo(f"  Mean B: {comparison['mean_b']:.1%}")
            typer.echo(f"  Difference: {comparison['mean_diff']:+.1%}")
            typer.echo(f"  p-value: {comparison['p_value']:.4f}")
            typer.echo(f"  Effect size (Cohen's d): {comparison['effect_size']:.3f}")
            typer.echo(f"  Significant (p < 0.05): {'Yes' if comparison['significant'] else 'No'}")

    # Export statistics
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    stats.export_json(output_path / "publication_statistics.json")

    typer.echo()
    typer.echo(f"Statistics exported to: {output_path / 'publication_statistics.json'}")


@publication_app.command("tables")
def publication_tables(
    results_dir: Annotated[
        Path,
        typer.Argument(
            help="Directory containing experiment result JSON files",
        ),
    ],
    output_dir: Annotated[
        str,
        typer.Option(
            "--output",
            "-o",
            help="Directory to save tables",
        ),
    ] = "results/publication/tables",
    formats: Annotated[
        str,
        typer.Option(
            "--formats",
            "-f",
            help="Output formats (comma-separated): latex,markdown",
        ),
    ] = "latex,markdown",
    pattern: Annotated[
        str,
        typer.Option(
            "--pattern",
            "-p",
            help="Glob pattern for result files",
        ),
    ] = "*.json",
) -> None:
    """Generate publication tables in LaTeX and Markdown (Task 3.3.2).

    Creates the following tables:
    - Main Results: Accuracy comparison across benchmarks and conditions
    - Ablation: Impact of removing individual system components
    - Category Breakdown: Accuracy per question category
    - Human Validation: Agreement metrics between annotators and LLM judge

    Example:
        benchmark publication tables results/ -o tables/
        benchmark publication tables results/ --formats latex
    """
    if not results_dir.exists():
        typer.echo(f"Error: Directory not found: {results_dir}")
        raise typer.Exit(1)

    typer.echo(f"Loading results from: {results_dir}")

    stats, loaded_files = _load_results_files(results_dir, pattern)

    if not stats.summaries:
        typer.echo("Error: No valid results found")
        raise typer.Exit(1)

    typer.echo(f"  Loaded {len(loaded_files)} result files")
    typer.echo()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    output_formats = [f.strip().lower() for f in formats.split(",")]

    # Main results table
    main_table = MainResultsTable(
        title="Main Results",
        caption="Accuracy comparison across benchmarks and memory conditions",
        label="tab:main-results",
        benchmark_order=[
            "longmemeval",
            "locomo",
            "contextbench",
            "memoryagentbench",
            "terminalbench",
        ],
    )
    main_data = stats.get_main_results_data()

    if main_data:
        typer.echo("Generating Main Results table...")
        if "latex" in output_formats:
            main_table.save_latex(main_data, output_path / "main_results.tex")
        if "markdown" in output_formats:
            main_table.save_markdown(main_data, output_path / "main_results.md")

    # Ablation table
    ablation_table = AblationTable(
        title="Ablation Study",
        caption="Impact of removing individual system components",
        label="tab:ablation",
    )
    ablation_data = stats.get_ablation_data()

    if ablation_data:
        typer.echo("Generating Ablation table...")
        if "latex" in output_formats:
            ablation_table.save_latex(ablation_data, output_path / "ablation.tex")
        if "markdown" in output_formats:
            ablation_table.save_markdown(ablation_data, output_path / "ablation.md")

    # Category breakdown table
    category_table = CategoryBreakdownTable(
        title="Category Breakdown",
        caption="Accuracy breakdown by question category",
        label="tab:category-breakdown",
    )
    category_data = stats.get_category_data()

    if category_data:
        typer.echo("Generating Category Breakdown table...")
        if "latex" in output_formats:
            category_table.save_latex(category_data, output_path / "category_breakdown.tex")
        if "markdown" in output_formats:
            category_table.save_markdown(category_data, output_path / "category_breakdown.md")

    typer.echo()
    typer.echo(f"Tables saved to: {output_path}")


@publication_app.command("figures")
def publication_figures(
    results_dir: Annotated[
        Path,
        typer.Argument(
            help="Directory containing experiment result JSON files",
        ),
    ],
    output_dir: Annotated[
        str,
        typer.Option(
            "--output",
            "-o",
            help="Directory to save figures",
        ),
    ] = "results/publication/figures",
    formats: Annotated[
        str,
        typer.Option(
            "--formats",
            "-f",
            help="Output formats (comma-separated): pdf,png,svg",
        ),
    ] = "pdf,png",
    pattern: Annotated[
        str,
        typer.Option(
            "--pattern",
            "-p",
            help="Glob pattern for result files",
        ),
    ] = "*.json",
    dpi: Annotated[
        int,
        typer.Option(
            "--dpi",
            help="Resolution for raster formats",
        ),
    ] = 300,
) -> None:
    """Generate publication figures with matplotlib (Task 3.3.3).

    Creates the following figures:
    - Performance Bar Chart: Grouped bar chart comparing adapter performance
    - Ablation Heatmap: Impact of removing each component
    - Category Radar Plot: Performance across question categories
    - Confidence Interval Plot: Accuracy with 95% CI error bars

    Requires matplotlib to be installed.

    Example:
        benchmark publication figures results/ -o figures/
        benchmark publication figures results/ --formats pdf --dpi 600
    """
    try:
        import importlib.util

        if importlib.util.find_spec("matplotlib") is None:
            raise ImportError("matplotlib not found")
    except ImportError as e:
        typer.echo("Error: matplotlib is required for figure generation")
        typer.echo("Install with: pip install matplotlib")
        raise typer.Exit(1) from e

    if not results_dir.exists():
        typer.echo(f"Error: Directory not found: {results_dir}")
        raise typer.Exit(1)

    typer.echo(f"Loading results from: {results_dir}")

    stats, loaded_files = _load_results_files(results_dir, pattern)

    if not stats.summaries:
        typer.echo("Error: No valid results found")
        raise typer.Exit(1)

    typer.echo(f"  Loaded {len(loaded_files)} result files")
    typer.echo()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    output_formats = [f.strip().lower() for f in formats.split(",")]

    # Performance bar chart
    bar_chart = PerformanceBarChart(
        title="Adapter Performance Comparison",
        dpi=dpi,
    )
    main_data = stats.get_main_results_data()

    if main_data:
        typer.echo("Generating Performance Bar Chart...")
        for fmt in output_formats:
            bar_chart.save(main_data, output_path / f"performance_comparison.{fmt}", format=fmt)

    # Ablation heatmap
    heatmap = AblationHeatmap(
        title="Ablation Study Impact",
        dpi=dpi,
    )
    ablation_data = stats.get_ablation_data()

    if ablation_data:
        typer.echo("Generating Ablation Heatmap...")
        for fmt in output_formats:
            heatmap.save(ablation_data, output_path / f"ablation_impact.{fmt}", format=fmt)

    # Category radar plot
    radar = CategoryRadarPlot(
        title="Performance by Category",
        dpi=dpi,
    )
    category_data = stats.get_category_data()

    if category_data and len(category_data) >= 3:
        typer.echo("Generating Category Radar Plot...")
        for fmt in output_formats:
            radar.save(category_data, output_path / f"category_radar.{fmt}", format=fmt)

    # Confidence interval plot
    ci_plot = ConfidenceIntervalPlot(
        title="Accuracy with 95% Confidence Intervals",
        dpi=dpi,
    )
    # Prepare CI data from summaries
    ci_data = []
    for s in stats.summaries:
        ci_data.append(
            {
                "adapter": f"{s.adapter_name} ({s.benchmark_name})",
                "accuracy": s.metrics.accuracy,
                "ci_lower": s.metrics.accuracy_ci[0]
                if s.metrics.accuracy_ci[0] > 0
                else s.metrics.accuracy,
                "ci_upper": s.metrics.accuracy_ci[1]
                if s.metrics.accuracy_ci[1] > 0
                else s.metrics.accuracy,
            }
        )

    if ci_data:
        typer.echo("Generating Confidence Interval Plot...")
        for fmt in output_formats:
            ci_plot.save(ci_data, output_path / f"confidence_intervals.{fmt}", format=fmt)

    typer.echo()
    typer.echo(f"Figures saved to: {output_path}")


@publication_app.command("analyze-cr")
def analyze_conflict_resolution(
    results_file: Annotated[
        Path,
        typer.Argument(
            help="Path to MemoryAgentBench results JSON file",
        ),
    ],
    output_dir: Annotated[
        str,
        typer.Option(
            "--output",
            "-o",
            help="Directory to save analysis",
        ),
    ] = "results/publication/conflict_resolution",
) -> None:
    """Analyze Conflict Resolution results from MemoryAgentBench (Task 2.4.2).

    Performs detailed analysis of the Conflict Resolution competency:
    - Git version history utilization analysis
    - Single-hop vs multi-hop performance comparison
    - Qualitative examples of version-aware resolution
    - Statistical comparison between memory conditions

    Example:
        benchmark publication analyze-cr results/memoryagentbench_exp.json
    """
    if not results_file.exists():
        typer.echo(f"Error: File not found: {results_file}")
        raise typer.Exit(1)

    with results_file.open() as f:
        data = json.load(f)

    typer.echo(f"Analyzing Conflict Resolution from: {results_file}")
    typer.echo()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Extract conflict resolution results
    analysis: dict[str, Any] = {
        "source_file": str(results_file),
        "benchmark": data.get("benchmark", "memoryagentbench"),
        "conditions": {},
        "comparisons": [],
    }

    trials = data.get("trials", {})
    cr_results_by_condition: dict[str, list[dict[str, Any]]] = {}

    for condition, trial_list in trials.items():
        condition_results: list[dict[str, Any]] = []

        for trial in trial_list:
            if not trial.get("success", True):
                continue

            raw_results = trial.get("raw_results", {})
            competency_results = raw_results.get("competency_results", {})

            # Find conflict resolution competency
            cr_data = competency_results.get("conflict_resolution", {})
            if cr_data:
                condition_results.append(
                    {
                        "trial_id": trial.get("trial_id"),
                        "accuracy": cr_data.get("accuracy", 0),
                        "total": cr_data.get("total_questions", 0),
                        "correct": cr_data.get("correct_count", 0),
                        "question_results": cr_data.get("question_results", []),
                    }
                )

        if condition_results:
            cr_results_by_condition[condition] = condition_results

            # Compute aggregate stats
            total_correct = sum(r["correct"] for r in condition_results)
            total_questions = sum(r["total"] for r in condition_results)
            avg_accuracy = total_correct / total_questions if total_questions > 0 else 0

            # Single vs multi-hop breakdown
            all_questions = []
            for r in condition_results:
                all_questions.extend(r.get("question_results", []))

            single_hop = [q for q in all_questions if q.get("difficulty") == "single_hop"]
            multi_hop = [q for q in all_questions if q.get("difficulty") == "multi_hop"]

            single_acc = (
                sum(1 for q in single_hop if q.get("correct")) / len(single_hop)
                if single_hop
                else None
            )
            multi_acc = (
                sum(1 for q in multi_hop if q.get("correct")) / len(multi_hop)
                if multi_hop
                else None
            )

            analysis["conditions"][condition] = {
                "trials": len(condition_results),
                "total_questions": total_questions,
                "total_correct": total_correct,
                "avg_accuracy": avg_accuracy,
                "single_hop_accuracy": single_acc,
                "multi_hop_accuracy": multi_acc,
                "single_hop_count": len(single_hop),
                "multi_hop_count": len(multi_hop),
            }

    # Display results
    typer.echo("=" * 60)
    typer.echo("CONFLICT RESOLUTION ANALYSIS")
    typer.echo("=" * 60)

    for condition, metrics in analysis["conditions"].items():
        typer.echo(f"\n{condition}:")
        typer.echo(f"  Overall Accuracy: {metrics['avg_accuracy']:.1%}")
        typer.echo(f"  Total Questions: {metrics['total_questions']}")
        if metrics["single_hop_accuracy"] is not None:
            typer.echo(
                f"  Single-hop ({metrics['single_hop_count']}): {metrics['single_hop_accuracy']:.1%}"
            )
        if metrics["multi_hop_accuracy"] is not None:
            typer.echo(
                f"  Multi-hop ({metrics['multi_hop_count']}): {metrics['multi_hop_accuracy']:.1%}"
            )

    # Compare conditions if we have both
    conditions = list(analysis["conditions"].keys())
    if len(conditions) >= 2:
        typer.echo()
        typer.echo("=" * 60)
        typer.echo("CONDITION COMPARISON")
        typer.echo("=" * 60)

        cond_a, cond_b = conditions[0], conditions[1]
        acc_a = analysis["conditions"][cond_a]["avg_accuracy"]
        acc_b = analysis["conditions"][cond_b]["avg_accuracy"]
        diff = acc_a - acc_b

        typer.echo(f"\n{cond_a} vs {cond_b}:")
        typer.echo(f"  {cond_a}: {acc_a:.1%}")
        typer.echo(f"  {cond_b}: {acc_b:.1%}")
        typer.echo(f"  Difference: {diff:+.1%}")

        analysis["comparisons"].append(
            {
                "condition_a": cond_a,
                "condition_b": cond_b,
                "accuracy_a": acc_a,
                "accuracy_b": acc_b,
                "difference": diff,
            }
        )

    # Export analysis
    output_file = output_path / "conflict_resolution_analysis.json"
    with output_file.open("w") as f:
        json.dump(analysis, f, indent=2)

    typer.echo()
    typer.echo(f"Analysis exported to: {output_file}")


@app.command("export-phase2-samples")
def export_phase2_samples(
    contextbench_results: Annotated[
        Path | None,
        typer.Option(
            "--contextbench",
            "-c",
            help="Path to Context-Bench experiment results JSON",
        ),
    ] = None,
    memoryagentbench_results: Annotated[
        Path | None,
        typer.Option(
            "--memoryagentbench",
            "-m",
            help="Path to MemoryAgentBench experiment results JSON",
        ),
    ] = None,
    output_dir: Annotated[
        str,
        typer.Option(
            "--output",
            "-o",
            help="Directory to save validation samples",
        ),
    ] = "results/validation/phase2",
    samples: Annotated[
        int,
        typer.Option(
            "--samples",
            "-n",
            help="Number of samples per benchmark (default: 100)",
        ),
    ] = 100,
    seed: Annotated[
        int,
        typer.Option(
            "--seed",
            "-s",
            help="Random seed for reproducible sampling",
        ),
    ] = 42,
    prioritize_cr: Annotated[
        bool,
        typer.Option(
            "--prioritize-cr",
            help="Prioritize Conflict Resolution samples from MemoryAgentBench",
        ),
    ] = True,
) -> None:
    """Export Phase 2 validation samples from Context-Bench and MemoryAgentBench (Task 2.4.3).

    Extracts 100 samples from each Phase 2 benchmark for human validation.
    When --prioritize-cr is set, MemoryAgentBench sampling weights Conflict
    Resolution cases more heavily.

    Example:
        benchmark export-phase2-samples \\
            --contextbench results/contextbench_exp.json \\
            --memoryagentbench results/mab_exp.json \\
            -o validation/phase2/
    """
    import csv
    import random

    if not contextbench_results and not memoryagentbench_results:
        typer.echo("Error: Must provide at least one results file")
        raise typer.Exit(1)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    all_samples: list[dict[str, Any]] = []

    # Extract Context-Bench samples
    if contextbench_results and contextbench_results.exists():
        typer.echo(f"Loading Context-Bench results from: {contextbench_results}")

        with contextbench_results.open() as f:
            cb_data = json.load(f)

        cb_samples: list[dict[str, Any]] = []
        trials = cb_data.get("trials", {})

        for condition, trial_list in trials.items():
            for trial in trial_list:
                if not trial.get("success", True):
                    continue

                raw_results = trial.get("raw_results", {})
                question_results = raw_results.get("question_results", [])

                for qr in question_results:
                    cb_samples.append(
                        {
                            "sample_id": f"cb_{condition}_{qr.get('question_id', len(cb_samples))}",
                            "benchmark": "contextbench",
                            "category": qr.get("category", "unknown"),
                            "condition": condition,
                            "question": qr.get("question", ""),
                            "reference_answer": qr.get("reference_answer", ""),
                            "model_response": qr.get("model_answer", qr.get("predicted", "")),
                            "llm_judgment": qr.get("judgment_text", ""),
                            "llm_score": 1.0 if qr.get("correct") else 0.0,
                            "hop_count": qr.get("hop_count", 1),
                        }
                    )

        # Sample
        if len(cb_samples) > samples:
            cb_samples = rng.sample(cb_samples, samples)

        all_samples.extend(cb_samples)
        typer.echo(f"  Extracted {len(cb_samples)} Context-Bench samples")

    # Extract MemoryAgentBench samples
    if memoryagentbench_results and memoryagentbench_results.exists():
        typer.echo(f"Loading MemoryAgentBench results from: {memoryagentbench_results}")

        with memoryagentbench_results.open() as f:
            mab_data = json.load(f)

        mab_samples: list[dict[str, Any]] = []
        cr_samples: list[dict[str, Any]] = []
        trials = mab_data.get("trials", {})

        for condition, trial_list in trials.items():
            for trial in trial_list:
                if not trial.get("success", True):
                    continue

                raw_results = trial.get("raw_results", {})
                competency_results = raw_results.get("competency_results", {})

                for competency, comp_data in competency_results.items():
                    question_results = comp_data.get("question_results", [])

                    for qr in question_results:
                        sample = {
                            "sample_id": f"mab_{condition}_{competency}_{qr.get('question_id', len(mab_samples))}",
                            "benchmark": "memoryagentbench",
                            "category": competency,
                            "condition": condition,
                            "question": qr.get("question", ""),
                            "reference_answer": qr.get("reference_answer", ""),
                            "model_response": qr.get("predicted", ""),
                            "llm_judgment": qr.get("judgment_text", ""),
                            "llm_score": 1.0 if qr.get("correct") else 0.0,
                            "difficulty": qr.get("difficulty", "unknown"),
                        }

                        if competency == "conflict_resolution":
                            cr_samples.append(sample)
                        else:
                            mab_samples.append(sample)

        # Prioritize CR if requested
        if prioritize_cr and cr_samples:
            # Take up to 50% from CR, rest from other competencies
            cr_count = min(len(cr_samples), samples // 2)
            other_count = samples - cr_count

            selected_cr = (
                rng.sample(cr_samples, cr_count) if len(cr_samples) > cr_count else cr_samples
            )
            selected_other = rng.sample(mab_samples, min(other_count, len(mab_samples)))

            mab_selected = selected_cr + selected_other
        else:
            all_mab = mab_samples + cr_samples
            mab_selected = rng.sample(all_mab, min(samples, len(all_mab)))

        all_samples.extend(mab_selected)
        typer.echo(f"  Extracted {len(mab_selected)} MemoryAgentBench samples")
        if prioritize_cr:
            cr_in_selected = sum(1 for s in mab_selected if s["category"] == "conflict_resolution")
            typer.echo(f"    (including {cr_in_selected} Conflict Resolution samples)")

    # Export to JSON
    json_path = output_path / "phase2_validation_samples.json"
    with json_path.open("w") as f:
        json.dump(
            {
                "export_metadata": {
                    "total_samples": len(all_samples),
                    "samples_per_benchmark": samples,
                    "prioritize_cr": prioritize_cr,
                    "seed": seed,
                },
                "samples": all_samples,
            },
            f,
            indent=2,
        )

    # Export to CSV
    csv_path = output_path / "phase2_validation_samples.csv"
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
        "human_score",
        "human_notes",
    ]

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for sample in all_samples:
            sample["human_score"] = ""
            sample["human_notes"] = ""
            writer.writerow(sample)

    typer.echo()
    typer.echo(f"Exported {len(all_samples)} total samples:")
    typer.echo(f"  JSON: {json_path}")
    typer.echo(f"  CSV: {csv_path}")


@publication_app.command("all")
def publication_all(
    results_dir: Annotated[
        Path,
        typer.Argument(
            help="Directory containing experiment result JSON files",
        ),
    ],
    output_dir: Annotated[
        str,
        typer.Option(
            "--output",
            "-o",
            help="Base directory for all publication outputs",
        ),
    ] = "results/publication",
    pattern: Annotated[
        str,
        typer.Option(
            "--pattern",
            "-p",
            help="Glob pattern for result files",
        ),
    ] = "*.json",
) -> None:
    """Generate all publication artifacts at once.

    Runs stats, tables, and figures commands in sequence, outputting
    to organized subdirectories.

    Example:
        benchmark publication all results/ -o publication/
    """
    output_base = Path(output_dir)

    typer.echo("=" * 60)
    typer.echo("GENERATING ALL PUBLICATION ARTIFACTS")
    typer.echo("=" * 60)

    # Stats
    typer.echo("\n[1/3] Computing statistics...")
    publication_stats(
        results_dir=results_dir,
        output_dir=str(output_base),
        pattern=pattern,
        compare_adapters=True,
    )

    # Tables
    typer.echo("\n[2/3] Generating tables...")
    publication_tables(
        results_dir=results_dir,
        output_dir=str(output_base / "tables"),
        formats="latex,markdown",
        pattern=pattern,
    )

    # Figures
    typer.echo("\n[3/3] Generating figures...")
    try:
        publication_figures(
            results_dir=results_dir,
            output_dir=str(output_base / "figures"),
            formats="pdf,png",
            pattern=pattern,
            dpi=300,
        )
    except typer.Exit:
        typer.echo("  Skipped (matplotlib not available)")

    typer.echo()
    typer.echo("=" * 60)
    typer.echo("ALL PUBLICATION ARTIFACTS GENERATED")
    typer.echo("=" * 60)
    typer.echo(f"Output directory: {output_base}")


if __name__ == "__main__":
    app()
