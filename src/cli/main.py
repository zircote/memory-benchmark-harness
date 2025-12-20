"""CLI entry point for memory benchmark harness."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated

import typer

from src.experiments.runner import AdapterCondition, ExperimentConfig, ExperimentRunner
from src.reporting import ResultsReporter, ValidationExporter

app = typer.Typer(
    name="benchmark",
    help="Benchmark harness for git-native semantic memory validation.",
    no_args_is_help=True,
)


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
    results = asyncio.run(runner.run())

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
        if data_a["benchmark"] == "longmemeval":
            metric_key = "accuracy"
        else:
            metric_key = "overall_accuracy"

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
    all_results: list[dict] = []
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


if __name__ == "__main__":
    app()
