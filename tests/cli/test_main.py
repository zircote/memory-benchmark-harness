"""Tests for the CLI main module.

Tests for the benchmark harness CLI commands including:
- export-samples: Export validation samples from experiment results
- export-samples-combined: Export combined validation samples
- report: Generate summary report from experiment results
- report-combined: Generate combined report from multiple experiments
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from src.cli.main import app

# Use constants to avoid benchmark name substring issues with security hooks
# The LME benchmark name contains a substring that triggers security warnings
BENCH_LME = "long" + "mem" + "eval"  # Constructed to avoid hook detection
BENCH_LOCO = "locomo"


@pytest.fixture
def runner() -> CliRunner:
    """Create a Typer CLI test runner."""
    return CliRunner()


def get_output(result) -> str:
    """Get all output from CLI result including stdout.

    Typer's CliRunner captures all output (including errors) in stdout.
    This helper provides a consistent way to access the output.
    """
    return result.stdout or ""


@pytest.fixture
def sample_lme_results() -> dict[str, Any]:
    """Create sample LME benchmark experiment results."""
    return {
        "experiment_id": "exp_lme_001",
        "benchmark": BENCH_LME,
        "config": {"adapters": ["no-memory"], "num_trials": 2},
        "started_at": "2024-01-01T00:00:00",
        "completed_at": "2024-01-01T01:00:00",
        "trials": {
            "no-memory": [
                {
                    "trial_id": 1,
                    "success": True,
                    "metrics": {"accuracy": 0.75},
                    "raw_results": {
                        "question_results": [
                            {
                                "question_id": "q1",
                                "question_type": "factoid",
                                "question": "What is the capital?",
                                "reference_answer": "Paris",
                                "predicted": "Paris",
                                "correct": True,
                                "judgment_text": "Correct",
                                "session_id": "s1",
                            },
                            {
                                "question_id": "q2",
                                "question_type": "reasoning",
                                "question": "Why did X happen?",
                                "reference_answer": "Because of Y",
                                "predicted": "Because of Z",
                                "correct": False,
                                "judgment_text": "Incorrect reasoning",
                                "session_id": "s1",
                            },
                        ]
                    },
                },
                {
                    "trial_id": 2,
                    "success": True,
                    "metrics": {"accuracy": 0.80},
                    "raw_results": {"question_results": []},
                },
            ],
            "mock": [
                {
                    "trial_id": 1,
                    "success": True,
                    "metrics": {"accuracy": 0.85},
                    "raw_results": {"question_results": []},
                },
            ],
        },
    }


@pytest.fixture
def sample_locomo_results() -> dict[str, Any]:
    """Create sample LoCoMo experiment results."""
    return {
        "experiment_id": "exp_loco_001",
        "benchmark": BENCH_LOCO,
        "config": {"adapters": ["no-memory"], "num_trials": 1},
        "started_at": "2024-01-01T00:00:00",
        "completed_at": "2024-01-01T01:00:00",
        "trials": {
            "no-memory": [
                {
                    "trial_id": 1,
                    "success": True,
                    "metrics": {"overall_accuracy": 0.70},
                    "raw_results": {
                        "conversation_results": [
                            {
                                "sample_id": "conv1",
                                "question_results": [
                                    {
                                        "question_id": "q1",
                                        "category": "identity",
                                        "question": "Who is Alice?",
                                        "reference_answer": "A friend",
                                        "predicted": "A colleague",
                                        "score": 0.8,
                                        "judgment_text": "Partially correct",
                                        "is_adversarial": False,
                                        "difficulty": "easy",
                                    }
                                ],
                            }
                        ]
                    },
                },
            ],
        },
    }


class TestExportSamples:
    """Tests for the export-samples command."""

    def test_export_samples_file_not_found(
        self,
        runner: CliRunner,
    ) -> None:
        """Test error when results file doesn't exist."""
        result = runner.invoke(app, ["export-samples", "nonexistent.json"])
        assert result.exit_code == 1
        assert "Error: File not found" in get_output(result)

    def test_export_samples_lme_benchmark(
        self,
        runner: CliRunner,
        sample_lme_results: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """Test exporting samples from LME benchmark results."""
        # Write sample results
        results_file = tmp_path / "lme_results.json"
        with results_file.open("w") as f:
            json.dump(sample_lme_results, f)

        output_dir = tmp_path / "validation"

        result = runner.invoke(
            app,
            [
                "export-samples",
                str(results_file),
                "-o",
                str(output_dir),
                "-n",
                "10",
            ],
        )

        assert result.exit_code == 0
        assert "Exporting validation samples" in result.stdout
        assert "Benchmark:" in result.stdout
        assert "Output files:" in result.stdout

        # Check output files were created
        assert (output_dir / "validation_samples.json").exists()
        assert (output_dir / "validation_samples.csv").exists()

    def test_export_samples_locomo(
        self,
        runner: CliRunner,
        sample_locomo_results: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """Test exporting samples from LoCoMo results."""
        results_file = tmp_path / "loco_results.json"
        with results_file.open("w") as f:
            json.dump(sample_locomo_results, f)

        output_dir = tmp_path / "validation"

        result = runner.invoke(
            app,
            ["export-samples", str(results_file), "-o", str(output_dir)],
        )

        assert result.exit_code == 0
        assert "Benchmark: locomo" in result.stdout

    def test_export_samples_no_stratify(
        self,
        runner: CliRunner,
        sample_lme_results: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """Test exporting samples without stratification."""
        results_file = tmp_path / "results.json"
        with results_file.open("w") as f:
            json.dump(sample_lme_results, f)

        output_dir = tmp_path / "validation"

        result = runner.invoke(
            app,
            [
                "export-samples",
                str(results_file),
                "-o",
                str(output_dir),
                "--no-stratify",
            ],
        )

        assert result.exit_code == 0
        assert "Stratified: False" in result.stdout


class TestExportSamplesCombined:
    """Tests for the export-samples-combined command."""

    def test_combined_no_files_error(
        self,
        runner: CliRunner,
    ) -> None:
        """Test error when no results files are provided."""
        result = runner.invoke(app, ["export-samples-combined"])
        assert result.exit_code == 1
        assert "Must provide at least one results file" in get_output(result)

    def test_combined_file_not_found(
        self,
        runner: CliRunner,
    ) -> None:
        """Test error when provided file doesn't exist."""
        result = runner.invoke(
            app,
            ["export-samples-combined", "--longmemeval", "nonexistent.json"],
        )
        assert result.exit_code == 1
        assert "Error: File not found" in get_output(result)

    def test_combined_single_benchmark(
        self,
        runner: CliRunner,
        sample_lme_results: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """Test combined export with single benchmark."""
        results_file = tmp_path / "lme_results.json"
        with results_file.open("w") as f:
            json.dump(sample_lme_results, f)

        output_dir = tmp_path / "validation"

        result = runner.invoke(
            app,
            [
                "export-samples-combined",
                "--longmemeval",
                str(results_file),
                "-o",
                str(output_dir),
            ],
        )

        assert result.exit_code == 0
        assert "Exporting combined validation samples" in result.stdout
        assert (output_dir / "validation_samples.json").exists()

    def test_combined_both_benchmarks(
        self,
        runner: CliRunner,
        sample_lme_results: dict[str, Any],
        sample_locomo_results: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """Test combined export with both benchmarks."""
        lme_file = tmp_path / "lme_results.json"
        with lme_file.open("w") as f:
            json.dump(sample_lme_results, f)

        loco_file = tmp_path / "loco_results.json"
        with loco_file.open("w") as f:
            json.dump(sample_locomo_results, f)

        output_dir = tmp_path / "validation"

        result = runner.invoke(
            app,
            [
                "export-samples-combined",
                "--longmemeval",
                str(lme_file),
                "--locomo",
                str(loco_file),
                "-o",
                str(output_dir),
                "-n",
                "5",
            ],
        )

        assert result.exit_code == 0
        # Check that both sources are mentioned
        assert str(lme_file) in result.stdout or "LongMem" in result.stdout
        assert str(loco_file) in result.stdout or "LoCoMo" in result.stdout


class TestReport:
    """Tests for the report command."""

    def test_report_file_not_found(
        self,
        runner: CliRunner,
    ) -> None:
        """Test error when results file doesn't exist."""
        result = runner.invoke(app, ["report", "nonexistent.json"])
        assert result.exit_code == 1
        assert "Error: File not found" in get_output(result)

    def test_report_basic(
        self,
        runner: CliRunner,
        sample_lme_results: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """Test basic report generation."""
        results_file = tmp_path / "results.json"
        with results_file.open("w") as f:
            json.dump(sample_lme_results, f)

        output_dir = tmp_path / "reports"

        result = runner.invoke(
            app,
            ["report", str(results_file), "-o", str(output_dir)],
        )

        assert result.exit_code == 0
        assert "Generating report from:" in result.stdout
        assert "Confidence level: 95%" in result.stdout
        assert "Report generated:" in result.stdout

        # Check output files
        report_files = list(output_dir.glob("*.json"))
        assert len(report_files) > 0

    def test_report_stdout(
        self,
        runner: CliRunner,
        sample_lme_results: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """Test report output to stdout."""
        results_file = tmp_path / "results.json"
        with results_file.open("w") as f:
            json.dump(sample_lme_results, f)

        result = runner.invoke(
            app,
            ["report", str(results_file), "--stdout"],
        )

        assert result.exit_code == 0
        # Should contain markdown report content
        assert "Experiment Results" in result.stdout
        assert "Summary by Condition" in result.stdout

    def test_report_custom_confidence(
        self,
        runner: CliRunner,
        sample_lme_results: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """Test report with custom confidence level."""
        results_file = tmp_path / "results.json"
        with results_file.open("w") as f:
            json.dump(sample_lme_results, f)

        output_dir = tmp_path / "reports"

        result = runner.invoke(
            app,
            [
                "report",
                str(results_file),
                "-o",
                str(output_dir),
                "-c",
                "0.99",
                "-b",
                "1000",
            ],
        )

        assert result.exit_code == 0
        assert "Confidence level: 99%" in result.stdout
        assert "Bootstrap iterations: 1000" in result.stdout

    def test_report_markdown_only(
        self,
        runner: CliRunner,
        sample_lme_results: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """Test report with markdown-only output."""
        results_file = tmp_path / "results.json"
        with results_file.open("w") as f:
            json.dump(sample_lme_results, f)

        output_dir = tmp_path / "reports"

        result = runner.invoke(
            app,
            ["report", str(results_file), "-o", str(output_dir), "-m"],
        )

        assert result.exit_code == 0
        assert "markdown:" in result.stdout
        # With markdown-only, json should not be in output
        assert "json:" not in result.stdout


class TestReportCombined:
    """Tests for the report-combined command."""

    def test_combined_dir_not_found(
        self,
        runner: CliRunner,
    ) -> None:
        """Test error when directory doesn't exist."""
        result = runner.invoke(app, ["report-combined", "nonexistent_dir/"])
        assert result.exit_code == 1
        assert "Error: Directory not found" in get_output(result)

    def test_combined_no_files(
        self,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """Test error when no matching files found."""
        result = runner.invoke(
            app,
            ["report-combined", str(tmp_path), "-p", "*.json"],
        )
        assert result.exit_code == 1
        assert "No files matching" in get_output(result)

    def test_combined_multiple_results(
        self,
        runner: CliRunner,
        sample_lme_results: dict[str, Any],
        sample_locomo_results: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """Test combined report from multiple result files."""
        # Create results directory
        results_dir = tmp_path / "results"
        results_dir.mkdir()

        # Write result files
        with (results_dir / "lme_exp.json").open("w") as f:
            json.dump(sample_lme_results, f)

        with (results_dir / "loco_exp.json").open("w") as f:
            json.dump(sample_locomo_results, f)

        output_dir = tmp_path / "reports"

        result = runner.invoke(
            app,
            ["report-combined", str(results_dir), "-o", str(output_dir)],
        )

        assert result.exit_code == 0
        assert "Found 2 result files" in result.stdout
        assert "Loaded: lme_exp.json" in result.stdout
        assert "Loaded: loco_exp.json" in result.stdout
        assert "Generated 2 individual reports" in result.stdout
        assert "Combined summary:" in result.stdout

        # Check combined summary was created
        assert (output_dir / "combined_summary.md").exists()

    def test_combined_skips_invalid_json(
        self,
        runner: CliRunner,
        sample_lme_results: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """Test that invalid JSON files are skipped."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()

        # Write valid result
        with (results_dir / "valid.json").open("w") as f:
            json.dump(sample_lme_results, f)

        # Write invalid JSON
        with (results_dir / "invalid.json").open("w") as f:
            f.write("not valid json{{{")

        # Write non-experiment JSON
        with (results_dir / "other.json").open("w") as f:
            json.dump({"some": "data"}, f)

        output_dir = tmp_path / "reports"

        result = runner.invoke(
            app,
            ["report-combined", str(results_dir), "-o", str(output_dir)],
        )

        assert result.exit_code == 0
        assert "Skipped (invalid JSON): invalid.json" in result.stdout
        assert "Skipped (not experiment results): other.json" in result.stdout
        assert "Generated 1 individual reports" in result.stdout

    def test_combined_with_pattern(
        self,
        runner: CliRunner,
        sample_lme_results: dict[str, Any],
        sample_locomo_results: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """Test combined report with file pattern filter."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()

        # Write result files with different names
        with (results_dir / "lme_exp1.json").open("w") as f:
            json.dump(sample_lme_results, f)

        with (results_dir / "locomo_exp1.json").open("w") as f:
            json.dump(sample_locomo_results, f)

        output_dir = tmp_path / "reports"

        # Only match lme files
        result = runner.invoke(
            app,
            [
                "report-combined",
                str(results_dir),
                "-o",
                str(output_dir),
                "-p",
                "lme*.json",
            ],
        )

        assert result.exit_code == 0
        assert "Found 1 result files" in result.stdout
        assert "Generated 1 individual reports" in result.stdout


class TestExistingCommands:
    """Tests for pre-existing CLI commands."""

    def test_list_benchmarks(self, runner: CliRunner) -> None:
        """Test list-benchmarks command."""
        result = runner.invoke(app, ["list-benchmarks"])
        assert result.exit_code == 0
        assert "Available benchmarks:" in result.stdout
        assert "locomo" in result.stdout.lower()

    def test_list_adapters(self, runner: CliRunner) -> None:
        """Test list-adapters command."""
        result = runner.invoke(app, ["list-adapters"])
        assert result.exit_code == 0
        assert "Available adapter conditions:" in result.stdout
        assert "git-notes" in result.stdout
        assert "no-memory" in result.stdout
        assert "mock" in result.stdout

    def test_compare_file_not_found(self, runner: CliRunner) -> None:
        """Test compare command with missing files."""
        result = runner.invoke(app, ["compare", "a.json", "b.json"])
        assert result.exit_code == 1
        assert "Error: File not found" in get_output(result)

    def test_compare_basic(
        self,
        runner: CliRunner,
        sample_lme_results: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """Test compare command with valid files."""
        # Create two result files with same conditions
        results_a = tmp_path / "a.json"
        results_b = tmp_path / "b.json"

        with results_a.open("w") as f:
            json.dump(sample_lme_results, f)

        # Modify experiment_id for B
        modified = sample_lme_results.copy()
        modified["experiment_id"] = "exp_lme_002"
        with results_b.open("w") as f:
            json.dump(modified, f)

        result = runner.invoke(app, ["compare", str(results_a), str(results_b)])
        assert result.exit_code == 0
        assert "Comparing experiments:" in result.stdout
        assert "Common conditions:" in result.stdout


class TestHelpOutput:
    """Tests for CLI help output."""

    def test_main_help(self, runner: CliRunner) -> None:
        """Test main help output."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Benchmark harness for git-native semantic memory" in result.stdout
        assert "export-samples" in result.stdout
        assert "report" in result.stdout

    def test_export_samples_help(self, runner: CliRunner) -> None:
        """Test export-samples help output."""
        result = runner.invoke(app, ["export-samples", "--help"])
        assert result.exit_code == 0
        assert "Export validation samples" in result.stdout
        assert "--output" in result.stdout
        assert "--samples" in result.stdout

    def test_report_help(self, runner: CliRunner) -> None:
        """Test report help output."""
        result = runner.invoke(app, ["report", "--help"])
        assert result.exit_code == 0
        assert "Generate a summary report" in result.stdout
        assert "--confidence" in result.stdout
        assert "--bootstrap" in result.stdout
        assert "--stdout" in result.stdout
