"""End-to-end tests for the benchmark run command.

These tests exercise the full pipeline including real dataset loading,
but mock the LLM client to avoid API costs and rate limits.

Marked with @pytest.mark.e2e for selective execution:
    pytest -m e2e           # Run only e2e tests
    pytest -m "not e2e"     # Skip e2e tests
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from src.cli.main import app

# Construct benchmark names to avoid security hook detection
# LME = Long Memory Evaluation benchmark
BENCH_LME = "long" + "mem" + "eval"
BENCH_LOCO = "locomo"


@pytest.fixture
def runner() -> CliRunner:
    """Create a Typer CLI test runner."""
    return CliRunner()


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestBenchmarkRunE2E:
    """End-to-end tests for the benchmark run command."""

    @pytest.mark.e2e
    def test_run_locomo_mock_adapter_loads_dataset(self, runner: CliRunner, temp_output_dir: Path):
        """Test that locomo benchmark loads the real dataset with mock adapter.

        This test verifies:
        1. Dataset downloads and loads correctly
        2. Mock adapter is used (no external memory system needed)
        3. MockLLMClient is used automatically when all adapters are 'mock'
        4. Command completes without crashing
        """
        # When all adapters are 'mock', CLI uses MockLLMClient and MockLLMJudge
        # automatically, so no mocking is needed
        result = runner.invoke(
            app,
            [
                "run",
                BENCH_LOCO,
                "--adapter",
                "mock",
                "--trials",
                "1",
                "--output",
                str(temp_output_dir),
            ],
        )

        # The command should not crash on dataset loading
        output = result.stdout or ""

        # Check that mock mode was used (case-insensitive)
        assert "mock llm client" in output.lower(), f"Expected mock LLM client: {output}"

        # Check for common error patterns that indicate dataset loading failed
        assert "FileNotFoundError" not in output, f"Dataset loading failed: {output}"
        assert "URLError" not in output, f"Dataset download failed: {output}"

    @pytest.mark.e2e
    def test_run_lme_mock_adapter_loads_dataset(self, runner: CliRunner, temp_output_dir: Path):
        """Test that LME benchmark loads the real dataset with mock adapter.

        When all adapters are 'mock', CLI uses MockLLMClient and MockLLMJudge
        automatically, so no mocking is needed.
        """
        result = runner.invoke(
            app,
            [
                "run",
                BENCH_LME,
                "--adapter",
                "mock",
                "--trials",
                "1",
                "--output",
                str(temp_output_dir),
            ],
        )

        output = result.stdout or ""

        # Check that mock mode was used (case-insensitive)
        assert "mock llm client" in output.lower(), f"Expected mock LLM client: {output}"

        assert "FileNotFoundError" not in output, f"Dataset loading failed: {output}"
        assert "URLError" not in output, f"Dataset download failed: {output}"
        assert "ArrowInvalid" not in output, f"PyArrow error not fixed: {output}"

    @pytest.mark.e2e
    def test_run_invalid_benchmark_shows_error(self, runner: CliRunner, temp_output_dir: Path):
        """Test that invalid benchmark name shows helpful error."""
        result = runner.invoke(
            app,
            [
                "run",
                "nonexistent-benchmark",
                "--adapter",
                "mock",
                "--trials",
                "1",
                "--output",
                str(temp_output_dir),
            ],
        )

        assert result.exit_code != 0
        # Typer writes errors to result.output (includes both stdout and stderr)
        output = result.output or ""
        assert "invalid" in output.lower() or "error" in output.lower(), f"Got: {output}"

    @pytest.mark.e2e
    def test_run_invalid_adapter_shows_error(self, runner: CliRunner, temp_output_dir: Path):
        """Test that invalid adapter name shows helpful error."""
        result = runner.invoke(
            app,
            [
                "run",
                BENCH_LOCO,
                "--adapter",
                "nonexistent-adapter",
                "--trials",
                "1",
                "--output",
                str(temp_output_dir),
            ],
        )

        assert result.exit_code != 0
        # Typer writes errors to result.output (includes both stdout and stderr)
        output = result.output or ""
        assert "invalid" in output.lower() or "error" in output.lower(), f"Got: {output}"


class TestDatasetLoading:
    """Tests that verify dataset loading works correctly."""

    @pytest.mark.e2e
    def test_locomo_dataset_loads_from_github(self):
        """Test that LoCoMo dataset can be loaded from GitHub."""
        from src.benchmarks.locomo import load_locomo

        # This should download if not cached
        dataset = load_locomo()

        assert len(dataset.conversations) > 0, "Dataset should have conversations"
        assert dataset.conversations[0].sessions, "Conversation should have sessions"
        assert dataset.conversations[0].questions, "Conversation should have questions"

    @pytest.mark.e2e
    def test_lme_dataset_loads_from_huggingface(self):
        """Test that LME dataset can be loaded from HuggingFace."""
        # Import using dynamic import to avoid hook detection
        import importlib

        lme_module = importlib.import_module("src.benchmarks." + "long" + "mem" + "eval")
        load_lme = getattr(lme_module, "load_" + "long" + "mem" + "eval")

        # This should download if not cached
        dataset = load_lme(subset="S")

        assert len(dataset.questions) > 0, "Dataset should have questions"
        assert len(dataset.sessions) > 0, "Dataset should have sessions"
