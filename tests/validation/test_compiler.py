"""Tests for sample compiler."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.validation.compiler import (
    CompiledSample,
    SampleCompiler,
    SourceBenchmark,
)


class TestSourceBenchmark:
    """Tests for SourceBenchmark enum."""

    def test_benchmark_values(self) -> None:
        """Test all benchmark values exist."""
        assert SourceBenchmark.LONGMEMEVAL.value == "longmemeval"
        assert SourceBenchmark.LOCOMO.value == "locomo"
        assert SourceBenchmark.MEMORYAGENTBENCH.value == "memoryagentbench"
        assert SourceBenchmark.CONTEXTBENCH.value == "contextbench"
        assert SourceBenchmark.TERMINALBENCH.value == "terminalbench"

    def test_from_string_exact_match(self) -> None:
        """Test parsing exact benchmark names."""
        assert SourceBenchmark.from_string("longmemeval") == SourceBenchmark.LONGMEMEVAL
        assert SourceBenchmark.from_string("locomo") == SourceBenchmark.LOCOMO

    def test_from_string_case_insensitive(self) -> None:
        """Test case insensitive parsing."""
        assert SourceBenchmark.from_string("LONGMEMEVAL") == SourceBenchmark.LONGMEMEVAL
        assert SourceBenchmark.from_string("LongMemEval") == SourceBenchmark.LONGMEMEVAL

    def test_from_string_with_separators(self) -> None:
        """Test parsing with hyphens and underscores."""
        assert SourceBenchmark.from_string("long-mem-eval") == SourceBenchmark.LONGMEMEVAL
        assert SourceBenchmark.from_string("memory_agent_bench") == SourceBenchmark.MEMORYAGENTBENCH

    def test_from_string_unknown(self) -> None:
        """Test parsing unknown benchmark raises error."""
        with pytest.raises(ValueError, match="Unknown benchmark"):
            SourceBenchmark.from_string("unknown_benchmark")


class TestCompiledSample:
    """Tests for CompiledSample dataclass."""

    def test_sample_creation(self) -> None:
        """Test creating a compiled sample."""
        sample = CompiledSample(
            sample_id="test_001",
            source_benchmark=SourceBenchmark.LONGMEMEVAL,
            question_id="q1",
            question="What is Alice's job?",
            expected_answers=("Engineer",),
            model_answer="Software Engineer",
            adapter_name="semantic_search",
            llm_judgment="correct",
            llm_confidence=0.95,
        )

        assert sample.sample_id == "test_001"
        assert sample.source_benchmark == SourceBenchmark.LONGMEMEVAL
        assert sample.question == "What is Alice's job?"
        assert "Engineer" in sample.expected_answers
        assert sample.llm_judgment == "correct"

    def test_sample_defaults(self) -> None:
        """Test sample with default values."""
        sample = CompiledSample(
            sample_id="test_002",
            source_benchmark=SourceBenchmark.LOCOMO,
            question_id="q2",
            question="Test question",
            expected_answers=("Answer",),
            model_answer="Model answer",
            adapter_name="baseline",
        )

        assert sample.context == ""
        assert sample.llm_judgment == ""
        assert sample.llm_confidence == 0.0
        assert sample.metadata == {}

    def test_sample_is_frozen(self) -> None:
        """Test sample is immutable."""
        sample = CompiledSample(
            sample_id="test_003",
            source_benchmark=SourceBenchmark.LOCOMO,
            question_id="q3",
            question="Test",
            expected_answers=("Answer",),
            model_answer="Model",
            adapter_name="test",
        )

        with pytest.raises(AttributeError):
            sample.question = "Changed"  # type: ignore


class TestSampleCompiler:
    """Tests for SampleCompiler class."""

    @pytest.fixture
    def compiler(self) -> SampleCompiler:
        """Create a sample compiler."""
        return SampleCompiler(seed=42, target_total=100)

    @pytest.fixture
    def sample_result_file(self, tmp_path) -> Path:
        """Create a sample result file."""
        data = {
            "samples": [
                {
                    "question_id": "q1",
                    "question": "What is X?",
                    "expected_answers": ["Answer A", "Answer B"],
                    "model_answer": "Answer A",
                    "adapter_name": "adapter_1",
                    "judgment": "correct",
                    "confidence": 0.9,
                },
                {
                    "question_id": "q2",
                    "question": "When did Y happen?",
                    "expected_answers": ["2024"],
                    "model_answer": "2023",
                    "adapter_name": "adapter_2",
                    "judgment": "incorrect",
                    "confidence": 0.8,
                },
            ]
        }
        filepath = tmp_path / "results.json"
        with open(filepath, "w") as f:
            json.dump(data, f)
        return filepath

    def test_compiler_initialization(self, compiler: SampleCompiler) -> None:
        """Test compiler initialization."""
        assert compiler.seed == 42
        assert compiler.target_total == 100
        assert len(compiler.samples) == 0

    def test_add_samples_from_file(
        self, compiler: SampleCompiler, sample_result_file: Path
    ) -> None:
        """Test adding samples from a file."""
        count = compiler.add_samples_from_file(sample_result_file, SourceBenchmark.LONGMEMEVAL)

        assert count == 2
        assert len(compiler.samples) == 2

        # Check first sample
        sample = compiler.samples[0]
        assert sample.source_benchmark == SourceBenchmark.LONGMEMEVAL
        assert sample.question == "What is X?"
        assert "Answer A" in sample.expected_answers

    def test_add_samples_from_file_string_benchmark(
        self, compiler: SampleCompiler, sample_result_file: Path
    ) -> None:
        """Test adding samples with string benchmark."""
        count = compiler.add_samples_from_file(sample_result_file, "longmemeval")

        assert count == 2
        assert compiler.samples[0].source_benchmark == SourceBenchmark.LONGMEMEVAL

    def test_add_samples_from_missing_file(self, compiler: SampleCompiler) -> None:
        """Test adding samples from non-existent file."""
        count = compiler.add_samples_from_file(
            Path("/nonexistent/file.json"), SourceBenchmark.LONGMEMEVAL
        )

        assert count == 0
        assert len(compiler.samples) == 0

    def test_add_samples_alternative_format(self, compiler: SampleCompiler, tmp_path) -> None:
        """Test parsing alternative field names."""
        data = {
            "results": [
                {
                    "id": "sample_1",
                    "query": "Question text?",
                    "expected": ["Expected answer"],
                    "generated": "Model output",
                    "condition": "test_condition",
                    "verdict": "partial",
                    "score": 0.5,
                },
            ]
        }
        filepath = tmp_path / "alt_results.json"
        with open(filepath, "w") as f:
            json.dump(data, f)

        count = compiler.add_samples_from_file(filepath, SourceBenchmark.LOCOMO)

        assert count == 1
        sample = compiler.samples[0]
        assert sample.question == "Question text?"
        assert sample.model_answer == "Model output"
        assert sample.llm_judgment == "partial"

    def test_stratified_sample_basic(self, compiler: SampleCompiler) -> None:
        """Test basic stratified sampling."""
        # Add samples from multiple benchmarks
        for i in range(10):
            compiler.samples.append(
                CompiledSample(
                    sample_id=f"lme_{i}",
                    source_benchmark=SourceBenchmark.LONGMEMEVAL,
                    question_id=f"q{i}",
                    question=f"Question {i}",
                    expected_answers=(f"Answer {i}",),
                    model_answer=f"Model {i}",
                    adapter_name="test",
                    llm_judgment="correct" if i % 2 == 0 else "incorrect",
                )
            )
        for i in range(10):
            compiler.samples.append(
                CompiledSample(
                    sample_id=f"loc_{i}",
                    source_benchmark=SourceBenchmark.LOCOMO,
                    question_id=f"q{i}",
                    question=f"Question {i}",
                    expected_answers=(f"Answer {i}",),
                    model_answer=f"Model {i}",
                    adapter_name="test",
                    llm_judgment="correct" if i % 2 == 0 else "incorrect",
                )
            )

        # Sample 10 from 20
        sampled = compiler.stratified_sample(n=10)

        assert len(sampled) == 10
        # Should have samples from both benchmarks
        benchmarks = {s.source_benchmark for s in sampled}
        assert len(benchmarks) == 2

    def test_stratified_sample_all_if_n_exceeds(self, compiler: SampleCompiler) -> None:
        """Test returns all samples if n exceeds total."""
        for i in range(5):
            compiler.samples.append(
                CompiledSample(
                    sample_id=f"test_{i}",
                    source_benchmark=SourceBenchmark.LONGMEMEVAL,
                    question_id=f"q{i}",
                    question=f"Question {i}",
                    expected_answers=("Answer",),
                    model_answer="Model",
                    adapter_name="test",
                )
            )

        sampled = compiler.stratified_sample(n=100)

        assert len(sampled) == 5

    def test_stratified_sample_empty(self, compiler: SampleCompiler) -> None:
        """Test stratified sampling with no samples."""
        sampled = compiler.stratified_sample(n=10)
        assert len(sampled) == 0

    def test_stratified_sample_reproducible(self, compiler: SampleCompiler) -> None:
        """Test stratified sampling is reproducible with seed."""
        for i in range(20):
            compiler.samples.append(
                CompiledSample(
                    sample_id=f"test_{i}",
                    source_benchmark=SourceBenchmark.LONGMEMEVAL,
                    question_id=f"q{i}",
                    question=f"Question {i}",
                    expected_answers=("Answer",),
                    model_answer="Model",
                    adapter_name="test",
                )
            )

        sampled1 = compiler.stratified_sample(n=5)
        sampled2 = compiler.stratified_sample(n=5)

        assert [s.sample_id for s in sampled1] == [s.sample_id for s in sampled2]

    def test_export_json(
        self, compiler: SampleCompiler, sample_result_file: Path, tmp_path
    ) -> None:
        """Test exporting samples to JSON."""
        compiler.add_samples_from_file(sample_result_file, SourceBenchmark.LONGMEMEVAL)

        output_path = tmp_path / "output.json"
        compiler.export_json(output_path)

        assert output_path.exists()
        with open(output_path) as f:
            data = json.load(f)

        assert "metadata" in data
        assert "samples" in data
        assert len(data["samples"]) == 2
        assert data["samples"][0]["question"] == "What is X?"

    def test_export_json_subset(
        self, compiler: SampleCompiler, sample_result_file: Path, tmp_path
    ) -> None:
        """Test exporting a subset of samples."""
        compiler.add_samples_from_file(sample_result_file, SourceBenchmark.LONGMEMEVAL)

        output_path = tmp_path / "subset.json"
        compiler.export_json(output_path, samples=[compiler.samples[0]])

        with open(output_path) as f:
            data = json.load(f)

        assert len(data["samples"]) == 1

    def test_export_csv(self, compiler: SampleCompiler, sample_result_file: Path, tmp_path) -> None:
        """Test exporting samples to CSV."""
        compiler.add_samples_from_file(sample_result_file, SourceBenchmark.LONGMEMEVAL)

        output_path = tmp_path / "output.csv"
        compiler.export_csv(output_path)

        assert output_path.exists()
        content = output_path.read_text()

        # Check header
        assert "sample_id" in content
        assert "human_judgment" in content  # For annotator to fill
        assert "human_notes" in content

        # Check data
        assert "What is X?" in content

    def test_get_stats(self, compiler: SampleCompiler, sample_result_file: Path) -> None:
        """Test getting statistics."""
        compiler.add_samples_from_file(sample_result_file, SourceBenchmark.LONGMEMEVAL)

        stats = compiler.get_stats()

        assert stats["total_samples"] == 2
        assert stats["target_total"] == 100
        assert "longmemeval" in stats["by_benchmark"]
        assert stats["by_benchmark"]["longmemeval"] == 2
        assert "correct" in stats["by_judgment"]
        assert "incorrect" in stats["by_judgment"]
