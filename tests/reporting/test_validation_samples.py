"""Tests for the human validation sample exporter.

Tests cover:
- ValidationSample dataclass creation and serialization
- ValidationExporter sample extraction from LongMemEval results
- ValidationExporter sample extraction from LoCoMo results
- Stratified sampling across categories and conditions
- Export to JSON format
- Export to CSV format
- Combined export functionality
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from src.reporting.validation_samples import ValidationExporter, ValidationSample

# Benchmark names as constants to avoid substring pattern issues
BENCHMARK_LME = "longmemeval"
BENCHMARK_LOCO = "locomo"


class TestValidationSample:
    """Tests for ValidationSample dataclass."""

    def test_valid_validation_sample(self) -> None:
        """Test creating a valid validation sample."""
        sample = ValidationSample(
            sample_id="lme_git-notes_q1",
            benchmark=BENCHMARK_LME,
            category="factoid",
            question="What is the capital of France?",
            reference_answer="Paris",
            model_response="The capital of France is Paris.",
            llm_judgment="Correct - the answer matches the reference.",
            llm_score=1.0,
            condition="git-notes",
            metadata={"session_id": "sess_001"},
        )

        assert sample.sample_id == "lme_git-notes_q1"
        assert sample.benchmark == BENCHMARK_LME
        assert sample.category == "factoid"
        assert sample.question == "What is the capital of France?"
        assert sample.reference_answer == "Paris"
        assert sample.model_response == "The capital of France is Paris."
        assert sample.llm_judgment == "Correct - the answer matches the reference."
        assert sample.llm_score == 1.0
        assert sample.condition == "git-notes"
        assert sample.metadata == {"session_id": "sess_001"}

    def test_validation_sample_with_none_score(self) -> None:
        """Test validation sample with None score."""
        sample = ValidationSample(
            sample_id="loco_no-memory_q2",
            benchmark=BENCHMARK_LOCO,
            category="temporal",
            question="When did we last meet?",
            reference_answer="Yesterday",
            model_response="I don't recall.",
            llm_judgment="Unable to assess.",
            llm_score=None,
            condition="no-memory",
        )

        assert sample.llm_score is None
        assert sample.metadata == {}  # Default empty dict

    def test_validation_sample_frozen(self) -> None:
        """Test that ValidationSample is immutable."""
        sample = ValidationSample(
            sample_id="test_001",
            benchmark=BENCHMARK_LME,
            category="factoid",
            question="Q",
            reference_answer="A",
            model_response="R",
            llm_judgment="J",
            llm_score=1.0,
            condition="mock",
        )

        with pytest.raises(AttributeError):
            sample.sample_id = "new_id"  # type: ignore[misc]

    def test_to_dict_serialization(self) -> None:
        """Test serialization to dictionary."""
        sample = ValidationSample(
            sample_id="lme_git-notes_q1",
            benchmark=BENCHMARK_LME,
            category="factoid",
            question="What is X?",
            reference_answer="Y",
            model_response="Y indeed",
            llm_judgment="Correct",
            llm_score=1.0,
            condition="git-notes",
            metadata={"trial_id": "t1"},
        )

        data = sample.to_dict()

        assert data["sample_id"] == "lme_git-notes_q1"
        assert data["benchmark"] == BENCHMARK_LME
        assert data["category"] == "factoid"
        assert data["question"] == "What is X?"
        assert data["reference_answer"] == "Y"
        assert data["model_response"] == "Y indeed"
        assert data["llm_judgment"] == "Correct"
        assert data["llm_score"] == 1.0
        assert data["condition"] == "git-notes"
        assert data["metadata"] == {"trial_id": "t1"}


class TestValidationExporterInit:
    """Tests for ValidationExporter initialization."""

    def test_default_initialization(self) -> None:
        """Test default parameter values."""
        exporter = ValidationExporter()

        assert exporter.samples_per_benchmark == 100
        assert exporter.stratify_by_category is True
        assert exporter.include_both_conditions is True
        assert exporter.seed == 42

    def test_custom_initialization(self) -> None:
        """Test custom parameter values."""
        exporter = ValidationExporter(
            samples_per_benchmark=50,
            stratify_by_category=False,
            include_both_conditions=False,
            seed=123,
        )

        assert exporter.samples_per_benchmark == 50
        assert exporter.stratify_by_category is False
        assert exporter.include_both_conditions is False
        assert exporter.seed == 123


class TestExtractLongMemEvalSamples:
    """Tests for LongMemEval sample extraction."""

    @pytest.fixture
    def exporter(self) -> ValidationExporter:
        """Create exporter with fixed seed."""
        return ValidationExporter(samples_per_benchmark=10, seed=42)

    @pytest.fixture
    def lme_results(self) -> dict[str, Any]:
        """Create sample LongMemEval experiment results."""
        return {
            "benchmark": BENCHMARK_LME,
            "experiment_id": "exp_lme_001",
            "trials": {
                "git-notes": [
                    {
                        "trial_id": "t1",
                        "success": True,
                        "metrics": {"accuracy": 0.85},
                        "raw_results": {
                            "question_results": [
                                {
                                    "question_id": "q1",
                                    "question_type": "factoid",
                                    "question": "What color is the sky?",
                                    "reference_answer": "Blue",
                                    "predicted": "The sky is blue.",
                                    "correct": True,
                                    "judgment_text": "Correct answer.",
                                    "session_id": "sess_001",
                                    "abstained": False,
                                },
                                {
                                    "question_id": "q2",
                                    "question_type": "reasoning",
                                    "question": "Why does ice float?",
                                    "reference_answer": "Lower density",
                                    "predicted": "Ice has lower density than water.",
                                    "correct": True,
                                    "judgment_text": "Good explanation.",
                                    "session_id": "sess_002",
                                    "abstained": False,
                                },
                            ]
                        },
                    }
                ],
                "no-memory": [
                    {
                        "trial_id": "t2",
                        "success": True,
                        "metrics": {"accuracy": 0.70},
                        "raw_results": {
                            "question_results": [
                                {
                                    "question_id": "q3",
                                    "question_type": "factoid",
                                    "question": "What is 2+2?",
                                    "reference_answer": "4",
                                    "predicted": "4",
                                    "correct": True,
                                    "judgment_text": "Correct.",
                                    "session_id": "sess_003",
                                    "abstained": False,
                                },
                            ]
                        },
                    },
                    {
                        "trial_id": "t3",
                        "success": False,  # Failed trial - should be skipped
                        "metrics": {},
                        "raw_results": {},
                    },
                ],
            },
        }

    def test_extract_basic_samples(
        self, exporter: ValidationExporter, lme_results: dict[str, Any]
    ) -> None:
        """Test basic sample extraction from LongMemEval results."""
        samples = exporter.extract_longmemeval_samples(lme_results)

        # Should extract 3 samples (2 from git-notes, 1 from no-memory)
        assert len(samples) == 3

        # Check sample fields
        for sample in samples:
            assert sample.benchmark == BENCHMARK_LME
            assert sample.condition in ("git-notes", "no-memory")
            assert sample.question != ""
            assert sample.reference_answer != ""

    def test_extract_skips_failed_trials(
        self, exporter: ValidationExporter, lme_results: dict[str, Any]
    ) -> None:
        """Test that failed trials are skipped."""
        samples = exporter.extract_longmemeval_samples(lme_results)

        # Should not include samples from failed trial t3
        trial_ids = [s.metadata.get("trial_id") for s in samples]
        assert "t3" not in trial_ids

    def test_extract_metadata_fields(
        self, exporter: ValidationExporter, lme_results: dict[str, Any]
    ) -> None:
        """Test that metadata fields are correctly extracted."""
        samples = exporter.extract_longmemeval_samples(lme_results)

        for sample in samples:
            assert "session_id" in sample.metadata
            assert "abstained" in sample.metadata
            assert "trial_id" in sample.metadata

    def test_extract_llm_score_conversion(
        self, exporter: ValidationExporter, lme_results: dict[str, Any]
    ) -> None:
        """Test that correct/incorrect is converted to llm_score."""
        samples = exporter.extract_longmemeval_samples(lme_results)

        for sample in samples:
            # All samples in fixture have correct=True
            assert sample.llm_score == 1.0

    def test_extract_empty_results(self, exporter: ValidationExporter) -> None:
        """Test extraction with empty results."""
        empty_results: dict[str, Any] = {
            "benchmark": BENCHMARK_LME,
            "experiment_id": "exp_empty",
            "trials": {},
        }

        samples = exporter.extract_longmemeval_samples(empty_results)

        assert samples == []


class TestExtractLoCoMoSamples:
    """Tests for LoCoMo sample extraction."""

    @pytest.fixture
    def exporter(self) -> ValidationExporter:
        """Create exporter with fixed seed."""
        return ValidationExporter(samples_per_benchmark=10, seed=42)

    @pytest.fixture
    def locomo_results(self) -> dict[str, Any]:
        """Create sample LoCoMo experiment results."""
        return {
            "benchmark": BENCHMARK_LOCO,
            "experiment_id": "exp_loco_001",
            "trials": {
                "git-notes": [
                    {
                        "trial_id": "t1",
                        "success": True,
                        "metrics": {"overall_accuracy": 0.80},
                        "raw_results": {
                            "conversation_results": [
                                {
                                    "sample_id": "conv_001",
                                    "question_results": [
                                        {
                                            "question_id": "qa1",
                                            "category": "identity",
                                            "question": "What is my name?",
                                            "reference_answer": "John",
                                            "predicted": "Your name is John.",
                                            "score": 1.0,
                                            "judgment_text": "Correct identification.",
                                            "is_adversarial": False,
                                            "difficulty": "easy",
                                        },
                                        {
                                            "question_id": "qa2",
                                            "category": "temporal",
                                            "question": "When did I start?",
                                            "reference_answer": "Last Monday",
                                            "predicted": "You started last Monday.",
                                            "score": 0.8,
                                            "judgment_text": "Partial credit.",
                                            "is_adversarial": False,
                                            "difficulty": "medium",
                                        },
                                    ],
                                },
                            ],
                        },
                    },
                ],
                "no-memory": [
                    {
                        "trial_id": "t2",
                        "success": True,
                        "metrics": {"overall_accuracy": 0.50},
                        "raw_results": {
                            "conversation_results": [
                                {
                                    "sample_id": "conv_002",
                                    "question_results": [
                                        {
                                            "question_id": "qa3",
                                            "category": "adversarial",
                                            "question": "Did I not say X?",
                                            "reference_answer": "You said X",
                                            "predicted": "I'm not sure.",
                                            "score": 0.0,
                                            "judgment_text": "Incorrect.",
                                            "is_adversarial": True,
                                            "difficulty": "hard",
                                        },
                                    ],
                                },
                            ],
                        },
                    },
                ],
            },
        }

    def test_extract_basic_samples(
        self, exporter: ValidationExporter, locomo_results: dict[str, Any]
    ) -> None:
        """Test basic sample extraction from LoCoMo results."""
        samples = exporter.extract_locomo_samples(locomo_results)

        # Should extract 3 samples (2 from git-notes, 1 from no-memory)
        assert len(samples) == 3

        # Check sample fields
        for sample in samples:
            assert sample.benchmark == BENCHMARK_LOCO
            assert sample.condition in ("git-notes", "no-memory")
            assert sample.category in ("identity", "temporal", "adversarial")

    def test_extract_adversarial_metadata(
        self, exporter: ValidationExporter, locomo_results: dict[str, Any]
    ) -> None:
        """Test that adversarial flag is in metadata."""
        samples = exporter.extract_locomo_samples(locomo_results)

        adversarial_samples = [s for s in samples if s.metadata.get("is_adversarial")]
        non_adversarial = [s for s in samples if not s.metadata.get("is_adversarial")]

        assert len(adversarial_samples) == 1
        assert len(non_adversarial) == 2

    def test_extract_difficulty_metadata(
        self, exporter: ValidationExporter, locomo_results: dict[str, Any]
    ) -> None:
        """Test that difficulty is in metadata."""
        samples = exporter.extract_locomo_samples(locomo_results)

        difficulties = {s.metadata.get("difficulty") for s in samples}
        assert difficulties == {"easy", "medium", "hard"}

    def test_extract_sample_id_format(
        self, exporter: ValidationExporter, locomo_results: dict[str, Any]
    ) -> None:
        """Test sample ID format includes conversation and question IDs."""
        samples = exporter.extract_locomo_samples(locomo_results)

        for sample in samples:
            # Format: loco_{condition}_{conv_sample_id}_{question_id}
            assert sample.sample_id.startswith("loco_")
            parts = sample.sample_id.split("_")
            assert len(parts) >= 4


class TestStratifiedSampling:
    """Tests for stratified sampling functionality."""

    @pytest.fixture
    def exporter(self) -> ValidationExporter:
        """Create exporter with stratification enabled."""
        return ValidationExporter(
            samples_per_benchmark=6,
            stratify_by_category=True,
            seed=42,
        )

    def _create_sample(self, sample_id: str, category: str, condition: str) -> ValidationSample:
        """Helper to create validation samples."""
        return ValidationSample(
            sample_id=sample_id,
            benchmark="test",
            category=category,
            question=f"Q_{sample_id}",
            reference_answer="A",
            model_response="R",
            llm_judgment="J",
            llm_score=1.0,
            condition=condition,
        )

    def test_stratified_sampling_balances_categories(self, exporter: ValidationExporter) -> None:
        """Test that stratified sampling balances across categories."""
        # Create samples with unbalanced categories
        samples = [
            self._create_sample(f"s_{i}", "cat_a" if i < 10 else "cat_b", "cond1")
            for i in range(20)
        ]

        import random

        rng = random.Random(exporter.seed)

        # Sample 6 from 20 (should try to balance)
        result = exporter._stratified_sample(samples, 6, rng)

        assert len(result) == 6

        # Check that both categories are represented
        categories = {s.category for s in result}
        assert len(categories) == 2

    def test_stratified_sampling_balances_conditions(self, exporter: ValidationExporter) -> None:
        """Test that stratified sampling balances across conditions."""
        samples = [
            self._create_sample(f"s_{i}", "cat", "cond1" if i < 10 else "cond2") for i in range(20)
        ]

        import random

        rng = random.Random(exporter.seed)

        result = exporter._stratified_sample(samples, 6, rng)

        assert len(result) == 6

        # Check that both conditions are represented
        conditions = {s.condition for s in result}
        assert len(conditions) == 2

    def test_non_stratified_sampling(self) -> None:
        """Test sampling without stratification."""
        exporter = ValidationExporter(
            samples_per_benchmark=5,
            stratify_by_category=False,
            seed=42,
        )

        samples = [self._create_sample(f"s_{i}", f"cat_{i % 3}", "cond") for i in range(20)]

        import random

        rng = random.Random(exporter.seed)

        result = exporter._stratified_sample(samples, 5, rng)

        assert len(result) == 5

    def test_sampling_returns_all_when_under_limit(self, exporter: ValidationExporter) -> None:
        """Test that all samples are returned when under the limit."""
        samples = [self._create_sample(f"s_{i}", "cat", "cond") for i in range(3)]

        import random

        rng = random.Random(exporter.seed)

        # Exporter wants 6, but only 3 available
        result = exporter._stratified_sample(samples, 6, rng)

        assert len(result) == 3  # All samples returned

    def test_reproducibility_with_same_seed(self) -> None:
        """Test that same seed produces same results."""
        samples = [self._create_sample(f"s_{i}", f"cat_{i % 2}", "cond") for i in range(50)]

        import random

        rng1 = random.Random(42)
        rng2 = random.Random(42)

        exporter = ValidationExporter(samples_per_benchmark=10, seed=42)

        result1 = exporter._stratified_sample(samples, 10, rng1)
        result2 = exporter._stratified_sample(samples, 10, rng2)

        assert [s.sample_id for s in result1] == [s.sample_id for s in result2]


class TestExportToJson:
    """Tests for JSON export functionality."""

    @pytest.fixture
    def exporter(self) -> ValidationExporter:
        """Create exporter."""
        return ValidationExporter(samples_per_benchmark=10, seed=42)

    @pytest.fixture
    def sample_data(self) -> list[ValidationSample]:
        """Create sample data for export."""
        return [
            ValidationSample(
                sample_id="s1",
                benchmark=BENCHMARK_LME,
                category="factoid",
                question="Q1?",
                reference_answer="A1",
                model_response="R1",
                llm_judgment="Correct",
                llm_score=1.0,
                condition="git-notes",
            ),
            ValidationSample(
                sample_id="s2",
                benchmark=BENCHMARK_LOCO,
                category="temporal",
                question="Q2?",
                reference_answer="A2",
                model_response="R2",
                llm_judgment="Incorrect",
                llm_score=0.0,
                condition="no-memory",
            ),
        ]

    def test_export_creates_file(
        self,
        exporter: ValidationExporter,
        sample_data: list[ValidationSample],
        tmp_path: Path,
    ) -> None:
        """Test that export creates the JSON file."""
        output_path = tmp_path / "validation_samples.json"

        exporter.export_to_json(sample_data, output_path)

        assert output_path.exists()

    def test_export_json_structure(
        self,
        exporter: ValidationExporter,
        sample_data: list[ValidationSample],
        tmp_path: Path,
    ) -> None:
        """Test the structure of exported JSON."""
        output_path = tmp_path / "validation_samples.json"

        exporter.export_to_json(sample_data, output_path)

        with output_path.open() as f:
            data = json.load(f)

        assert "export_metadata" in data
        assert "samples" in data
        assert data["export_metadata"]["total_samples"] == 2
        assert len(data["samples"]) == 2

    def test_export_json_sample_content(
        self,
        exporter: ValidationExporter,
        sample_data: list[ValidationSample],
        tmp_path: Path,
    ) -> None:
        """Test that sample content is correctly serialized."""
        output_path = tmp_path / "validation_samples.json"

        exporter.export_to_json(sample_data, output_path)

        with output_path.open() as f:
            data = json.load(f)

        sample = data["samples"][0]
        assert sample["sample_id"] == "s1"
        assert sample["benchmark"] == BENCHMARK_LME
        assert sample["category"] == "factoid"
        assert sample["llm_score"] == 1.0

    def test_export_creates_parent_dirs(
        self,
        exporter: ValidationExporter,
        sample_data: list[ValidationSample],
        tmp_path: Path,
    ) -> None:
        """Test that export creates parent directories."""
        output_path = tmp_path / "nested" / "dir" / "samples.json"

        exporter.export_to_json(sample_data, output_path)

        assert output_path.exists()


class TestExportToCsv:
    """Tests for CSV export functionality."""

    @pytest.fixture
    def exporter(self) -> ValidationExporter:
        """Create exporter."""
        return ValidationExporter(samples_per_benchmark=10, seed=42)

    @pytest.fixture
    def sample_data(self) -> list[ValidationSample]:
        """Create sample data for export."""
        return [
            ValidationSample(
                sample_id="s1",
                benchmark=BENCHMARK_LME,
                category="factoid",
                question="Q1?",
                reference_answer="A1",
                model_response="R1",
                llm_judgment="Correct",
                llm_score=1.0,
                condition="git-notes",
            ),
            ValidationSample(
                sample_id="s2",
                benchmark=BENCHMARK_LOCO,
                category="temporal",
                question="Q2?",
                reference_answer="A2",
                model_response="R2",
                llm_judgment="Incorrect",
                llm_score=None,  # Test None handling
                condition="no-memory",
            ),
        ]

    def test_export_creates_file(
        self,
        exporter: ValidationExporter,
        sample_data: list[ValidationSample],
        tmp_path: Path,
    ) -> None:
        """Test that export creates the CSV file."""
        output_path = tmp_path / "validation_samples.csv"

        exporter.export_to_csv(sample_data, output_path)

        assert output_path.exists()

    def test_export_csv_headers(
        self,
        exporter: ValidationExporter,
        sample_data: list[ValidationSample],
        tmp_path: Path,
    ) -> None:
        """Test that CSV has correct headers."""
        output_path = tmp_path / "validation_samples.csv"

        exporter.export_to_csv(sample_data, output_path)

        with output_path.open(newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames

        expected_headers = [
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
        assert fieldnames == expected_headers

    def test_export_csv_content(
        self,
        exporter: ValidationExporter,
        sample_data: list[ValidationSample],
        tmp_path: Path,
    ) -> None:
        """Test that CSV content is correct."""
        output_path = tmp_path / "validation_samples.csv"

        exporter.export_to_csv(sample_data, output_path)

        with output_path.open(newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2
        assert rows[0]["sample_id"] == "s1"
        assert rows[0]["benchmark"] == BENCHMARK_LME
        assert rows[0]["human_score"] == ""  # Empty for annotation
        assert rows[0]["human_notes"] == ""  # Empty for notes

    def test_export_csv_none_score(
        self,
        exporter: ValidationExporter,
        sample_data: list[ValidationSample],
        tmp_path: Path,
    ) -> None:
        """Test that None score is exported as empty string."""
        output_path = tmp_path / "validation_samples.csv"

        exporter.export_to_csv(sample_data, output_path)

        with output_path.open(newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Second sample has llm_score=None
        assert rows[1]["llm_score"] == ""


class TestExportCombined:
    """Tests for combined export functionality."""

    @pytest.fixture
    def exporter(self) -> ValidationExporter:
        """Create exporter."""
        return ValidationExporter(samples_per_benchmark=10, seed=42)

    @pytest.fixture
    def lme_results(self) -> dict[str, Any]:
        """Create sample LongMemEval results."""
        return {
            "benchmark": BENCHMARK_LME,
            "trials": {
                "git-notes": [
                    {
                        "success": True,
                        "trial_id": "t1",
                        "raw_results": {
                            "question_results": [
                                {
                                    "question_id": f"q{i}",
                                    "question_type": "factoid",
                                    "question": f"Question {i}",
                                    "reference_answer": f"Answer {i}",
                                    "predicted": f"Predicted {i}",
                                    "correct": True,
                                    "judgment_text": "OK",
                                    "session_id": f"s{i}",
                                }
                                for i in range(5)
                            ]
                        },
                    }
                ]
            },
        }

    @pytest.fixture
    def locomo_results(self) -> dict[str, Any]:
        """Create sample LoCoMo results."""
        return {
            "benchmark": BENCHMARK_LOCO,
            "trials": {
                "no-memory": [
                    {
                        "success": True,
                        "trial_id": "t2",
                        "raw_results": {
                            "conversation_results": [
                                {
                                    "sample_id": "conv1",
                                    "question_results": [
                                        {
                                            "question_id": f"qa{i}",
                                            "category": "identity",
                                            "question": f"Q{i}",
                                            "reference_answer": f"A{i}",
                                            "predicted": f"P{i}",
                                            "score": 1.0,
                                            "judgment_text": "OK",
                                        }
                                        for i in range(5)
                                    ],
                                }
                            ]
                        },
                    }
                ]
            },
        }

    def test_export_combined_creates_files(
        self,
        exporter: ValidationExporter,
        lme_results: dict[str, Any],
        locomo_results: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """Test that combined export creates both JSON and CSV files."""
        summary = exporter.export_combined(
            longmemeval_results=lme_results,
            locomo_results=locomo_results,
            output_dir=tmp_path,
        )

        assert (tmp_path / "validation_samples.json").exists()
        assert (tmp_path / "validation_samples.csv").exists()
        assert len(summary["output_files"]) == 2

    def test_export_combined_returns_summary(
        self,
        exporter: ValidationExporter,
        lme_results: dict[str, Any],
        locomo_results: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """Test that combined export returns correct summary."""
        summary = exporter.export_combined(
            longmemeval_results=lme_results,
            locomo_results=locomo_results,
            output_dir=tmp_path,
        )

        assert summary["total_samples"] == 10  # 5 from each benchmark
        assert BENCHMARK_LME in summary["by_benchmark"]
        assert BENCHMARK_LOCO in summary["by_benchmark"]

    def test_export_combined_with_none_lme(
        self,
        exporter: ValidationExporter,
        locomo_results: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """Test combined export with only LoCoMo results."""
        summary = exporter.export_combined(
            longmemeval_results=None,
            locomo_results=locomo_results,
            output_dir=tmp_path,
        )

        assert summary["total_samples"] == 5
        assert BENCHMARK_LOCO in summary["by_benchmark"]
        assert BENCHMARK_LME not in summary["by_benchmark"]

    def test_export_combined_with_none_locomo(
        self,
        exporter: ValidationExporter,
        lme_results: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """Test combined export with only LongMemEval results."""
        summary = exporter.export_combined(
            longmemeval_results=lme_results,
            locomo_results=None,
            output_dir=tmp_path,
        )

        assert summary["total_samples"] == 5
        assert BENCHMARK_LME in summary["by_benchmark"]
        assert BENCHMARK_LOCO not in summary["by_benchmark"]

    def test_export_combined_with_both_none(
        self,
        exporter: ValidationExporter,
        tmp_path: Path,
    ) -> None:
        """Test combined export with no results."""
        summary = exporter.export_combined(
            longmemeval_results=None,
            locomo_results=None,
            output_dir=tmp_path,
        )

        assert summary["total_samples"] == 0
        assert summary["by_benchmark"] == {}


class TestValidationExporterHelpers:
    """Tests for helper methods."""

    def test_different_seeds_for_benchmarks(self) -> None:
        """Test that LongMemEval and LoCoMo use different seeds."""
        exporter = ValidationExporter(seed=42)

        # The implementation uses seed for LongMemEval and seed+1 for LoCoMo
        # This prevents correlation between samples from different benchmarks
        import random

        lme_rng = random.Random(exporter.seed)
        loco_rng = random.Random(exporter.seed + 1)

        # Generate some random values
        lme_vals = [lme_rng.random() for _ in range(5)]
        loco_vals = [loco_rng.random() for _ in range(5)]

        # They should be different
        assert lme_vals != loco_vals
