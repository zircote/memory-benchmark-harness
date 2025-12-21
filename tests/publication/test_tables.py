"""Tests for publication table generators."""

from __future__ import annotations

import pytest

from src.publication.tables import (
    AblationTable,
    CategoryBreakdownTable,
    HumanValidationTable,
    MainResultsTable,
)


class TestMainResultsTable:
    """Tests for MainResultsTable generator."""

    @pytest.fixture
    def table(self) -> MainResultsTable:
        """Create a main results table."""
        return MainResultsTable()

    @pytest.fixture
    def sample_data(self) -> list[dict]:
        """Sample data for main results."""
        return [
            {
                "adapter": "git_notes",
                "overall_accuracy": 0.85,
                "f1_score": 0.83,
                "abstention_rate": 0.05,
                "benchmarks": {
                    "longmemeval": 0.88,
                    "locomo": 0.82,
                },
            },
            {
                "adapter": "no_memory",
                "overall_accuracy": 0.70,
                "f1_score": 0.68,
                "abstention_rate": 0.10,
                "benchmarks": {
                    "longmemeval": 0.72,
                    "locomo": 0.68,
                },
            },
        ]

    def test_generate_latex(self, table: MainResultsTable, sample_data: list[dict]) -> None:
        """Test LaTeX generation."""
        latex = table.generate_latex(sample_data)

        assert r"\begin{table}" in latex
        assert r"\end{table}" in latex
        assert r"\toprule" in latex
        assert r"\bottomrule" in latex
        assert "git\\_notes" in latex or "git_notes" in latex.replace(r"\_", "_")
        assert "88.0%" in latex or "88%" in latex
        assert "LME" in latex or "longmemeval" in latex.lower()

    def test_generate_markdown(self, table: MainResultsTable, sample_data: list[dict]) -> None:
        """Test Markdown generation."""
        markdown = table.generate_markdown(sample_data)

        assert "## Main Results" in markdown
        assert "| Condition |" in markdown
        assert "git_notes" in markdown
        assert "85.0%" in markdown

    def test_empty_data(self, table: MainResultsTable) -> None:
        """Test with empty data."""
        assert table.generate_latex([]) == ""
        assert table.generate_markdown([]) == ""

    def test_save_latex(self, table: MainResultsTable, sample_data: list[dict], tmp_path) -> None:
        """Test saving LaTeX file."""
        output_path = tmp_path / "table.tex"
        table.save_latex(sample_data, output_path)

        assert output_path.exists()
        content = output_path.read_text()
        assert r"\begin{table}" in content

    def test_save_markdown(
        self, table: MainResultsTable, sample_data: list[dict], tmp_path
    ) -> None:
        """Test saving Markdown file."""
        output_path = tmp_path / "table.md"
        table.save_markdown(sample_data, output_path)

        assert output_path.exists()
        content = output_path.read_text()
        assert "## Main Results" in content

    def test_custom_benchmark_order(self, sample_data: list[dict]) -> None:
        """Test custom benchmark ordering."""
        table = MainResultsTable(benchmark_order=["locomo", "longmemeval"])
        markdown = table.generate_markdown(sample_data)

        # LoCoMo should appear before LME in the headers
        locomo_pos = markdown.find("LoCoMo")
        lme_pos = markdown.find("LME")
        assert locomo_pos < lme_pos


class TestAblationTable:
    """Tests for AblationTable generator."""

    @pytest.fixture
    def table(self) -> AblationTable:
        """Create an ablation table."""
        return AblationTable()

    @pytest.fixture
    def sample_data(self) -> list[dict]:
        """Sample ablation data."""
        return [
            {
                "ablation": "no_semantic_search",
                "baseline": 0.85,
                "ablated": 0.70,
                "delta": -0.15,
                "delta_pct": -17.6,
                "p_value": 0.001,
                "effect_size": 0.8,
                "significant": True,
            },
            {
                "ablation": "no_version_history",
                "baseline": 0.85,
                "ablated": 0.82,
                "delta": -0.03,
                "delta_pct": -3.5,
                "p_value": 0.12,
                "effect_size": 0.2,
                "significant": False,
            },
        ]

    def test_generate_latex(self, table: AblationTable, sample_data: list[dict]) -> None:
        """Test LaTeX generation."""
        latex = table.generate_latex(sample_data)

        assert r"\begin{table}" in latex
        assert r"\checkmark" in latex  # Significant marker
        assert "-15.0%" in latex or "-0.15" in latex
        assert "w/o Semantic Search" in latex

    def test_generate_markdown(self, table: AblationTable, sample_data: list[dict]) -> None:
        """Test Markdown generation."""
        markdown = table.generate_markdown(sample_data)

        assert "## Ablation Study" in markdown
        assert "✓" in markdown  # Significant marker
        assert "-15.0%" in markdown
        assert "w/o Semantic Search" in markdown

    def test_ablation_name_formatting(self, table: AblationTable) -> None:
        """Test ablation name formatting."""
        data = [
            {
                "ablation": "recency_only",
                "baseline": 0.85,
                "ablated": 0.60,
                "delta": -0.25,
                "delta_pct": -29.4,
                "significant": True,
            },
        ]
        markdown = table.generate_markdown(data)

        assert "Recency Only" in markdown


class TestCategoryBreakdownTable:
    """Tests for CategoryBreakdownTable generator."""

    @pytest.fixture
    def table(self) -> CategoryBreakdownTable:
        """Create a category breakdown table."""
        return CategoryBreakdownTable()

    @pytest.fixture
    def sample_data(self) -> list[dict]:
        """Sample category data."""
        return [
            {"category": "Single-Hop QA", "git_notes": 0.92, "no_memory": 0.75},
            {"category": "Multi-Hop QA", "git_notes": 0.78, "no_memory": 0.55},
            {"category": "Temporal", "git_notes": 0.85, "no_memory": 0.62},
        ]

    def test_generate_latex(self, table: CategoryBreakdownTable, sample_data: list[dict]) -> None:
        """Test LaTeX generation."""
        latex = table.generate_latex(sample_data)

        assert r"\begin{table}" in latex
        assert "Single-Hop QA" in latex
        assert "92.0%" in latex or "0.92" in latex

    def test_generate_markdown(
        self, table: CategoryBreakdownTable, sample_data: list[dict]
    ) -> None:
        """Test Markdown generation."""
        markdown = table.generate_markdown(sample_data)

        assert "## Category Breakdown" in markdown
        assert "Single-Hop QA" in markdown
        assert "92.0%" in markdown

    def test_custom_adapters(self, sample_data: list[dict]) -> None:
        """Test custom adapter ordering."""
        table = CategoryBreakdownTable(adapters=["no_memory", "git_notes"])
        markdown = table.generate_markdown(sample_data)

        # Check header order
        lines = markdown.split("\n")
        header = next(l for l in lines if l.startswith("| Category"))
        assert header.index("no_memory") < header.index("git_notes")


class TestHumanValidationTable:
    """Tests for HumanValidationTable generator."""

    @pytest.fixture
    def table(self) -> HumanValidationTable:
        """Create a human validation table."""
        return HumanValidationTable()

    @pytest.fixture
    def sample_data(self) -> list[dict]:
        """Sample validation data."""
        return [
            {
                "benchmark": "LongMemEval",
                "agreement_rate": 0.92,
                "kappa": 0.85,
                "weighted_kappa": 0.88,
                "n_samples": 100,
            },
            {
                "benchmark": "LoCoMo",
                "agreement_rate": 0.88,
                "kappa": 0.78,
                "weighted_kappa": 0.82,
                "n_samples": 100,
            },
        ]

    def test_generate_latex(self, table: HumanValidationTable, sample_data: list[dict]) -> None:
        """Test LaTeX generation."""
        latex = table.generate_latex(sample_data)

        assert r"\begin{table}" in latex
        assert r"\kappa" in latex
        assert "LongMemEval" in latex
        assert "92.0%" in latex or "0.92" in latex
        assert "0.850" in latex or "0.85" in latex

    def test_generate_markdown(self, table: HumanValidationTable, sample_data: list[dict]) -> None:
        """Test Markdown generation."""
        markdown = table.generate_markdown(sample_data)

        assert "## Human Validation Agreement" in markdown
        assert "κ" in markdown  # Kappa symbol
        assert "LongMemEval" in markdown
        assert "92.0%" in markdown
        assert "0.850" in markdown
