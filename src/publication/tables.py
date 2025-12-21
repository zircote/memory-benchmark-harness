"""Publication table generators.

This module generates publication-ready tables in LaTeX and Markdown formats.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TableGenerator(ABC):
    """Base class for table generators.

    Attributes:
        title: Table title
        caption: Table caption
        label: LaTeX label for references
    """

    title: str = ""
    caption: str = ""
    label: str = ""

    @abstractmethod
    def generate_latex(self, data: list[dict[str, Any]]) -> str:
        """Generate LaTeX table.

        Args:
            data: Table data as list of row dictionaries

        Returns:
            LaTeX string
        """
        pass

    @abstractmethod
    def generate_markdown(self, data: list[dict[str, Any]]) -> str:
        """Generate Markdown table.

        Args:
            data: Table data as list of row dictionaries

        Returns:
            Markdown string
        """
        pass

    def save_latex(self, data: list[dict[str, Any]], path: Path | str) -> None:
        """Save table as LaTeX file.

        Args:
            data: Table data
            path: Output path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        content = self.generate_latex(data)
        with open(path, "w") as f:
            f.write(content)
        logger.info(f"Saved LaTeX table to {path}")

    def save_markdown(self, data: list[dict[str, Any]], path: Path | str) -> None:
        """Save table as Markdown file.

        Args:
            data: Table data
            path: Output path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        content = self.generate_markdown(data)
        with open(path, "w") as f:
            f.write(content)
        logger.info(f"Saved Markdown table to {path}")


@dataclass
class MainResultsTable(TableGenerator):
    """Generates main results comparison table.

    Shows accuracy across benchmarks for each adapter condition.
    """

    title: str = "Main Results"
    caption: str = "Accuracy comparison across benchmarks and conditions"
    label: str = "tab:main-results"
    benchmark_order: list[str] = field(default_factory=list)

    def generate_latex(self, data: list[dict[str, Any]]) -> str:
        """Generate LaTeX table for main results.

        Args:
            data: List of {adapter, overall_accuracy, f1_score, abstention_rate, benchmarks}

        Returns:
            LaTeX string
        """
        if not data:
            return ""

        # Determine benchmark columns
        all_benchmarks = set()
        for row in data:
            all_benchmarks.update(row.get("benchmarks", {}).keys())

        benchmarks = self.benchmark_order if self.benchmark_order else sorted(all_benchmarks)

        # Build column spec
        num_cols = 1 + len(benchmarks) + 2  # Adapter + benchmarks + Overall + F1
        col_spec = "l" + "c" * (num_cols - 1)

        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            f"\\caption{{{self.caption}}}",
            f"\\label{{{self.label}}}",
            f"\\begin{{tabular}}{{{col_spec}}}",
            r"\toprule",
        ]

        # Header row
        benchmark_headers = " & ".join(self._format_header(b) for b in benchmarks)
        header = f"Condition & {benchmark_headers} & Overall & F1 \\\\"
        lines.append(header)
        lines.append(r"\midrule")

        # Data rows
        for row in data:
            adapter = self._escape_latex(row["adapter"])
            benchmark_values = []
            for b in benchmarks:
                val = row.get("benchmarks", {}).get(b, 0.0)
                benchmark_values.append(f"{val:.1%}")

            overall = f"{row['overall_accuracy']:.1%}"
            f1 = f"{row['f1_score']:.2f}"

            values = " & ".join(benchmark_values)
            lines.append(f"{adapter} & {values} & {overall} & {f1} \\\\")

        lines.extend(
            [
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{table}",
            ]
        )

        return "\n".join(lines)

    def generate_markdown(self, data: list[dict[str, Any]]) -> str:
        """Generate Markdown table for main results.

        Args:
            data: List of result dictionaries

        Returns:
            Markdown string
        """
        if not data:
            return ""

        # Determine benchmark columns
        all_benchmarks = set()
        for row in data:
            all_benchmarks.update(row.get("benchmarks", {}).keys())

        benchmarks = self.benchmark_order if self.benchmark_order else sorted(all_benchmarks)

        lines = [f"## {self.title}", ""]

        # Header
        headers = ["Condition"] + [self._format_header(b) for b in benchmarks] + ["Overall", "F1"]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "|".join(["---"] * len(headers)) + "|")

        # Data rows
        for row in data:
            values = [row["adapter"]]
            for b in benchmarks:
                val = row.get("benchmarks", {}).get(b, 0.0)
                values.append(f"{val:.1%}")
            values.append(f"{row['overall_accuracy']:.1%}")
            values.append(f"{row['f1_score']:.2f}")
            lines.append("| " + " | ".join(values) + " |")

        return "\n".join(lines)

    def _format_header(self, name: str) -> str:
        """Format benchmark name for header."""
        # Shorten long names
        replacements = {
            "longmemeval": "LME",
            "locomo": "LoCoMo",
            "memoryagentbench": "MAB",
            "contextbench": "CB",
            "terminalbench": "TB",
        }
        return replacements.get(name.lower(), name)

    def _escape_latex(self, text: str) -> str:
        """Escape special LaTeX characters."""
        replacements = [
            ("_", r"\_"),
            ("&", r"\&"),
            ("%", r"\%"),
            ("#", r"\#"),
        ]
        for old, new in replacements:
            text = text.replace(old, new)
        return text


@dataclass
class AblationTable(TableGenerator):
    """Generates ablation study results table.

    Shows impact of removing each component.
    """

    title: str = "Ablation Study"
    caption: str = "Impact of removing individual system components"
    label: str = "tab:ablation"

    def generate_latex(self, data: list[dict[str, Any]]) -> str:
        """Generate LaTeX table for ablation results.

        Args:
            data: List of {ablation, baseline, ablated, delta, delta_pct, p_value, significant}

        Returns:
            LaTeX string
        """
        if not data:
            return ""

        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            f"\\caption{{{self.caption}}}",
            f"\\label{{{self.label}}}",
            r"\begin{tabular}{lccccc}",
            r"\toprule",
            r"Ablation & Baseline & Ablated & $\Delta$ & $\Delta\%$ & Sig. \\",
            r"\midrule",
        ]

        for row in data:
            ablation = self._format_ablation(row["ablation"])
            baseline = f"{row['baseline']:.1%}"
            ablated = f"{row['ablated']:.1%}"
            delta = f"{row['delta']:+.1%}"
            delta_pct = f"{row['delta_pct']:+.1f}\\%"
            sig = r"\checkmark" if row.get("significant", False) else ""

            lines.append(
                f"{ablation} & {baseline} & {ablated} & {delta} & {delta_pct} & {sig} \\\\"
            )

        lines.extend(
            [
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{table}",
            ]
        )

        return "\n".join(lines)

    def generate_markdown(self, data: list[dict[str, Any]]) -> str:
        """Generate Markdown table for ablation results.

        Args:
            data: List of ablation dictionaries

        Returns:
            Markdown string
        """
        if not data:
            return ""

        lines = [f"## {self.title}", ""]

        headers = ["Ablation", "Baseline", "Ablated", "Δ", "Δ%", "Sig."]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "|".join(["---"] * len(headers)) + "|")

        for row in data:
            values = [
                self._format_ablation(row["ablation"]),
                f"{row['baseline']:.1%}",
                f"{row['ablated']:.1%}",
                f"{row['delta']:+.1%}",
                f"{row['delta_pct']:+.1f}%",
                "✓" if row.get("significant", False) else "",
            ]
            lines.append("| " + " | ".join(values) + " |")

        return "\n".join(lines)

    def _format_ablation(self, name: str) -> str:
        """Format ablation name for display."""
        replacements = {
            "no_semantic_search": "w/o Semantic Search",
            "no_metadata_filter": "w/o Metadata Filter",
            "no_version_history": "w/o Version History",
            "fixed_window": "Fixed Window Only",
            "recency_only": "Recency Only",
        }
        return replacements.get(name.lower(), name.replace("_", " ").title())


@dataclass
class CategoryBreakdownTable(TableGenerator):
    """Generates per-category accuracy breakdown table.

    Shows accuracy for each question category/type.
    """

    title: str = "Category Breakdown"
    caption: str = "Accuracy breakdown by question category"
    label: str = "tab:category-breakdown"
    adapters: list[str] = field(default_factory=list)

    def generate_latex(self, data: list[dict[str, Any]]) -> str:
        """Generate LaTeX table for category breakdown.

        Args:
            data: List of {category, adapter1: acc, adapter2: acc, ...}

        Returns:
            LaTeX string
        """
        if not data:
            return ""

        # Determine adapter columns
        adapters = self.adapters
        if not adapters:
            adapters = sorted(k for k in data[0] if k != "category")

        col_spec = "l" + "c" * len(adapters)

        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            f"\\caption{{{self.caption}}}",
            f"\\label{{{self.label}}}",
            f"\\begin{{tabular}}{{{col_spec}}}",
            r"\toprule",
        ]

        # Header
        adapter_headers = " & ".join(self._escape_latex(a) for a in adapters)
        lines.append(f"Category & {adapter_headers} \\\\")
        lines.append(r"\midrule")

        # Data rows
        for row in data:
            category = self._escape_latex(row["category"])
            values = []
            for adapter in adapters:
                val = row.get(adapter, 0.0)
                values.append(f"{val:.1%}")

            lines.append(f"{category} & {' & '.join(values)} \\\\")

        lines.extend(
            [
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{table}",
            ]
        )

        return "\n".join(lines)

    def generate_markdown(self, data: list[dict[str, Any]]) -> str:
        """Generate Markdown table for category breakdown.

        Args:
            data: List of category dictionaries

        Returns:
            Markdown string
        """
        if not data:
            return ""

        # Determine adapter columns
        adapters = self.adapters
        if not adapters:
            adapters = sorted(k for k in data[0] if k != "category")

        lines = [f"## {self.title}", ""]

        # Header
        headers = ["Category"] + adapters
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "|".join(["---"] * len(headers)) + "|")

        # Data rows
        for row in data:
            values = [row["category"]]
            for adapter in adapters:
                val = row.get(adapter, 0.0)
                values.append(f"{val:.1%}")
            lines.append("| " + " | ".join(values) + " |")

        return "\n".join(lines)

    def _escape_latex(self, text: str) -> str:
        """Escape special LaTeX characters."""
        replacements = [
            ("_", r"\_"),
            ("&", r"\&"),
            ("%", r"\%"),
            ("#", r"\#"),
        ]
        for old, new in replacements:
            text = text.replace(old, new)
        return text


@dataclass
class HumanValidationTable(TableGenerator):
    """Generates human validation agreement table.

    Shows inter-annotator and human-LLM agreement metrics.
    """

    title: str = "Human Validation Agreement"
    caption: str = "Agreement metrics between annotators and LLM judges"
    label: str = "tab:human-validation"

    def generate_latex(self, data: list[dict[str, Any]]) -> str:
        """Generate LaTeX table for human validation.

        Args:
            data: List of {benchmark, agreement_rate, kappa, weighted_kappa, n_samples}

        Returns:
            LaTeX string
        """
        if not data:
            return ""

        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            f"\\caption{{{self.caption}}}",
            f"\\label{{{self.label}}}",
            r"\begin{tabular}{lcccc}",
            r"\toprule",
            r"Benchmark & Agreement & $\kappa$ & Weighted $\kappa$ & $n$ \\",
            r"\midrule",
        ]

        for row in data:
            benchmark = self._escape_latex(row["benchmark"])
            agreement = f"{row['agreement_rate']:.1%}"
            kappa = f"{row['kappa']:.3f}"
            weighted = f"{row['weighted_kappa']:.3f}"
            n = str(row["n_samples"])

            lines.append(f"{benchmark} & {agreement} & {kappa} & {weighted} & {n} \\\\")

        lines.extend(
            [
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{table}",
            ]
        )

        return "\n".join(lines)

    def generate_markdown(self, data: list[dict[str, Any]]) -> str:
        """Generate Markdown table for human validation.

        Args:
            data: List of validation dictionaries

        Returns:
            Markdown string
        """
        if not data:
            return ""

        lines = [f"## {self.title}", ""]

        headers = ["Benchmark", "Agreement", "κ", "Weighted κ", "n"]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "|".join(["---"] * len(headers)) + "|")

        for row in data:
            values = [
                row["benchmark"],
                f"{row['agreement_rate']:.1%}",
                f"{row['kappa']:.3f}",
                f"{row['weighted_kappa']:.3f}",
                str(row["n_samples"]),
            ]
            lines.append("| " + " | ".join(values) + " |")

        return "\n".join(lines)

    def _escape_latex(self, text: str) -> str:
        """Escape special LaTeX characters."""
        replacements = [
            ("_", r"\_"),
            ("&", r"\&"),
        ]
        for old, new in replacements:
            text = text.replace(old, new)
        return text
