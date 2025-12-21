"""Publication figure generators.

This module generates publication-ready figures using Matplotlib.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

# Optional matplotlib import
try:
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import numpy as np

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None  # type: ignore[assignment, unused-ignore]
    np = None  # type: ignore[assignment, unused-ignore]


@dataclass
class FigureGenerator(ABC):
    """Base class for figure generators.

    Attributes:
        title: Figure title
        figsize: Figure size (width, height) in inches
        dpi: Resolution for saved figures
        style: Matplotlib style to use
    """

    title: str = ""
    figsize: tuple[float, float] = (8, 6)
    dpi: int = 300
    style: str = "seaborn-v0_8-whitegrid"

    def __post_init__(self) -> None:
        """Initialize matplotlib style."""
        if MATPLOTLIB_AVAILABLE:
            try:
                plt.style.use(self.style)
            except Exception:
                # Fall back to default style
                pass

    @abstractmethod
    def generate(self, data: list[dict[str, Any]]) -> Any:
        """Generate the figure.

        Args:
            data: Figure data

        Returns:
            Matplotlib figure object or None if unavailable
        """
        pass

    def save(
        self,
        data: list[dict[str, Any]],
        path: Path | str,
        format: str = "pdf",
    ) -> bool:
        """Save figure to file.

        Args:
            data: Figure data
            path: Output path
            format: Output format (pdf, png, svg)

        Returns:
            True if saved successfully
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available, cannot save figure")
            return False

        fig = self.generate(data)
        if fig is None:
            return False

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(path, format=format, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Saved figure to {path}")
        return True

    def show(self, data: list[dict[str, Any]]) -> None:
        """Display figure interactively.

        Args:
            data: Figure data
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available")
            return

        fig = self.generate(data)
        if fig is not None:
            plt.show()


@dataclass
class PerformanceBarChart(FigureGenerator):
    """Generates grouped bar chart comparing adapter performance.

    Shows accuracy for each adapter across benchmarks.
    """

    title: str = "Adapter Performance Comparison"
    xlabel: str = "Benchmark"
    ylabel: str = "Accuracy"
    colors: list[str] = field(
        default_factory=lambda: [
            "#2ecc71",  # Green
            "#3498db",  # Blue
            "#9b59b6",  # Purple
            "#e74c3c",  # Red
            "#f39c12",  # Orange
        ]
    )
    bar_width: float = 0.2

    def generate(self, data: list[dict[str, Any]]) -> Any:
        """Generate grouped bar chart.

        Args:
            data: List of {adapter, benchmarks: {name: accuracy}}

        Returns:
            Matplotlib figure
        """
        if not MATPLOTLIB_AVAILABLE or not data:
            return None

        # Extract benchmarks and adapters
        benchmarks = sorted({b for row in data for b in row.get("benchmarks", {})})
        adapters = [row["adapter"] for row in data]

        if not benchmarks:
            return None

        fig, ax = plt.subplots(figsize=self.figsize)

        x = np.arange(len(benchmarks))
        n_adapters = len(adapters)
        total_width = self.bar_width * n_adapters
        offset = -total_width / 2 + self.bar_width / 2

        for i, row in enumerate(data):
            adapter = row["adapter"]
            values = [row.get("benchmarks", {}).get(b, 0.0) for b in benchmarks]
            color = self.colors[i % len(self.colors)]

            ax.bar(
                x + offset + i * self.bar_width,
                values,
                self.bar_width,
                label=adapter,
                color=color,
                alpha=0.85,
            )

        ax.set_xlabel(self.xlabel, fontsize=12)
        ax.set_ylabel(self.ylabel, fontsize=12)
        ax.set_title(self.title, fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(self._format_labels(benchmarks), rotation=45, ha="right")
        ax.set_ylim(0, 1.0)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax.legend(loc="upper right", framealpha=0.9)
        ax.grid(axis="y", alpha=0.3)

        fig.tight_layout()
        return fig

    def _format_labels(self, labels: list[str]) -> list[str]:
        """Format benchmark labels."""
        replacements = {
            "longmemeval": "LongMemEval",
            "locomo": "LoCoMo",
            "memoryagentbench": "MemoryAgentBench",
            "contextbench": "Context-Bench",
            "terminalbench": "Terminal-Bench",
        }
        return [replacements.get(label.lower(), label) for label in labels]


@dataclass
class AblationHeatmap(FigureGenerator):
    """Generates heatmap showing ablation impact.

    Shows performance delta when each component is removed.
    """

    title: str = "Ablation Study Impact"
    cmap: str = "RdYlGn_r"  # Red for negative impact
    annot_fmt: str = ".1%"

    def generate(self, data: list[dict[str, Any]]) -> Any:
        """Generate ablation heatmap.

        Args:
            data: List of {ablation, delta, delta_pct, ...}

        Returns:
            Matplotlib figure
        """
        if not MATPLOTLIB_AVAILABLE or not data:
            return None

        ablations = [self._format_ablation(row["ablation"]) for row in data]
        deltas = [row["delta"] for row in data]

        fig, ax = plt.subplots(figsize=(8, max(4, len(ablations) * 0.6)))

        # Create horizontal bar chart styled like a heatmap
        colors = [
            plt.cm.RdYlGn_r((d + 0.5) / 1.0)  # Normalize around 0
            for d in deltas
        ]

        y_pos = np.arange(len(ablations))
        bars = ax.barh(y_pos, deltas, color=colors, alpha=0.85, height=0.6)

        # Add annotations
        for i, (_bar, delta) in enumerate(zip(bars, deltas, strict=False)):
            color = "white" if abs(delta) > 0.15 else "black"
            ax.annotate(
                f"{delta:+.1%}",
                xy=(delta, i),
                ha="left" if delta >= 0 else "right",
                va="center",
                fontsize=10,
                fontweight="bold",
                color=color,
            )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(ablations)
        ax.set_xlabel("Accuracy Change", fontsize=12)
        ax.set_title(self.title, fontsize=14, fontweight="bold")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlim(min(deltas) - 0.05, max(max(deltas) + 0.05, 0.05))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:+.0%}"))

        fig.tight_layout()
        return fig

    def _format_ablation(self, name: str) -> str:
        """Format ablation name."""
        replacements = {
            "no_semantic_search": "w/o Semantic Search",
            "no_metadata_filter": "w/o Metadata Filter",
            "no_version_history": "w/o Version History",
            "fixed_window": "Fixed Window Only",
            "recency_only": "Recency Only",
        }
        return replacements.get(name.lower(), name.replace("_", " ").title())


@dataclass
class CategoryRadarPlot(FigureGenerator):
    """Generates radar/spider plot for category breakdown.

    Shows performance across question categories for each adapter.
    """

    title: str = "Performance by Category"
    colors: list[str] = field(
        default_factory=lambda: [
            "#2ecc71",
            "#3498db",
            "#9b59b6",
        ]
    )
    fill_alpha: float = 0.25

    def generate(self, data: list[dict[str, Any]]) -> Any:
        """Generate radar plot.

        Args:
            data: List of {category, adapter1: acc, adapter2: acc, ...}

        Returns:
            Matplotlib figure
        """
        if not MATPLOTLIB_AVAILABLE or not data:
            return None

        categories = [row["category"] for row in data]
        adapters = sorted(k for k in data[0] if k != "category")

        if len(categories) < 3:
            logger.warning("Radar plot requires at least 3 categories")
            return None

        # Compute angles
        n_cats = len(categories)
        angles = [n / n_cats * 2 * np.pi for n in range(n_cats)]
        angles += angles[:1]  # Close the polygon

        fig, ax = plt.subplots(
            figsize=self.figsize,
            subplot_kw={"projection": "polar"},
        )

        for i, adapter in enumerate(adapters):
            values = [row.get(adapter, 0.0) for row in data]
            values += values[:1]  # Close the polygon

            color = self.colors[i % len(self.colors)]
            ax.plot(angles, values, "o-", linewidth=2, label=adapter, color=color)
            ax.fill(angles, values, alpha=self.fill_alpha, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(self._format_categories(categories), fontsize=10)
        ax.set_ylim(0, 1.0)
        ax.set_title(self.title, fontsize=14, fontweight="bold", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))

        fig.tight_layout()
        return fig

    def _format_categories(self, categories: list[str]) -> list[str]:
        """Format category labels."""
        # Shorten long category names
        return [c[:15] + "..." if len(c) > 15 else c for c in categories]


@dataclass
class ConfidenceIntervalPlot(FigureGenerator):
    """Generates plot showing accuracy with confidence intervals.

    Shows point estimate and 95% CI for each condition.
    """

    title: str = "Accuracy with 95% Confidence Intervals"
    color: str = "#3498db"
    marker_size: int = 100

    def generate(self, data: list[dict[str, Any]]) -> Any:
        """Generate CI plot.

        Args:
            data: List of {adapter, accuracy, ci_lower, ci_upper}

        Returns:
            Matplotlib figure
        """
        if not MATPLOTLIB_AVAILABLE or not data:
            return None

        adapters = [row["adapter"] for row in data]
        accuracies = [row["accuracy"] for row in data]
        ci_lower = [row.get("ci_lower", row["accuracy"]) for row in data]
        ci_upper = [row.get("ci_upper", row["accuracy"]) for row in data]

        fig, ax = plt.subplots(figsize=self.figsize)

        y_pos = np.arange(len(adapters))

        # Error bars
        errors = np.array(
            [
                [acc - lower for acc, lower in zip(accuracies, ci_lower, strict=False)],
                [upper - acc for acc, upper in zip(accuracies, ci_upper, strict=False)],
            ]
        )

        ax.errorbar(
            accuracies,
            y_pos,
            xerr=errors,
            fmt="none",
            ecolor="gray",
            elinewidth=2,
            capsize=5,
        )

        # Point estimates
        ax.scatter(
            accuracies,
            y_pos,
            s=self.marker_size,
            c=self.color,
            zorder=5,
        )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(adapters)
        ax.set_xlabel("Accuracy", fontsize=12)
        ax.set_title(self.title, fontsize=14, fontweight="bold")
        ax.set_xlim(0, 1.0)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
        ax.grid(axis="x", alpha=0.3)

        fig.tight_layout()
        return fig


@dataclass
class HumanAgreementPlot(FigureGenerator):
    """Generates plot showing human validation agreement.

    Shows agreement rate and Kappa for each benchmark.
    """

    title: str = "Human Validation Agreement"
    colors: dict[str, str] = field(
        default_factory=lambda: {
            "agreement": "#2ecc71",
            "kappa": "#3498db",
        }
    )

    def generate(self, data: list[dict[str, Any]]) -> Any:
        """Generate agreement plot.

        Args:
            data: List of {benchmark, agreement_rate, kappa}

        Returns:
            Matplotlib figure
        """
        if not MATPLOTLIB_AVAILABLE or not data:
            return None

        benchmarks = [row["benchmark"] for row in data]
        agreement = [row["agreement_rate"] for row in data]
        kappa = [row["kappa"] for row in data]

        fig, ax = plt.subplots(figsize=self.figsize)

        x = np.arange(len(benchmarks))
        width = 0.35

        ax.bar(
            x - width / 2,
            agreement,
            width,
            label="Agreement Rate",
            color=self.colors["agreement"],
            alpha=0.85,
        )
        ax.bar(
            x + width / 2,
            kappa,
            width,
            label="Cohen's κ",
            color=self.colors["kappa"],
            alpha=0.85,
        )

        ax.set_xlabel("Benchmark", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title(self.title, fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(benchmarks, rotation=45, ha="right")
        ax.set_ylim(0, 1.0)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        # Add reference lines for Kappa interpretation
        ax.axhline(0.8, color="gray", linestyle="--", alpha=0.5, linewidth=1)
        ax.axhline(0.6, color="gray", linestyle=":", alpha=0.5, linewidth=1)

        fig.tight_layout()
        return fig
