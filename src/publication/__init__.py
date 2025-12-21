"""Publication artifacts generation.

This module provides tools for generating publication-ready artifacts
including tables, figures, and statistical summaries.
"""

from __future__ import annotations

from src.publication.figures import (
    AblationHeatmap,
    CategoryRadarPlot,
    FigureGenerator,
    PerformanceBarChart,
)
from src.publication.statistics import (
    BenchmarkSummary,
    PublicationStatistics,
    UnifiedMetrics,
)
from src.publication.tables import (
    AblationTable,
    CategoryBreakdownTable,
    MainResultsTable,
    TableGenerator,
)

__all__ = [
    # Tables
    "TableGenerator",
    "MainResultsTable",
    "AblationTable",
    "CategoryBreakdownTable",
    # Figures
    "FigureGenerator",
    "PerformanceBarChart",
    "AblationHeatmap",
    "CategoryRadarPlot",
    # Statistics
    "PublicationStatistics",
    "UnifiedMetrics",
    "BenchmarkSummary",
]
