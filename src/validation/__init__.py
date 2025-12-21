"""Human validation infrastructure for benchmark evaluation.

This module provides tools for collecting and analyzing human validation
of LLM-generated answers, including:
- Sample compilation from multiple benchmarks
- Annotation guidelines and rubrics
- Validation data collection and analysis
- Inter-annotator agreement calculation
"""

from __future__ import annotations

from src.validation.analysis import (
    AgreementMetrics,
    ValidationAnalyzer,
    ValidationReport,
)
from src.validation.annotation import (
    AnnotationGuidelines,
    AnnotationRubric,
    RubricLevel,
    create_default_guidelines,
)
from src.validation.collector import (
    AnnotatedSample,
    AnnotationSession,
    ValidationCollector,
)
from src.validation.compiler import (
    CompiledSample,
    SampleCompiler,
    SourceBenchmark,
)

__all__ = [
    # Annotation
    "AnnotationGuidelines",
    "AnnotationRubric",
    "RubricLevel",
    "create_default_guidelines",
    # Collector
    "AnnotatedSample",
    "AnnotationSession",
    "ValidationCollector",
    # Compiler
    "CompiledSample",
    "SampleCompiler",
    "SourceBenchmark",
    # Analysis
    "AgreementMetrics",
    "ValidationAnalyzer",
    "ValidationReport",
]
