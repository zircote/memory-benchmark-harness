"""MemoryAgentBench benchmark implementation.

This module provides the dataset loader, agent wrapper, evaluation pipeline,
and metrics calculation for the MemoryAgentBench benchmark.

Dataset source: ai-hyz/MemoryAgentBench on HuggingFace
Paper: https://arxiv.org/abs/2507.05257

The benchmark evaluates four core competencies:
1. Accurate Retrieval (AR) - Single/multi-hop information retrieval
2. Test-Time Learning (TTL) - Learning from examples during interaction
3. Long-Range Understanding (LRU) - Forming global cognition from long conversations
4. Conflict Resolution (CR) - Identifying and updating outdated information
"""

from src.benchmarks.memoryagentbench.dataset import (
    Competency,
    MemoryAgentBenchDataset,
    MemoryAgentBenchQuestion,
    MemoryAgentBenchSplit,
    load_memoryagentbench,
    load_memoryagentbench_from_file,
)
from src.benchmarks.memoryagentbench.metrics import (
    CompetencyMetrics,
    MemoryAgentBenchMetrics,
    MetricsCalculator,
)
from src.benchmarks.memoryagentbench.pipeline import (
    CompetencyResult,
    MemoryAgentBenchPipeline,
    SplitResult,
)
from src.benchmarks.memoryagentbench.wrapper import (
    MemoryAgentBenchAgent,
)

__all__ = [
    # Dataset
    "Competency",
    "MemoryAgentBenchDataset",
    "MemoryAgentBenchQuestion",
    "MemoryAgentBenchSplit",
    "load_memoryagentbench",
    "load_memoryagentbench_from_file",
    # Wrapper
    "MemoryAgentBenchAgent",
    # Pipeline
    "CompetencyResult",
    "MemoryAgentBenchPipeline",
    "SplitResult",
    # Metrics
    "CompetencyMetrics",
    "MemoryAgentBenchMetrics",
    "MetricsCalculator",
]
