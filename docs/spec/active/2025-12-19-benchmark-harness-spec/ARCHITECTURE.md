---
document_type: architecture
project_id: SPEC-2025-12-19-001
version: 1.0.0
last_updated: 2025-12-19T19:00:00Z
status: approved
---

# Benchmark Harness - Technical Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Benchmark Harness                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐             │
│  │   LongMemEval   │  │     LoCoMo      │  │  Context-Bench  │             │
│  │     Wrapper     │  │     Wrapper     │  │     Wrapper     │             │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘             │
│           │                    │                    │                       │
│  ┌────────┴────────────────────┴────────────────────┴────────┐             │
│  │                    MemorySystemAdapter                     │             │
│  │              (Unified Abstract Base Class)                 │             │
│  └────────────────────────┬───────────────────────────────────┘             │
│                           │                                                  │
│           ┌───────────────┴───────────────┐                                 │
│           ▼                               ▼                                 │
│  ┌─────────────────────┐        ┌─────────────────────┐                    │
│  │  GitNotesAdapter    │        │  NoMemoryAdapter    │                    │
│  │  (with-memory)      │        │  (baseline)         │                    │
│  └─────────────────────┘        └─────────────────────┘                    │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                          Evaluation Layer                                   │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌───────────────────┐   │
│  │   LLM-as-Judge      │  │   Statistics        │  │  Human Validation │   │
│  │   (GPT-4o cached)   │  │   (Bootstrap CI)    │  │  (Sample Export)  │   │
│  └─────────────────────┘  └─────────────────────┘  └───────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. MemorySystemAdapter (Base Class)

The unified interface for all memory systems, implementing ADR-005.

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class MemoryItem:
    """Single memory item returned from search."""
    memory_id: str
    content: str
    metadata: dict
    score: float  # Relevance score (0.0 - 1.0)
    created_at: datetime
    updated_at: Optional[datetime] = None


@dataclass
class MemoryOperationResult:
    """Result of a memory operation."""
    success: bool
    memory_id: Optional[str] = None
    error: Optional[str] = None


class MemorySystemAdapter(ABC):
    """
    Abstract base class for memory system adapters.

    All benchmark-specific wrappers delegate to this interface,
    enabling consistent testing and baseline comparison.
    """

    @abstractmethod
    def add(self, content: str, metadata: Optional[dict] = None) -> MemoryOperationResult:
        """
        Add a new memory entry.

        Args:
            content: The memory content to store
            metadata: Optional metadata (timestamps, session_id, tags)

        Returns:
            MemoryOperationResult with the assigned memory_id
        """
        ...

    @abstractmethod
    def search(
        self,
        query: str,
        limit: int = 10,
        min_score: float = 0.0,
        metadata_filter: Optional[dict] = None
    ) -> list[MemoryItem]:
        """
        Search memories by semantic similarity.

        Args:
            query: The search query
            limit: Maximum number of results
            min_score: Minimum relevance score threshold
            metadata_filter: Optional metadata constraints

        Returns:
            List of MemoryItem ordered by relevance
        """
        ...

    @abstractmethod
    def update(self, memory_id: str, content: str, metadata: Optional[dict] = None) -> MemoryOperationResult:
        """
        Update an existing memory entry.

        Args:
            memory_id: The ID of the memory to update
            content: New content
            metadata: Optional updated metadata

        Returns:
            MemoryOperationResult indicating success/failure
        """
        ...

    @abstractmethod
    def delete(self, memory_id: str) -> MemoryOperationResult:
        """
        Delete a memory entry.

        Args:
            memory_id: The ID of the memory to delete

        Returns:
            MemoryOperationResult indicating success/failure
        """
        ...

    @abstractmethod
    def clear(self) -> MemoryOperationResult:
        """
        Clear all memories (for test isolation).

        Returns:
            MemoryOperationResult indicating success/failure
        """
        ...

    @abstractmethod
    def get_stats(self) -> dict:
        """
        Get adapter statistics (memory count, storage size, etc.).

        Returns:
            Dictionary with adapter-specific statistics
        """
        ...
```

### 2. GitNotesAdapter

Implements the memory adapter for `git-notes-memory-manager`.

```python
from git_notes_memory import MemoryManager


class GitNotesAdapter(MemorySystemAdapter):
    """
    Adapter wrapping git-notes-memory-manager plugin.

    Uses git notes for persistent storage and sqlite-vec for embeddings.
    """

    def __init__(self, repo_path: str, model: str = "bge-m3"):
        """
        Initialize the git-notes memory system.

        Args:
            repo_path: Path to git repository for notes storage
            model: Embedding model name (default: bge-m3)
        """
        self._manager = MemoryManager(repo_path, embedding_model=model)

    def add(self, content: str, metadata: Optional[dict] = None) -> MemoryOperationResult:
        try:
            memory_id = self._manager.add(content, metadata or {})
            return MemoryOperationResult(success=True, memory_id=memory_id)
        except Exception as e:
            return MemoryOperationResult(success=False, error=str(e))

    def search(
        self,
        query: str,
        limit: int = 10,
        min_score: float = 0.0,
        metadata_filter: Optional[dict] = None
    ) -> list[MemoryItem]:
        results = self._manager.search(query, k=limit)
        return [
            MemoryItem(
                memory_id=r.id,
                content=r.content,
                metadata=r.metadata,
                score=r.score,
                created_at=r.created_at,
                updated_at=r.updated_at
            )
            for r in results
            if r.score >= min_score
        ]

    # ... remaining methods follow same pattern
```

### 3. NoMemoryAdapter (Baseline)

Implements a no-retrieval baseline per ADR-012.

```python
class NoMemoryAdapter(MemorySystemAdapter):
    """
    No-memory baseline adapter.

    Stores memories but search always returns empty.
    Used to measure the benefit of memory retrieval.
    """

    def __init__(self):
        self._memories: list[tuple[str, str, dict]] = []
        self._counter = 0

    def add(self, content: str, metadata: Optional[dict] = None) -> MemoryOperationResult:
        self._counter += 1
        memory_id = f"baseline_{self._counter}"
        self._memories.append((memory_id, content, metadata or {}))
        return MemoryOperationResult(success=True, memory_id=memory_id)

    def search(
        self,
        query: str,
        limit: int = 10,
        min_score: float = 0.0,
        metadata_filter: Optional[dict] = None
    ) -> list[MemoryItem]:
        # Baseline: never retrieve memories
        return []

    def clear(self) -> MemoryOperationResult:
        self._memories.clear()
        self._counter = 0
        return MemoryOperationResult(success=True)

    def get_stats(self) -> dict:
        return {
            "memory_count": len(self._memories),
            "type": "no-memory-baseline"
        }
```

### 4. Benchmark Wrapper Pattern

Each benchmark has unique interface requirements. Wrappers adapt the unified interface.

```python
# LongMemEval Wrapper
class LongMemEvalAgent:
    """Wrapper adapting MemorySystemAdapter to LongMemEval interface."""

    def __init__(self, adapter: MemorySystemAdapter, llm_client):
        self._adapter = adapter
        self._llm = llm_client

    def ingest_history(self, session_id: str, messages: list[dict]) -> None:
        """Ingest conversation history into memory."""
        for msg in messages:
            self._adapter.add(
                content=f"{msg['role']}: {msg['content']}",
                metadata={
                    "session_id": session_id,
                    "role": msg["role"],
                    "timestamp": msg.get("timestamp")
                }
            )

    def answer_question(self, question: str, session_id: str) -> str:
        """Answer a question using retrieved memories."""
        memories = self._adapter.search(
            query=question,
            limit=10,
            metadata_filter={"session_id": session_id}
        )

        context = "\n\n".join([m.content for m in memories])

        response = self._llm.complete(
            system="Answer based on the conversation history.",
            messages=[
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
            ]
        )
        return response.content


# LoCoMo Wrapper
class LoCoMoAgent:
    """Wrapper adapting MemorySystemAdapter to LoCoMo interface."""

    def __init__(self, adapter: MemorySystemAdapter, llm_client):
        self._adapter = adapter
        self._llm = llm_client

    def process_conversation(self, conversation_id: str, turns: list[dict]) -> None:
        """Process and store conversation turns."""
        for turn in turns:
            self._adapter.add(
                content=turn["text"],
                metadata={
                    "conversation_id": conversation_id,
                    "speaker": turn["speaker"],
                    "turn_id": turn["turn_id"]
                }
            )

    def answer_qa(self, question: str, conversation_id: str, category: str) -> str:
        """Answer QA question for specific category."""
        memories = self._adapter.search(
            query=question,
            limit=15,
            metadata_filter={"conversation_id": conversation_id}
        )

        context = "\n".join([f"[{m.metadata['speaker']}]: {m.content}" for m in memories])

        return self._llm.complete(
            system=f"Answer the {category} question from the conversation.",
            messages=[{"role": "user", "content": f"Conversation:\n{context}\n\nQuestion: {question}"}]
        ).content
```

### 5. LLM-as-Judge with Caching

Implements ADR-007 for reliable and efficient evaluation.

```python
import hashlib
import json
import time
from pathlib import Path
from dataclasses import dataclass
from enum import Enum


class JudgmentResult(Enum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    PARTIALLY_CORRECT = "partially_correct"


@dataclass
class CachedJudgment:
    result: JudgmentResult
    confidence: float
    reasoning: str
    model_version: str
    cached_at: float


class LLMJudge:
    """
    GPT-4o based answer evaluation with caching and retry.

    Implements ADR-007:
    - Cache key: hash(question + response + judge_model_version)
    - Retry: exponential backoff (1s, 2s, 4s, 8s, max 60s)
    - Max retries: 5
    - Cache TTL: 30 days
    """

    CACHE_TTL_DAYS = 30
    MAX_RETRIES = 5

    def __init__(
        self,
        openai_client,
        cache_dir: Path,
        model: str = "gpt-4o"
    ):
        self._client = openai_client
        self._cache_dir = cache_dir
        self._model = model
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(self, question: str, response: str, reference: str) -> str:
        """Generate deterministic cache key."""
        content = json.dumps({
            "question": question,
            "response": response,
            "reference": reference,
            "model": self._model
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _get_cached(self, cache_key: str) -> Optional[CachedJudgment]:
        """Retrieve cached judgment if valid."""
        cache_file = self._cache_dir / f"{cache_key}.json"
        if not cache_file.exists():
            return None

        with open(cache_file) as f:
            data = json.load(f)

        # Check TTL
        age_days = (time.time() - data["cached_at"]) / 86400
        if age_days > self.CACHE_TTL_DAYS:
            cache_file.unlink()
            return None

        return CachedJudgment(
            result=JudgmentResult(data["result"]),
            confidence=data["confidence"],
            reasoning=data["reasoning"],
            model_version=data["model_version"],
            cached_at=data["cached_at"]
        )

    def _save_cache(self, cache_key: str, judgment: CachedJudgment) -> None:
        """Save judgment to cache."""
        cache_file = self._cache_dir / f"{cache_key}.json"
        with open(cache_file, "w") as f:
            json.dump({
                "result": judgment.result.value,
                "confidence": judgment.confidence,
                "reasoning": judgment.reasoning,
                "model_version": judgment.model_version,
                "cached_at": judgment.cached_at
            }, f)

    def judge(
        self,
        question: str,
        response: str,
        reference: str
    ) -> CachedJudgment:
        """
        Evaluate response correctness with caching and retry.

        Args:
            question: The original question
            response: Model's answer to evaluate
            reference: Ground truth reference answer

        Returns:
            CachedJudgment with result and reasoning
        """
        cache_key = self._cache_key(question, response, reference)

        # Check cache
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        # Exponential backoff retry
        for attempt in range(self.MAX_RETRIES):
            try:
                result = self._call_judge(question, response, reference)
                self._save_cache(cache_key, result)
                return result
            except Exception as e:
                if attempt == self.MAX_RETRIES - 1:
                    raise
                wait = min(2 ** attempt, 60)  # 1, 2, 4, 8, ..., max 60
                time.sleep(wait)

        raise RuntimeError("Failed to get judgment after max retries")

    def _call_judge(
        self,
        question: str,
        response: str,
        reference: str
    ) -> CachedJudgment:
        """Make actual API call to GPT-4o."""
        prompt = f"""Evaluate if the response correctly answers the question.

Question: {question}

Reference Answer: {reference}

Model Response: {response}

Respond in JSON:
{{
    "result": "correct" | "incorrect" | "partially_correct",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}}
"""
        completion = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )

        data = json.loads(completion.choices[0].message.content)

        return CachedJudgment(
            result=JudgmentResult(data["result"]),
            confidence=data["confidence"],
            reasoning=data["reasoning"],
            model_version=self._model,
            cached_at=time.time()
        )
```

### 6. Statistical Analysis Module

Implements ADR-003, ADR-010 for rigorous statistical analysis.

```python
import numpy as np
from scipy import stats
from dataclasses import dataclass
from typing import Callable


@dataclass
class ConfidenceInterval:
    """Bootstrap confidence interval result."""
    mean: float
    lower: float
    upper: float
    std_error: float
    n_iterations: int


@dataclass
class ComparisonResult:
    """Statistical comparison between two conditions."""
    effect_size: float  # Cohen's d
    p_value: float
    p_value_corrected: float  # After Holm-Bonferroni
    is_significant: bool
    ci_difference: ConfidenceInterval


class StatisticalAnalyzer:
    """
    Statistical analysis with bootstrap CI and Holm-Bonferroni correction.

    Implements:
    - ADR-003: Statistical significance (p < 0.05) as primary criterion
    - ADR-010: Holm-Bonferroni for multiple comparison control
    """

    def __init__(
        self,
        n_bootstrap: int = 2000,
        confidence_level: float = 0.95,
        random_seed: int = 42
    ):
        self._n_bootstrap = n_bootstrap
        self._confidence = confidence_level
        self._rng = np.random.default_rng(random_seed)

    def bootstrap_ci(
        self,
        data: np.ndarray,
        statistic: Callable = np.mean
    ) -> ConfidenceInterval:
        """
        Compute BCa (Bias-Corrected and Accelerated) bootstrap CI.

        Args:
            data: Sample data
            statistic: Function to compute statistic (default: mean)

        Returns:
            ConfidenceInterval with bounds
        """
        # Bootstrap resampling
        bootstrap_stats = np.array([
            statistic(self._rng.choice(data, size=len(data), replace=True))
            for _ in range(self._n_bootstrap)
        ])

        # BCa correction
        alpha = 1 - self._confidence
        original_stat = statistic(data)

        # Bias correction factor
        z0 = stats.norm.ppf(np.mean(bootstrap_stats < original_stat))

        # Acceleration factor (jackknife)
        jackknife_stats = np.array([
            statistic(np.delete(data, i))
            for i in range(len(data))
        ])
        jackknife_mean = np.mean(jackknife_stats)
        num = np.sum((jackknife_mean - jackknife_stats) ** 3)
        den = 6 * np.sum((jackknife_mean - jackknife_stats) ** 2) ** 1.5
        a = num / den if den != 0 else 0

        # Adjusted percentiles
        z_lower = stats.norm.ppf(alpha / 2)
        z_upper = stats.norm.ppf(1 - alpha / 2)

        lower_pct = stats.norm.cdf(z0 + (z0 + z_lower) / (1 - a * (z0 + z_lower)))
        upper_pct = stats.norm.cdf(z0 + (z0 + z_upper) / (1 - a * (z0 + z_upper)))

        lower_bound = np.percentile(bootstrap_stats, lower_pct * 100)
        upper_bound = np.percentile(bootstrap_stats, upper_pct * 100)

        return ConfidenceInterval(
            mean=original_stat,
            lower=lower_bound,
            upper=upper_bound,
            std_error=np.std(bootstrap_stats),
            n_iterations=self._n_bootstrap
        )

    def paired_comparison(
        self,
        memory_scores: np.ndarray,
        baseline_scores: np.ndarray
    ) -> ComparisonResult:
        """
        Compare memory system vs baseline with paired test.

        Args:
            memory_scores: Scores for memory condition
            baseline_scores: Scores for no-memory baseline

        Returns:
            ComparisonResult with effect size and p-value
        """
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(memory_scores, baseline_scores)

        # Cohen's d for paired samples
        diff = memory_scores - baseline_scores
        cohens_d = np.mean(diff) / np.std(diff, ddof=1)

        # CI on the difference
        diff_ci = self.bootstrap_ci(diff)

        return ComparisonResult(
            effect_size=cohens_d,
            p_value=p_value,
            p_value_corrected=p_value,  # Corrected in batch
            is_significant=p_value < 0.05,
            ci_difference=diff_ci
        )

    def holm_bonferroni_correction(
        self,
        p_values: list[float]
    ) -> list[float]:
        """
        Apply Holm-Bonferroni step-down correction.

        Implements ADR-010 for FWER control.

        Args:
            p_values: List of raw p-values

        Returns:
            List of corrected p-values
        """
        n = len(p_values)
        sorted_indices = np.argsort(p_values)
        sorted_p = np.array(p_values)[sorted_indices]

        corrected = np.zeros(n)
        for i, (idx, p) in enumerate(zip(sorted_indices, sorted_p)):
            corrected[idx] = min(1.0, p * (n - i))

        # Enforce monotonicity
        for i in range(1, n):
            corrected[sorted_indices[i]] = max(
                corrected[sorted_indices[i]],
                corrected[sorted_indices[i - 1]]
            )

        return corrected.tolist()
```

## Directory Structure

```
memory-benchmark-harness/
├── src/
│   ├── adapters/
│   │   ├── __init__.py
│   │   ├── base.py              # MemorySystemAdapter ABC
│   │   ├── git_notes.py         # GitNotesAdapter
│   │   ├── no_memory.py         # NoMemoryAdapter
│   │   └── mock.py              # MockAdapter for testing
│   │
│   ├── benchmarks/
│   │   ├── __init__.py
│   │   ├── longmemeval/
│   │   │   ├── __init__.py
│   │   │   ├── wrapper.py       # LongMemEvalAgent
│   │   │   ├── pipeline.py      # Evaluation orchestration
│   │   │   └── metrics.py       # LongMemEval-specific metrics
│   │   │
│   │   ├── locomo/
│   │   │   ├── __init__.py
│   │   │   ├── wrapper.py       # LoCoMoAgent
│   │   │   ├── pipeline.py
│   │   │   └── categories.py    # 5 QA categories
│   │   │
│   │   ├── context_bench/
│   │   │   ├── __init__.py
│   │   │   ├── wrapper.py       # Letta Benchmark integration
│   │   │   └── pipeline.py
│   │   │
│   │   ├── memory_agent_bench/
│   │   │   ├── __init__.py
│   │   │   ├── wrapper.py       # 4 competencies
│   │   │   └── conflict.py      # Conflict resolution focus
│   │   │
│   │   └── terminal_bench/
│   │       ├── __init__.py
│   │       ├── agent.py         # AbstractInstalledAgent
│   │       └── tasks.py         # Memory-relevant task subset
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── judge.py             # LLMJudge
│   │   ├── statistics.py        # StatisticalAnalyzer
│   │   ├── human_export.py      # Human validation export
│   │   └── ablation.py          # LOCO ablation framework
│   │
│   ├── reporting/
│   │   ├── __init__.py
│   │   ├── tables.py            # LaTeX table generation
│   │   ├── figures.py           # Matplotlib figures
│   │   └── publication.py       # arXiv formatting
│   │
│   └── cli/
│       ├── __init__.py
│       └── main.py              # Click CLI entrypoint
│
├── tests/
│   ├── unit/
│   │   ├── test_adapters.py
│   │   ├── test_judge.py
│   │   └── test_statistics.py
│   │
│   ├── integration/
│   │   ├── test_longmemeval.py
│   │   ├── test_locomo.py
│   │   └── test_full_pipeline.py
│   │
│   └── fixtures/
│       ├── sample_conversations.json
│       └── mock_memories.json
│
├── docker/
│   ├── Dockerfile
│   ├── Dockerfile.gpu
│   └── docker-compose.yml
│
├── data/
│   └── .gitkeep               # Downloaded datasets go here
│
├── results/
│   └── .gitkeep               # Evaluation results
│
├── pyproject.toml
├── uv.lock
└── README.md
```

## Docker Architecture

### Base Image (CPU)

```dockerfile
# docker/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/

# Copy project files
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

COPY src/ ./src/
COPY tests/ ./tests/

ENTRYPOINT ["uv", "run", "python", "-m", "src.cli.main"]
```

### GPU Image

```dockerfile
# docker/Dockerfile.gpu
FROM nvidia/cuda:12.2-runtime-ubuntu22.04

WORKDIR /app

# Install Python and uv
RUN apt-get update && apt-get install -y python3.11 python3.11-venv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/

# Copy and install
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

COPY src/ ./src/

ENTRYPOINT ["uv", "run", "python", "-m", "src.cli.main"]
```

### Docker Compose

```yaml
# docker/docker-compose.yml
version: '3.8'

services:
  benchmark:
    build:
      context: ..
      dockerfile: docker/Dockerfile.gpu
    volumes:
      - ../data:/app/data
      - ../results:/app/results
      - ~/.cache/huggingface:/root/.cache/huggingface
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - HF_HOME=/root/.cache/huggingface
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    command: ["run", "--benchmark", "all"]

  benchmark-lite:
    extends: benchmark
    command: ["run", "--benchmark", "lite", "--samples", "10"]
```

## Configuration

### Environment Variables

```bash
# Required
OPENAI_API_KEY=sk-...        # For LLM-as-Judge
HF_HOME=/path/to/cache       # HuggingFace cache

# Optional
ANTHROPIC_API_KEY=sk-...     # For Terminal-Bench (Claude)
CUDA_VISIBLE_DEVICES=0       # GPU selection
BENCHMARK_SEED=42            # Random seed
BENCHMARK_CACHE_TTL=30       # Judge cache TTL (days)
```

### Configuration File

```yaml
# config/benchmark.yaml
adapters:
  git_notes:
    embedding_model: bge-m3
    repo_path: /tmp/benchmark-repo

benchmarks:
  longmemeval:
    enabled: true
    subset: ["S", "M"]  # Skip "L" for faster runs
  locomo:
    enabled: true
    categories: all
  context_bench:
    enabled: true
  memory_agent_bench:
    enabled: true
    focus: conflict_resolution
  terminal_bench:
    enabled: true
    task_subset: memory_relevant

evaluation:
  judge:
    model: gpt-4o
    cache_ttl_days: 30
    max_retries: 5
  statistics:
    n_bootstrap: 2000
    confidence_level: 0.95
    n_runs: 5
  human_validation:
    samples_per_benchmark: 100

output:
  results_dir: ./results
  latex_tables: true
  figures: true
```

## Data Flow

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Dataset     │────▶│  Adapter     │────▶│  Benchmark   │
│  (HF/GitHub) │     │  (Memory)    │     │  Wrapper     │
└──────────────┘     └──────────────┘     └──────────────┘
                                                  │
                                                  ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Results     │◀────│  Statistics  │◀────│  LLM Judge   │
│  (JSON/CSV)  │     │  Analysis    │     │  (GPT-4o)    │
└──────────────┘     └──────────────┘     └──────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────┐
│  Publication Artifacts                                │
│  - LaTeX Tables (with CI)                            │
│  - Matplotlib Figures                                 │
│  - Human Validation Export                            │
└──────────────────────────────────────────────────────┘
```

## Integration Points

### git-notes-memory-manager

```python
# Expected API (per ADR-002)
from git_notes_memory import MemoryManager

manager = MemoryManager(repo_path="/path/to/repo")

# Add memory
memory_id = manager.add(
    content="User prefers dark mode",
    metadata={"session": "abc123", "type": "preference"}
)

# Search memories
results = manager.search(
    query="user interface preferences",
    k=10
)
for result in results:
    print(f"{result.id}: {result.content} (score: {result.score})")

# Update memory
manager.update(memory_id, content="User prefers light mode")

# Delete memory
manager.delete(memory_id)
```

### HuggingFace Datasets

```python
from datasets import load_dataset

# LongMemEval
longmemeval = load_dataset("xiaowu0162/longmemeval-cleaned")

# MemoryAgentBench
mab = load_dataset("ai-hyz/MemoryAgentBench")
```

### Letta/Context-Bench

```python
from letta_evals import Benchmark

benchmark = Benchmark(
    agent=our_agent,
    memory_system=git_notes_adapter
)
results = benchmark.run()
```

## Security Considerations

1. **API Key Management**: All API keys via environment variables, never in code
2. **Docker Isolation**: Benchmarks run in containers with minimal privileges
3. **Cache Security**: Judge cache stored locally, not committed to git
4. **Data Privacy**: Test datasets are public academic benchmarks only

## Performance Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| Full suite | < 48 hours | Single GPU run |
| Lite benchmark | < 30 minutes | CI validation |
| Judge cache hit | > 80% | Reruns |
| Memory search | < 100ms | P95 latency |
| Embedding generation | < 50ms | Per document |

## Appendix: Sequence Diagrams

### Benchmark Execution Flow

```
User                 CLI              Adapter          Benchmark          Judge
  │                   │                  │                │                 │
  │  run benchmark    │                  │                │                 │
  ├──────────────────▶│                  │                │                 │
  │                   │  init adapter    │                │                 │
  │                   ├─────────────────▶│                │                 │
  │                   │                  │                │                 │
  │                   │  load dataset    │                │                 │
  │                   ├──────────────────────────────────▶│                 │
  │                   │                  │                │                 │
  │                   │                  │  ingest memories                 │
  │                   │                  │◀───────────────┤                 │
  │                   │                  │                │                 │
  │                   │                  │  answer question                 │
  │                   │                  │◀───────────────┤                 │
  │                   │                  │                │                 │
  │                   │                  │                │  evaluate       │
  │                   │                  │                ├────────────────▶│
  │                   │                  │                │                 │
  │                   │                  │                │  judgment       │
  │                   │                  │                │◀────────────────┤
  │                   │                  │                │                 │
  │  results          │                  │                │                 │
  │◀──────────────────┤                  │                │                 │
```
