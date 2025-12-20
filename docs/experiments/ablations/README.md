# Ablation Studies

Measures the contribution of individual components to overall system performance.

## Overview

Ablation studies systematically disable components to understand:
- Which features contribute most to performance
- Trade-offs between complexity and accuracy
- Minimal viable configuration for different use cases

## Quick Start

```bash
# Run all ablations on LongMemEval
uv run benchmark run-ablations --benchmark longmemeval --trials 3

# Run specific ablation
uv run benchmark run longmemeval --adapter no_semantic_search --trials 3
```

## Ablation Configurations

| Adapter | Disabled Feature | Description |
|---------|-----------------|-------------|
| `no_semantic_search` | Embedding retrieval | Keyword matching only |
| `no_metadata_filter` | Metadata filtering | No temporal/type filters |
| `no_version_history` | Version tracking | No edit history |
| `fixed_window` | Dynamic windowing | Fixed context window |
| `recency_only` | Relevance scoring | Most recent only |

## Running Ablations

### All Ablations

```bash
uv run benchmark run-ablations \
  --benchmark longmemeval \
  --trials 3 \
  --output-dir results/ablations/
```

### Specific Ablation

```bash
uv run benchmark run longmemeval \
  --adapter no_semantic_search \
  --trials 3 \
  --output-dir results/ablations/no_semantic_search/
```

### Cross-Benchmark Ablations

```bash
# Run same ablation across multiple benchmarks
for bm in longmemeval locomo contextbench; do
  uv run benchmark run $bm \
    --adapter no_semantic_search \
    --trials 3 \
    --output-dir results/ablations/no_semantic_search/
done
```

## Output

### File Structure

```
results/ablations/
├── no_semantic_search/
│   ├── longmemeval_trial_0.json
│   ├── longmemeval_trial_1.json
│   └── ...
├── no_metadata_filter/
│   └── ...
├── no_version_history/
│   └── ...
├── fixed_window/
│   └── ...
└── recency_only/
    └── ...
```

### Result Schema

```json
{
  "benchmark": "longmemeval",
  "adapter": "no_semantic_search",
  "ablation": {
    "disabled_feature": "semantic_search",
    "baseline_adapter": "git_notes"
  },
  "metrics": {
    "accuracy": 0.58,
    "delta_from_baseline": -0.14
  }
}
```

## Analysis

### Compare All Ablations

```bash
uv run benchmark compare-ablations \
  --results-dir results/ablations/ \
  --baseline git_notes \
  --output analysis/ablation_report.md
```

### Expected Results (LongMemEval)

| Configuration | Accuracy | Δ from Baseline |
|---------------|----------|-----------------|
| `git_notes` (baseline) | ~72% | — |
| `no_semantic_search` | ~58% | -14% |
| `no_metadata_filter` | ~65% | -7% |
| `no_version_history` | ~70% | -2% |
| `fixed_window` | ~62% | -10% |
| `recency_only` | ~55% | -17% |

### Statistical Significance

```bash
uv run benchmark report \
  --results-dir results/ablations/ \
  --significance-test mcnemar \
  --baseline git_notes
```

## Interpreting Results

### Feature Importance

Larger accuracy drops indicate more important features:

1. **High Impact** (>10% drop): Core functionality
2. **Medium Impact** (5-10% drop): Important but not critical
3. **Low Impact** (<5% drop): Nice-to-have features

### Trade-off Analysis

```
┌──────────────────────────────────────────────────────────────┐
│                    Feature vs. Performance                    │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  Accuracy │                                    ┌─────────────│
│    80% ───┤                                    │ git_notes   │
│           │                          ┌─────────┤             │
│    70% ───┤                          │ no_ver  │             │
│           │            ┌─────────────┤         │             │
│    60% ───┤ ┌──────────┤ no_meta     │         │             │
│           │ │ no_sem   │             │         │             │
│    50% ───┤ │          │             │         │             │
│           │ │          │             │         │             │
│    40% ───┼─┴──────────┴─────────────┴─────────┴─────────────│
│           Low          ←  Complexity  →           High       │
└──────────────────────────────────────────────────────────────┘
```

## Performance

### Time Estimates

| Scope | Estimated Time |
|-------|---------------|
| Single ablation (3 trials) | ~1 hour |
| All ablations (5 configs × 3 trials) | ~5 hours |
| Full ablation suite (5 benchmarks) | ~25 hours |

### Cost Estimates

~$10-20 per ablation configuration (3 trials)
~$50-100 for full ablation suite

## Advanced Usage

### Custom Ablation

Create a custom ablation adapter:

```python
from src.adapters.base import MemorySystemAdapter
from src.adapters.git_notes import GitNotesAdapter

class CustomAblation(GitNotesAdapter):
    """Example: Disable both semantic search and metadata filtering."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.use_semantic_search = False
        self.use_metadata_filter = False
```

Register in `src/adapters/__init__.py`:

```python
ADAPTERS["custom_ablation"] = CustomAblation
```

## Code References

- Adapter base: `src/adapters/base.py`
- Ablation configs: `src/adapters/ablations.py`
- Analysis: `src/analysis/ablation_analysis.py`

## See Also

- [Experiments Overview](../README.md)
- [LongMemEval](../longmemeval/README.md)
- [Publication Artifacts](../../publication/REPRODUCIBILITY.md)
