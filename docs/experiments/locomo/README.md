# LoCoMo Benchmark

Evaluates longitudinal conversation memory using the LoCoMo (Longitudinal Conversation Memory) dataset.

## Overview

LoCoMo tests an agent's ability to:
- Maintain coherent memory across extended time periods
- Handle temporal references and updates
- Manage evolving entity relationships over time

## Quick Start

```bash
# Single trial baseline
uv run benchmark run locomo --adapter no_memory --trials 1

# Full experiment
uv run benchmark run locomo --adapter git_notes --trials 5

# Compare adapters
uv run benchmark run locomo --adapter git_notes,no_memory --trials 5
```

## Dataset

**Source**: [HuggingFace: locomo](https://huggingface.co/datasets/locomo/locomo)

### Structure

```
Dataset:
├── conversations/       # Long-form conversations
│   ├── conv_id
│   ├── turns[]         # Conversation turns
│   ├── timestamps[]    # Temporal markers
│   └── entities{}      # Entity tracking
└── questions/
    ├── question_id
    ├── question_text
    ├── ground_truth
    ├── temporal_reference
    └── required_context[]
```

### Question Categories

| Category | Description |
|----------|-------------|
| `current_state` | Current state of an entity |
| `temporal_evolution` | How something changed over time |
| `relationship` | Entity relationships |
| `aggregate` | Summary across time periods |

## Running Experiments

### Basic Usage

```bash
uv run benchmark run locomo \
  --adapter git_notes \
  --trials 5 \
  --output-dir results/locomo/
```

### Advanced Options

```bash
uv run benchmark run locomo \
  --adapter git_notes \
  --trials 5 \
  --max-concurrent 5 \
  --batch-size 25 \
  --temporal-window 30d \
  --output-dir results/locomo/
```

## Output

### File Structure

```
results/locomo/
├── git_notes_trial_0.json
├── git_notes_trial_1.json
└── ...
```

### Result Schema

```json
{
  "benchmark": "locomo",
  "adapter": "git_notes",
  "trial": 0,
  "timestamp": "2025-12-20T10:30:00Z",
  "metrics": {
    "accuracy": 0.68,
    "temporal_accuracy": 0.72,
    "entity_consistency": 0.85
  },
  "questions": [...]
}
```

## Analysis

### Generate Statistics

```bash
uv run benchmark report \
  --results-dir results/locomo/ \
  --output-format json,latex
```

### Expected Results

| Adapter | Accuracy (95% CI) |
|---------|------------------|
| `git_notes` | ~65-75% |
| `no_memory` | ~35-45% |

## Performance Considerations

### Time Estimates

| Configuration | Estimated Time |
|--------------|----------------|
| 1 trial | ~30 minutes |
| 5 trials | ~3 hours |

### Cost Estimates

~$40-60 total for 5 trials

## Troubleshooting

### Dataset Download

```bash
uv run python -c "
from src.benchmarks.locomo import load_dataset
load_dataset()
"
```

## Code References

- Dataset loader: `src/benchmarks/locomo/dataset.py`
- Pipeline: `src/benchmarks/locomo/pipeline.py`

## See Also

- [Experiments Overview](../README.md)
- [LongMemEval Benchmark](../longmemeval/README.md)
