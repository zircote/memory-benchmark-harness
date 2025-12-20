# LongMemEval Benchmark

Evaluates long-term memory capabilities using the LongMemEval dataset from HuggingFace.

## Overview

LongMemEval tests an agent's ability to:
- Remember information across long conversation histories
- Retrieve relevant context for answering questions
- Handle different question types (factual, temporal, multi-hop)

## Quick Start

```bash
# Single trial with no_memory baseline
uv run benchmark run longmemeval --adapter no_memory --trials 1

# Full experiment with git_notes
uv run benchmark run longmemeval --adapter git_notes --trials 5

# Compare adapters
uv run benchmark run longmemeval --adapter git_notes,no_memory --trials 5
```

## Dataset

**Source**: [HuggingFace: longmemeval](https://huggingface.co/datasets/longmemeval/longmemeval)

### Structure

```
Dataset:
├── sessions/           # Conversation histories
│   ├── session_1/
│   │   ├── messages[]  # List of messages
│   │   └── metadata    # Session context
│   └── ...
└── questions/          # Evaluation questions
    ├── question_id
    ├── question_text
    ├── ground_truth[]  # Acceptable answers
    ├── question_type   # Category
    └── relevant_session_ids[]
```

### Question Types

| Type | Description |
|------|-------------|
| `SINGLE_SESSION_USER` | Answer from single user session |
| `SINGLE_SESSION_ASSISTANT` | Answer from assistant response |
| `MULTI_SESSION` | Requires multiple sessions |
| `TEMPORAL` | Time-based reasoning |
| `AGGREGATION` | Summary across sessions |

### Subsets

| Subset | Questions | Sessions | Use Case |
|--------|-----------|----------|----------|
| `dev` | 100 | 50 | Development/debugging |
| `test` | 500 | 250 | Full evaluation |
| `sample` | 20 | 10 | Quick smoke test |

## Running Experiments

### Basic Usage

```bash
uv run benchmark run longmemeval \
  --adapter git_notes \
  --trials 5 \
  --output-dir results/longmemeval/
```

### Advanced Options

```bash
uv run benchmark run longmemeval \
  --adapter git_notes \
  --trials 5 \
  --subset test \
  --max-concurrent 5 \
  --batch-size 25 \
  --output-dir results/longmemeval/ \
  --skip-cache false
```

### Available Adapters

| Adapter | Description |
|---------|-------------|
| `git_notes` | Git-notes based memory system |
| `no_memory` | Baseline without memory |
| `no_semantic_search` | Ablation: no semantic retrieval |
| `no_metadata_filter` | Ablation: no metadata filtering |
| `recency_only` | Ablation: recency-based retrieval |

## Output

### File Structure

```
results/longmemeval/
├── git_notes_trial_0.json
├── git_notes_trial_1.json
├── git_notes_trial_2.json
├── git_notes_trial_3.json
├── git_notes_trial_4.json
├── no_memory_trial_0.json
└── ...
```

### Result Schema

```json
{
  "benchmark": "longmemeval",
  "adapter": "git_notes",
  "trial": 0,
  "timestamp": "2025-12-20T10:30:00Z",
  "metrics": {
    "accuracy": 0.72,
    "precision": 0.75,
    "recall": 0.70,
    "f1_score": 0.72
  },
  "timing": {
    "total_seconds": 3600,
    "questions_per_second": 0.14
  },
  "questions": [
    {
      "question_id": "q_001",
      "question_text": "What restaurant did Alice recommend?",
      "ground_truth": ["Luigi's Italian", "Luigi's"],
      "model_answer": "Alice recommended Luigi's Italian restaurant.",
      "judgment": {
        "result": "correct",
        "score": 1.0,
        "reasoning": "..."
      }
    }
  ]
}
```

## Analysis

### Generate Statistics

```bash
uv run benchmark report \
  --results-dir results/longmemeval/ \
  --output-format json,latex,csv
```

### Compare Adapters

```bash
uv run benchmark compare \
  --results-dir results/longmemeval/ \
  --adapters git_notes,no_memory \
  --output analysis/longmemeval_comparison.md
```

### Expected Results

| Adapter | Accuracy (95% CI) | Notes |
|---------|------------------|-------|
| `git_notes` | ~70-80% | With semantic search |
| `no_memory` | ~40-50% | Context-window only |

*Results may vary based on LLM non-determinism.*

## Performance Considerations

### Time Estimates

| Configuration | Estimated Time |
|--------------|----------------|
| 1 trial, dev subset | ~20 minutes |
| 5 trials, dev subset | ~2 hours |
| 5 trials, test subset | ~10 hours |

### Cost Estimates

| Component | Cost per Trial |
|-----------|---------------|
| LLM Judge (GPT-4o) | ~$5-10 |
| Embeddings | ~$0.50 |
| **Total (5 trials)** | ~$30-50 |

## Troubleshooting

### Dataset Download Slow

```bash
# Pre-download the dataset
uv run python -c "
from src.benchmarks.longmemeval import load_dataset
load_dataset()
"
```

### Rate Limiting

```bash
# Reduce concurrency
uv run benchmark run longmemeval \
  --adapter git_notes \
  --max-concurrent 2
```

### Memory Issues

```bash
# Reduce batch size
uv run benchmark run longmemeval \
  --adapter git_notes \
  --batch-size 10
```

## Code References

- Dataset loader: `src/benchmarks/longmemeval/dataset.py`
- Pipeline: `src/benchmarks/longmemeval/pipeline.py`
- Question types: `src/benchmarks/longmemeval/dataset.py:QuestionType`

## See Also

- [Experiments Overview](../README.md)
- [LoCoMo Benchmark](../locomo/README.md)
- [Ablations Documentation](../ablations/README.md)
