# Context-Bench

Evaluates context utilization efficiency and retrieval accuracy.

## Overview

Context-Bench tests:
- How effectively agents utilize available context
- Retrieval precision and recall
- Context window optimization

## Quick Start

```bash
# Single trial
uv run benchmark run contextbench --adapter no_memory --trials 1

# Full experiment
uv run benchmark run contextbench --adapter git_notes --trials 5
```

## Dataset

Context-Bench uses synthetic scenarios designed to test specific retrieval patterns:

| Scenario | Description |
|----------|-------------|
| `needle_in_haystack` | Find specific information in large context |
| `multi_hop` | Chain reasoning across multiple documents |
| `temporal_ordering` | Reason about time-ordered events |
| `entity_tracking` | Track entity state changes |

## Running Experiments

### Basic Usage

```bash
uv run benchmark run contextbench \
  --adapter git_notes \
  --trials 5 \
  --output-dir results/contextbench/
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--context-sizes` | Context sizes to test | `4k,8k,16k,32k` |
| `--scenarios` | Specific scenarios | all |
| `--retrieval-limit` | Max retrieved chunks | `10` |

## Output

### File Structure

```
results/contextbench/
├── git_notes_trial_0.json
└── ...
```

### Metrics

| Metric | Description |
|--------|-------------|
| `retrieval_precision` | Relevant items / retrieved items |
| `retrieval_recall` | Retrieved relevant / total relevant |
| `context_utilization` | Effective use of retrieved context |
| `answer_accuracy` | Correctness of final answers |

## Analysis

### Generate Report

```bash
uv run benchmark report \
  --results-dir results/contextbench/ \
  --breakdown-by scenario
```

### Expected Results

| Adapter | Precision | Recall |
|---------|-----------|--------|
| `git_notes` | ~80% | ~75% |
| `no_memory` | N/A | N/A |

## Performance

| Configuration | Time |
|--------------|------|
| 1 trial | ~15 minutes |
| 5 trials | ~1 hour |

Cost: ~$15-25 for 5 trials

## Code References

- Pipeline: `src/benchmarks/contextbench/pipeline.py`
- Scenarios: `src/benchmarks/contextbench/scenarios.py`

## See Also

- [Experiments Overview](../README.md)
- [MemoryAgentBench](../memoryagentbench/README.md)
