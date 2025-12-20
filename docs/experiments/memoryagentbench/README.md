# MemoryAgentBench

Evaluates memory coordination in multi-agent scenarios.

## Overview

MemoryAgentBench tests:
- Memory sharing between agents
- Conflict resolution in concurrent updates
- Coordination efficiency across agent boundaries

## Quick Start

```bash
# Single trial
uv run benchmark run memoryagentbench --adapter no_memory --trials 1

# Full experiment
uv run benchmark run memoryagentbench --adapter git_notes --trials 5
```

## Dataset

Synthetic multi-agent coordination scenarios:

| Scenario | Agents | Description |
|----------|--------|-------------|
| `handoff` | 2 | Sequential agent handoff |
| `collaboration` | 3 | Parallel collaboration |
| `conflict` | 2 | Conflicting memory updates |
| `hierarchy` | 4 | Hierarchical agent structure |

## Running Experiments

### Basic Usage

```bash
uv run benchmark run memoryagentbench \
  --adapter git_notes \
  --trials 5 \
  --output-dir results/memoryagentbench/
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--scenarios` | Specific scenarios | all |
| `--agent-count` | Number of simulated agents | varies |
| `--conflict-mode` | How to handle conflicts | `last_write_wins` |

## Output

### File Structure

```
results/memoryagentbench/
├── git_notes_trial_0.json
└── ...
```

### Metrics

| Metric | Description |
|--------|-------------|
| `coordination_accuracy` | Correct inter-agent communication |
| `memory_consistency` | State consistency across agents |
| `handoff_success` | Successful context handoffs |
| `conflict_resolution` | Properly resolved conflicts |

## Analysis

### Generate Report

```bash
uv run benchmark report \
  --results-dir results/memoryagentbench/ \
  --breakdown-by scenario
```

### Expected Results

| Adapter | Coordination | Consistency |
|---------|-------------|-------------|
| `git_notes` | ~85% | ~90% |
| `no_memory` | ~50% | ~40% |

## Performance

| Configuration | Time |
|--------------|------|
| 1 trial | ~15 minutes |
| 5 trials | ~1 hour |

Cost: ~$15-25 for 5 trials

## Code References

- Pipeline: `src/benchmarks/memoryagentbench/pipeline.py`
- Scenarios: `src/benchmarks/memoryagentbench/scenarios.py`

## See Also

- [Experiments Overview](../README.md)
- [Terminal-Bench](../terminalbench/README.md)
