# Terminal-Bench

Evaluates agent performance on terminal-based tasks with long-running contexts.

## Overview

Terminal-Bench tests:
- Command execution and output interpretation
- Session state management
- Error recovery and debugging workflows

**Requires Docker** for sandboxed command execution.

## Prerequisites

```bash
# Verify Docker is available
docker --version

# Add user to docker group (if needed)
sudo usermod -aG docker $USER
newgrp docker
```

## Quick Start

```bash
# Single trial
uv run benchmark run terminalbench --adapter no_memory --trials 1

# Full experiment
uv run benchmark run terminalbench --adapter git_notes --trials 5
```

## Dataset

Terminal-based task scenarios:

| Scenario | Description |
|----------|-------------|
| `file_manipulation` | Create, edit, move files |
| `process_management` | Start/stop processes |
| `debugging` | Interpret error messages |
| `configuration` | System configuration tasks |
| `git_operations` | Git workflow tasks |

## Running Experiments

### Basic Usage

```bash
uv run benchmark run terminalbench \
  --adapter git_notes \
  --trials 5 \
  --output-dir results/terminalbench/
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--docker-image` | Container image | `ubuntu:22.04` |
| `--timeout` | Task timeout (seconds) | `300` |
| `--scenarios` | Specific scenarios | all |

## Output

### File Structure

```
results/terminalbench/
├── git_notes_trial_0.json
└── ...
```

### Metrics

| Metric | Description |
|--------|-------------|
| `task_completion` | Successfully completed tasks |
| `command_accuracy` | Correct commands issued |
| `state_tracking` | Accurate session state tracking |
| `error_recovery` | Recovery from errors |

## Analysis

### Generate Report

```bash
uv run benchmark report \
  --results-dir results/terminalbench/ \
  --breakdown-by scenario
```

### Expected Results

| Adapter | Completion | Accuracy |
|---------|-----------|----------|
| `git_notes` | ~75% | ~80% |
| `no_memory` | ~45% | ~50% |

## Performance

| Configuration | Time |
|--------------|------|
| 1 trial | ~45 minutes |
| 5 trials | ~4 hours |

Cost: ~$60-80 for 5 trials (longer LLM interactions)

## Troubleshooting

### Docker Permission Denied

```bash
sudo usermod -aG docker $USER
newgrp docker
```

### Container Timeout

```bash
# Increase timeout
uv run benchmark run terminalbench \
  --adapter git_notes \
  --timeout 600
```

### Docker Not Found

Ensure Docker Desktop or Docker Engine is installed and running.

## Code References

- Pipeline: `src/benchmarks/terminalbench/pipeline.py`
- Docker runner: `src/benchmarks/terminalbench/docker_runner.py`
- Scenarios: `src/benchmarks/terminalbench/scenarios.py`

## See Also

- [Experiments Overview](../README.md)
- [Ablations](../ablations/README.md)
