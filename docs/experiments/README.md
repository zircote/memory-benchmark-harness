# Experiment Documentation

This directory contains comprehensive documentation for running experiments with the Memory Benchmark Harness.

## Quick Start

```bash
# 1. Validate the setup
uv run python scripts/minimal_validation.py

# 2. Run all benchmarks
uv run benchmark run-all --adapters git_notes,no_memory

# 3. Generate analysis
uv run benchmark report --results-dir results/
```

## Experiment Categories

| Category | Directory | Description |
|----------|-----------|-------------|
| [Validation](./validation/) | `validation/` | API and setup verification |
| [LongMemEval](./longmemeval/) | `longmemeval/` | Long-term memory evaluation |
| [LoCoMo](./locomo/) | `locomo/` | Longitudinal conversation memory |
| [Context-Bench](./contextbench/) | `contextbench/` | Context utilization evaluation |
| [MemoryAgentBench](./memoryagentbench/) | `memoryagentbench/` | Multi-agent memory coordination |
| [Terminal-Bench](./terminalbench/) | `terminalbench/` | Terminal-based task evaluation |
| [Ablations](./ablations/) | `ablations/` | Component contribution analysis |

## Prerequisites

### Environment Variables

```bash
export OPENAI_API_KEY="your-api-key"  # Required for LLM Judge
```

### Dependencies

```bash
uv sync  # Install all dependencies
```

### Optional: Git-Notes Memory Manager

```bash
pip install git-notes-memory-manager  # For git_notes adapter
```

## Workflow Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Experiment Pipeline                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   1. VALIDATION                                              │
│   ├── API connectivity check                                 │
│   ├── LLM Judge functionality test                          │
│   └── Component integration verification                     │
│                                                              │
│   2. BENCHMARK EXECUTION                                     │
│   ├── Load dataset (HuggingFace or local)                   │
│   ├── Initialize memory adapter                              │
│   ├── Process each question:                                 │
│   │   ├── Populate memory with context                       │
│   │   ├── Query memory for relevant information              │
│   │   ├── Generate answer using LLM                          │
│   │   └── Judge answer correctness                           │
│   └── Save results to JSON                                   │
│                                                              │
│   3. ANALYSIS                                                │
│   ├── Compute accuracy metrics                               │
│   ├── Bootstrap confidence intervals                         │
│   ├── Statistical significance tests                         │
│   └── Generate publication artifacts                         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Results Structure

```
results/
├── validation_test/
│   └── minimal_validation.json
├── longmemeval/
│   ├── git_notes_trial_0.json
│   ├── git_notes_trial_1.json
│   └── ...
├── locomo/
│   └── ...
├── contextbench/
│   └── ...
├── memoryagentbench/
│   └── ...
├── terminalbench/
│   └── ...
└── ablations/
    ├── no_semantic_search/
    ├── no_metadata_filter/
    └── ...
```

## Common Options

All benchmark commands support these options:

| Option | Description | Default |
|--------|-------------|---------|
| `--adapter` | Memory system adapter | `git_notes` |
| `--trials` | Number of experimental trials | `5` |
| `--max-concurrent` | Maximum concurrent API calls | `10` |
| `--batch-size` | Batch size for processing | `50` |
| `--output-dir` | Results output directory | `results/` |
| `--skip-cache` | Bypass judgment cache | `false` |

## Troubleshooting

See individual experiment documentation for specific troubleshooting guides.

### Common Issues

1. **API Rate Limits**: Reduce `--max-concurrent`
2. **Out of Memory**: Reduce `--batch-size`
3. **Slow Downloads**: Pre-download datasets (see individual docs)
4. **Docker Errors**: For Terminal-Bench, ensure Docker is running

## See Also

- [Reproducibility Package](../publication/REPRODUCIBILITY.md) - Full reproducibility instructions
- [GPU Setup Guide](../GPU_RUNNER_SETUP.md) - GPU configuration for embeddings
- [Git-Notes API](../GIT_NOTES_API.md) - Memory adapter interface
