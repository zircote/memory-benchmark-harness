# Experiment Documentation

This directory contains comprehensive documentation for running experiments with the Memory Benchmark Harness.

## Quick Start

```bash
# 1. Validate the setup
uv run python scripts/minimal_validation.py

# 2. Run benchmarks (each benchmark runs separately)
uv run benchmark run longmemeval --adapter git-notes,no-memory --trials 5 --output results/
uv run benchmark run locomo --adapter git-notes,no-memory --trials 5 --output results/

# 3. Generate reports
uv run benchmark report results/exp_longmemeval_*.json --output results/reports/
uv run benchmark report results/exp_locomo_*.json --output results/reports/

# 4. Generate publication artifacts (tables, figures, statistics)
uv run benchmark publication all results/ --output results/publication/
```

## Experiment Categories

| Category                                | Directory           | Description                      |
| --------------------------------------- | ------------------- | -------------------------------- |
| [Validation](./validation/)             | `validation/`       | API and setup verification       |
| [LongMemEval](./longmemeval/)           | `longmemeval/`      | Long-term memory evaluation      |
| [LoCoMo](./locomo/)                     | `locomo/`           | Longitudinal conversation memory |
| [Context-Bench](./contextbench/)        | `contextbench/`     | Context utilization evaluation   |
| [MemoryAgentBench](./memoryagentbench/) | `memoryagentbench/` | Multi-agent memory coordination  |
| [Terminal-Bench](./terminalbench/)      | `terminalbench/`    | Terminal-based task evaluation   |
| [Ablations](./ablations/)               | `ablations/`        | Component contribution analysis  |

## Prerequisites

### Environment Variables

```bash
export OPENAI_API_KEY="your-api-key"  # pragma: allowlist secret - Required for LLM Judge
```

### Dependencies

```bash
uv sync  # Install all dependencies
```

### Optional: Git-Notes Memory Manager

```bash
pip install git-notes-memory-manager  # For git-notes adapter
```

## Workflow Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Experiment Pipeline                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   1. VALIDATION                                             │
│   ├── API connectivity check                                │
│   ├── LLM Judge functionality test                          │
│   └── Component integration verification                    │
│                                                             │
│   2. BENCHMARK EXECUTION                                    │
│   ├── Load dataset (HuggingFace or local)                   │
│   ├── Initialize memory adapter                             │
│   ├── Process each question:                                │
│   │   ├── Populate memory with context                      │
│   │   ├── Query memory for relevant information             │
│   │   ├── Generate answer using LLM                         │
│   │   └── Judge answer correctness                          │
│   └── Save results to JSON                                  │
│                                                             │
│   3. ANALYSIS                                               │
│   ├── Compute accuracy metrics                              │
│   ├── Bootstrap confidence intervals                        │
│   ├── Statistical significance tests                        │
│   └── Generate publication artifacts                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Results Structure

```
results/
├── exp_longmemeval_<uuid>.json      # Experiment results (all trials, all conditions)
├── exp_locomo_<uuid>.json
├── reports/
│   ├── summary_exp_longmemeval_<uuid>.md
│   ├── summary_exp_longmemeval_<uuid>.json
│   └── ...
├── publication/
│   ├── publication_statistics.json   # Aggregated metrics with CIs
│   ├── tables/
│   │   ├── main_results.tex          # LaTeX tables
│   │   ├── main_results.md           # Markdown tables
│   │   ├── ablation.tex
│   │   └── category_breakdown.tex
│   └── figures/
│       ├── performance_comparison.pdf
│       ├── performance_comparison.png
│       ├── ablation_impact.pdf
│       └── confidence_intervals.pdf
└── validation/
    ├── longmemeval_samples.csv       # Human annotation samples
    ├── locomo_samples.csv
    └── phase2_validation_samples.json
```

## Common Options

The `benchmark run` command supports these options:

| Option      | Short | Description                                                   | Default     |
| ----------- | ----- | ------------------------------------------------------------- | ----------- |
| `--adapter` | `-a`  | Memory adapter(s): `git-notes`, `no-memory`, `mock`           | `no-memory` |
| `--trials`  | `-n`  | Number of trials per condition                                | `5`         |
| `--seed`    | `-s`  | Base random seed for reproducibility                          | `42`        |
| `--output`  | `-o`  | Directory to save results                                     | `results`   |
| `--dataset` | `-d`  | Path to dataset file (uses default download if not specified) | -           |
| `--quiet`   | `-q`  | Suppress progress output                                      | `false`     |

### Available Adapters

| Adapter     | Description                                           |
| ----------- | ----------------------------------------------------- |
| `git-notes` | GitNotesAdapter using git-notes-memory-manager        |
| `no-memory` | NoMemoryAdapter baseline (stores but never retrieves) |
| `mock`      | MockAdapter for testing with configurable behavior    |

### Available Benchmarks

| Benchmark     | Description                                                     |
| ------------- | --------------------------------------------------------------- |
| `longmemeval` | Long-term memory QA with factoid, reasoning, temporal questions |
| `locomo`      | Long conversational memory with 5 QA categories                 |

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
