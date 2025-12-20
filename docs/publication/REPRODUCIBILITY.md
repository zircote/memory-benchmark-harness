# Reproducibility Package

This document describes how to reproduce the experimental results presented in our paper.

## Requirements

### Hardware

- **CPU**: 8+ cores recommended for parallel benchmark execution
- **RAM**: 16GB minimum, 32GB recommended
- **GPU**: Optional, improves embedding computation speed
- **Storage**: 50GB free space for datasets and results

### Software

- Python 3.11+
- Git 2.30+
- Docker (optional, for Terminal-Bench tasks)
- OpenAI API key (for GPT-4o judge)

## Quick Start

```bash
# Clone the repository
git clone https://github.com/zircote/memory-benchmark-harness
cd memory-benchmark-harness

# Install dependencies
uv sync

# Set API keys
export OPENAI_API_KEY="your-key-here"

# Run all benchmarks
uv run benchmark run-all --adapters git_notes,no_memory
```

## Detailed Instructions

### 1. Environment Setup

```bash
# Create virtual environment
uv venv
source .venv/bin/activate

# Install all dependencies including dev
uv sync --all-extras

# Verify installation
uv run pytest tests/ -v --tb=short
```

### 2. Dataset Preparation

Datasets are automatically downloaded on first use. To pre-download:

```bash
# Download LongMemEval
uv run python -c "from src.benchmarks.longmemeval import load_dataset; load_dataset()"

# Download LoCoMo
uv run python -c "from src.benchmarks.locomo import load_dataset; load_dataset()"
```

### 3. Running Individual Benchmarks

```bash
# LongMemEval
uv run benchmark run longmemeval --adapter git_notes --trials 5

# LoCoMo
uv run benchmark run locomo --adapter git_notes --trials 5

# Context-Bench
uv run benchmark run contextbench --adapter git_notes --trials 5

# MemoryAgentBench
uv run benchmark run memoryagentbench --adapter git_notes --trials 5

# Terminal-Bench (requires Docker)
uv run benchmark run terminalbench --adapter git_notes --trials 5
```

### 4. Running Ablation Studies

```bash
# All ablations
uv run benchmark run-ablations --benchmark longmemeval --trials 3

# Specific ablation
uv run benchmark run longmemeval --adapter no_semantic_search --trials 3
```

### 5. Statistical Analysis

```bash
# Generate comparison report
uv run benchmark compare --results-dir results/ --output analysis/

# Bootstrap confidence intervals
uv run benchmark report --results-dir results/ --bootstrap-n 2000
```

### 6. Human Validation

```bash
# Export validation samples
uv run benchmark export-samples --results-dir results/ --n-samples 200 --output validation/

# After annotation, compute agreement
uv run benchmark compute-agreement --annotations validation/annotated.json
```

## Expected Results

After running all experiments, you should have:

```
results/
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

## Result Verification

### Checksums

After running experiments, verify result integrity:

```bash
# Generate checksums
sha256sum results/**/*.json > checksums.txt

# Verify against reference
diff checksums.txt docs/publication/reference_checksums.txt
```

### Statistical Bounds

Results should fall within these ranges (95% CI):

| Benchmark | git_notes Accuracy | no_memory Accuracy |
|-----------|-------------------|-------------------|
| LongMemEval | XX.X% ± Y.Y% | XX.X% ± Y.Y% |
| LoCoMo | XX.X% ± Y.Y% | XX.X% ± Y.Y% |
| Context-Bench | XX.X% ± Y.Y% | XX.X% ± Y.Y% |
| MemoryAgentBench | XX.X% ± Y.Y% | XX.X% ± Y.Y% |
| Terminal-Bench | XX.X% ± Y.Y% | XX.X% ± Y.Y% |

*Note: Results may vary slightly due to LLM non-determinism. We report averages across 5 trials.*

## Troubleshooting

### Common Issues

**API Rate Limits**

```bash
# Reduce concurrency
uv run benchmark run longmemeval --adapter git_notes --max-concurrent 2
```

**Out of Memory**

```bash
# Process in smaller batches
uv run benchmark run longmemeval --adapter git_notes --batch-size 10
```

**Docker Permission Errors**

```bash
# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker
```

### Cache Management

```bash
# Clear judgment cache (forces re-evaluation)
rm -rf .cache/judgments/

# Keep cache (recommended for reproducibility)
# Cache is content-addressed and should produce identical results
```

## Resource Estimates

### Time

| Benchmark | Approximate Time (5 trials) |
|-----------|----------------------------|
| LongMemEval | ~2 hours |
| LoCoMo | ~3 hours |
| Context-Bench | ~1 hour |
| MemoryAgentBench | ~1 hour |
| Terminal-Bench | ~4 hours |
| Ablations (5x) | ~10 hours |
| **Total** | **~21 hours** |

### Cost

| Component | Estimated Cost |
|-----------|----------------|
| OpenAI API (judgments) | ~$50-100 |
| OpenAI API (embeddings) | ~$5-10 |
| Compute (if cloud) | ~$20-50 |
| **Total** | **~$75-160** |

## Configuration Files

### `config/experiment.yaml`

```yaml
# Experiment configuration
experiments:
  trials: 5
  random_seed: 42

benchmarks:
  - longmemeval
  - locomo
  - contextbench
  - memoryagentbench
  - terminalbench

adapters:
  primary:
    - git_notes
    - no_memory
  ablations:
    - no_semantic_search
    - no_metadata_filter
    - no_version_history
    - fixed_window
    - recency_only

statistics:
  bootstrap_iterations: 2000
  confidence_level: 0.95
  correction: holm_bonferroni
```

### `config/judge.yaml`

```yaml
# LLM Judge configuration
judge:
  model: gpt-4o
  temperature: 0.0
  max_retries: 5

cache:
  enabled: true
  ttl_days: 30
  algorithm: sha256
```

## Citation

If you use this benchmark harness, please cite:

```bibtex
@article{author2024gitnotes,
  title={Git-Notes Memory: A Git-Based Memory System for LLM Agents},
  author={Author Name},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```

## Contact

For questions about reproducibility:
- Open an issue: https://github.com/zircote/memory-benchmark-harness/issues
- Email: [contact email]
