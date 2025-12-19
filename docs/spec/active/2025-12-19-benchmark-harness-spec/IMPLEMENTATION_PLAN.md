---
document_type: implementation_plan
project_id: SPEC-2025-12-19-001
version: 1.0.0
last_updated: 2025-12-19T19:30:00Z
status: draft
---

# Benchmark Harness - Implementation Plan

## Overview

This document provides a detailed task breakdown for the benchmark harness implementation across four phases. Each phase has explicit gate criteria that must be met before proceeding.

**Total Estimated Duration**: 14-18 weeks
- Phase 0: 2 weeks (Infrastructure)
- Phase 1: 4-6 weeks (Core Evaluation)
- Phase 2: 4-5 weeks (Extended Evaluation)
- Phase 3: 4-5 weeks (Real-World Validation)

---

## Phase 0: Infrastructure Foundation (2 weeks)

**Gate Criteria:**
1. [ ] MemorySystemAdapter ABC defined with full type hints
2. [ ] GitNotesAdapter passes integration tests with git-notes-memory-manager
3. [ ] NoMemoryAdapter passes unit tests
4. [ ] MockAdapter available for test isolation
5. [ ] CI pipeline runs on every PR
6. [ ] Docker base image builds successfully

### Task 0.1: Project Setup

- [ ] **0.1.1** Initialize repository with `uv init`
  - Create `pyproject.toml` with Python 3.11+ requirement
  - Add development dependencies (pytest, mypy, ruff)
  - Configure `uv.lock` for reproducibility

- [ ] **0.1.2** Create directory structure
  - `src/adapters/`, `src/benchmarks/`, `src/evaluation/`, `src/reporting/`, `src/cli/`
  - `tests/unit/`, `tests/integration/`, `tests/fixtures/`
  - `docker/`, `data/`, `results/`, `config/`

- [ ] **0.1.3** Configure development tooling
  - `ruff` for linting and formatting
  - `mypy` for type checking
  - `pytest` with coverage reporting
  - Pre-commit hooks

### Task 0.2: MemorySystemAdapter Base Class

- [ ] **0.2.1** Define `MemorySystemAdapter` ABC in `src/adapters/base.py`
  - `add(content, metadata) -> MemoryOperationResult`
  - `search(query, limit, min_score, metadata_filter) -> list[MemoryItem]`
  - `update(memory_id, content, metadata) -> MemoryOperationResult`
  - `delete(memory_id) -> MemoryOperationResult`
  - `clear() -> MemoryOperationResult`
  - `get_stats() -> dict`

- [ ] **0.2.2** Define data classes
  - `MemoryItem` dataclass
  - `MemoryOperationResult` dataclass
  - Type aliases and protocols

- [ ] **0.2.3** Write unit tests for base class contract
  - Test abstract method signatures
  - Test dataclass serialization
  - Test type hints with mypy

### Task 0.3: GitNotesAdapter Implementation

- [ ] **0.3.1** Document git-notes-memory-manager API
  - Export current API signatures
  - Document initialization parameters
  - Document expected return types

- [ ] **0.3.2** Implement `GitNotesAdapter` in `src/adapters/git_notes.py`
  - Import and wrap `git_notes_memory.MemoryManager`
  - Implement all 6 adapter methods
  - Handle exceptions and return `MemoryOperationResult`

- [ ] **0.3.3** Write integration tests
  - Test with real git repository
  - Test add/search/update/delete cycle
  - Test error handling for invalid operations

### Task 0.4: NoMemoryAdapter Implementation

- [ ] **0.4.1** Implement `NoMemoryAdapter` in `src/adapters/no_memory.py`
  - Store memories in-memory list
  - `search()` always returns empty list
  - Track statistics for validation

- [ ] **0.4.2** Write unit tests
  - Verify memories are stored
  - Verify search returns empty
  - Verify clear works correctly

### Task 0.5: MockAdapter for Testing

- [ ] **0.5.1** Implement `MockAdapter` in `src/adapters/mock.py`
  - Configurable search results
  - Operation recording for assertions
  - Latency simulation

- [ ] **0.5.2** Write test utilities
  - Fixture factories
  - Sample memory data generators
  - Assertion helpers

### Task 0.6: Docker Infrastructure

- [ ] **0.6.1** Create CPU Dockerfile
  - Base Python 3.11 image
  - Install uv and dependencies
  - Configure entrypoint

- [ ] **0.6.2** Create GPU Dockerfile
  - NVIDIA CUDA 12.2 base image
  - Same dependency chain as CPU
  - GPU passthrough configuration

- [ ] **0.6.3** Create docker-compose.yml
  - `benchmark` service with GPU
  - `benchmark-lite` service for CI
  - Volume mounts for data and results

- [ ] **0.6.4** Test Docker builds
  - Build both images
  - Run smoke tests in containers
  - Verify GPU access in GPU image

### Task 0.7: CI/CD Pipeline

- [ ] **0.7.1** Create GitHub Actions workflow
  - Lint and type check on every PR
  - Unit tests on every PR
  - Integration tests on merge to main

- [ ] **0.7.2** Configure self-hosted GPU runner documentation
  - RunPod setup instructions
  - Lambda Labs setup instructions
  - GitHub Actions runner installation

---

## Phase 1: Core Evaluation (4-6 weeks)

**Gate Criteria:**
1. [ ] LongMemEval pipeline produces results for S and M subsets
2. [ ] LoCoMo pipeline produces results for all 5 QA categories
3. [ ] LLM-as-Judge achieves >80% cache hit rate on reruns
4. [ ] Bootstrap CI computed with 2000 iterations
5. [ ] 5 runs completed per benchmark
6. [ ] 200 samples exported for human validation

### Task 1.1: LLM-as-Judge Implementation

- [ ] **1.1.1** Implement `LLMJudge` class in `src/evaluation/judge.py`
  - OpenAI GPT-4o API integration
  - JSON response parsing
  - Judgment result dataclass

- [ ] **1.1.2** Implement caching layer
  - Content-addressed cache keys
  - File-based cache storage
  - TTL expiration (30 days default)

- [ ] **1.1.3** Implement retry logic
  - Exponential backoff (1s, 2s, 4s, 8s, max 60s)
  - Max 5 retries per judgment
  - Error classification

- [ ] **1.1.4** Write tests for judge
  - Mock API responses
  - Test cache hit/miss scenarios
  - Test retry behavior

### Task 1.2: Statistical Analysis Module

- [ ] **1.2.1** Implement `StatisticalAnalyzer` in `src/evaluation/statistics.py`
  - BCa bootstrap confidence intervals
  - Paired t-test for comparisons
  - Cohen's d effect size

- [ ] **1.2.2** Implement Holm-Bonferroni correction
  - Step-down procedure
  - Monotonicity enforcement
  - Integration with comparison results

- [ ] **1.2.3** Write statistical tests
  - Test against known distributions
  - Verify CI coverage
  - Test correction math

### Task 1.3: LongMemEval Pipeline

- [ ] **1.3.1** Download and parse LongMemEval dataset
  - `xiaowu0162/longmemeval-cleaned` from HuggingFace
  - Parse S and M subsets
  - Extract session histories and questions

- [ ] **1.3.2** Implement `LongMemEvalAgent` wrapper
  - `ingest_history()` method
  - `answer_question()` method
  - Memory retrieval integration

- [ ] **1.3.3** Implement evaluation pipeline
  - Run memory ingestion
  - Answer all questions
  - Collect judgments

- [ ] **1.3.4** Implement metrics calculation
  - Accuracy per memory ability (5 abilities)
  - Aggregate accuracy
  - Per-subset breakdown

- [ ] **1.3.5** Run LongMemEval experiments
  - 5 runs with different seeds
  - Both git-notes and no-memory conditions
  - Record all raw results

### Task 1.4: LoCoMo Pipeline

- [ ] **1.4.1** Download and parse LoCoMo dataset
  - Clone `snap-research/locomo` repository
  - Parse 10 conversations
  - Extract 5 QA categories

- [ ] **1.4.2** Implement `LoCoMoAgent` wrapper
  - `process_conversation()` method
  - `answer_qa()` method per category
  - Handle long conversations

- [ ] **1.4.3** Implement evaluation pipeline
  - Process each conversation
  - Answer QA across categories
  - Judge answers

- [ ] **1.4.4** Implement metrics calculation
  - Per-category accuracy
  - Aggregate accuracy
  - Conversation difficulty analysis

- [ ] **1.4.5** Run LoCoMo experiments
  - 5 runs with different seeds
  - Both conditions
  - Record all raw results

### Task 1.5: Phase 1 Analysis

- [ ] **1.5.1** Compute bootstrap confidence intervals
  - 95% CI for all metrics
  - BCa correction applied
  - Per-benchmark and aggregate

- [ ] **1.5.2** Compute statistical comparisons
  - Paired tests for each benchmark
  - Effect sizes (Cohen's d)
  - Holm-Bonferroni correction

- [ ] **1.5.3** Export human validation samples
  - 100 samples from LongMemEval
  - 100 samples from LoCoMo
  - Include question, reference, response, judgment

- [ ] **1.5.4** Generate preliminary results
  - Summary statistics table
  - Accuracy bar charts with CI
  - Raw data export (JSON)

---

## Phase 2: Extended Evaluation (4-5 weeks)

**Gate Criteria:**
1. [ ] Context-Bench evaluation complete with Letta integration
2. [ ] MemoryAgentBench evaluation complete for all 4 competencies
3. [ ] Conflict resolution results demonstrate git version advantage
4. [ ] Ablation study framework operational
5. [ ] 200 additional samples exported for human validation

### Task 2.1: Context-Bench Integration

- [ ] **2.1.1** Clone and analyze letta-evals repository
  - `letta-ai/letta-evals` from GitHub
  - Understand Benchmark class interface
  - Identify integration points

- [ ] **2.1.2** Implement Context-Bench wrapper
  - Adapt MemorySystemAdapter to Letta interface
  - Implement agent configuration
  - Handle context window management

- [ ] **2.1.3** Run Context-Bench evaluation
  - Both memory conditions
  - 5 runs per condition
  - Collect all metrics

### Task 2.2: MemoryAgentBench Integration

- [ ] **2.2.1** Download and parse MemoryAgentBench
  - `ai-hyz/MemoryAgentBench` from HuggingFace
  - Parse 4 competencies
  - Focus on Conflict Resolution tasks

- [ ] **2.2.2** Implement competency evaluation
  - Knowledge Acquisition
  - Preference Consistency
  - Knowledge Update
  - **Conflict Resolution** (primary focus)

- [ ] **2.2.3** Implement git version history integration
  - Query git notes history for conflicts
  - Present version timeline to agent
  - Evaluate resolution accuracy

- [ ] **2.2.4** Run MemoryAgentBench evaluation
  - Both conditions
  - 5 runs per condition
  - Focus analysis on Conflict Resolution

### Task 2.3: Ablation Study Framework

- [ ] **2.3.1** Design LOCO ablation structure
  - Identify components to ablate
  - Define ablation adapter variants
  - Plan p-value computation

- [ ] **2.3.2** Implement ablation adapters
  - No semantic search (random retrieval)
  - No metadata filtering
  - No version history

- [ ] **2.3.3** Run ablation studies
  - One benchmark (suggest LoCoMo)
  - All ablation conditions
  - Compute contribution percentages

### Task 2.4: Phase 2 Analysis

- [ ] **2.4.1** Compute statistical analysis
  - Bootstrap CI for new benchmarks
  - Paired comparisons
  - Holm-Bonferroni across all Phase 1+2 comparisons

- [ ] **2.4.2** Analyze Conflict Resolution results
  - Compare git-notes vs no-memory specifically
  - Identify cases where version history helped
  - Document qualitative examples

- [ ] **2.4.3** Export human validation samples
  - 100 from Context-Bench
  - 100 from MemoryAgentBench
  - Prioritize conflict resolution cases

---

## Phase 3: Real-World Validation (4-5 weeks)

**Gate Criteria:**
1. [ ] Terminal-Bench 2.0 AbstractInstalledAgent implemented
2. [ ] Docker task execution environment operational
3. [ ] Memory-relevant task subset identified and evaluated
4. [ ] Final comparative analysis complete
5. [ ] arXiv paper draft complete
6. [ ] Blog post draft complete
7. [ ] All human validation samples collected (500 total)

### Task 3.1: Terminal-Bench 2.0 Integration

- [ ] **3.1.1** Analyze Terminal-Bench 2.0 requirements
  - Review `terminal-bench.com` documentation
  - Understand Harbor containerization
  - Identify AbstractInstalledAgent interface

- [ ] **3.1.2** Implement AbstractInstalledAgent subclass
  - Integrate MemorySystemAdapter
  - Handle task execution lifecycle
  - Implement memory persistence across tasks

- [ ] **3.1.3** Set up Docker execution environment
  - Harbor-compatible container configuration
  - Resource limits and timeouts
  - Result capture and logging

- [ ] **3.1.4** Select memory-relevant task subset
  - Review full task list
  - Filter for tasks benefiting from memory
  - Document selection criteria

- [ ] **3.1.5** Run Terminal-Bench evaluation
  - Both conditions
  - 5 runs per condition
  - Record task-level results

### Task 3.2: Final Human Validation

- [ ] **3.2.1** Compile all validation samples
  - 100 from LongMemEval (Phase 1)
  - 100 from LoCoMo (Phase 1)
  - 100 from Context-Bench (Phase 2)
  - 100 from MemoryAgentBench (Phase 2)
  - 100 from Terminal-Bench (Phase 3)

- [ ] **3.2.2** Create annotation guidelines
  - Define correct/incorrect/partial criteria
  - Provide examples for each category
  - Document edge cases

- [ ] **3.2.3** Conduct human validation
  - Annotate samples (or coordinate with team)
  - Calculate inter-annotator agreement
  - Compare with LLM-as-Judge results

### Task 3.3: Final Analysis

- [ ] **3.3.1** Compute final statistics
  - All 5 benchmarks combined
  - Holm-Bonferroni across all comparisons
  - Effect sizes with interpretation

- [ ] **3.3.2** Generate publication tables
  - LaTeX-formatted main results table
  - Per-benchmark breakdown tables
  - Ablation study table

- [ ] **3.3.3** Generate publication figures
  - Bar charts with 95% CI error bars
  - Radar chart of memory abilities
  - Ablation component contribution

### Task 3.4: Publication Artifacts

- [ ] **3.4.1** Draft arXiv paper
  - Abstract and introduction
  - Related work (memory systems, benchmarks)
  - Methodology (adapters, benchmarks, statistics)
  - Results with tables and figures
  - Discussion and limitations
  - Conclusion and future work

- [ ] **3.4.2** Create appendix
  - Full result tables
  - Hyperparameter configurations
  - Human validation details

- [ ] **3.4.3** Draft blog post
  - Accessible summary of findings
  - Visual highlights
  - Call to action (try git-notes-memory-manager)

- [ ] **3.4.4** Prepare reproducibility package
  - Docker image published
  - All raw results archived
  - Analysis notebooks included

---

## Task Dependencies

```
Phase 0
├── 0.1 Project Setup (no deps)
├── 0.2 MemorySystemAdapter (depends on 0.1)
├── 0.3 GitNotesAdapter (depends on 0.2)
├── 0.4 NoMemoryAdapter (depends on 0.2)
├── 0.5 MockAdapter (depends on 0.2)
├── 0.6 Docker (depends on 0.1)
└── 0.7 CI/CD (depends on 0.1, 0.6)

Phase 1
├── 1.1 LLM-as-Judge (depends on Phase 0)
├── 1.2 Statistics (depends on Phase 0)
├── 1.3 LongMemEval (depends on 1.1, 1.2)
├── 1.4 LoCoMo (depends on 1.1, 1.2)
└── 1.5 Analysis (depends on 1.3, 1.4)

Phase 2
├── 2.1 Context-Bench (depends on Phase 1)
├── 2.2 MemoryAgentBench (depends on Phase 1)
├── 2.3 Ablation (depends on Phase 1)
└── 2.4 Analysis (depends on 2.1, 2.2, 2.3)

Phase 3
├── 3.1 Terminal-Bench (depends on Phase 2)
├── 3.2 Human Validation (depends on Phase 1, 2, 3.1)
├── 3.3 Final Analysis (depends on 3.1, 3.2)
└── 3.4 Publication (depends on 3.3)
```

---

## Risk Mitigation Tasks

### High Priority

- [ ] **R1** Document git-notes-memory-manager API before Phase 0.3
- [ ] **R2** Test GPT-4o rate limits with sample batch before Phase 1.1
- [ ] **R3** Verify HuggingFace dataset availability before Phase 1.3, 1.4

### Medium Priority

- [ ] **R4** Create fallback GPU runner documentation (RunPod/Lambda)
- [ ] **R5** Pin Context-Bench to specific commit
- [ ] **R6** Set up cost monitoring for OpenAI API

### Low Priority

- [ ] **R7** Document upgrade path for API version changes
- [ ] **R8** Create lite benchmark mode for faster CI runs

---

## Success Checkpoints

| Checkpoint | Target Date | Criteria |
|------------|-------------|----------|
| Infrastructure Ready | Week 2 | All Phase 0 gates passed |
| First Results | Week 6 | LongMemEval + LoCoMo complete |
| Extended Results | Week 11 | Context-Bench + MemoryAgentBench complete |
| Publication Ready | Week 16 | All drafts complete, human validation done |

---

## Appendix: Task Estimation Notes

**Phase 0 (2 weeks)**
- Setup tasks: 2-3 days
- Adapter implementation: 3-4 days
- Docker/CI: 2-3 days
- Buffer: 1-2 days

**Phase 1 (4-6 weeks)**
- Judge + Statistics: 1 week
- LongMemEval: 1.5-2 weeks
- LoCoMo: 1.5-2 weeks
- Analysis: 0.5-1 week

**Phase 2 (4-5 weeks)**
- Context-Bench: 1.5-2 weeks
- MemoryAgentBench: 1.5-2 weeks
- Ablation: 0.5-1 week
- Analysis: 0.5-1 week

**Phase 3 (4-5 weeks)**
- Terminal-Bench: 2 weeks
- Human validation: 0.5-1 week
- Analysis: 0.5-1 week
- Publication: 1-2 weeks
