---
document_type: requirements
project_id: SPEC-2025-12-19-001
version: 1.0.0
last_updated: 2025-12-19T18:30:00Z
status: completed
---

# Benchmark Harness for Git-Native Semantic Memory Validation

## Product Requirements Document

## Executive Summary

This document specifies requirements for a benchmark harness that validates the benefits of git-native semantic memory for AI coding agents. The harness compares the `git-notes-memory-manager` Claude Code plugin against a no-memory baseline across five established academic benchmarks.

**Primary deliverable:** A reproducible evaluation framework demonstrating statistically significant memory benefits, published as an arXiv paper with accompanying blog post.

**Scope:** Two-way comparison (memory vs. no-memory) across 5 benchmarks with ablation studies. Future work may extend to multi-system comparisons.

## Problem Statement

### The Problem

AI coding agents lack rigorous validation of long-horizon memory benefits. While memory systems like git-notes-memory-manager exist, there is no defensible evidence that semantic memory provides measurable advantages over context-only approaches.

### Impact

- **Researchers:** Cannot cite memory benefits without reproducible benchmarks
- **Users:** Cannot make informed decisions about memory system adoption
- **Developers:** Cannot prioritize memory features without validated impact data

### Current State

The git-notes-memory-manager plugin exists with a stable API, but:
- No systematic benchmarking against established datasets
- No statistical validation of claimed benefits
- No comparison against baseline approaches

## Goals and Success Criteria

### Primary Goal

Demonstrate whether git-native semantic memory provides statistically significant benefits for AI coding agents across long-horizon memory tasks.

### Success Metrics

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| Statistical significance | p < 0.05 (Holm-Bonferroni corrected) | Bootstrap CI + paired tests |
| Reproducibility | Single-command Docker execution | CI validation |
| Human validation | 100 samples/benchmark validated | Inter-annotator agreement |
| Publication | arXiv submission + blog post | Completed artifacts |

### Non-Goals (Explicit Exclusions)

- Multi-system comparison (Mem0, MemGPT, etc.) - deferred to future work
- Production deployment of harness
- Real-time benchmarking
- Non-English language evaluation

## User Analysis

### Primary Users

**Researchers/Authors**
- **Need:** Reproducible benchmark results for publication
- **Context:** Validating git-notes-memory-manager claims

**Plugin Developers**
- **Need:** Performance regression detection
- **Context:** Continuous integration during development

### User Stories

1. As a **researcher**, I want to run the complete benchmark suite with a single command so that I can reproduce published results.

2. As a **developer**, I want to run a lite benchmark on every PR so that I can detect performance regressions early.

3. As an **author**, I want publication-ready tables and figures so that I can include them directly in the paper.

4. As a **reviewer**, I want access to raw results and analysis code so that I can verify the claims.

## Functional Requirements

### Must Have (P0)

| ID | Requirement | Rationale | Acceptance Criteria |
|----|-------------|-----------|---------------------|
| FR-001 | Unified MemorySystemAdapter base class | Consistent interface across benchmarks | Abstract class with add/search/update/delete/clear methods |
| FR-002 | Git-notes-memory-manager adapter | Core evaluation target | Implements MemorySystemAdapter, passes integration tests |
| FR-003 | No-memory baseline adapter | Comparison baseline | Full-context approach without retrieval |
| FR-004 | LongMemEval evaluation pipeline | ICLR 2025 benchmark | 500 questions, 5 memory abilities evaluated |
| FR-005 | LoCoMo evaluation pipeline | ACL 2024 benchmark | 10 conversations, 5 QA categories |
| FR-006 | LLM-as-Judge with caching | Evaluation method | GPT-4o with exponential backoff, 30-day cache |
| FR-007 | Bootstrap confidence intervals | Statistical rigor | BCa bootstrap, 2000+ iterations |
| FR-008 | Holm-Bonferroni correction | Multiple comparison control | Automatic adjustment for all comparisons |
| FR-009 | Docker-based execution | Reproducibility | Single `docker compose run` command |
| FR-010 | Human validation sample export | Judge validation | Export 100 samples/benchmark for annotation |

### Should Have (P1)

| ID | Requirement | Rationale | Acceptance Criteria |
|----|-------------|-----------|---------------------|
| FR-101 | Context-Bench evaluation | Agentic memory skills | Letta benchmark class implementation |
| FR-102 | MemoryAgentBench evaluation | Conflict resolution testing | Four competencies evaluated |
| FR-103 | Terminal-Bench 2.0 integration | Real-world coding tasks | AbstractInstalledAgent implementation |
| FR-104 | Ablation study framework | Component contribution analysis | LOCO ablation with p-values |
| FR-105 | Publication table generator | arXiv formatting | LaTeX-ready tables with CI |
| FR-106 | GitHub Actions CI | Automated lite benchmark | PR validation workflow |

### Nice to Have (P2)

| ID | Requirement | Rationale | Acceptance Criteria |
|----|-------------|-----------|---------------------|
| FR-201 | Interactive results dashboard | Exploration | Streamlit or notebook |
| FR-202 | Cost tracking | Budget monitoring | Per-benchmark API cost reports |
| FR-203 | Benchmark leaderboard | Community engagement | Web page with results |
| FR-204 | Multi-GPU parallelization | Faster execution | Scale across available GPUs |

## Non-Functional Requirements

### Performance

- Full benchmark suite completion: < 48 hours on single GPU
- Lite benchmark (CI): < 30 minutes
- LLM-as-Judge cache hit rate: > 80% on reruns

### Reliability

- Interrupted runs resumable via caching
- 5 automatic retries for transient API failures
- Results persisted to disk after each benchmark

### Reproducibility

- All random seeds documented and configurable
- Dependency versions locked
- Docker images tagged and published
- Results reproducible to 3 decimal places

### Maintainability

- 80%+ test coverage for core modules
- Type hints throughout
- Documentation for all public APIs

## Technical Constraints

### Hardware Requirements

- NVIDIA GPU with ≥8GB VRAM (embedding models)
- 32GB RAM minimum
- 100GB disk space for datasets and results

### Software Stack

- Python 3.11+
- CUDA 12.2+
- Docker with GPU support
- uv for dependency management

### API Dependencies

- OpenAI API (GPT-4o for LLM-as-Judge)
- Anthropic API (Claude for Terminal-Bench)
- HuggingFace Datasets

## Dependencies

### Internal Dependencies

- git-notes-memory-manager plugin (stable API)
- Existing benchmark datasets (HuggingFace, GitHub)

### External Dependencies

| Dependency | Version | Purpose | Risk |
|------------|---------|---------|------|
| OpenAI API | GPT-4o | LLM-as-Judge | Rate limits, cost |
| HuggingFace Datasets | Latest | LongMemEval, MemoryAgentBench | Download availability |
| Context-Bench | Pinned commit | Letta evaluation | API changes |
| Terminal-Bench 2.0 | v2.0.0 | Coding task evaluation | Harbor maturity |

## Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| GPT-4o rate limits | Medium | High | Caching, exponential backoff |
| No significant improvement found | Medium | High | Frame as "validation", report effect sizes |
| Context-Bench API changes | Medium | Medium | Pin to commit, document upgrade path |
| GPU runner unavailability | Low | High | RunPod/Lambda fallback instructions |
| git-notes API changes | Low | Medium | API documentation, version pinning |

## Revised Phase Structure

Based on decisions captured in ADRs:

### Phase 0: Infrastructure Foundation (2 weeks)

**Gate Criteria:** Adapter contracts defined, CI pipeline working

**Deliverables:**
- [ ] Unified MemorySystemAdapter base class
- [ ] Git-notes-memory-manager adapter
- [ ] No-memory baseline adapter
- [ ] Mock memory system for testing
- [ ] Docker base image with dependencies
- [ ] GitHub Actions CI skeleton
- [ ] git-notes-memory-manager API documentation

### Phase 1: Core Evaluation (4-6 weeks)

**Gate Criteria:** LongMemEval and LoCoMo results complete with statistics

**Deliverables:**
- [ ] LongMemEval evaluation pipeline (LongMemEval_S and LongMemEval_M)
- [ ] LoCoMo evaluation pipeline (all 5 QA categories)
- [ ] LLM-as-Judge with caching
- [ ] Statistical analysis module (bootstrap CI)
- [ ] Results with 5 runs, 95% CI
- [ ] Human validation sample export (200 samples)

### Phase 2: Extended Evaluation (4-5 weeks)

**Gate Criteria:** Context-Bench and MemoryAgentBench complete

**Deliverables:**
- [ ] Context-Bench evaluation (Letta Benchmark class)
- [ ] MemoryAgentBench evaluation (4 competencies, esp. Conflict Resolution)
- [ ] Ablation study framework
- [ ] Human validation sample export (200 samples)

### Phase 3: Real-World Validation (4-5 weeks)

**Gate Criteria:** Terminal-Bench 2.0 integration complete

**Deliverables:**
- [ ] Terminal-Bench 2.0 AbstractInstalledAgent
- [ ] Docker environment for task execution
- [ ] Task subset selection (memory-relevant)
- [ ] Final comparative analysis
- [ ] Human validation sample export (100 samples)
- [ ] arXiv paper draft
- [ ] Blog post draft

## Open Questions

- [ ] What is the exact git-notes-memory-manager Python API signature?
- [ ] Which Terminal-Bench 2.0 tasks are most relevant to memory benefits?
- [ ] What is the cost budget for LLM-as-Judge API calls?
- [ ] Who will perform human validation annotation?

## Appendix

### Glossary

| Term | Definition |
|------|------------|
| LLM-as-Judge | Using GPT-4o to evaluate answer correctness |
| Holm-Bonferroni | Step-down procedure for multiple comparison correction |
| BCa Bootstrap | Bias-corrected and accelerated confidence interval method |
| LOCO | Leave-One-Component-Out ablation study design |
| arXiv | Open-access preprint server for academic papers |

### References

- [LongMemEval Paper (ICLR 2025)](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned)
- [LoCoMo Paper (ACL 2024)](https://github.com/snap-research/locomo)
- [Context-Bench (Letta)](https://github.com/letta-ai/letta-evals)
- [Terminal-Bench 2.0](https://terminal-bench.com)
- [MemoryAgentBench (arXiv:2507.05257)](https://huggingface.co/datasets/ai-hyz/MemoryAgentBench)
- [git-notes-memory-manager](https://github.com/zircote/git-notes-memory-manager)
