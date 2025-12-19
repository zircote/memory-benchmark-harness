---
project_id: SPEC-2025-12-19-001
project_name: "Benchmark Harness for Git-Native Semantic Memory Validation"
slug: benchmark-harness-spec
status: draft
created: 2025-12-19T18:05:00Z
approved: null
started: null
completed: null
expires: 2026-03-19T18:05:00Z
superseded_by: null
tags: [benchmarking, memory-systems, ai-agents, academic-publication, testing-framework]
stakeholders: []
worktree:
  branch: plan/review-file-prepare-plan-actio
  base_branch: main
  created_from_commit: 9c9d205
---

# Benchmark Harness for Git-Native Semantic Memory Validation

## Project Overview

A comprehensive benchmark harness to validate the benefits of git-native semantic memory for AI coding agents. This system enables academic publication with reproducible results across five established benchmarks:

- **LongMemEval** (ICLR 2025) - Long-term memory evaluation
- **LoCoMo** (ACL 2024) - Long-context conversation memory
- **Context-Bench** (Letta, Oct 2025) - Agentic context engineering
- **Terminal-Bench 2.0** - Real-world coding task validation
- **MemoryAgentBench** (arXiv, July 2025) - Four competencies evaluation

## Current Status

🟡 **DRAFT** - Specification complete, awaiting approval

## Key Documents

| Document | Status | Description |
|----------|--------|-------------|
| [REQUIREMENTS.md](./REQUIREMENTS.md) | ✅ Complete | Product requirements with revised 4-phase structure |
| [ARCHITECTURE.md](./ARCHITECTURE.md) | ✅ Complete | Technical design with MemorySystemAdapter pattern |
| [IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md) | ✅ Complete | Detailed task breakdown for all phases |
| [DECISIONS.md](./DECISIONS.md) | ✅ Complete | 12 Architecture Decision Records |
| [CHANGELOG.md](./CHANGELOG.md) | ✅ Complete | Change history |

## Quick Links

- Source Specification: `/docs/BenchmarkHarnessSpecificationforGi.md`
- Target Repository: `zircote/memory-benchmark-harness`

## Success Criteria

1. ✅ Statistical significance (p < 0.05 with Holm-Bonferroni correction) as primary metric
2. ✅ Two-way comparison: git-notes-memory-manager vs no-memory baseline
3. ✅ All evaluations reproducible via Docker with single command
4. ✅ Results formatted for arXiv publication with bootstrap confidence intervals
5. ✅ Phase-gated development: Phase 0 → Phase 1 → Phase 2 → Phase 3
6. ✅ 500 samples validated by human annotators (100 per benchmark)

## Key Decisions (ADRs)

| ADR | Decision | Impact |
|-----|----------|--------|
| ADR-012 | Simplified to 2-way comparison | Faster delivery, cleaner narrative |
| ADR-005 | Unified MemorySystemAdapter | Single interface for all benchmarks |
| ADR-004 | Added Phase 0 (2 weeks) | Reduced Phase 1 risk |
| ADR-001 | MemoryAgentBench in Phase 2 | Validates git version advantage |
