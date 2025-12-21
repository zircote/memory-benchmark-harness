---
document_type: decisions
project_id: SPEC-2025-12-19-001
---

# Benchmark Harness - Architecture Decision Records

## ADR-001: MemoryAgentBench Phase Allocation

**Date**: 2025-12-19
**Status**: Accepted
**Deciders**: Project Owner

### Context
The architect review identified MemoryAgentBench's Conflict Resolution competency as the strongest differentiation opportunity for git-native memory (leveraging git's version history). However, it was not assigned to any implementation phase in the original specification.

### Decision
Add MemoryAgentBench to Phase 2, extending the phase scope alongside Context-Bench.

### Consequences
**Positive:**
- Validates the core differentiation hypothesis early
- Phase 2 becomes the "memory competency" validation phase
- Research narrative strengthens around version control advantages

**Negative:**
- Phase 2 duration increases (~2 additional weeks)
- Additional adapter/evaluation infrastructure required

---

## ADR-002: Git-Notes-Memory-Manager API Status

**Date**: 2025-12-19
**Status**: Accepted
**Deciders**: Project Owner

### Context
The benchmark harness depends on the git-notes-memory-manager plugin's Python API for memory operations (add, search, update, delete). The architect review flagged this as undefined.

### Decision
The API exists and is stable. Documentation and formalization can proceed immediately without design phase.

### Consequences
**Positive:**
- No Phase 0 delay for API design
- Can begin adapter development immediately
- Known interface reduces integration risk

**Negative:**
- Must document current API before implementation
- Any needed API changes may require plugin updates

---

## ADR-003: Success Threshold Definition

**Date**: 2025-12-19
**Status**: Accepted
**Deciders**: Project Owner

### Context
The original specification used vague success metrics ("measurable improvement"). The architect review recommended quantitative thresholds for publication defensibility.

### Decision
Focus on statistical significance (p < 0.05) as the primary success criterion, rather than fixed effect size thresholds.

### Consequences
**Positive:**
- More flexible - doesn't commit to specific improvement percentages
- Academically defensible if properly reported
- Allows the data to speak for itself

**Negative:**
- Reviewers may request effect sizes alongside p-values
- Small but significant improvements may have limited practical value
- Must report effect sizes in paper even if not gated on them

**Mitigation:**
- Report both p-values AND effect sizes (Cohen's d, absolute improvement)
- Include practical significance discussion in paper

---

## ADR-004: Timeline Adjustment with Phase 0

**Date**: 2025-12-19
**Status**: Accepted
**Deciders**: Project Owner

### Context
The architect review flagged Phase 1 (4-6 weeks for 4 adapters + 2 benchmarks + statistics) as aggressive. Infrastructure foundation work was not explicitly scoped.

### Decision
Add Phase 0 (2 weeks) for infrastructure foundation before Phase 1.

### Consequences
**Positive:**
- Reduces Phase 1 risk
- Establishes adapter contracts and CI/CD before benchmark work
- Creates test fixtures and mock infrastructure
- Total timeline more realistic

**Negative:**
- Adds 2 weeks to overall project timeline
- Delays first benchmark results by 2 weeks

### Phase 0 Scope (Proposed)
- [ ] Unified MemorySystemAdapter base class
- [ ] Mock memory system for testing
- [ ] CI/CD pipeline foundation
- [ ] Docker base image with dependencies
- [ ] Test fixture infrastructure
- [ ] git-notes-memory-manager API documentation

---

## ADR-005: Unified Adapter Interface

**Date**: 2025-12-19
**Status**: Accepted
**Deciders**: Project Owner

### Context
The specification shows three different adapter patterns:
- LongMemEval: `ingest_history()`, `answer_question()`
- Context-Bench: `setup_agent()`, `metric()`
- Terminal-Bench: `AbstractInstalledAgent` subclass

### Decision
Create a unified `MemorySystemAdapter` base class with benchmark-specific wrappers.

### Consequences
**Positive:**
- Single source of truth for memory operations
- Easier testing and mocking
- Consistent initialization across benchmarks
- Reduced code duplication

**Negative:**
- Additional abstraction layer
- Wrapper classes needed for each benchmark

### Implementation
```python
class MemorySystemAdapter(ABC):
    @abstractmethod
    def add(self, content: str, metadata: dict) -> str: ...
    @abstractmethod
    def search(self, query: str, limit: int = 10) -> list[MemoryItem]: ...
    @abstractmethod
    def update(self, memory_id: str, content: str) -> None: ...
    @abstractmethod
    def delete(self, memory_id: str) -> None: ...
    @abstractmethod
    def clear(self) -> None: ...
```

---

## ADR-006: GPU Hosting Strategy

**Date**: 2025-12-19
**Status**: Accepted
**Deciders**: Project Owner

### Context
Embedding models require 44MB - 2.5GB VRAM. Options include local GPU, cloud GPU, or API-based embeddings.

### Decision
Local GPU required for all embedding generation.

### Consequences
**Positive:**
- Full reproducibility (no API versioning issues)
- No rate limits or API costs for embeddings
- Consistent latency across runs

**Negative:**
- Requires GPU hardware availability
- Self-hosted GPU runner needed for GitHub Actions
- Higher barrier to entry for contributors

### Mitigation
- Document GPU requirements clearly
- Provide RunPod/Lambda setup instructions as alternative
- Consider lite mode with smaller embedding models for CI

---

## ADR-007: LLM-as-Judge Error Handling

**Date**: 2025-12-19
**Status**: Accepted
**Deciders**: Project Owner

### Context
GPT-4o is the specified LLM-as-Judge. Rate limits, outages, or cost overruns could halt evaluation.

### Decision
Implement caching with exponential backoff retry.

### Consequences
**Positive:**
- Resilient to transient failures
- Reduced API costs on reruns
- Can resume interrupted evaluations

**Negative:**
- Cache invalidation complexity
- Stale cached judgments if test data changes
- Storage requirements for judgment cache

### Implementation
- Cache key: hash(question + model_response + judge_model_version)
- Retry: exponential backoff (1s, 2s, 4s, 8s, max 60s)
- Max retries: 5 per judgment
- Cache TTL: 30 days (configurable)

---

## ADR-008: Baseline Hyperparameter Tuning

**Date**: 2025-12-19
**Status**: Accepted
**Deciders**: Project Owner

### Context
Fair comparison requires baselines (Mem0, MemGPT) to be reasonably optimized. Using poor defaults could inflate git-notes-memory advantages.

### Decision
Light tuning: test 2-3 configurations per baseline.

### Consequences
**Positive:**
- Fair comparison - reviewers cannot claim baseline handicap
- Manageable tuning effort
- Configurations can be documented for reproducibility

**Negative:**
- Not exhaustive search - some reviewer skepticism possible
- Additional development time

### Tuning Parameters per Baseline
**Mem0:**
1. Default configuration
2. chunk_size: 512 → 1024
3. top_k: 5 → 10

**MemGPT/Letta:**
1. Default configuration
2. Larger archival memory allocation
3. Different embedding model

**RAG:**
1. chunk_size: 256, 512, 1024
2. top_k: 5, 10
3. embedding: BGE-M3, text-embedding-3-small

---

## ADR-009: Publication Target

**Date**: 2025-12-19
**Status**: Accepted
**Deciders**: Project Owner

### Context
Different publication venues have different rigor requirements. This affects statistical methodology, reproducibility standards, and timeline.

### Decision
Target arXiv + Blog for community visibility without formal peer review.

### Consequences
**Positive:**
- Faster time to publication
- More flexibility in presentation
- Community feedback before potential venue submission
- No reviewer gatekeeping

**Negative:**
- Less academic prestige
- May need additional work for later venue submission
- No formal peer review validation

---

## ADR-010: Multiple Comparison Correction

**Date**: 2025-12-19
**Status**: Accepted
**Deciders**: Project Owner

### Context
Comparing across multiple baselines and benchmarks creates multiple hypothesis tests. Raw p-values without correction inflate Type I error rate.

### Decision
Use Holm-Bonferroni correction for family-wise error rate control.

### Consequences
**Positive:**
- Controls FWER properly
- Less conservative than Bonferroni
- Step-down procedure preserves power

**Negative:**
- Some comparisons may lose significance after correction
- More complex to explain than raw p-values

---

## ADR-011: Human Validation Scope

**Date**: 2025-12-19
**Status**: Accepted
**Deciders**: Project Owner

### Context
GPT-4o as LLM-as-Judge has 97% agreement with human experts. Human validation provides additional credibility but requires manual effort.

### Decision
Validate 100 samples per benchmark (~500 total judgments).

### Consequences
**Positive:**
- Strong defense against LLM-as-Judge criticism
- Can report human agreement rate
- Identifies systematic judge errors

**Negative:**
- Manual annotation effort required
- Need annotation guidelines
- Inter-annotator agreement tracking

---

## ADR-012: Comparison Scope Simplification

**Date**: 2025-12-19
**Status**: Accepted
**Deciders**: Project Owner

### Context
Original spec compared against 4 baselines (Mem0, MemGPT, RAG, Full-context). This creates complexity in implementation and analysis.

### Decision
Simplify to two-way comparison:
1. **git-notes-memory-manager** (with memory)
2. **No-memory baseline** (full context, no retrieval)

### Consequences
**Positive:**
- Dramatically simplified implementation
- Cleaner research narrative
- Faster time to results
- No dependency on third-party memory systems

**Negative:**
- Loses direct comparison to commercial alternatives
- May limit publication impact
- Cannot claim superiority to Mem0/MemGPT

**Mitigation:**
- Frame as "memory benefit validation" not "system comparison"
- Future work section can propose multi-system comparison
- Ablation studies still valuable without baselines

### Updated Phase Scope
- Phase 0: Infrastructure (git-notes + no-memory adapters only)
- Phase 1: LongMemEval + LoCoMo
- Phase 2: Context-Bench + MemoryAgentBench
- Phase 3: Terminal-Bench 2.0
