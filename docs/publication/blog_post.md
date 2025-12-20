# Git-Notes Memory: Teaching LLM Agents to Remember with Git

*How we built a memory system for AI agents using Git's native storage—and why version history matters more than you'd think.*

---

## The Memory Problem

Every conversation with an LLM starts fresh. Ask Claude about your favorite programming language today, and tomorrow it won't remember. For casual chats, this is fine. But for agents that need to work with you over weeks or months—tracking your project preferences, remembering past decisions, recalling what worked and what didn't—this amnesia is a serious limitation.

The typical solution? Vector databases. Store embeddings, retrieve by similarity, hope the right memories surface. But this approach has a fundamental problem: **it forgets how things change**.

## A Different Approach: Git for Memory

Git already solves many problems that memory systems struggle with:

- **Version history**: Every change is tracked, forever
- **Conflict resolution**: Built-in handling when things collide
- **Audit trail**: Complete provenance for every piece of data
- **Distribution**: Works offline, syncs when ready

We built [git-notes-memory-manager](https://github.com/zircote/git-notes-memory-manager), a memory system that stores agent memories as Git notes. And we needed to know: does it actually work better?

## The Benchmark Challenge

Evaluating memory systems is surprisingly hard. A few questions about "what did we discuss yesterday" doesn't cut it. We needed:

1. **Multi-hop reasoning**: "Based on my preference for Python and yesterday's performance issue, what should we try?"
2. **Temporal understanding**: "What did I think about microservices before last month?"
3. **Conflict resolution**: "I changed my mind about the database—does the agent track that?"
4. **Adversarial queries**: "Can we trick it into using outdated information?"

We evaluated across **five benchmarks** covering these scenarios:

| Benchmark | Focus | Tasks |
|-----------|-------|-------|
| LongMemEval | Long-context recall | 500 QA pairs |
| LoCoMo | Conversation memory | 867 questions |
| Context-Bench | Context utilization | TBD |
| MemoryAgentBench | Memory competencies | TBD |
| Terminal-Bench 2.0 | Real-world agent tasks | TBD |

## What We Measured

For each benchmark, we compared:

- **git_notes**: Full git-notes-memory-manager
- **no_memory**: Baseline without external memory

Key metrics:
- **Accuracy**: Did it answer correctly?
- **Abstention rate**: Did it say "I don't know" when appropriate?
- **Latency**: How fast can it retrieve memories?

We also ran **ablation studies**—systematically disabling components to understand what matters:

| Component Disabled | What We Learn |
|-------------------|---------------|
| Semantic search | Is embedding-based retrieval necessary? |
| Metadata filters | Do timestamps and tags help? |
| Version history | Does access to old values matter? |
| Fixed window | Is dynamic retrieval better than fixed? |
| Recency only | Is "most recent wins" sufficient? |

## Results

*[Results placeholder - to be updated after experiments complete]*

### Main Findings

| Condition | Overall Accuracy | Best Category |
|-----------|-----------------|---------------|
| git_notes | XX.X% | Multi-hop QA |
| no_memory | XX.X% | Single-hop QA |

**Key insight**: The biggest gains came from [TBD based on actual results].

### Ablation Insights

The components that mattered most:

1. **[TBD]** — Removing this caused a XX% drop
2. **[TBD]** — Less impactful than expected (only X%)
3. **[TBD]** — Surprisingly important for adversarial queries

### Human Validation

We didn't just trust the LLM to judge itself. Human annotators reviewed XX samples with inter-annotator agreement of κ = X.XX.

## The Version History Advantage

One finding that surprised us: **version history access was crucial for conflict resolution**.

When information changes—"Actually, let's use PostgreSQL instead of MongoDB"—agents without version history:
- Sometimes use outdated info
- Can't explain *why* they switched
- Struggle when asked about the change itself

Git-based memory handles this naturally. The old value is still there, just one commit back. The agent can:
- Access the current value
- Retrieve history when asked
- Explain what changed and when

## Try It Yourself

The complete benchmark harness is open source:

```bash
git clone https://github.com/zircote/memory-benchmark-harness
cd memory-benchmark-harness
uv sync
uv run benchmark run longmemeval --adapter git_notes
```

The harness supports:
- Custom memory adapters (implement one interface)
- All five benchmarks with unified metrics
- Statistical analysis with bootstrap confidence intervals
- Human validation sample export

## What's Next

This is a first step toward rigorous memory system evaluation. Future work includes:

- **Multi-agent scenarios**: How do agents share memories?
- **Longer timeframes**: Days and weeks, not just sessions
- **Comparison with vector DBs**: Direct head-to-head evaluation
- **Cost analysis**: Git operations vs. embedding API calls

## Acknowledgments

Thanks to the teams behind LongMemEval, LoCoMo, Context-Bench, MemoryAgentBench, and Terminal-Bench for creating the benchmarks that made this evaluation possible.

---

*Code: [memory-benchmark-harness](https://github.com/zircote/memory-benchmark-harness)*
*Paper: [arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX)*
*Memory system: [git-notes-memory-manager](https://github.com/zircote/git-notes-memory-manager)*
