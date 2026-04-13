# Memory Log

> Append-only chronological record of all operations.
> Format: `## [YYYY-MM-DD] operation | subject`
> Operations: `ingest`, `analyze`, `synthesize`, `lint`, `update`, `create`, `merge`

---

## [2026-04-13] create | Research goal: LLM Architecture for Simple Math
- Issue: GH-1 (parent), GH-2/3/4 (sub-issues)
- Branch: research/GH-1-math-llm-architecture
- Scope: architecture, tokenization, training for add/sub/mul

## [2026-04-13] ingest | Tokenization Strategies (Hypothesis A — GH-2)
- Source: sources/tokenization-arithmetic-research.md
- Entities created: 6 (papers + researcher Nogueira)
- Findings created: 6 (digit-level, reversed-order, BPE, index-hints, scratchpad, padding)
- Open questions: 3 (combining techniques, length-gen necessity, scratchpad format)
- Agent: hypothesis-tokenization

## [2026-04-13] ingest | Architecture Sizing (Hypothesis B — GH-3)
- Source: sources/architecture-sizing-research.md
- Entities created: 6 (papers + Karpathy)
- Findings created: 7 (depth-vs-width, min-viable, positional-encoding, attention-heads, data-formatting, looped-transformers, normalization)
- Open questions: 3 (mixed-operation arch, position-coupling-vs-abacus, mult-scaling)
- Agent: hypothesis-architecture

## [2026-04-13] ingest | Training Curriculum (Hypothesis C — GH-4)
- Source: sources/training-curriculum-research.md
- Entities created: 6 (papers: Lee, Power, Nanda, Cho, Jelassi, McLeish)
- Findings created: 8 (curriculum, data-format, grokking, length-gen, optimizer, scratchpad, balanced-sampling, augmentation)
- Open questions: 4 (mult-length-gen, weight-decay, scratchpad-vs-reverse, grokking-integer-vs-modular)
- Agent: hypothesis-training
