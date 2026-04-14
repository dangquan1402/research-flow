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

## [2026-04-13] synthesize | Cross-hypothesis synthesis — themes + architecture spec
- Themes created: 5 (data-format-supremacy, shallow-wide-architecture, position-encoding-unlocks-generalization, training-data-distribution, regularization-drives-generalization)
- Contradictions resolved: scratchpad-vs-reversed (complementary not competing), depth-vs-looped (loops provide computational depth without parameter depth), NoPE-vs-position-aware (scope-dependent choice)
- Output: outputs/math-llm-architecture-spec.md
- Spec: 4L/4H/256d/~3-5M params, digit-level tokenization, reversed LSB-first output, AdamW wd=0.5, balanced carry sampling
- 5 experiments planned for M4 Mac validation
- Agent: synthesizer

## [2026-04-13] experiment | Baseline addition — reversed output, 5-digit
- Model: 4L/4H/256D (3.16M params), MLX on Apple M4
- Result: 99.9% accuracy (peak 100% at epoch 45), 517s training
- All digit counts 1-6 at 100% by epoch 50
- Findings created: experiment-baseline-addition-reversed
- Agent: experimenter

## [2026-04-13] experiment | Tokenization comparison — reversed vs plain
- Same model (4L/4H/256D), same data (50K, 5-digit addition)
- Reversed: 99.9% (peak 100%), 95% at epoch 15
- Plain: 97.85% (never 100%), 95% at epoch 30
- Reversed +2.05% accuracy, 2x faster learning
- Findings created: experiment-tokenization-comparison
- Agent: experimenter

## [2026-04-13] experiment | Depth vs width ablation — 4 configurations
- 8L/4H/128D (1.58M): 99.95%, peak 100% ep 30, some instability
- 4L/4H/256D (3.16M): 99.90%, peak 100% ep 45, stable
- 2L/4H/384D (3.56M): 99.90%, peak 100% ep 20, most stable — SWEET SPOT
- 2L/8H/512D (6.32M): 99.65%, peak 100% ep 35, diminishing returns
- All configs >99.5% — data format matters more than architecture for addition
- Findings created: experiment-depth-vs-width-ablation
- Agent: experimenter

## [2026-04-13] update | Architecture spec updated with experimental results
- Added Section 10 to outputs/math-llm-architecture-spec.md
- All 3 experiments passed success criteria
- Recommended V1a update: 2L/4H/384D instead of 2L/4H/256D
- Agent: experimenter

## [2026-04-13] synthesize | Final spec revision — experimental validation integrated
- Revised V1 primary config: 4L/4H/256D → 2L/4H/384D (based on Exp 3 sweet spot)
- Added V1-safe (4L/4H/256D) for mixed operations, V1-min (8L/4H/128D) for parameter constraints
- Updated Decision Log with experimental status for all 11 decisions
- Key insight: balanced carry sampling closes 13% of the reversed-vs-plain gap (97.85% vs literature's 85%)
- Commented on GH-1 with final results summary
- Agent: synthesizer

## [2026-04-13] experiment | Subtraction — 5-digit, 2L/384D
- Model: 2L/4H/384D (3.56M params), MLX on Apple M4
- Result: 99.90% accuracy, 461s training
- Identical to addition — borrow = carry computationally
- Findings created: experiment-subtraction-5digit
- Agent: experimenter

## [2026-04-13] experiment | Multiplication baseline — 3-digit, 2L/384D (no scratchpad)
- Model: 2L/4H/384D (3.56M params), MLX on Apple M4
- Result: 85.15% accuracy, 607s training
- Accuracy cliff by output length: 1-3d=100%, 4d=96%, 5d=66%, 6d=32%
- Multiplication fundamentally harder — model capacity-limited at 2 layers
- Findings created: experiment-multiplication-baseline
- Agent: experimenter

## [2026-04-13] experiment | Multiplication deep — 3-digit, 4L/256D
- Model: 4L/4H/256D (3.16M params), MLX on Apple M4
- Result: 94.90% accuracy, 664s training — +9.75pp over 2L/384D
- 5d=93%, 6d=60% (vs 2L: 5d=66%, 6d=32%)
- Depth >> width for multiplication; partial-product accumulation needs layers
- Still not fully solved at 6-digit — scratchpad likely needed
- Findings created: experiment-multiplication-4L256D
- Agent: experimenter

## [2026-04-13] update | Spec updated with sub/mul experimental results
- Operation-specific architecture recommendation added
- Add/Sub: 2L/384D (99.9%), Mul: 4L/256D (94.9%)
- Key finding: "data format > architecture" holds for add/sub but NOT multiplication
- V1-safe promoted to primary config for multiplication workloads
- Agent: team-lead

## [2026-04-14] analyze | Scratchpad format research
- Researched scratchpad formats from Lee et al. (ICLR 2024), ICoT (Deng 2024), and others
- Recommended format: aligned reversed partial products with `|` separator
- Each partial product = single-digit × multi-digit (easy), shifted by position (zero-prepend)
- ~2.3x sequence length overhead for 3-digit multiplication
- Findings created: scratchpad-format-research
- Agent: scratchpad-engineer

## [2026-04-14] experiment | Multiplication scratchpad — 4L/256D
- Model: 4L/4H/256D (3.17M params), MLX on Apple M4
- Result: **100.00%** accuracy (peak at epoch 70), 2340s training
- 6-digit results: 60% → 100% (vs non-scratchpad baseline)
- +5.1pp over non-scratchpad (94.9% → 100%)
- Findings created: experiment-scratchpad-4L256D
- Agent: scratchpad-engineer

## [2026-04-14] experiment | Multiplication scratchpad — 2L/384D
- Model: 2L/4H/384D (3.57M params), MLX on Apple M4
- Result: **100.00%** accuracy (first 100% at epoch 35), 1970s training
- 6-digit results: 32% → 100% (+68pp, most dramatic improvement)
- +14.85pp over non-scratchpad (85.15% → 100%)
- **KEY RESULT**: scratchpad eliminates depth bottleneck — 2L/384D now universal
- Findings created: experiment-scratchpad-2L384D
- Agent: scratchpad-engineer

## [2026-04-14] update | Spec updated with scratchpad results
- Operation-specific architecture recommendation unified: 2L/384D for all ops
- Scratchpad format validated and documented
- Decision log updated: mult format validated
- Open question #1 (scratchpad format) marked resolved
- Agent: scratchpad-engineer

## [2026-04-14] create | experiment-mixed-operations

- Added `--op mixed` mode to training script
- Generates equal proportions of add, sub, mul with per-op data formatting
- Trained 2L/4H/384D for 80 epochs on 50k mixed examples
- Results: add 99.1%, sub 99.7%, mul 97.5% (overall 98.75%)
- No catastrophic interference between operations
- Created finding: experiment-mixed-operations.md
- Created results: mixed_ops_results.md
- Agent: mixed-ops
