---
title: "Open Question: Do We Need Length Generalization?"
created: 2026-04-13
tags: [length-generalization, scope, design-decision, open-question]
priority: high
---

## Question

Should our small arithmetic LLM be designed to generalize to number lengths beyond the training distribution? Or is it acceptable to define a fixed maximum digit count and train on the full range?

## Context

Position-aware embeddings (Abacus, Position Coupling) are the key to length generalization but add architectural complexity. If we define our scope as "integers up to N digits" and train on all lengths 1--N, standard positional encodings with digit-level tokenization and reversed output may suffice.

The tradeoff:
- **With length generalization:** More complex architecture, but train on shorter numbers and generalize. Potentially smaller training set.
- **Without length generalization:** Simpler architecture, but must train on the full range. Training set grows with max digit count.

## Why It Matters

This is a fundamental scope decision that affects architecture, training data, and evaluation. It should be decided before implementation begins.

## Factors to Consider
- What is the target use case? If "simple math" means up to ~10-digit numbers, training on the full range is feasible.
- Abacus Embeddings are a relatively small modification (embedding layer only). The cost/benefit may favor including them even if not strictly necessary.
- Position Coupling requires task-specific position ID logic, which is more complex to implement.
