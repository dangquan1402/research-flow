---
title: "Paper: Understanding Addition and Subtraction in Transformers"
created: 2026-04-13
updated: 2026-04-13
type: paper
tags: [arithmetic, transformer, interpretability, carry-mechanism]
staleness_days: 0
---

## Overview

- **Authors:** Nikankin, Dotan, Kibriya (and collaborators)
- **Published:** 2024 (arXiv: 2402.02619)
- **URL:** https://arxiv.org/abs/2402.02619

## Key Contribution

Reverse-engineered how transformers implement addition and subtraction, discovering that 2 layers and 3 attention heads are the minimal configuration for >99.999% accuracy. Identified 3 specialized head roles (base addition, carry detection, carry cascading) and showed all independently trained models converge on the same algorithmic solution.

## Architecture

- 2 layers, 3 attention heads — minimal for near-perfect addition
- Models up to 10M parameters tested

## Referenced In

- [attention-heads-carry-operations]
- [minimum-viable-math-transformer]
- [depth-vs-width-arithmetic]
