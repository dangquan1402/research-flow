---
title: "Open Question: Optimal Architecture for Mixed Arithmetic Operations"
created: 2026-04-13
updated: 2026-04-13
source: architecture-sizing-research
priority: high
research_goal: "GH-3-architecture-sizing"
tags: [architecture, arithmetic, mixed-operations]
staleness_days: 0
---

## Question

What is the minimum architecture that achieves >95% accuracy on addition, subtraction, AND multiplication simultaneously? Published work mostly studies these operations in isolation. The few mixed-operation experiments used 2-3 layers with 3-4 heads, but comprehensive benchmarks are lacking.

## Why It Matters

Our arithmetic LLM needs all three operations. If addition needs 2L/3H and multiplication needs 2L/3H with auxiliary losses, does a combined model need 3L/6H? Or can a single 2L/4H model handle all three? The answer determines our parameter budget and architecture choice.

## Current Evidence

- Addition: 2L/3H sufficient (Nikankin 2024)
- Multiplication: 2L with auxiliary losses achieves 99% on 4x4 (2025)
- Mixed: "14 mixed models with various architectures, including 6- and 10-digit models with 2-3 layers and 3-4 attention heads" (Nikankin 2024) — but detailed results not fully extracted
- No published ablation specifically testing how many extra heads/layers mixing operations requires

## Possible Approaches

1. Train multiple single-operation models and compare against mixed model of same total size
2. Progressive training: learn addition first, then add subtraction, then multiplication
3. Task-specific tokens to route to different attention patterns
