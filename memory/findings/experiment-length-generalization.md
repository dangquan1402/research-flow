---
title: "Experiment: Length Generalization (5-digit → 6-10 digit addition)"
created: 2026-04-14
updated: 2026-04-14
source: experiment
confidence: high
tags: [experiment, length-generalization, position-coupling, APE, addition, OOD]
---

# Length Generalization Experiment

## Setup
- **Task**: Integer addition, trained on 1–5 digits, evaluated OOD on 6–10 digits
- **Architecture**: 2L / 4H / 384D, reversed (LSB-first) output
- **Conditions**: Learned APE vs Position Coupling (Cho 2024)
- **Training**: 50 epochs, batch 256, 50k training samples
- **OOD eval**: 500 samples per digit count (1–10)

## Results

| Digits | APE | PC | ID/OOD |
|--------|------:|------:|--------|
| 1 | 100.0% | 20.0% | ID |
| 2 | 99.8% | 28.8% | ID |
| 3 | 100.0% | 87.2% | ID |
| 4 | 100.0% | 90.8% | ID |
| 5 | 100.0% | 88.0% | ID |
| 6 | 0.0% | 4.6% | OOD |
| 7 | 0.0% | 0.8% | OOD |
| 8 | 0.0% | 0.2% | OOD |
| 9 | 0.0% | 0.0% | OOD |
| 10 | 0.0% | 0.0% | OOD |

## Key Findings

1. **Learned APE exhibits a hard length generalization cliff**: Perfect ID accuracy collapses to exactly 0% at 6+ digits. The model has memorized position-specific patterns that don't transfer.

2. **Position Coupling did not converge in 50 epochs**: Final accuracy 77% vs 99.95% for APE. Loss was still decreasing at epoch 50. This is an **inconclusive** comparison for OOD — the PC model hasn't mastered the ID task.

3. **Weak directional OOD signal from PC**: 4.6% at 6 digits vs 0% for APE. The position coupling mechanism provides some length transfer even without convergence, suggesting the approach has merit but needs more training.

4. **PC has anomalous low accuracy on 1–2 digit examples**: 20% and 28.8% despite these being trivially easy for APE. Possible causes: random offset [1, 100] creates too much positional noise for short sequences, or the position embedding space (256 entries) is too sparse.

## Confidence Assessment

- **APE fails at length generalization**: HIGH confidence — clear 100%→0% cliff, consistent with literature
- **PC enables length generalization**: INCONCLUSIVE — model didn't converge, needs more epochs
- **PC mechanism is directionally correct**: MEDIUM — the 4.6% OOD signal is above random chance but below practical utility

## What Would Disprove This?

If PC with sufficient training (200+ epochs, full convergence) still shows <10% accuracy on 6-10 digit OOD, the implementation may be incorrect or the approach may not work with reversed output format (the paper uses standard left-to-right decoding).

## Next Steps

- Run PC for 200+ epochs to achieve convergence
- Ablate random offset range (try [1, 20] for short sequences)
- Try standard (non-reversed) output format as in the original paper
