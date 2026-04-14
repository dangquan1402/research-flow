# Length Generalization: Position Coupling vs Learned APE

## Experiment Setup

| Parameter | Value |
|---|---|
| Task | Integer addition |
| Training digits | 1–5 |
| OOD eval digits | 6–10 |
| Architecture | 2L / 4H / 384D |
| Epochs | 50 |
| Batch size | 256 |
| Output format | Reversed (LSB-first) |
| Framework | MLX |
| OOD eval samples | 500 per digit count |

## OOD Accuracy by Digit Count

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

## Summary

```
Digits: 1      2      3      4      5     | 6     7     8     9     10
APE:    100.0  99.8   100.0  100.0  100.0 | 0.0   0.0   0.0   0.0   0.0
PC:     20.0   28.8   87.2   90.8   88.0  | 4.6   0.8   0.2   0.0   0.0
```

## Key Findings

1. **APE achieves perfect in-distribution accuracy** (99.95% final) but completely fails on OOD lengths (0% for 6–10 digits). This confirms the known limitation of learned positional encodings for length generalization.

2. **Position Coupling did NOT converge in 50 epochs** (77% final accuracy vs 99.95% for APE). The model was still improving at epoch 50 (loss: 1.09, trending down). This makes the OOD comparison unfair — the PC model hasn't mastered the ID task.

3. **PC shows a weak OOD signal** (4.6% at 6 digits, 0.8% at 7) compared to APE's flat 0%. This is directionally correct — the position coupling mechanism does provide some length generalization — but the effect is minimal without full convergence.

4. **PC struggles on 1–2 digit examples** (20%, 28.8%) despite these being the simplest. This may be an artifact of the random position offset [1, 100] interacting poorly with very short sequences, or insufficient training.

## Next Steps

- **Run PC for more epochs** (200–500) to achieve convergence before comparing OOD performance
- **Investigate the 1–2 digit accuracy drop** in PC — may need to adjust random offset range
- **Try larger model** — PC may need more capacity to learn the position-coupled representations
- **Compare with other PE methods**: RoPE, ALiBi, NoPE

## Raw Data Files

- APE: `add_5d_reversed_2L4H384D_ape_lengthgen.json`
- PC: `add_5d_reversed_2L4H384D_position_coupling_pc_lengthgen.json`
