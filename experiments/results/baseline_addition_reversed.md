# Baseline Experiment: 5-Digit Addition with Reversed Output

## Configuration

| Parameter | Value |
|---|---|
| Operation | Addition |
| Max digits | 5 |
| Tokenizer | Reversed (LSB-first) |
| Balanced carry | Yes |
| Architecture | 4L / 4H / 256D / 1024 FFN |
| Parameters | 3,161,856 |
| Train samples | 50,000 |
| Test samples | 2,000 |
| Epochs | 50 |
| Batch size | 256 |
| Optimizer | AdamW (lr=1e-3, wd=0.5, beta2=0.99) |
| Schedule | Cosine with 100-step warmup |
| Framework | MLX 0.31.1 (Apple M4, Metal) |
| Total time | 517s (~8.6 min) |

## Results

| Metric | Value |
|---|---|
| Final accuracy | **99.90%** |
| Peak accuracy | **100.00%** (epoch 45) |
| Final loss | 1.3006 |

### Accuracy by digit count (epoch 50)

| Digits | Accuracy |
|---|---|
| 1 | 100% |
| 2 | 100% |
| 3 | 100% |
| 4 | 100% |
| 5 | 100% |
| 6 | 100% |

### Training progression

- Epoch 5: 49.45% — model starts learning, 1-digit and 3-digit strong
- Epoch 10: 86.95% — rapid improvement across all digit counts
- Epoch 15: 95.50% — near-converged
- Epoch 25: 97.90% — continuing gradual improvement
- Epoch 35: 99.00% — approaching perfect
- Epoch 45: 100.00% — perfect accuracy
- Epoch 50: 99.90% — slight regression (1 error in 2000)

## Key Observations

1. **Reversed output works as predicted.** The model achieves near-perfect accuracy on 5-digit addition with reversed (LSB-first) output, confirming the research findings.
2. **Small model suffices.** Only 3.16M parameters needed for >99% accuracy.
3. **Training is fast.** Under 9 minutes on Apple M4 with MLX.
4. **Balanced carry sampling ensures uniform performance** across all digit counts.
5. **6-digit answers** (from carry out of 5+5 digit addition) are handled well, showing some generalization beyond training operand length.
