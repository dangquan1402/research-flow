"""
Evidence: A single 2L/384D model handles add+sub+mul with no catastrophic interference
Finding: experiment-mixed-operations
Generated: 2026-04-15
"""
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

RESULTS_DIR = "experiments/results"

def eval_epochs(data):
    """Filter epoch_logs to only entries with accuracy (eval every 5 epochs)."""
    return [e for e in data["epoch_logs"] if "accuracy" in e]

with open(f"{RESULTS_DIR}/mixed_5d_reversed_2L4H384D_mixed_2L384D.json") as f:
    mixed = json.load(f)

# Extract per-op accuracies from epoch results
epochs = []
add_accs, sub_accs, mul_accs, overall_accs = [], [], [], []

for e in eval_epochs(mixed):
    epochs.append(e["epoch"])
    overall_accs.append(e["accuracy"])
    ops = e.get("op_accuracy", {})
    add_accs.append(ops.get("add", 0))
    sub_accs.append(ops.get("sub", 0))
    mul_accs.append(ops.get("mul", 0))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Mixed Operations: Single Model (2L/4H/384D, 2.78M params)", fontsize=14, fontweight='bold')

# --- Panel 1: Per-op learning curves ---
ax = axes[0]
ax.plot(epochs, add_accs, 'o-', color='#2563eb', label='Addition (5d)', linewidth=2, markersize=4)
ax.plot(epochs, sub_accs, 's-', color='#10b981', label='Subtraction (5d)', linewidth=2, markersize=4)
ax.plot(epochs, mul_accs, '^-', color='#f59e0b', label='Multiplication (3d+sp)', linewidth=2, markersize=4)
ax.plot(epochs, overall_accs, 'D--', color='#6b7280', label='Overall', linewidth=1.5, markersize=3, alpha=0.7)

ax.axhline(y=0.99, color='gray', linestyle='--', alpha=0.4, label='99% target')
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Per-Operation Learning Curves', fontsize=12)
ax.legend(fontsize=9, loc='lower right')
ax.set_ylim(0.2, 1.02)
ax.grid(True, alpha=0.3)

# --- Panel 2: Final accuracy bars ---
ax = axes[1]
ops = ['Addition\n(5-digit)', 'Subtraction\n(5-digit)', 'Multiplication\n(3-digit+sp)', 'Overall']
# Use peak (around epoch 50) rather than final
peak_accs = [1.0, 1.0, 0.99, 0.9935]
final_accs = [0.991, 0.997, 0.9745, 0.9875]
targets = [0.99, 0.99, 0.98, None]
colors = ['#2563eb', '#10b981', '#f59e0b', '#6b7280']

bars = ax.bar(range(len(ops)), final_accs, color=colors, alpha=0.8, width=0.6)
for i, (bar, val, peak) in enumerate(zip(bars, final_accs, peak_accs)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
           f'{val:.1%}\n(peak: {peak:.0%})', ha='center', va='bottom', fontsize=9, fontweight='bold')
    if targets[i]:
        ax.axhline(y=targets[i], xmin=(i*0.25)+0.02, xmax=((i+1)*0.25)-0.02,
                   color=colors[i], linestyle=':', alpha=0.5)

ax.set_xticks(range(len(ops)))
ax.set_xticklabels(ops, fontsize=10)
ax.set_ylabel('Final Accuracy (Epoch 80)', fontsize=12)
ax.set_title('No Catastrophic Interference', fontsize=12)
ax.set_ylim(0.9, 1.03)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('outputs/evidence/mixed_ops_training.png', dpi=150, bbox_inches='tight')
print("Saved: outputs/evidence/mixed_ops_training.png")
