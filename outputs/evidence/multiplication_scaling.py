"""
Evidence: Multiplication difficulty scales steeply with digit count; scratchpad is essential
Finding: experiment-scratchpad-2L384D, scratchpad-chain-of-thought, multiplication-scaling-requirements
Generated: 2026-04-15
"""
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np

RESULTS_DIR = "experiments/results"

def eval_epochs(data):
    """Filter epoch_logs to only entries with accuracy (eval every 5 epochs)."""
    return [e for e in data["epoch_logs"] if "accuracy" in e]

# 3-digit mul experiments
configs_3d = [
    ("2L/384D\nno scratch", "mul_3d_reversed_2L4H384D_mul_baseline_2L384D.json", '#dc2626'),
    ("4L/256D\nno scratch", "mul_3d_reversed_4L4H256D_mul_4L256D.json", '#f59e0b'),
    ("2L/384D\n+ scratch", "mul_3d_reversed_2L4H384D_scratchpad_2L384D.json", '#2563eb'),
    ("4L/256D\n+ scratch", "mul_3d_reversed_4L4H256D_scratchpad_4L256D.json", '#10b981'),
    ("Looped 192D\n+ scratch", "mul_3d_reversed_1L4H192D_looped_mul_scratchpad.json", '#8b5cf6'),
]

# 5-digit mul
configs_5d = [
    ("2L/384D 5d\n+ scratch", "mul_5d_reversed_2L4H384D_mul_5digit_2L384D.json", '#06b6d4'),
]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Multiplication: Difficulty Scaling & Scratchpad Necessity", fontsize=14, fontweight='bold')

# --- Panel 1: 3-digit mul convergence ---
ax = axes[0]
for label, fname, color in configs_3d:
    with open(f"{RESULTS_DIR}/{fname}") as f:
        d = json.load(f)
    epochs = [e["epoch"] for e in eval_epochs(d)]
    accs = [e["accuracy"] for e in eval_epochs(d)]
    ls = '-' if 'scratch' in label.lower() else '--'
    ax.plot(epochs, accs, ls, color=color, label=label.replace('\n', ' '), linewidth=2, markersize=3, marker='o')

ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('3-Digit Multiplication: All Configs', fontsize=12)
ax.legend(fontsize=8, loc='lower right')
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3)

# --- Panel 2: Digit scaling ---
ax = axes[1]

# Collect best accuracy at each digit count
digit_counts = [3, 5]
best_no_scratch = [0.949, None]  # 3d: 4L256D no scratch; no 5d without scratch tested
best_with_scratch = [1.0, 0.7785]  # 3d: 2L384D+sp; 5d: 2L384D+sp

x = np.arange(len(digit_counts))
width = 0.3

bars_sp = ax.bar(x, [b if b else 0 for b in best_with_scratch], width,
                 label='Best with Scratchpad', color='#2563eb', alpha=0.85)
# Only plot no-scratch where we have data
bars_nosp = ax.bar([0 + width], [0.949], width,
                   label='Best without Scratchpad', color='#dc2626', alpha=0.85)

for i, (sp, nsp) in enumerate(zip(best_with_scratch, best_no_scratch)):
    if sp:
        ax.text(i, sp + 0.02, f'{sp:.1%}', ha='center', fontsize=11, fontweight='bold', color='#2563eb')
    if nsp:
        ax.text(i + width, nsp + 0.02, f'{nsp:.1%}', ha='center', fontsize=11, fontweight='bold', color='#dc2626')

ax.set_xlabel('Digit Count', fontsize=12)
ax.set_ylabel('Best Accuracy Achieved', fontsize=12)
ax.set_title('Multiplication Scaling: 3-digit vs 5-digit', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels([f'{d}-digit' for d in digit_counts], fontsize=12)
ax.set_ylim(0, 1.15)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

ax.annotate('5-digit mul remains\nan open challenge', xy=(1, 0.7785),
           xytext=(1.3, 0.5), fontsize=10, color='#dc2626',
           arrowprops=dict(arrowstyle='->', color='#dc2626', lw=1.5),
           bbox=dict(boxstyle='round', facecolor='#fef2f2'))

plt.tight_layout()
plt.savefig('outputs/evidence/multiplication_scaling.png', dpi=150, bbox_inches='tight')
print("Saved: outputs/evidence/multiplication_scaling.png")
