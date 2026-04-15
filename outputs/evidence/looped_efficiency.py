"""
Evidence: Looped transformers achieve near-baseline accuracy with 87% fewer parameters
Finding: experiment-looped-192D-addition, experiment-looped-128D-addition, experiment-looped-mul-scratchpad
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

# All configs: (label, filename, params_M, color, marker)
configs = [
    ("1Lx4 128D\n(add)", "add_5d_reversed_1L4H128D_looped_1Lx4_128D.json", 0.204, '#dc2626', 'X'),
    ("1Lx4 192D\n(add)", "add_5d_reversed_1L4H192D_looped_1Lx4_192D.json", 0.454, '#10b981', 'o'),
    ("1Lx4 192D\n(mul+sp)", "mul_3d_reversed_1L4H192D_looped_mul_scratchpad.json", 0.456, '#8b5cf6', 's'),
    ("2L 384D\n(add)", "add_5d_reversed_2L4H384D_width_2L384D.json", 3.56, '#2563eb', 'D'),
    ("2L 384D\n(mul)", "mul_3d_reversed_2L4H384D_mul_baseline_2L384D.json", 3.56, '#f59e0b', '^'),
    ("2L 384D\n(mul+sp)", "mul_3d_reversed_2L4H384D_scratchpad_2L384D.json", 3.56, '#06b6d4', 'v'),
]

data = []
for label, fname, params, color, marker in configs:
    with open(f"{RESULTS_DIR}/{fname}") as f:
        d = json.load(f)
    peak = max(e["accuracy"] for e in eval_epochs(d))
    final = eval_epochs(d)[-1]["accuracy"]
    data.append((label, params, peak, final, color, marker))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Parameter Efficiency: Looped vs Standard Transformers", fontsize=14, fontweight='bold')

# --- Panel 1: Params vs Peak Accuracy scatter ---
ax = axes[0]
for label, params, peak, final, color, marker in data:
    ax.scatter(params, peak, color=color, marker=marker, s=120, zorder=5, edgecolors='black', linewidth=0.5)
    # offset label based on position
    offset_x = 0.1 if params < 1 else -0.3
    offset_y = -0.03 if 'mul' in label and 'sp' not in label else 0.01
    ax.annotate(label.replace('\n', ' '), (params, peak),
               xytext=(params + offset_x, peak + offset_y), fontsize=8)

ax.set_xlabel('Parameters (M)', fontsize=12)
ax.set_ylabel('Peak Accuracy', fontsize=12)
ax.set_title('Accuracy vs Model Size', fontsize=12)
ax.set_xscale('log')
ax.set_xlim(0.15, 5)
ax.set_ylim(0.3, 1.05)
ax.grid(True, alpha=0.3)

# Add efficiency frontier annotation
ax.annotate('87% fewer params\nsame accuracy', xy=(0.454, 0.992), xytext=(1.0, 0.85),
           fontsize=10, fontweight='bold', color='#059669',
           arrowprops=dict(arrowstyle='->', color='#059669', lw=2),
           bbox=dict(boxstyle='round,pad=0.3', facecolor='#ecfdf5', edgecolor='#059669'))

# --- Panel 2: Comparison bars for looped vs standard ---
ax = axes[1]
comparisons = [
    ("Addition\n5-digit", 0.454, 0.992, 3.56, 1.0),
    ("Mul+Scratch\n3-digit", 0.456, 0.994, 3.56, 1.0),
]

x = range(len(comparisons))
width = 0.3
for i, (task, lp, la, sp, sa) in enumerate(comparisons):
    ax.bar(i - width/2, la, width, color='#10b981', alpha=0.8,
           label='Looped (~0.45M)' if i == 0 else '')
    ax.bar(i + width/2, sa, width, color='#2563eb', alpha=0.8,
           label='Standard (~3.6M)' if i == 0 else '')
    ax.text(i - width/2, la + 0.003, f'{la:.1%}\n{lp:.0f}K', ha='center', fontsize=9, fontweight='bold')
    ax.text(i + width/2, sa + 0.003, f'{sa:.0%}\n{sp:.1f}M', ha='center', fontsize=9, fontweight='bold')

ax.set_xticks(list(x))
ax.set_xticklabels([c[0] for c in comparisons], fontsize=11)
ax.set_ylabel('Peak Accuracy', fontsize=12)
ax.set_title('Looped vs Standard: Near-Parity at 8x Fewer Params', fontsize=12)
ax.set_ylim(0.95, 1.03)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('outputs/evidence/looped_efficiency.png', dpi=150, bbox_inches='tight')
print("Saved: outputs/evidence/looped_efficiency.png")
