"""
Evidence: Scratchpad dramatically improves multiplication accuracy
Finding: experiment-scratchpad-2L384D, experiment-scratchpad-4L256D, scratchpad-chain-of-thought
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

experiments = {
    "2L/384D\nno scratch": ("mul_3d_reversed_2L4H384D_mul_baseline_2L384D.json", '#dc2626'),
    "2L/384D\n+ scratch": ("mul_3d_reversed_2L4H384D_scratchpad_2L384D.json", '#2563eb'),
    "4L/256D\nno scratch": ("mul_3d_reversed_4L4H256D_mul_4L256D.json", '#f59e0b'),
    "4L/256D\n+ scratch": ("mul_3d_reversed_4L4H256D_scratchpad_4L256D.json", '#10b981'),
}

data = {}
for label, (fname, color) in experiments.items():
    with open(f"{RESULTS_DIR}/{fname}") as f:
        data[label] = (json.load(f), color)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Scratchpad Impact on 3-Digit Multiplication", fontsize=14, fontweight='bold')

# --- Panel 1: Learning curves ---
ax = axes[0]
for label, (d, color) in data.items():
    epochs = [e["epoch"] for e in eval_epochs(d)]
    accs = [e["accuracy"] for e in eval_epochs(d)]
    linestyle = '-' if 'scratch' in label else '--'
    ax.plot(epochs, accs, linestyle, color=color, label=label.replace('\n', ' '),
            linewidth=2, markersize=3, marker='o')

ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Learning Curves', fontsize=12)
ax.legend(fontsize=9)
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3)

# --- Panel 2: Before/After bar chart ---
ax = axes[1]
labels_short = ['2L/384D', '4L/256D']
no_scratch = [0.8515, 0.949]
with_scratch = [1.0, 1.0]
deltas = [with_scratch[i] - no_scratch[i] for i in range(2)]

x = range(len(labels_short))
width = 0.3
bars1 = ax.bar([i - width/2 for i in x], no_scratch, width, label='No Scratchpad', color='#dc2626', alpha=0.8)
bars2 = ax.bar([i + width/2 for i in x], with_scratch, width, label='+ Scratchpad', color='#2563eb', alpha=0.8)

for i, (ns, ws, delta) in enumerate(zip(no_scratch, with_scratch, deltas)):
    ax.annotate(f'+{delta:.1%}', xy=(i + width/2, ws),
               xytext=(i + width/2 + 0.15, ws - 0.05), fontsize=11, fontweight='bold', color='#059669',
               arrowprops=dict(arrowstyle='->', color='#059669', lw=1.5))

for bar, val in zip(bars1, no_scratch):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.1%}', ha='center', va='bottom', fontsize=10)
for bar, val in zip(bars2, with_scratch):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.0%}', ha='center', va='bottom', fontsize=10)

ax.set_xlabel('Architecture', fontsize=12)
ax.set_ylabel('Peak Accuracy', fontsize=12)
ax.set_title('Scratchpad: Before vs After', fontsize=12)
ax.set_xticks(list(x))
ax.set_xticklabels(labels_short, fontsize=11)
ax.set_ylim(0.7, 1.08)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('outputs/evidence/scratchpad_impact.png', dpi=150, bbox_inches='tight')
print("Saved: outputs/evidence/scratchpad_impact.png")
