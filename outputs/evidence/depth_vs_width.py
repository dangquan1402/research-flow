"""
Evidence: Width matters more than depth; 2L/384D is the sweet spot
Finding: depth-vs-width-arithmetic, experiment-depth-vs-width-ablation
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

configs = [
    ("8L/4H/128D\n(1.58M)", "add_5d_reversed_8L4H128D_depth_8L128D.json", '#f59e0b'),
    ("4L/4H/256D\n(3.16M)", "add_5d_reversed_4L4H256D_baseline.json", '#8b5cf6'),
    ("2L/4H/384D\n(3.56M)", "add_5d_reversed_2L4H384D_width_2L384D.json", '#2563eb'),
    ("2L/8H/512D\n(6.32M)", "add_5d_reversed_2L8H512D_width_2L512D.json", '#dc2626'),
]

data = []
for label, fname, color in configs:
    with open(f"{RESULTS_DIR}/{fname}") as f:
        d = json.load(f)
    data.append((label, d, color))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Depth vs Width Ablation: 5-Digit Addition", fontsize=14, fontweight='bold')

# --- Panel 1: Convergence curves ---
ax = axes[0]
for label, d, color in data:
    epochs = [e["epoch"] for e in eval_epochs(d)]
    accs = [e["accuracy"] for e in eval_epochs(d)]
    ax.plot(epochs, accs, 'o-', color=color, label=label.replace('\n', ' '), linewidth=2, markersize=3)

ax.axhline(y=0.95, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Convergence Curves', fontsize=12)
ax.legend(fontsize=9, loc='lower right')
ax.set_ylim(0.3, 1.02)
ax.grid(True, alpha=0.3)

# --- Panel 2: Params vs Peak Accuracy + Time to 95% ---
ax = axes[1]
params = [1.58, 3.16, 3.56, 6.32]
peak_accs = []
time_to_95 = []
colors = [c for _, _, c in configs]

for label, d, color in data:
    accs = [e["accuracy"] for e in eval_epochs(d)]
    peak_accs.append(max(accs))
    t95 = None
    for e in eval_epochs(d):
        if e["accuracy"] >= 0.95:
            t95 = e["epoch"]
            break
    time_to_95.append(t95 if t95 else 50)

ax2 = ax.twinx()
bars = ax.bar(range(len(params)), peak_accs, color=colors, alpha=0.7, width=0.5)
ax.set_ylim(0.99, 1.002)
ax.set_ylabel('Peak Accuracy', fontsize=12, color='#2563eb')
ax.set_xticks(range(len(params)))
ax.set_xticklabels([f'{p}M' for p in params], fontsize=10)
ax.set_xlabel('Parameters', fontsize=12)

ax2.plot(range(len(params)), time_to_95, 's--', color='#dc2626', linewidth=2, markersize=8)
ax2.set_ylabel('Epochs to 95%', fontsize=12, color='#dc2626')
ax2.set_ylim(0, 25)

ax.set_title('Peak Accuracy & Convergence Speed', fontsize=12)

for i, (peak, t95) in enumerate(zip(peak_accs, time_to_95)):
    ax.text(i, peak + 0.0005, f'{peak:.1%}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('outputs/evidence/depth_vs_width.png', dpi=150, bbox_inches='tight')
print("Saved: outputs/evidence/depth_vs_width.png")
