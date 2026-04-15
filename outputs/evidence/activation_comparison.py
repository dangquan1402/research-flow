"""
Evidence: SwiGLU converges faster than GELU/ReLU² at equal parameter count
Finding: experiment-swiglu-fair-comparison
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

activations = {
    "GELU": ("add_5d_reversed_2L4H384D_act_gelu.json", '#2563eb'),
    "SwiGLU": ("add_5d_reversed_2L4H384D_swiglu_fair_swiglu.json", '#10b981'),
    "ReLU²": ("add_5d_reversed_2L4H384D_act_relu2.json", '#f59e0b'),
}

data = {}
for label, (fname, color) in activations.items():
    with open(f"{RESULTS_DIR}/{fname}") as f:
        data[label] = (json.load(f), color)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Activation Function Comparison: 5-Digit Addition (2L/4H/384D, ~3.56M params)", fontsize=14, fontweight='bold')

# --- Panel 1: Learning curves ---
ax = axes[0]
for label, (d, color) in data.items():
    epochs = [e["epoch"] for e in eval_epochs(d)]
    accs = [e["accuracy"] for e in eval_epochs(d)]
    ax.plot(epochs, accs, 'o-', color=color, label=label, linewidth=2, markersize=4)

ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Learning Curves', fontsize=12)
ax.legend(fontsize=11)
ax.set_ylim(0.3, 1.02)
ax.grid(True, alpha=0.3)

# --- Panel 2: Loss curves ---
ax = axes[1]
for label, (d, color) in data.items():
    epochs = [e["epoch"] for e in eval_epochs(d)]
    losses = [e["loss"] for e in eval_epochs(d)]
    ax.plot(epochs, losses, 'o-', color=color, label=label, linewidth=2, markersize=4)

ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Eval Loss', fontsize=12)
ax.set_title('Loss Curves', fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/evidence/activation_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: outputs/evidence/activation_comparison.png")
