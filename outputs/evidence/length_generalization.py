"""
Evidence: Standard APE fails completely on OOD lengths; Position Coupling shows weak signal
Finding: experiment-length-generalization, experiment-position-coupling-vs-ape
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

with open(f"{RESULTS_DIR}/add_5d_reversed_2L4H384D_ape_lengthgen.json") as f:
    ape_data = json.load(f)
with open(f"{RESULTS_DIR}/add_5d_reversed_2L4H384D_position_coupling_pc_lengthgen.json") as f:
    pc_data = json.load(f)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Length Generalization: APE vs Position Coupling (2L/4H/384D)", fontsize=14, fontweight='bold')

# --- Panel 1: OOD accuracy by digit count ---
ax = axes[0]

# Get OOD accuracy (top-level key with digits 1-10)
ape_ood = ape_data["ood_accuracy"]
pc_ood = pc_data["ood_accuracy"]

digits = list(range(1, 11))
ape_by_digit = [ape_ood.get(str(d), 0) for d in digits]
pc_by_digit = [pc_ood.get(str(d), 0) for d in digits]

x = np.arange(len(digits))
width = 0.35
bars1 = ax.bar(x - width/2, ape_by_digit, width, label='APE (Learned)', color='#2563eb', alpha=0.85)
bars2 = ax.bar(x + width/2, pc_by_digit, width, label='Position Coupling', color='#f59e0b', alpha=0.85)

# Mark ID vs OOD boundary
ax.axvline(x=4.5, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax.text(2.0, 0.55, 'In-Distribution\n(trained)', ha='center', fontsize=10, color='#059669',
       bbox=dict(boxstyle='round', facecolor='#ecfdf5', alpha=0.8))
ax.text(7.0, 0.55, 'Out-of-Distribution\n(never seen)', ha='center', fontsize=10, color='#dc2626',
       bbox=dict(boxstyle='round', facecolor='#fef2f2', alpha=0.8))

ax.set_xlabel('Digit Count', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Accuracy by Digit Count (Final Epoch)', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels([str(d) for d in digits])
ax.set_ylim(-0.05, 1.1)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

# --- Panel 2: Learning curves ---
ax = axes[1]
ape_epochs = [e["epoch"] for e in eval_epochs(ape_data)]
ape_accs = [e["accuracy"] for e in eval_epochs(ape_data)]
pc_epochs = [e["epoch"] for e in eval_epochs(pc_data)]
pc_accs = [e["accuracy"] for e in eval_epochs(pc_data)]

ax.plot(ape_epochs, ape_accs, 'o-', color='#2563eb', label='APE', linewidth=2, markersize=4)
ax.plot(pc_epochs, pc_accs, 's-', color='#f59e0b', label='Position Coupling', linewidth=2, markersize=4)

ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Overall Accuracy (ID + OOD)', fontsize=12)
ax.set_title('Convergence (50 epochs)', fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

ax.annotate('PC not converged\nat 50 epochs', xy=(50, pc_accs[-1]),
           xytext=(35, pc_accs[-1] - 0.15), fontsize=10, color='#f59e0b',
           arrowprops=dict(arrowstyle='->', color='#f59e0b', lw=1.5))

plt.tight_layout()
plt.savefig('outputs/evidence/length_generalization.png', dpi=150, bbox_inches='tight')
print("Saved: outputs/evidence/length_generalization.png")
