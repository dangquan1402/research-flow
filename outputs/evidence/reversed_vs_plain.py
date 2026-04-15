"""
Evidence: Reversed (LSB-first) output is faster and more accurate than plain (MSB-first)
Finding: reversed-digit-order, experiment-tokenization-comparison
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

with open(f"{RESULTS_DIR}/add_5d_reversed_4L4H256D_baseline.json") as f:
    reversed_data = json.load(f)
with open(f"{RESULTS_DIR}/add_5d_plain_4L4H256D_tokenizer_plain.json") as f:
    plain_data = json.load(f)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Reversed vs Plain Output: 5-Digit Addition (4L/4H/256D)", fontsize=14, fontweight='bold')

# --- Panel 1: Learning curves ---
ax = axes[0]
rev_epochs = [e["epoch"] for e in eval_epochs(reversed_data)]
rev_acc = [e["accuracy"] for e in eval_epochs(reversed_data)]
plain_epochs = [e["epoch"] for e in eval_epochs(plain_data)]
plain_acc = [e["accuracy"] for e in eval_epochs(plain_data)]

ax.plot(rev_epochs, rev_acc, 'o-', color='#2563eb', label='Reversed (LSB-first)', linewidth=2, markersize=4)
ax.plot(plain_epochs, plain_acc, 's-', color='#dc2626', label='Plain (MSB-first)', linewidth=2, markersize=4)
ax.axhline(y=0.95, color='gray', linestyle='--', alpha=0.5, label='95% threshold')
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Learning Curves', fontsize=12)
ax.legend(fontsize=10)
ax.set_ylim(0.2, 1.02)
ax.grid(True, alpha=0.3)

# Annotate 95% crossing points
for epochs, accs, color, name in [(rev_epochs, rev_acc, '#2563eb', 'Rev'), (plain_epochs, plain_acc, '#dc2626', 'Plain')]:
    for i in range(len(accs)):
        if accs[i] >= 0.95:
            ax.annotate(f'{name}: ep {epochs[i]}', xy=(epochs[i], accs[i]),
                       xytext=(epochs[i]+3, accs[i]-0.05), fontsize=8, color=color,
                       arrowprops=dict(arrowstyle='->', color=color, lw=0.8))
            break

# --- Panel 2: Accuracy by digit count ---
ax = axes[1]
rev_final = eval_epochs(reversed_data)[-1]
plain_final = eval_epochs(plain_data)[-1]

digits = sorted(rev_final["digit_accuracy"].keys(), key=int)
rev_by_digit = [rev_final["digit_accuracy"][d] for d in digits]
plain_by_digit = [plain_final["digit_accuracy"][d] for d in digits]

x = range(len(digits))
width = 0.35
bars1 = ax.bar([i - width/2 for i in x], rev_by_digit, width, label='Reversed', color='#2563eb', alpha=0.85)
bars2 = ax.bar([i + width/2 for i in x], plain_by_digit, width, label='Plain', color='#dc2626', alpha=0.85)

ax.set_xlabel('Digit Count', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Final Accuracy by Digit Count (Epoch 50)', fontsize=12)
ax.set_xticks(list(x))
ax.set_xticklabels([f'{d}d' for d in digits])
ax.set_ylim(0.9, 1.005)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

for bar, val in zip(bars1, rev_by_digit):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, f'{val:.0%}', ha='center', va='bottom', fontsize=8, color='#2563eb')
for bar, val in zip(bars2, plain_by_digit):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, f'{val:.0%}', ha='center', va='bottom', fontsize=8, color='#dc2626')

plt.tight_layout()
plt.savefig('outputs/evidence/reversed_vs_plain.png', dpi=150, bbox_inches='tight')
print("Saved: outputs/evidence/reversed_vs_plain.png")
