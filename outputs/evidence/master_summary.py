"""
Evidence: Master summary of all experiments — one view of the full research
Finding: all findings
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

# All experiments with metadata
experiments = [
    # (short_label, filename, op, arch_type, params_M)
    ("Baseline add 4L256D", "add_5d_reversed_4L4H256D_baseline.json", "add", "standard", 3.16),
    ("Add 8L128D (deep)", "add_5d_reversed_8L4H128D_depth_8L128D.json", "add", "standard", 1.58),
    ("Add 2L384D (wide)", "add_5d_reversed_2L4H384D_width_2L384D.json", "add", "standard", 3.56),
    ("Add 2L512D (v.wide)", "add_5d_reversed_2L8H512D_width_2L512D.json", "add", "standard", 6.32),
    ("Add plain output", "add_5d_plain_4L4H256D_tokenizer_plain.json", "add", "standard", 3.16),
    ("Add GELU", "add_5d_reversed_2L4H384D_act_gelu.json", "add", "standard", 3.56),
    ("Add SwiGLU", "add_5d_reversed_2L4H384D_swiglu_fair_swiglu.json", "add", "standard", 3.56),
    ("Add ReLU²", "add_5d_reversed_2L4H384D_act_relu2.json", "add", "standard", 3.56),
    ("Add pretrain", "add_5d_reversed_2L4H384D_pretrain_add.json", "add", "standard", 3.56),
    ("Add looped 192D", "add_5d_reversed_1L4H192D_looped_1Lx4_192D.json", "add", "looped", 0.454),
    ("Add looped 128D", "add_5d_reversed_1L4H128D_looped_1Lx4_128D.json", "add", "looped", 0.204),
    ("Sub 2L384D", "sub_5d_reversed_2L4H384D_sub_2L384D.json", "sub", "standard", 3.56),
    ("Mul 4L256D", "mul_3d_reversed_4L4H256D_mul_4L256D.json", "mul", "standard", 3.16),
    ("Mul 2L384D", "mul_3d_reversed_2L4H384D_mul_baseline_2L384D.json", "mul", "standard", 3.56),
    ("Mul 4L256D+sp", "mul_3d_reversed_4L4H256D_scratchpad_4L256D.json", "mul", "standard", 3.16),
    ("Mul 2L384D+sp", "mul_3d_reversed_2L4H384D_scratchpad_2L384D.json", "mul", "standard", 3.56),
    ("Mul looped+sp", "mul_3d_reversed_1L4H192D_looped_mul_scratchpad.json", "mul", "looped", 0.456),
    ("Mul 5-digit", "mul_5d_reversed_2L4H384D_mul_5digit_2L384D.json", "mul", "standard", 3.56),
    ("Mixed 2L384D", "mixed_5d_reversed_2L4H384D_mixed_2L384D.json", "mixed", "standard", 2.78),
    ("Mixed scratch", "mixed_5d_reversed_2L4H384D_mixed_scratch.json", "mixed", "standard", 3.56),
    ("Mixed SwiGLU", "mixed_5d_reversed_2L4H384D_swiglu_fair_mixed.json", "mixed", "standard", 3.56),
    ("Mixed looped", "mixed_5d_reversed_1L4H192D_looped_finetune_mixed.json", "mixed", "looped", 0.463),
]

rows = []
for label, fname, op, arch, params in experiments:
    with open(f"{RESULTS_DIR}/{fname}") as f:
        d = json.load(f)
    accs = [e["accuracy"] for e in eval_epochs(d)]
    peak = max(accs)
    final = accs[-1]
    peak_ep = [e["epoch"] for e in eval_epochs(d)][accs.index(peak)]
    rows.append((label, op, arch, params, final, peak, peak_ep))

# Sort by operation then accuracy
rows.sort(key=lambda r: ({"add": 0, "sub": 1, "mul": 2, "mixed": 3}[r[1]], -r[5]))

fig, ax = plt.subplots(figsize=(16, 10))
ax.axis('off')
ax.set_title("Master Experiment Summary — All Results", fontsize=16, fontweight='bold', pad=20)

headers = ["Experiment", "Op", "Type", "Params", "Final Acc", "Peak Acc", "Peak Ep"]
cell_text = []
cell_colors = []

for label, op, arch, params, final, peak, peak_ep in rows:
    param_str = f"{params:.2f}M" if params >= 1 else f"{params*1000:.0f}K"
    row = [label, op, arch, param_str, f"{final:.1%}", f"{peak:.1%}", str(peak_ep)]
    cell_text.append(row)

    # Color code by peak accuracy
    if peak >= 0.99:
        bg = '#ecfdf5'  # green
    elif peak >= 0.95:
        bg = '#fefce8'  # yellow
    elif peak >= 0.80:
        bg = '#fff7ed'  # orange
    else:
        bg = '#fef2f2'  # red
    cell_colors.append([bg] * len(headers))

table = ax.table(cellText=cell_text, colLabels=headers, cellColours=cell_colors,
                cellLoc='center', loc='center',
                colColours=['#e2e8f0'] * len(headers))

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.0, 1.4)

# Bold headers
for j in range(len(headers)):
    table[0, j].set_text_props(fontweight='bold', fontsize=10)

plt.savefig('outputs/evidence/master_summary.png', dpi=150, bbox_inches='tight')
print("Saved: outputs/evidence/master_summary.png")
