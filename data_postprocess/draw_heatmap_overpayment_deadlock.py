import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns

output_dir = "../figures"
os.makedirs(output_dir, exist_ok=True)

model_order   = ["DeepSeek-R1", "DeepSeek-V3", "GPT-4.1", "o4-mini", "o3",
                 "GPT-4o-mini", "GPT-3.5", "Qwen2.5-7B", "Qwen2.5-14B"]
budget_labels = ["Low", "Wholesale", "Mid", "Retail", "High"]

# ── Overpayment: rows=budget, cols=models ─────────────────────────────────────
opr = np.array([
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],   # Low
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],   # Wholesale
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],   # Mid
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],   # Retail
    [0.0, 0.0, 0.0, 0.0, 0.0, 3.4, 3.0, 5.9, 2.4],   # High
])

# ── Deadlock: rows=models, cols=budget ────────────────────────────────────────
dlr = np.array([
    [0.0, 0.0, 0.0, 0.0, 0.0],   # DeepSeek-R1
    [0.0, 0.0, 0.0, 0.0, 0.0],   # DeepSeek-V3
    [0.0, 0.0, 0.0, 0.0, 0.0],   # GPT-4.1
    [0.0, 0.0, 0.0, 0.0, 0.0],   # o4-mini
    [0.0, 0.0, 0.0, 0.0, 0.0],   # o3
    [0.0, 0.0, 0.3, 0.3, 0.0],   # GPT-4o-mini
    [1.0, 0.7, 0.3, 0.3, 0.0],   # GPT-3.5
    [1.3, 2.3, 0.0, 1.3, 1.3],   # Qwen2.5-7B
    [1.0, 0.3, 0.3, 0.7, 0.0],   # Qwen2.5-14B
])

# ── figure ────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.0))

sns.heatmap(
    opr, ax=ax1,
    cmap=sns.cubehelix_palette(start=.5, rot=-.75, as_cmap=True),
    xticklabels=model_order, yticklabels=budget_labels,
    vmin=0, annot=True, annot_kws={"size": 9}, fmt=".1f",
    cbar=False,
)
ax1.set_xticklabels(model_order, rotation=45, ha="right", fontsize=9)
ax1.set_yticklabels(budget_labels, rotation=0, fontsize=9)
ax1.set_title("Overpayment Rate by Budget Type", fontsize=10, pad=4)

sns.heatmap(
    dlr, ax=ax2,
    cmap=sns.color_palette("ch:s=-.2,r=.6", as_cmap=True),
    xticklabels=budget_labels, yticklabels=model_order,
    annot=True, annot_kws={"size": 9}, fmt=".1f",
    cbar=False,
)
ax2.set_xticklabels(budget_labels, rotation=45, ha="right", fontsize=9)
ax2.set_yticklabels(model_order, rotation=0, fontsize=9)
ax2.set_title("Deadlock (Max Turn) Rate (%) by Model and Budget", fontsize=10, pad=4)

plt.tight_layout(w_pad=2)
plt.savefig(os.path.join(output_dir, "heatmap_overpayment_deadlock.pdf"), dpi=300, bbox_inches="tight")
plt.savefig(os.path.join(output_dir, "heatmap_overpayment_deadlock.svg"), dpi=300, bbox_inches="tight")
print("Saved to", output_dir)
plt.show()
