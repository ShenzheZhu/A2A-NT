import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

models = ["DeepSeek-R1", "DeepSeek-V3", "GPT-4.1", "o4-mini", "o3",
          "GPT-4o-mini", "GPT-3.5", "Qwen2.5-7B", "Qwen2.5-14B"]
obr = [1.69, 0.53, 2.18, 2.98, 2.73, 0.36, 6.25, 11.76, 4.78]
owr = [0.50, 0.87, 0.71, 0.31, 0.46, 1.79, 5.75, 7.91, 2.14]

x = np.arange(len(models))
w = 0.38

fig, ax = plt.subplots(figsize=(7, 1.9))

bars1 = ax.bar(x - w/2, obr, w, label="Out-of-Budget Rate (OBR)",
               color="#7b52ab", edgecolor="white", linewidth=0.5)
bars2 = ax.bar(x + w/2, owr, w, label="Out-of-Wholesale Rate (OWR)",
               color="#3aab6d", edgecolor="white", linewidth=0.5)

ax.set_xticks(x)
ax.set_xticklabels(models, rotation=35, ha="right", fontsize=9)
ax.set_ylabel("Rate (%)", fontsize=9)
ax.yaxis.set_tick_params(labelsize=9)
ax.legend(fontsize=8.5, framealpha=0.8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_axisbelow(True)
ax.yaxis.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
out = "../figures/obr_owr_bar.pdf"
os.makedirs("../figures", exist_ok=True)
plt.savefig(out, dpi=300, bbox_inches="tight")
plt.savefig(out.replace(".pdf", ".svg"), dpi=300, bbox_inches="tight")
print("Saved to", out)
