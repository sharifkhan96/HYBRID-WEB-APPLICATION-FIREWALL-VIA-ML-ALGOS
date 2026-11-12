"""Figures comparing untuned vs. hyperparameter-tuned models, ID and OOD."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
METRICDIR = ROOT / "results" / "metrics"
FIGDIR = ROOT / "results" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

PALETTE = ["#2a78d6", "#008300", "#e87ba4", "#eda100",
           "#1baf7a", "#eb6834", "#4a3aa7", "#e34948"]
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#c3c2b7", "axes.grid": True,
    "grid.color": "#e1e0d9", "grid.linewidth": 0.6, "axes.axisbelow": True,
})

df = pd.read_csv(METRICDIR / "hyperparameter_tuning.csv")
df = df.sort_values("tuned_ood_acc", ascending=False).reset_index(drop=True)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

x = np.arange(len(df))
w = 0.35

ax = axes[0]
ax.bar(x - w/2, df["untuned_id_acc"], width=w, label="Untuned", color=PALETTE[0])
ax.bar(x + w/2, df["tuned_id_acc"], width=w, label="Tuned", color=PALETTE[4])
ax.set_xticks(x)
ax.set_xticklabels(df["model"], rotation=20, ha="right")
ax.set_ylabel("Accuracy")
ax.set_ylim(0.99, 1.001)
ax.set_title("In-distribution: untuned vs. tuned")
ax.legend(loc="lower right", fontsize=8)

ax = axes[1]
ax.bar(x - w/2, df["untuned_ood_acc"], width=w, label="Untuned", color=PALETTE[0])
ax.bar(x + w/2, df["tuned_ood_acc"], width=w, label="Tuned", color=PALETTE[4])
ax.set_xticks(x)
ax.set_xticklabels(df["model"], rotation=20, ha="right")
ax.set_ylabel("Accuracy")
ax.set_ylim(0, 1.0)
ax.set_title("Out-of-distribution: untuned vs. tuned")
ax.legend(loc="upper right", fontsize=8)

fig.tight_layout()
fig.savefig(FIGDIR / "hyperparameter_tuning_comparison.png", bbox_inches="tight")
plt.close(fig)

print("Wrote", FIGDIR / "hyperparameter_tuning_comparison.png")
print(df[["model", "untuned_id_acc", "tuned_id_acc", "untuned_ood_acc", "tuned_ood_acc"]].to_string(index=False))
