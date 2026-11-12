"""Figures summarizing black-box adversarial evasion rates."""
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

df = pd.read_csv(METRICDIR / "adversarial_evasion.csv")

MODEL_ORDER = ["RandomForest", "XGBoost", "LinearSVM", "MLP", "LogisticRegression", "MultinomialNB"]
TRANSFORM_ORDER = ["case_randomization", "whitespace_insertion", "url_encoding",
                   "comment_injection", "padding_dilution", "combined"]

# ---- Figure 1: heatmap, in-distribution, model x transform ----
idd = df[df.split == "in_distribution_test"].copy()
pivot = idd.pivot(index="model", columns="transform", values="evasion_rate")
pivot = pivot.reindex(index=MODEL_ORDER, columns=TRANSFORM_ORDER)

fig, ax = plt.subplots(figsize=(8, 4.5))
im = ax.imshow(pivot.values, cmap="Reds", vmin=0, vmax=max(0.05, np.nanmax(pivot.values)), aspect="auto")
ax.set_xticks(range(len(TRANSFORM_ORDER)))
ax.set_xticklabels(TRANSFORM_ORDER, rotation=30, ha="right")
ax.set_yticks(range(len(MODEL_ORDER)))
ax.set_yticklabels(MODEL_ORDER)
for i in range(pivot.shape[0]):
    for j in range(pivot.shape[1]):
        v = pivot.values[i, j]
        if not np.isnan(v):
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                     color="white" if v > pivot.values[~np.isnan(pivot.values)].max() * 0.6 else "black",
                     fontsize=9)
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Evasion rate")
ax.set_title("Adversarial evasion rate: in-distribution test set")
fig.tight_layout()
fig.savefig(FIGDIR / "adversarial_evasion_heatmap_id.png", bbox_inches="tight")
plt.close(fig)

# ---- Figure 2: heatmap, OOD ----
ood = df[df.split == "ood_lexical_test"].copy()
pivot_ood = ood.pivot(index="model", columns="transform", values="evasion_rate")
pivot_ood = pivot_ood.reindex(index=MODEL_ORDER, columns=TRANSFORM_ORDER)

fig, ax = plt.subplots(figsize=(8, 4.5))
im = ax.imshow(pivot_ood.values, cmap="Reds", vmin=0, vmax=max(0.05, np.nanmax(pivot_ood.values)), aspect="auto")
ax.set_xticks(range(len(TRANSFORM_ORDER)))
ax.set_xticklabels(TRANSFORM_ORDER, rotation=30, ha="right")
ax.set_yticks(range(len(MODEL_ORDER)))
ax.set_yticklabels(MODEL_ORDER)
for i in range(pivot_ood.shape[0]):
    for j in range(pivot_ood.shape[1]):
        v = pivot_ood.values[i, j]
        if not np.isnan(v):
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                     color="white" if v > np.nanmax(pivot_ood.values) * 0.6 else "black",
                     fontsize=9)
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Evasion rate")
ax.set_title("Adversarial evasion rate: OOD lexical test set")
fig.tight_layout()
fig.savefig(FIGDIR / "adversarial_evasion_heatmap_ood.png", bbox_inches="tight")
plt.close(fig)

print("Wrote heatmaps.")
print("\n=== ID: mean evasion rate per model ===")
print(idd.groupby("model")["evasion_rate"].mean().reindex(MODEL_ORDER).to_string())
print("\n=== OOD: mean evasion rate per model ===")
print(ood.groupby("model")["evasion_rate"].mean().reindex(MODEL_ORDER).to_string())
print("\n=== combined transform, per model, ID vs OOD ===")
comb = df[df["transform"] == "combined"].pivot(index="model", columns="split", values="evasion_rate")
print(comb.reindex(MODEL_ORDER).to_string())
