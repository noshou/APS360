"""Quick summary plot from baseline_results.json."""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_PATH = "Baselines/baseline_results.json"
OUT_PATH     = "Baselines/baseline_summary.png"

LABELS = {
    "mean_iq":     "MeanIQ",
    "composition": "Composition",
    "mean_ff":     "MeanFF",
    "rg":          "Rg",
    "nn_dist":     "NNDist",
    "pair_peak":   "PairPeak",
    "power_law":   "PowerLaw",
    "atom_count":  "AtomCount",
}

with open(RESULTS_PATH) as f:
    results = json.load(f)

keys   = list(LABELS.keys())
names  = [LABELS[k] for k in keys]
r2     = [results[k]["test"]["r2"]   for k in keys]
msle   = [results[k]["test"]["msle"] for k in keys]

order  = np.argsort(r2)[::-1]
names  = [names[i]  for i in order]
r2     = [r2[i]     for i in order]
msle   = [msle[i]   for i in order]

colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in r2]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

bars = ax1.barh(names, r2, color=colors, edgecolor="white", linewidth=0.5)
ax1.axvline(0, color="black", linewidth=0.8)
ax1.set_xlabel("Test R²  (log1p scale)")
ax1.set_title("Baseline R² (higher = better)")
ax1.grid(axis="x", alpha=0.3)
for bar, v in zip(bars, r2):
    ax1.text(v + (0.02 if v >= 0 else -0.02), bar.get_y() + bar.get_height() / 2,
             f"{v:.3f}", va="center", ha="left" if v >= 0 else "right", fontsize=8)

bars2 = ax2.barh(names, msle, color="#4c9be8", edgecolor="white", linewidth=0.5)
ax2.set_xlabel("Test MSLE  (log1p scale, lower = better)")
ax2.set_title("Baseline MSLE (lower = better)")
ax2.grid(axis="x", alpha=0.3)
for bar, v in zip(bars2, msle):
    ax2.text(v + 0.1, bar.get_y() + bar.get_height() / 2,
             f"{v:.3f}", va="center", ha="left", fontsize=8)

fig.suptitle("Baseline Comparison — Test Set", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"Saved → {OUT_PATH}")
