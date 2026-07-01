"""
Baseline runner with per-batch and global RAM profiling.

Saves to Baselines/:
  baseline_results.json        - msle / r2 for every baseline × split
  memory_profile.png           - per-batch RAM + global timeline
  plots/<name>.png             - per-baseline scatter + example curves
"""

import json
import os
import time
import threading

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import psutil
import torch
from dataclasses import replace as dc_replace
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Preprocess              import Encoding
from ScatterNet.batching     import Batcher, Batch
from Baselines               import Baseline, MeanIQBaseline, AtomCountBaseline
from Baselines.chemistry_only  import CompositionBaseline, MeanFFBaseline
from Baselines.global_geometry import RgBaseline, PowerLawBaseline
from Baselines.local_geometry  import NNBaseline, PairPeakBaseline

# ── config ────────────────────────────────────────────────────────────────────

HDF5_PATH    = "Preprocess/I(q)@L=50.h5"
DB_NAME      = "Preprocess/scatternet"
OUT_DIR      = "Baselines"
PLOTS_DIR    = f"{OUT_DIR}/plots"
RESULTS_PATH = f"{OUT_DIR}/baseline_results.json"
MEMORY_PLOT  = f"{OUT_DIR}/memory_profile.png"
BATCHER_SEED = 0
ATOM_SIZE_CEIL: int = -1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# How many molecules to collect for per-baseline scatter/curve plots
N_SCATTER_MOLS  = 2000
N_CURVE_EXAMPLES = 9   # shown in a 3×3 grid

BUCKETS: list[tuple[int, int]] = [
    (     1,      3), (     4,      6), (     7,     12), (    13,     14),
    (    15,     16), (    17,     17), (    18,     18), (    19,     19),
    (    20,     20), (    21,     21), (    22,     23), (    24,     26),
    (    27,     33), (    34,     40), (    41,     45), (    46,     50),
    (    51,     55), (    56,     60), (    61,     64), (    65,     69),
    (    70,     74), (    75,     80), (    81,     84), (    85,     90),
    (    91,     96), (    97,    102), (   103,    108), (   109,    116),
    (   117,    124), (   125,    132), (   133,    142), (   143,    152),
    (   153,    160), (   161,    170), (   171,    180), (   181,    192),
    (   193,    202), (   203,    216), (   217,    228), (   229,    242),
    (   243,    258), (   259,    276), (   277,    296), (   297,    316),
    (   317,    336), (   337,    364), (   365,    392), (   393,    428),
    (   429,    472), (   473,    524), (   525,    596), (   597,    696),
    (   697,    856), (   857,   1208), (  1209,   3177), (  3178,   4251),
    (  4252,   6046), (  6047,  78819),
]

# ── memory helpers ────────────────────────────────────────────────────────────

_proc = psutil.Process()

def rss_gb() -> float:
    return _proc.memory_info().rss / 1024**3

_timeline: list[tuple[float, float]] = []
_stop_sampling = threading.Event()

def _sampler(interval: float = 0.5):
    t0 = time.perf_counter()
    while not _stop_sampling.is_set():
        _timeline.append((time.perf_counter() - t0, rss_gb()))
        time.sleep(interval)

_batch_records: list[dict] = []
_batch_counter = 0

# ── setup ─────────────────────────────────────────────────────────────────────

os.makedirs(PLOTS_DIR, exist_ok=True)

print("Loading HDF5 metadata...")
with h5py.File(HDF5_PATH, "r") as f:
    q_grid = torch.tensor(f["q_grid"][:]).float()
energy = 12_500.0  # eV; not stored in HDF5, fixed pipeline constant

q_np = q_grid.numpy()

print("Building encoding (or reusing existing)...")
enc = Encoding(DB_NAME, HDF5_PATH)

print("Building batcher...")
batcher = Batcher(
    hdf5_db        = HDF5_PATH,
    enc            = enc,
    batches        = BUCKETS,
    seed           = BATCHER_SEED,
    atom_size_ceil = ATOM_SIZE_CEIL,
)
train_set, val_set, test_set = batcher.get_sets()

def _loader(dataset):
    return DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x[0])

train_loader = _loader(train_set)
val_loader   = _loader(val_set)
test_loader  = _loader(test_set)

print(f"Batches - train: {len(train_set)}  val: {len(val_set)}  test: {len(test_set)}")

# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate_baseline(
    baseline:      Baseline,
    loader:        DataLoader,
    baseline_name: str,
    record_memory: bool = False,
    collect_samples: bool = False,
) -> tuple[float, float, list, list, list]:
    """Returns (msle, r2, pred_curves, actual_curves, atom_counts) where
    pred/actual_curves are lists of (Q,) numpy arrays (up to N_SCATTER_MOLS total)."""
    global _batch_counter
    ss_res = 0.0
    sum_y  = 0.0
    sum_y2 = 0.0
    n_elem = 0

    pred_curves:   list = []
    actual_curves: list = []
    atom_counts:   list = []

    with torch.no_grad():
        for batch in loader:
            batch = dc_replace(
                batch,
                vocab=batch.vocab.to(DEVICE),
                iqval=batch.iqval.to(DEVICE),
                coord=batch.coord.to(DEVICE),
            )

            if record_memory:
                mem_before = rss_gb()

            pred       = baseline(batch)
            log_pred   = torch.log1p(pred.clamp(min=0))
            log_target = torch.log1p(batch.iqval)

            if record_memory:
                mem_after = rss_gb()
                n_mols  = batch.iqval.shape[0]
                n_atoms = int(batch.coord.shape[0] * batch.coord.shape[1])
                _batch_records.append({
                    "idx":        _batch_counter,
                    "baseline":   baseline_name,
                    "n_mols":     n_mols,
                    "n_atoms":    n_atoms,
                    "mem_before": mem_before,
                    "mem_after":  mem_after,
                    "mem_delta":  mem_after - mem_before,
                })
                _batch_counter += 1

            ss_res += ((log_pred - log_target) ** 2).sum().item()
            sum_y  += log_target.sum().item()
            sum_y2 += (log_target ** 2).sum().item()
            n_elem += log_target.numel()

            if collect_samples and len(pred_curves) < N_SCATTER_MOLS:
                mask = batch.padding_mask()  # (N, M)
                for i in range(batch.iqval.shape[0]):
                    if len(pred_curves) >= N_SCATTER_MOLS:
                        break
                    pred_curves.append(pred[i].cpu().numpy())
                    actual_curves.append(batch.iqval[i].cpu().numpy())
                    n_real = mask[i].sum().item()
                    atom_counts.append(int(n_real))

    if n_elem == 0:
        return float("nan"), float("nan"), [], [], []

    msle   = ss_res / n_elem
    ss_tot = sum_y2 - sum_y ** 2 / n_elem
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return msle, r2, pred_curves, actual_curves, atom_counts


def plot_baseline(name: str, pred_curves: list, actual_curves: list, atom_counts: list,
                  test_msle: float, test_r2: float) -> None:
    """Save a 2-panel figure: scatter (log1p pred vs actual) + 9 example curves."""
    fig = plt.figure(figsize=(18, 9))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[1, 1.6], wspace=0.35)

    # ── left: scatter log1p(pred) vs log1p(actual) ──────────────────────────
    ax_sc = fig.add_subplot(gs[0])
    preds_flat   = np.concatenate([np.log1p(np.maximum(p, 0)) for p in pred_curves])
    actuals_flat = np.concatenate([np.log1p(a) for a in actual_curves])
    counts_rep   = np.repeat(atom_counts, len(q_np))

    sc = ax_sc.scatter(actuals_flat, preds_flat, s=0.3, alpha=0.15,
                       c=np.log1p(counts_rep), cmap="plasma", rasterized=True)
    lim = (min(actuals_flat.min(), preds_flat.min()),
           max(actuals_flat.max(), preds_flat.max()))
    ax_sc.plot(lim, lim, "r--", linewidth=1, label="ideal")
    ax_sc.set_xlim(lim); ax_sc.set_ylim(lim)
    ax_sc.set_xlabel("log1p(actual I(q))")
    ax_sc.set_ylabel("log1p(predicted I(q))")
    ax_sc.set_title(f"Predicted vs actual\nMSLE={test_msle:.4f}  R²={test_r2:.4f}")
    ax_sc.legend(fontsize=8)
    ax_sc.grid(True, alpha=0.2)
    plt.colorbar(sc, ax=ax_sc, label="log1p(atom count)", pad=0.01)

    # ── right: 3×3 example I(q) curves ──────────────────────────────────────
    ax_c = fig.add_subplot(gs[1])
    ax_c.set_visible(False)
    inner = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=gs[1],
                                             hspace=0.55, wspace=0.4)

    # pick examples spanning small / medium / large atom counts
    order  = np.argsort(atom_counts)
    step   = max(1, len(order) // N_CURVE_EXAMPLES)
    chosen = [order[i * step] for i in range(N_CURVE_EXAMPLES)]

    for k, idx in enumerate(chosen):
        ax = fig.add_subplot(inner[k // 3, k % 3])
        ax.plot(q_np, actual_curves[idx],   color="#333333", linewidth=1.2, label="actual")
        ax.plot(q_np, np.maximum(pred_curves[idx], 0), color="#e05a2b",
                linewidth=1.0, linestyle="--", label="pred")
        ax.set_title(f"{atom_counts[idx]} atoms", fontsize=7, pad=2)
        ax.set_xlabel("q (Å⁻¹)", fontsize=6)
        ax.set_ylabel("I(q)", fontsize=6)
        ax.tick_params(labelsize=5)
        ax.grid(True, alpha=0.2)
        if k == 0:
            ax.legend(fontsize=5, loc="upper right")

    fig.suptitle(name, fontsize=14, fontweight="bold")
    out = f"{PLOTS_DIR}/{name}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  plot saved → {out}")


def run(name: str, baseline: Baseline, needs_fit: bool = False) -> dict:
    if needs_fit:
        print(f"  fitting {name}...")
        baseline.fit(train_loader)

    print(f"  evaluating {name}...")
    train_msle, train_r2, _, _, _ = evaluate_baseline(
        baseline, train_loader, name, record_memory=True)
    val_msle, val_r2, _, _, _ = evaluate_baseline(
        baseline, val_loader, name)
    test_msle, test_r2, pred_c, act_c, atoms = evaluate_baseline(
        baseline, test_loader, name, collect_samples=True)

    print(
        f"  {name}\n"
        f"    train  msle={train_msle:.4f}  r2={train_r2:.4f}\n"
        f"    val    msle={val_msle:.4f}  r2={val_r2:.4f}\n"
        f"    test   msle={test_msle:.4f}  r2={test_r2:.4f}"
    )

    if pred_c:
        plot_baseline(name, pred_c, act_c, atoms, test_msle, test_r2)

    return {
        "train": {"msle": train_msle, "r2": train_r2},
        "val":   {"msle": val_msle,   "r2": val_r2},
        "test":  {"msle": test_msle,  "r2": test_r2},
    }

# ── run ───────────────────────────────────────────────────────────────────────

print("\nStarting memory sampler...")
sampler_thread = threading.Thread(target=_sampler, args=(0.5,), daemon=True)
sampler_thread.start()

print("Running baselines...\n")
t_start = time.perf_counter()

results: dict = {}
results["mean_iq"]     = run("MeanIQBaseline",     MeanIQBaseline(),                   needs_fit=True)
results["atom_count"]  = run("AtomCountBaseline",  AtomCountBaseline(),                needs_fit=True)
results["composition"] = run("CompositionBaseline",CompositionBaseline(),              needs_fit=True)
results["mean_ff"]     = run("MeanFFBaseline",     MeanFFBaseline(q_grid, energy))
results["rg"]          = run("RgBaseline",         RgBaseline(q_grid, energy))
results["power_law"]   = run("PowerLawBaseline",   PowerLawBaseline(q_grid),           needs_fit=True)
results["nn_dist"]     = run("NNBaseline",         NNBaseline(q_grid, energy))
results["pair_peak"]   = run("PairPeakBaseline",   PairPeakBaseline(q_grid, energy))

t_total = time.perf_counter() - t_start
_stop_sampling.set()
sampler_thread.join()

# ── save results ──────────────────────────────────────────────────────────────

with open(RESULTS_PATH, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved → {RESULTS_PATH}")

# ── memory plot ───────────────────────────────────────────────────────────────

BASELINE_COLORS = {
    "MeanIQBaseline":      "#4c9be8",
    "AtomCountBaseline":   "#e07b39",
    "CompositionBaseline": "#9b59b6",
    "MeanFFBaseline":      "#5cb85c",
    "RgBaseline":          "#e74c3c",
    "PowerLawBaseline":    "#f0c040",
    "NNBaseline":          "#1abc9c",
    "PairPeakBaseline":    "#d35400",
}

fig = plt.figure(figsize=(18, 12))
gs  = gridspec.GridSpec(3, 1, hspace=0.45)

ax1 = fig.add_subplot(gs[0])
if _timeline:
    ts, mems = zip(*_timeline)
    ax1.plot(ts, mems, color="#4c9be8", linewidth=1.2, label="RSS GB")
    ax1.axhline(max(mems), color="red", linestyle="--", linewidth=1,
                label=f"Peak {max(mems):.2f} GB")
    ax1.fill_between(ts, mems, alpha=0.15, color="#4c9be8")
ax1.set_xlabel("Wall time (s)")
ax1.set_ylabel("RAM (GB)")
ax1.set_title(f"Global RAM usage over full run  (total {t_total:.0f}s)")
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

ax2 = fig.add_subplot(gs[1])
if _batch_records:
    seen = set()
    for rec in _batch_records:
        bl    = rec["baseline"]
        color = BASELINE_COLORS.get(bl, "#888888")
        label = bl if bl not in seen else None
        ax2.scatter(rec["idx"], rec["mem_after"], s=4, color=color,
                    alpha=0.5, label=label)
        seen.add(bl)
    ax2.set_xlabel("Batch index (train pass only)")
    ax2.set_ylabel("RAM after batch (GB)")
    ax2.set_title("Per-batch RAM - each dot = one training batch, coloured by baseline")
    ax2.legend(loc="upper left", fontsize=7, markerscale=3)
    ax2.grid(True, alpha=0.3)

ax3 = fig.add_subplot(gs[2])
if _batch_records:
    seen = set()
    for rec in _batch_records:
        bl    = rec["baseline"]
        color = BASELINE_COLORS.get(bl, "#888888")
        label = bl if bl not in seen else None
        ax3.scatter(rec["n_atoms"], rec["mem_delta"] * 1024,
                    s=4, color=color, alpha=0.4, label=label)
        seen.add(bl)
    ax3.set_xlabel("Atoms in batch (padded total)")
    ax3.set_ylabel("ΔRAM per batch (MB)")
    ax3.set_title("Memory delta per batch vs batch size")
    ax3.legend(loc="upper left", fontsize=7, markerscale=3)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(0, color="gray", linewidth=0.8)

plt.suptitle("Baseline Run - Memory Profile", fontsize=14, fontweight="bold", y=1.01)
plt.savefig(MEMORY_PLOT, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Memory plot saved → {MEMORY_PLOT}")

if _batch_records:
    peak_batch = max(_batch_records, key=lambda r: r["mem_after"])
    print(f"\nMemory summary:")
    print(f"  Peak RSS:            {max(m for _,m in _timeline):.2f} GB")
    print(f"  Peak batch (atoms):  {peak_batch['n_atoms']:,}  ({peak_batch['baseline']})")
    print(f"  Total batches:       {len(_batch_records):,}")
    print(f"  Total wall time:     {t_total:.0f}s")
