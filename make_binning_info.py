"""Regenerate Preprocess/binning_info/ from the encoding DB.

Usage:
    python make_binning_info.py [--n-bins N]

Reads atom counts from Preprocess/iq_train_set-ENCODING.sqlite3, computes
N equal-molecule-count bins, writes all_counts.npy + four plots +
binning_info.txt, and patches DEFAULT_BUCKETS in ScatterNet/utils/config.py.
"""

import argparse, sqlite3, textwrap, datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

DB       = "Preprocess/iq_train_set-ENCODING.sqlite3"
OUT_DIR  = Path("Preprocess/binning_info")
CFG_PATH = Path("ScatterNet/utils/config.py")

# ── load ─────────────────────────────────────────────────────────────────────

conn  = sqlite3.connect(DB)
rows  = conn.execute("SELECT atoms, grp FROM items ORDER BY atoms").fetchall()
conn.close()

atoms_all  = np.array([r[0] for r in rows], dtype=np.int32)
grp_all    = [r[1] for r in rows]
N          = len(atoms_all)
print(f"Loaded {N:,} molecules  min={atoms_all.min()}  max={atoms_all.max()}")

np.save(OUT_DIR / "all_counts.npy", atoms_all)

# ── bins ──────────────────────────────────────────────────────────────────────

# Fixed bins — same boundaries as before, recompute stats against new distribution
BINS = [
    (    1,    3), (    4,    6), (    7,   12), (   13,   14), (   15,   16),
    (   17,   17), (   18,   18), (   19,   19), (   20,   20), (   21,   21),
    (   22,   23), (   24,   26), (   27,   33), (   34,   40), (   41,   45),
    (   46,   50), (   51,   55), (   56,   60), (   61,   64), (   65,   69),
    (   70,   74), (   75,   80), (   81,   84), (   85,   90), (   91,   96),
    (   97,  102), (  103,  108), (  109,  116), (  117,  124), (  125,  132),
    (  133,  142), (  143,  152), (  153,  160), (  161,  170), (  171,  180),
    (  181,  192), (  193,  202), (  203,  216), (  217,  228), (  229,  242),
    (  243,  258), (  259,  276), (  277,  296), (  297,  316), (  317,  336),
    (  337,  364), (  365,  392), (  393,  428), (  429,  472), (  473,  524),
    (  525,  596), (  597,  696), (  697,  856), (  857, 1208), ( 1209, 3177),
    ( 3178, 4251), ( 4252, 6046),
]
print(f"Using {len(BINS)} fixed bins")

# ── per-bin stats ─────────────────────────────────────────────────────────────

bin_lo   = np.array([b[0] for b in BINS], dtype=np.int32)
bin_hi   = np.array([b[1] for b in BINS], dtype=np.int32)

mol_counts = np.zeros(len(BINS), dtype=np.int64)
atom_totals= np.zeros(len(BINS), dtype=np.int64)
bin_means  = np.zeros(len(BINS))
bin_p95    = np.zeros(len(BINS))
bin_max    = np.zeros(len(BINS), dtype=np.int32)
pad_ratios = np.zeros(len(BINS))

for i, (lo, hi) in enumerate(BINS):
    mask = (atoms_all >= lo) & (atoms_all <= hi)
    sub  = atoms_all[mask]
    if len(sub) == 0:
        continue
    mol_counts[i]  = len(sub)
    atom_totals[i] = sub.sum()
    bin_means[i]   = sub.mean()
    bin_p95[i]     = np.percentile(sub, 95)
    bin_max[i]     = sub.max()
    pad_ratios[i]  = bin_means[i] / bin_max[i] if bin_max[i] > 0 else 1.0

# ── group breakdown ───────────────────────────────────────────────────────────

from collections import Counter
grp_counts = Counter(grp_all)
groups_sorted = sorted(grp_counts.items(), key=lambda x: -x[1])

# per-group atom stats
grp_atoms = {}
for a, g in zip(atoms_all, grp_all):
    grp_atoms.setdefault(g, []).append(a)
grp_stats = {}
for g, alist in grp_atoms.items():
    arr = np.array(alist)
    grp_stats[g] = (len(arr), arr.sum(), arr.min(), arr.max(), arr.mean())

# ── plots ─────────────────────────────────────────────────────────────────────

COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]

def vlines(ax, color="gray"):
    for lo, hi in BINS:
        ax.axvline(lo, color=color, alpha=0.25, lw=0.5)

# 1. atom count histogram (all groups, log-scale)
fig, ax = plt.subplots(figsize=(14, 5))
edges = np.linspace(0, atoms_all.max() + 1, 300)
for idx, (g, _) in enumerate(groups_sorted[:10]):
    mask = np.array([x == g for x in grp_all])
    ax.hist(atoms_all[mask], bins=edges, alpha=0.7,
            label=g.split("/")[-1][:28], color=COLORS[idx % len(COLORS)])
vlines(ax)
ax.set_xlabel("Atom count")
ax.set_ylabel("Molecules (log)")
ax.set_yscale("log")
ax.legend(fontsize=6, ncol=2)
ax.set_title(f"Atom-count distribution — {N:,} molecules, {len(BINS)} bins")
fig.tight_layout()
fig.savefig(OUT_DIR / "atom_count_histogram.png", dpi=150)
plt.close(fig)

# 2. bin molecule count
fig, ax = plt.subplots(figsize=(14, 4))
x = np.arange(len(BINS))
ax.bar(x, mol_counts, color="steelblue")
ax.set_xlabel("Bin index")
ax.set_ylabel("Molecules in bin")
ax.set_title("Molecule count per bin")
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
fig.tight_layout()
fig.savefig(OUT_DIR / "bin_molecules.png", dpi=150)
plt.close(fig)

# 3. bin padding stats
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
x = np.arange(len(BINS))
ax1.plot(x, bin_max,   label="max",  marker=".", ms=3)
ax1.plot(x, bin_p95,   label="p95",  marker=".", ms=3)
ax1.plot(x, bin_means, label="mean", marker=".", ms=3)
ax1.set_ylabel("Atom count")
ax1.legend()
ax1.set_title("Bin atom-count stats + padding ratio")
ax2.bar(x, pad_ratios, color="orange")
ax2.axhline(0.6, color="red", lw=0.8, linestyle="--", label="0.6 floor")
ax2.set_xlabel("Bin index")
ax2.set_ylabel("Pad ratio (mean/max)")
ax2.set_ylim(0, 1.05)
ax2.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "bin_padding.png", dpi=150)
plt.close(fig)

# 4. dataset CDF with bin boundaries
fig, ax = plt.subplots(figsize=(12, 5))
cdf_x = np.sort(atoms_all)
cdf_y = np.arange(1, N + 1) / N
ax.plot(cdf_x, cdf_y, lw=1.2, label="Full dataset")
for lo, _ in BINS:
    ax.axvline(lo, color="gray", alpha=0.2, lw=0.6)
ax.set_xlabel("Atom count")
ax.set_ylabel("Cumulative fraction")
ax.set_title("Dataset CDF with bin boundaries")
ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "dataset_cdf.png", dpi=150)
plt.close(fig)

# 5. large molecule zoom (500+)
fig, ax = plt.subplots(figsize=(12, 5))
mask_lg = atoms_all >= 500
ax.hist(atoms_all[mask_lg], bins=100, color="steelblue", alpha=0.8)
for lo, _ in BINS:
    if lo >= 500:
        ax.axvline(lo, color="gray", alpha=0.3, lw=0.7)
ax.set_xlabel("Atom count")
ax.set_ylabel("Molecules")
ax.set_title(f"Large molecule zoom (≥500 atoms, n={mask_lg.sum():,})")
fig.tight_layout()
fig.savefig(OUT_DIR / "large_mol_zoom.png", dpi=150)
plt.close(fig)

print("Plots written.")

# ── binning_info.txt ──────────────────────────────────────────────────────────

total_atoms = int(atoms_all.sum())
min_mol = int(mol_counts[mol_counts > 0].min())
max_mol = int(mol_counts.max())
mean_mol= int(mol_counts[mol_counts > 0].mean())
min_pad = float(pad_ratios[pad_ratios > 0].min())
max_pad = float(pad_ratios.max())
mean_pad= float(pad_ratios[pad_ratios > 0].mean())
all_pass = bool((pad_ratios[pad_ratios > 0] >= 0.60).all())

lines = []
lines.append("=" * 80)
lines.append("BINNING ANALYSIS - I(q)@L=50.h5")
lines.append(f"Generated: {datetime.date.today()}")
lines.append("=" * 80)
lines.append("")
lines.append("DATASET SUMMARY")
lines.append("─" * 15)
lines.append(f"Total molecules : {N:,}")
lines.append(f"Total atoms     : {total_atoms:,}")

lines.append("")
lines.append("Group breakdown:")
hdr = f"  {'Group':<44} {'Molecules':>9}  {'Atoms total':>12}  {'Min':>6}  {'Max':>6}  {'Mean':>6}"
lines.append(hdr)
lines.append("  " + "─" * 86)
for g, _ in groups_sorted:
    n, at, mn, mx, me = grp_stats[g]
    lines.append(f"  {g:<44} {n:>9,}  {at:>12,}  {mn:>6}  {mx:>6}  {me:>6.0f}")

lines.append("")
lines.append("BINNING STRATEGY")
lines.append("─" * 16)
lines.append("Objective: equal molecule count per bin (not equal atom count).")
lines.append("Rationale: loss is per I(q) curve, so molecule count determines training signal.")
lines.append("Padding:   handled by Batcher's atom_size_ceil.")

sep = "=" * 85
lines.append("")
lines.append(sep)
lines.append(f"{'=' * 27} FINAL BINS ({len(BINS)} total) {'=' * 27}")
lines.append(sep)
lines.append("")
lines.append(f" {'Bin':>3}  {'Range':<16}  {'Molecules':>9}  {'Total atoms':>12}  {'Mean':>8}  {'p95':>8}  {'Max':>8}  {'Pad':>5}")
lines.append(f" {'─'*3}  {'─'*16}  {'─'*9}  {'─'*12}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*5}")
for i, (lo, hi) in enumerate(BINS):
    if mol_counts[i] == 0:
        continue
    lines.append(
        f" {i+1:>3}  ({lo:>7,},{hi:>7,})  {mol_counts[i]:>9,}  {atom_totals[i]:>12,}"
        f"  {bin_means[i]:>8.1f}  {bin_p95[i]:>8.1f}  {bin_max[i]:>8,}  {pad_ratios[i]:>5.2f}"
    )

lines.append("")
lines.append(f"Molecule count range: {min_mol:,} – {max_mol:,}  (mean {mean_mol:,})")
lines.append(f"Pad ratio range:      {min_pad:.2f} – {max_pad:.2f}  (mean {mean_pad:.2f})")
lines.append(f"All bins >= 0.60 pad: {all_pass}")
lines.append("")
lines.append(f"NOTE ON atom_size_ceil:")
lines.append(f"  atom_size_ceil is set to {atoms_all.max():,} -- all bins are within this ceiling.")
lines.append("")
lines.append("=" * 80)
lines.append("FILES")
lines.append("=" * 80)
lines.append("  atom_count_histogram.png - full distribution, all groups, bin lines")
lines.append("  bin_padding.png          - max/p95/mean + pad ratio per bin")
lines.append("  bin_molecules.png        - molecule count per bin")
lines.append("  dataset_cdf.png          - dataset CDF with bin boundaries")
lines.append("  large_mol_zoom.png       - 500+ atom detail")

txt = "\n".join(lines) + "\n"
(OUT_DIR / "binning_info.txt").write_text(txt)
print("binning_info.txt written.")

# ── patch config.py DEFAULT_BUCKETS ──────────────────────────────────────────

bucket_lines = ["DEFAULT_BUCKETS: list[tuple[int, int]] = ["]
for i, (lo, hi) in enumerate(BINS):
    sep2 = "," if i < len(BINS) - 1 else ""
    bucket_lines.append(f"    ({lo:5d},{hi:5d}){sep2}")
bucket_lines.append("    ]")
new_block = "\n".join(bucket_lines)

cfg_text = CFG_PATH.read_text()
import re
cfg_text = re.sub(
    r"DEFAULT_BUCKETS: list\[tuple\[int, int\]\] = \[.*?\]",
    new_block,
    cfg_text,
    flags=re.DOTALL,
)
CFG_PATH.write_text(cfg_text)
print(f"Patched DEFAULT_BUCKETS in {CFG_PATH}  ({len(BINS)} bins)")
print("Done.")
