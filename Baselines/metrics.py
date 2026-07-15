"""Shared evaluation metrics and plots for baseline scattering-curve predictors.
"""

import matplotlib
matplotlib.use("pgf")  # must happen before the first `import matplotlib.pyplot` anywhere in
                        # the process -- lets JuliaMono be used as an arbitrary system font via
                        # fontspec/xelatex, which the default dvipng+latex usetex path can't do

import numpy as np
import torch
import math
import re

from dataclasses         import dataclass, field
from collections.abc     import Iterable
from ScatterNet.batching import Batch
from Baselines.baseline  import Baseline

# Fixed atom-count buckets for the error-vs-size diagnostic. Molecules span
# 1 to ~6046 atoms (see Preprocess/README.md); log-spaced buckets keep both
# the small-molecule and large-molecule regimes visible on one plot.
ATOM_BINS = [(1, 10), (11, 50), (51, 200), (201, 1000), (1001, 100_000)]
ATOM_BIN_LABELS = ["1-10", "11-50", "51-200", "201-1000", "1001+"]

_PCT_ERR_FLOOR = 1e-3   # I(q) denominator floor for percent-error, avoids /0 blowup


@dataclass
class EvalResult:
    """All metrics collected for one baseline over one evaluation pass."""

    name:         str
    msle:         float   # mean squared log1p error, global
    r2_log1p:     float   # R^2 in log1p(I) space, global
    r2_raw:       float   # R^2 in raw I(q) space, global
    us_per_atom:  float

    r2_per_q:            np.ndarray  # (Q,) R^2 in log1p space at each q
    pct_err_per_q:        np.ndarray  # (Q,) mean |pred-true|/true * 100 at each q
    mean_true_log1p_per_q: np.ndarray  # (Q,) mean log1p(true) at each q -- Kratky-style
    mean_pred_log1p_per_q: np.ndarray  # (Q,) mean log1p(pred) at each q -- Kratky-style

    msle_by_atom_bin: dict = field(default_factory=dict)  # label -> mean per-molecule MSLE
    n_by_atom_bin:     dict = field(default_factory=dict)  # label -> molecule count in that bin

    # ── per-molecule error distributions (the honest ones) ──────────────────
    # Unlike the per-q means above, these are collected one value per molecule
    # (aggregated over that molecule's q-points), so their spread and skew
    # describe the actual model error rather than the q-dependence of a mean.
    # Stored as pre-binned histograms + summary moments rather than raw arrays
    # so the checkpoint stays small no matter how large the test set is.
    resid_hist:  np.ndarray = field(default_factory=lambda: np.array([]))  # counts, signed per-mol resid
    resid_edges: np.ndarray = field(default_factory=lambda: np.array([]))  # len = len(resid_hist)+1
    resid_stats: dict       = field(default_factory=dict)  # mean/std/median/skew/kurtosis/p05/p95/n

    msle_hist:  np.ndarray  = field(default_factory=lambda: np.array([]))  # counts, per-mol MSLE
    msle_edges: np.ndarray  = field(default_factory=lambda: np.array([]))
    msle_stats: dict        = field(default_factory=dict)  # mean/std/median/skew/p05/p95/n

    # per atom-count bin: mean signed residual, its std, and molecule count.
    # label -> [mean, std, n]. Complements msle_by_atom_bin (magnitude) with
    # direction (does the model over- or under-predict as molecules grow?).
    resid_by_atom_bin: dict = field(default_factory=dict)

    # least-squares fit log10(per-mol MSLE) ~ slope*log10(atoms) + intercept.
    # slope is the empirical error-scaling exponent with molecule size.
    atom_scaling_fit: dict  = field(default_factory=dict)  # slope/intercept/r2/n

    def to_json(self) -> dict:
        """Serialize to plain JSON-able types, for checkpointing across sessions."""
        return {
            "name": self.name, "msle": self.msle, "r2_log1p": self.r2_log1p,
            "r2_raw": self.r2_raw, "us_per_atom": self.us_per_atom,
            "r2_per_q": np.asarray(self.r2_per_q).tolist(),
            "pct_err_per_q": np.asarray(self.pct_err_per_q).tolist(),
            "mean_true_log1p_per_q": np.asarray(self.mean_true_log1p_per_q).tolist(),
            "mean_pred_log1p_per_q": np.asarray(self.mean_pred_log1p_per_q).tolist(),
            "msle_by_atom_bin": self.msle_by_atom_bin,
            "n_by_atom_bin": self.n_by_atom_bin,
            "resid_hist": np.asarray(self.resid_hist).tolist(),
            "resid_edges": np.asarray(self.resid_edges).tolist(),
            "resid_stats": self.resid_stats,
            "msle_hist": np.asarray(self.msle_hist).tolist(),
            "msle_edges": np.asarray(self.msle_edges).tolist(),
            "msle_stats": self.msle_stats,
            "resid_by_atom_bin": self.resid_by_atom_bin,
            "atom_scaling_fit": self.atom_scaling_fit,
        }

    @classmethod
    def from_json(cls, data: dict) -> "EvalResult":
        """Inverse of `to_json` -- rebuilds the per-q numpy arrays.

        Fields added after the first checkpoint schema are read with ``.get``
        so an older checkpoint (missing the per-molecule distributions) still
        loads; those baselines just come back with empty distribution fields
        and get re-run this session if their plots are needed.
        """
        return cls(
            name=data["name"], msle=data["msle"], r2_log1p=data["r2_log1p"],
            r2_raw=data["r2_raw"], us_per_atom=data["us_per_atom"],
            r2_per_q=np.array(data["r2_per_q"]),
            pct_err_per_q=np.array(data["pct_err_per_q"]),
            mean_true_log1p_per_q=np.array(data["mean_true_log1p_per_q"]),
            mean_pred_log1p_per_q=np.array(data["mean_pred_log1p_per_q"]),
            msle_by_atom_bin=data["msle_by_atom_bin"],
            n_by_atom_bin=data["n_by_atom_bin"],
            resid_hist=np.array(data.get("resid_hist", [])),
            resid_edges=np.array(data.get("resid_edges", [])),
            resid_stats=data.get("resid_stats", {}),
            msle_hist=np.array(data.get("msle_hist", [])),
            msle_edges=np.array(data.get("msle_edges", [])),
            msle_stats=data.get("msle_stats", {}),
            resid_by_atom_bin=data.get("resid_by_atom_bin", {}),
            atom_scaling_fit=data.get("atom_scaling_fit", {}),
        )


def _dist_stats(x: np.ndarray) -> dict:
    """Summary moments of a 1-D sample: mean, std, median, skew, excess kurtosis, 5/95 pctiles, n.

    Skewness and excess kurtosis are computed directly from centered moments
    (numpy only, no scipy dependency in this widely-imported module). Skew > 0
    means a longer right tail, < 0 a longer left tail. Returns an all-``nan``
    dict (n=0) for an empty sample so callers never special-case it.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n == 0:
        return {k: float("nan") for k in
                ("mean", "std", "median", "skew", "kurtosis", "p05", "p95")} | {"n": 0}
    mu    = float(x.mean())
    sigma = float(x.std())  # population std (ddof=0), matches the moment definitions below
    if sigma > 0:
        z    = (x - mu) / sigma
        skew = float((z ** 3).mean())
        kurt = float((z ** 4).mean() - 3.0)
    else:
        skew = kurt = 0.0
    return {
        "mean": mu, "std": sigma, "median": float(np.median(x)),
        "skew": skew, "kurtosis": kurt,
        "p05": float(np.percentile(x, 5)), "p95": float(np.percentile(x, 95)),
        "n": n,
    }


@torch.no_grad()
def evaluate(baseline: Baseline, loader: Iterable[Batch], q_grid: torch.Tensor, name: str) -> EvalResult:
    """Evaluate a baseline over every batch in ``loader``, in one streaming pass.

    All sums are accumulated globally across the whole evaluation set (not
    averaged per-batch), since batches vary hugely in molecule count and
    target variance -- averaging a per-batch statistic would let one
    low-variance batch swing the aggregate wildly.

    Parameters
    ----------
    baseline : Baseline
        A fitted baseline to evaluate.
    loader : Iterable[Batch]
        Evaluation batches (e.g. a materialized test set).
    q_grid : torch.Tensor
        q-point grid, shape ``(Q,)``. Only used for its length here; kept as
        an explicit argument so callers don't have to thread it separately.
    name : str
        Display name for this baseline, used as the plot legend/label.

    Returns
    -------
    EvalResult
    """
    Q = len(q_grid)

    sum_sq_err_log1p, sum_y_log1p, sum_y2_log1p = 0.0, 0.0, 0.0
    sum_sq_err_raw,   sum_y_raw,   sum_y2_raw   = 0.0, 0.0, 0.0
    n_points = 0

    sum_sq_err_log1p_q = torch.zeros(Q)
    sum_true_log1p_q   = torch.zeros(Q)
    sum_pred_log1p_q   = torch.zeros(Q)
    sum_y2_log1p_q      = torch.zeros(Q)
    sum_pct_err_q       = torch.zeros(Q)
    n_mols = 0

    bin_sq_errs  = {label: [] for label in ATOM_BIN_LABELS}  # per-molecule MSLE, per atom-count bucket
    bin_resids   = {label: [] for label in ATOM_BIN_LABELS}  # per-molecule signed residual, same bucketing

    all_msle:  list = []  # per-molecule MSLE, every molecule (for the error distribution)
    all_resid: list = []  # per-molecule mean signed log1p residual (for bias/skew)
    all_atoms: list = []  # per-molecule atom count (for the error-vs-size scaling fit)

    tpa_weighted, total_atoms = 0.0, 0

    for batch in loader:
        pred, tpa = baseline.timed_call(batch)
        pred = pred.clamp(min=0)
        true = batch.iqval.clamp(min=0)

        pred_log1p = torch.log1p(pred)
        true_log1p = torch.log1p(true)
        sq_err_log1p = (pred_log1p - true_log1p) ** 2
        sq_err_raw   = (pred - true) ** 2

        sum_sq_err_log1p += sq_err_log1p.sum().item()
        sum_y_log1p      += true_log1p.sum().item()
        sum_y2_log1p      += (true_log1p ** 2).sum().item()

        sum_sq_err_raw += sq_err_raw.sum().item()
        sum_y_raw      += true.sum().item()
        sum_y2_raw      += (true ** 2).sum().item()

        n_points += true.numel()

        sum_sq_err_log1p_q += sq_err_log1p.sum(dim=0).cpu()
        sum_true_log1p_q   += true_log1p.sum(dim=0).cpu()
        sum_pred_log1p_q   += pred_log1p.sum(dim=0).cpu()
        sum_y2_log1p_q      += (true_log1p ** 2).sum(dim=0).cpu()

        pct_err = 100.0 * (pred - true).abs() / true.clamp(min=_PCT_ERR_FLOOR)
        sum_pct_err_q += pct_err.sum(dim=0).cpu()

        n_mols += true.shape[0]

        per_mol_msle  = sq_err_log1p.mean(dim=1).cpu()                 # (N,) magnitude
        per_mol_resid = (pred_log1p - true_log1p).mean(dim=1).cpu()    # (N,) signed direction
        atom_counts   = batch.padding_mask().sum(dim=1).cpu().tolist()
        for a, m, s in zip(atom_counts, per_mol_msle.tolist(), per_mol_resid.tolist()):
            all_msle.append(m)
            all_resid.append(s)
            all_atoms.append(a)
            for (lo, hi), label in zip(ATOM_BINS, ATOM_BIN_LABELS):
                if lo <= a <= hi:
                    bin_sq_errs[label].append(m)
                    bin_resids[label].append(s)
                    break

        n_atoms = int(batch.padding_mask().sum().item())
        tpa_weighted += tpa * n_atoms
        total_atoms  += n_atoms

    msle = sum_sq_err_log1p / n_points if n_points > 0 else float("nan")

    ss_tot_log1p = sum_y2_log1p - (sum_y_log1p ** 2) / n_points if n_points > 0 else 0.0
    r2_log1p = 1 - sum_sq_err_log1p / ss_tot_log1p if ss_tot_log1p > 0 else float("nan")

    ss_tot_raw = sum_y2_raw - (sum_y_raw ** 2) / n_points if n_points > 0 else 0.0
    r2_raw = 1 - sum_sq_err_raw / ss_tot_raw if ss_tot_raw > 0 else float("nan")

    us_per_atom = (tpa_weighted / total_atoms * 1e6) if total_atoms > 0 else float("nan")

    if n_mols > 0:
        ss_tot_q = sum_y2_log1p_q - (sum_true_log1p_q ** 2) / n_mols
        r2_per_q = torch.where(
            ss_tot_q > 0, 1 - sum_sq_err_log1p_q / ss_tot_q, torch.full_like(ss_tot_q, float("nan"))
        ).numpy()
        pct_err_per_q = (sum_pct_err_q / n_mols).numpy()
        mean_true_log1p_per_q = (sum_true_log1p_q / n_mols).numpy()
        mean_pred_log1p_per_q = (sum_pred_log1p_q / n_mols).numpy()
    else:
        r2_per_q = np.full(Q, float("nan"))
        pct_err_per_q = np.full(Q, float("nan"))
        mean_true_log1p_per_q = np.full(Q, float("nan"))
        mean_pred_log1p_per_q = np.full(Q, float("nan"))

    # mean, not median -- a baseline that quietly fails on rare large molecules is
    # a real problem worth seeing, not noise to smooth away. Sparse buckets (few
    # molecules of that size in this evaluation set) are a real risk for this
    # statistic, so n_by_atom_bin is reported alongside it and plotted as a
    # per-point sample count rather than papered over with a different aggregator.
    msle_by_atom_bin = {
        label: (sum(bin_sq_errs[label]) / len(bin_sq_errs[label]) if bin_sq_errs[label] else float("nan"))
        for label in ATOM_BIN_LABELS
    }
    n_by_atom_bin = {label: len(bin_sq_errs[label]) for label in ATOM_BIN_LABELS}

    # ── per-molecule error distributions ────────────────────────────────────
    resid_arr = np.asarray(all_resid, dtype=float)
    msle_arr  = np.asarray(all_msle,  dtype=float)
    atoms_arr = np.asarray(all_atoms, dtype=float)

    resid_stats = _dist_stats(resid_arr)
    msle_stats  = _dist_stats(msle_arr)

    # 40 fixed-width bins over the 1/99 percentile range keeps the histogram
    # informative even when a handful of blown-up molecules would otherwise
    # stretch the axis flat; the tails past the clip still count into the edge
    # bins (np.histogram clips out-of-range, so widen the range slightly).
    def _hist(a: np.ndarray, lo_pct=1.0, hi_pct=99.0, bins=40):
        a = a[np.isfinite(a)]
        if a.size == 0:
            return np.array([]), np.array([])
        lo, hi = np.percentile(a, [lo_pct, hi_pct])
        if not (hi > lo):
            lo, hi = float(a.min()), float(a.max()) if a.max() > a.min() else float(a.min()) + 1.0
        counts, edges = np.histogram(a.clip(lo, hi), bins=bins, range=(lo, hi))
        return counts, edges

    resid_hist, resid_edges = _hist(resid_arr)
    msle_hist,  msle_edges  = _hist(msle_arr, lo_pct=0.0, hi_pct=99.0)

    resid_by_atom_bin = {
        label: [
            float(np.mean(bin_resids[label])) if bin_resids[label] else float("nan"),
            float(np.std(bin_resids[label]))  if bin_resids[label] else float("nan"),
            len(bin_resids[label]),
        ]
        for label in ATOM_BIN_LABELS
    }

    # error-scaling fit: log10(MSLE) ~ slope*log10(atoms) + intercept. A slope
    # near 0 means error is size-independent; a positive slope means the model
    # degrades on larger molecules. Floor MSLE so perfect (0) predictions don't
    # send the log to -inf and poison the fit.
    atom_scaling_fit: dict = {"slope": float("nan"), "intercept": float("nan"),
                              "r2": float("nan"), "n": 0}
    fit_mask = np.isfinite(msle_arr) & np.isfinite(atoms_arr) & (atoms_arr > 0)
    if fit_mask.sum() >= 3:
        lx = np.log10(atoms_arr[fit_mask])
        ly = np.log10(np.clip(msle_arr[fit_mask], 1e-12, None))
        if lx.std() > 0:
            slope, intercept = np.polyfit(lx, ly, 1)
            resid_fit = ly - (slope * lx + intercept)
            ss_res = float((resid_fit ** 2).sum())
            ss_tot = float(((ly - ly.mean()) ** 2).sum())
            r2_fit = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            atom_scaling_fit = {"slope": float(slope), "intercept": float(intercept),
                                "r2": float(r2_fit), "n": int(fit_mask.sum())}

    return EvalResult(
        name=name,
        msle=msle,
        r2_log1p=r2_log1p,
        r2_raw=r2_raw,
        us_per_atom=us_per_atom,
        r2_per_q=r2_per_q,
        pct_err_per_q=pct_err_per_q,
        mean_true_log1p_per_q=mean_true_log1p_per_q,
        mean_pred_log1p_per_q=mean_pred_log1p_per_q,
        msle_by_atom_bin=msle_by_atom_bin,
        n_by_atom_bin=n_by_atom_bin,
        resid_hist=resid_hist,
        resid_edges=resid_edges,
        resid_stats=resid_stats,
        msle_hist=msle_hist,
        msle_edges=msle_edges,
        msle_stats=msle_stats,
        resid_by_atom_bin=resid_by_atom_bin,
        atom_scaling_fit=atom_scaling_fit,
    )


# ── plotting ──────────────────────────────────────────────────────────────

# Matplotlib qualitative "Accent" colormap, darkened ~18% (the stock palette reads
# too bright/pastel at plot scale) and with the stock yellow (#ffff99, nearly
# invisible on white) replaced by a gold that still fits the Accent family before
# darkening.
PALETTE = [
    "#55b755", "#9a81bc", "#fc9c41", "#a58520",
    "#2e5990", "#c50268", "#9d4b13", "#545454",
]
TEXT_PRIMARY, TEXT_MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"

_Q_LABEL = r"$q\ (\mathrm{\AA}^{-1})$"


def _configure_mpl():
    """LaTeX text rendering with JuliaMono (text + math), used by every plot in
    this module. Applied lazily (not at import time) so importing this module
    doesn't require a LaTeX toolchain unless a plot is actually made.

    Runs through xelatex (not the default dvipng+latex usetex path), because
    JuliaMono is an arbitrary system font only reachable via fontspec, which
    dvipng+latex can't load. See the ``matplotlib.use("pgf")`` call at the top
    of this module -- that backend choice is what makes fontspec available.
    """
    import matplotlib.pyplot as plt
    plt.rcParams["pgf.texsystem"] = "xelatex"
    plt.rcParams["text.usetex"] = True
    plt.rcParams["pgf.rcfonts"] = False
    plt.rcParams["pgf.preamble"] = r"\usepackage{fontspec}\setmainfont{JuliaMono}\usepackage{amsmath}"
    plt.rcParams["font.size"] = 14


def _style_axes(ax):
    ax.tick_params(colors=TEXT_MUTED)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def _colors(results: list) -> dict:
    return {r.name: PALETTE[i % len(PALETTE)] for i, r in enumerate(results)}


def _slug(name: str) -> str:
    """Baseline display name -> filesystem-safe filename fragment."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def plot_summary(results: list, out_path: str):
    """Bar chart: MSLE, R² (raw), R² (log1p), µs/atom -- one row per baseline."""
    import matplotlib.pyplot as plt
    _configure_mpl()

    ordered = sorted(results, key=lambda r: r.msle if math.isfinite(r.msle) else float("inf"))
    names   = [r.name for r in ordered]
    colors  = [PALETTE[i % len(PALETTE)] for i in range(len(ordered))]

    specs = [
        ("msle",        r"MSLE ($\ln(1+I)$)",  "{:.4f}"),
        ("r2_raw",       r"$R^2$ (raw)",          "{:.3f}"),
        ("r2_log1p",     r"$R^2$ ($\ln(1+I)$)", "{:.3f}"),
        ("us_per_atom", r"$\mu\mathrm{s}$ / atom", "{:.1f}"),
    ]

    n = len(ordered)
    fig, axes = plt.subplots(1, len(specs), figsize=(4 * len(specs), 0.85 * n + 1.8))
    fig.subplots_adjust(wspace=0.5)

    y = range(n)
    for ax, (attr, title, fmt) in zip(axes, specs):
        vals = [getattr(r, attr) for r in ordered]
        finite_vals = [v for v in vals if math.isfinite(v)]
        xmax = max(finite_vals) if finite_vals else 1
        # clamp bar geometry to [0, xmax] -- a metric like R^2 can be arbitrarily
        # negative for a badly-fit baseline, and plotting that raw value as bar
        # width blows the bbox_inches="tight" figure size out to cover it even
        # though it's clipped out of the visible xlim. The printed label below
        # still shows the true (possibly very negative) value.
        plot_vals = [min(max(v, 0.0), xmax) if math.isfinite(v) else 0 for v in vals]

        bars = ax.barh(y, plot_vals, color=colors, height=0.68, zorder=3)
        ax.set_yticks(list(y))
        ax.set_yticklabels(names if ax is axes[0] else [], color=TEXT_PRIMARY)
        ax.set_ylim(n - 0.5, -0.5)
        ax.set_title(title, color=TEXT_PRIMARY, pad=10)
        ax.tick_params(colors=TEXT_MUTED, length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for bar, v in zip(bars, vals):
            label = "n/a" if not math.isfinite(v) else fmt.format(v)
            ax.text(bar.get_width() + 0.02 * xmax, bar.get_y() + bar.get_height() / 2,
                     label, va="center", ha="left", color=TEXT_PRIMARY)
        ax.set_xlim(0, xmax * 1.22 if xmax > 0 else 1)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_per_q_r2(results: list, q_grid: torch.Tensor, out_path: str):
    """Per-q R² (log1p space): does performance hold up at high q?

    Pooled R² is dominated by molecule-to-molecule scale variance (I(0)
    roughly tracks atom count). This plot breaks R² out per q-point instead:
    low q is where that easy scale signal lives, high q is where fine
    structural shape has to be predicted. A line that collapses at high q is
    winning on scale alone, not on physics.
    """
    import matplotlib.pyplot as plt
    _configure_mpl()

    q_np = q_grid.numpy() if isinstance(q_grid, torch.Tensor) else np.asarray(q_grid)
    colors = _colors(results)

    fig, ax = plt.subplots(figsize=(11, 6))
    for r in results:
        ax.plot(q_np, r.r2_per_q, color=colors[r.name], linewidth=2, label=r.name)
    ax.axhline(0, color=TEXT_MUTED, linewidth=1, linestyle="--", zorder=1)
    ax.set_xlabel(_Q_LABEL, color=TEXT_PRIMARY)
    ax.set_ylabel(r"$R^2$ at this $q$-point ($\ln(1+I)$ space)", color=TEXT_PRIMARY)
    _style_axes(ax)
    finite = [v for r in results for v in r.r2_per_q if math.isfinite(v)]
    floor = max(-3.0, min(finite)) if finite else -3.0
    ax.set_ylim(floor - 0.1, 1.05)
    # legend outside the axes (right side), never over the data itself
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False,
              labelcolor=TEXT_PRIMARY)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_per_q_percent_error(results: list, q_grid: torch.Tensor, out_path: str):
    """Per-q mean absolute percent error: |pred-true|/true * 100 at each q-point.

    Complements per-q R² with a directly interpretable "how far off, in
    percent" curve. Percent error blows up where true I(q) is near zero
    (denominator floored at 1e-3), so treat very high values there as a
    scale artifact rather than a real failure -- cross-check against the
    Kratky-style curve to see if that's what's happening.
    """
    import matplotlib.pyplot as plt
    _configure_mpl()

    q_np = q_grid.numpy() if isinstance(q_grid, torch.Tensor) else np.asarray(q_grid)
    colors = _colors(results)

    fig, ax = plt.subplots(figsize=(11, 6))
    for r in results:
        ax.plot(q_np, r.pct_err_per_q, color=colors[r.name], linewidth=2, label=r.name)
    ax.set_xlabel(_Q_LABEL, color=TEXT_PRIMARY)
    ax.set_ylabel(r"mean $\dfrac{|\mathrm{pred} - \mathrm{true}|}{\mathrm{true}} \times 100\%$",
                  color=TEXT_PRIMARY)
    _style_axes(ax)
    finite = [v for r in results for v in r.pct_err_per_q if math.isfinite(v)]
    ymax = min(500.0, max(finite) * 1.1) if finite else 100.0
    ax.set_ylim(0, ymax)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False,
              labelcolor=TEXT_PRIMARY)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_kratky(results: list, q_grid: torch.Tensor, out_dir: str) -> list:
    """Kratky-style overlay: mean log1p(I(q)) vs q, true vs. each baseline's prediction.

    One PNG per baseline (``kratky_<name>.png``) rather than one shared grid --
    each gets its full-size figure instead of a cramped subplot. Every baseline
    is evaluated on the same test set, so the true curve (dashed black) is
    identical across files, plotted against that baseline's predicted mean
    curve (solid, colored). A baseline whose solid line tracks the dashed line
    across the whole q range is capturing the actual curve shape, not just its
    scale.

    Returns the list of file paths written.
    """
    import os
    import matplotlib.pyplot as plt
    _configure_mpl()

    q_np = q_grid.numpy() if isinstance(q_grid, torch.Tensor) else np.asarray(q_grid)
    colors = _colors(results)
    written = []

    for r in results:
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(q_np, r.mean_true_log1p_per_q, color=TEXT_PRIMARY, linewidth=1.5,
                linestyle="--", label="true")
        ax.plot(q_np, r.mean_pred_log1p_per_q, color=colors[r.name], linewidth=2, label="pred")
        ax.set_title(r.name, color=TEXT_PRIMARY, pad=10)
        ax.set_xlabel(_Q_LABEL, color=TEXT_PRIMARY)
        ax.set_ylabel(r"mean $\ln(1+I(q))$", color=TEXT_PRIMARY)
        _style_axes(ax)
        ax.legend(loc="upper right", frameon=False, labelcolor=TEXT_PRIMARY)

        fig.tight_layout()
        out_path = os.path.join(out_dir, f"kratky_{_slug(r.name)}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(out_path)

    return written


def _draw_hist(ax, counts, edges, color):
    """Draw a pre-binned histogram (counts over edges) as filled bars on ``ax``."""
    counts = np.asarray(counts)
    edges  = np.asarray(edges)
    if counts.size == 0 or edges.size < 2:
        return
    ax.bar(edges[:-1], counts, width=np.diff(edges), align="edge",
           color=color, zorder=3)


def plot_residual_histogram(results: list, out_dir: str) -> list:
    """Per-MOLECULE signed residual distribution, one PNG per baseline.

    This is the honest error distribution: for each molecule, the mean over its
    q-points of ``ln(1+pred) - ln(1+true)``, collected one value per molecule by
    ``evaluate`` (see ``EvalResult.resid_hist``). Unlike the earlier per-q proxy,
    per-molecule errors are NOT averaged together first, so the spread and skew
    here describe the model, not the q-dependence of a mean.

    Read it as: the distribution's center is the systematic bias (negative =
    under-predicts, positive = over-predicts), its width is the per-molecule
    variability, and its skew (printed in the title) is which tail is longer --
    positive skew means a long right tail of over-predicted molecules, negative
    a long left tail of under-predicted ones.

    Reference lines: muted dashed at zero (no bias), solid at the mean, dotted
    at the median. Title reports mean, median, skew, and n.

    Returns the list of file paths written.
    """
    import os
    import matplotlib.pyplot as plt
    _configure_mpl()

    colors = _colors(results)
    written = []

    for r in results:
        fig, ax = plt.subplots(figsize=(7, 5.5))
        _draw_hist(ax, r.resid_hist, r.resid_edges, colors[r.name])
        ax.axvline(0, color=TEXT_MUTED, linewidth=1, linestyle="--", zorder=1)

        s = r.resid_stats or {}
        title = r.name
        if s.get("n", 0) > 0:
            ax.axvline(s["mean"],   color=TEXT_PRIMARY, linewidth=1.5, linestyle="-",  zorder=2)
            ax.axvline(s["median"], color=TEXT_PRIMARY, linewidth=1.2, linestyle=":",  zorder=2)
            title = (rf"{r.name}: $\mu={s['mean']:.3f}$, "
                     rf"med$={s['median']:.3f}$, skew$={s['skew']:.2f}$ ($n={s['n']}$)")
        ax.set_title(title, color=TEXT_PRIMARY, pad=10, fontsize=12)
        ax.set_xlabel(r"per-molecule mean $\ln(1+\mathrm{pred}) - \ln(1+\mathrm{true})$",
                      color=TEXT_PRIMARY)
        ax.set_ylabel("molecules", color=TEXT_PRIMARY)
        ax.tick_params(colors=TEXT_MUTED)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)

        fig.tight_layout()
        out_path = os.path.join(out_dir, f"residual_histogram_{_slug(r.name)}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(out_path)

    return written


def plot_per_molecule_msle(results: list, out_dir: str) -> list:
    """Per-molecule MSLE distribution, one PNG per baseline.

    The magnitude companion to the signed-residual histogram: for each molecule,
    the mean squared log1p error over its q-points. Non-negative by construction,
    so a heavy right tail is expected and the interesting question is how heavy
    -- a long tail means a subset of molecules the baseline fails badly on even
    when the bulk are fine (which a single pooled MSLE hides). Title reports
    mean vs median (mean >> median flags exactly that tail) and skew.

    Returns the list of file paths written.
    """
    import os
    import matplotlib.pyplot as plt
    _configure_mpl()

    colors = _colors(results)
    written = []

    for r in results:
        fig, ax = plt.subplots(figsize=(7, 5.5))
        _draw_hist(ax, r.msle_hist, r.msle_edges, colors[r.name])

        s = r.msle_stats or {}
        title = r.name
        if s.get("n", 0) > 0:
            ax.axvline(s["mean"],   color=TEXT_PRIMARY, linewidth=1.5, linestyle="-", zorder=2)
            ax.axvline(s["median"], color=TEXT_PRIMARY, linewidth=1.2, linestyle=":", zorder=2)
            title = (rf"{r.name}: mean$={s['mean']:.3f}$, "
                     rf"med$={s['median']:.3f}$, skew$={s['skew']:.2f}$ ($n={s['n']}$)")
        ax.set_title(title, color=TEXT_PRIMARY, pad=10, fontsize=12)
        ax.set_xlabel(r"per-molecule MSLE ($\ln(1+I)$ space)", color=TEXT_PRIMARY)
        ax.set_ylabel("molecules", color=TEXT_PRIMARY)
        ax.tick_params(colors=TEXT_MUTED)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)

        fig.tight_layout()
        out_path = os.path.join(out_dir, f"per_molecule_msle_{_slug(r.name)}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(out_path)

    return written


def plot_error_vs_atom_count(results: list, out_path: str):
    """Mean per-molecule MSLE by atom-count bucket -- does error grow with molecule size?

    Mean, not median: a baseline that quietly fails on rare large molecules is a
    real problem worth surfacing, not noise to smooth away. The real risk here is
    a sparsely-populated bucket (few molecules of that size in this evaluation
    set) making the mean unreliable -- so bucket sizes are printed on the x-axis
    (shared across baselines, since they're evaluated on the same test set) so a
    thin bucket is visible rather than silently trusted.

    A baseline whose error rises steeply on large molecules is failing to
    generalize its size-scaling, not just noisier on individual points.
    """
    import matplotlib.pyplot as plt
    _configure_mpl()

    colors = _colors(results)

    # drop buckets with zero molecules in this evaluation set entirely, rather
    # than showing an empty "n=0" tick -- a small smoke-test sample only ever
    # populates the first bucket or two, and reserving x-axis space for buckets
    # with nothing plotted just reads as confusing dead space
    counts = results[0].n_by_atom_bin if results else {}
    present_labels = [label for label in ATOM_BIN_LABELS if counts.get(label, 0) > 0]

    # widen the figure as buckets accumulate so the rotated "bucket (n=...)"
    # ticks stay legible instead of overlapping on a fixed-width axis
    width = max(9.0, 1.1 * len(present_labels) + 2.5)
    fig, ax = plt.subplots(figsize=(width, 5.5))

    x = range(len(present_labels))
    for r in results:
        vals = [r.msle_by_atom_bin.get(label, float("nan")) for label in present_labels]
        # fold the fitted error-scaling exponent into the legend: slope of
        # log10(MSLE) vs log10(atoms) across all molecules (not just these bins),
        # so "does error grow with size?" gets a number, not just eyeballing
        slope = (r.atom_scaling_fit or {}).get("slope", float("nan"))
        label = r.name if not math.isfinite(slope) else rf"{r.name} (slope $={slope:.2f}$)"
        ax.plot(x, vals, marker="o", color=colors[r.name], linewidth=2, label=label)
    # log y: MSLE spans orders of magnitude across buckets, and a linear axis
    # flattens every well-fit baseline into an indistinguishable band near zero
    if any(math.isfinite(v) and v > 0
           for r in results for v in r.msle_by_atom_bin.values()):
        ax.set_yscale("log")
    xticklabels = [rf"{label} ($n={counts[label]}$)" for label in present_labels]
    ax.set_xticks(list(x))
    ax.set_xticklabels(xticklabels, color=TEXT_PRIMARY, rotation=45, ha="right")
    # pad the categorical x-range so end points aren't glued to the plot edges
    pad = 0.3 if len(present_labels) > 1 else 0.6
    ax.set_xlim(-pad, len(present_labels) - 1 + pad)
    ax.set_xlabel("atom count bucket", color=TEXT_PRIMARY)
    ax.set_ylabel(r"mean per-molecule $\mathrm{MSLE}$", color=TEXT_PRIMARY)
    ax.tick_params(colors=TEXT_MUTED)
    for spine in ax.spines.values():
        spine.set_visible(False)
    # categorical x -- vertical gridlines at each bucket don't carry extra
    # information (unlike a continuous axis) and just add clutter
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False,
              labelcolor=TEXT_PRIMARY)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_residual_by_atom_count(results: list, out_path: str):
    """Mean SIGNED per-molecule residual by atom-count bucket, with +/-1 std bands.

    error_vs_atom_count shows error *magnitude* (MSLE) vs size; this shows its
    *direction*. A line that drifts below zero on large buckets means the
    baseline systematically under-predicts big molecules (and vice versa) --
    a bias that a squared-error plot cannot reveal because it discards sign.
    The shaded +/-1 std band is the per-molecule spread within each bucket, so a
    line hugging zero with a wide band is unbiased-but-noisy, while a line
    riding off zero with a narrow band is a consistent, correctable offset.
    """
    import matplotlib.pyplot as plt
    _configure_mpl()

    colors = _colors(results)

    counts = results[0].n_by_atom_bin if results else {}
    present_labels = [label for label in ATOM_BIN_LABELS if counts.get(label, 0) > 0]

    width = max(9.0, 1.1 * len(present_labels) + 2.5)
    fig, ax = plt.subplots(figsize=(width, 5.5))

    x = np.arange(len(present_labels))
    for r in results:
        means = np.array([(r.resid_by_atom_bin.get(label, [np.nan, np.nan, 0]))[0]
                          for label in present_labels], dtype=float)
        stds  = np.array([(r.resid_by_atom_bin.get(label, [np.nan, np.nan, 0]))[1]
                          for label in present_labels], dtype=float)
        ax.plot(x, means, marker="o", color=colors[r.name], linewidth=2, label=r.name)
        ax.fill_between(x, means - stds, means + stds, color=colors[r.name],
                        alpha=0.12, linewidth=0, zorder=2)
    ax.axhline(0, color=TEXT_MUTED, linewidth=1, linestyle="--", zorder=1)

    xticklabels = [rf"{label} ($n={counts[label]}$)" for label in present_labels]
    ax.set_xticks(list(x))
    ax.set_xticklabels(xticklabels, color=TEXT_PRIMARY, rotation=45, ha="right")
    pad = 0.3 if len(present_labels) > 1 else 0.6
    ax.set_xlim(-pad, len(present_labels) - 1 + pad)
    ax.set_xlabel("atom count bucket", color=TEXT_PRIMARY)
    ax.set_ylabel(r"mean signed residual (log1p space)", color=TEXT_PRIMARY)
    ax.tick_params(colors=TEXT_MUTED)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False,
              labelcolor=TEXT_PRIMARY)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_per_q_summary(results: list, out_path: str):
    """One-glance summary of each baseline's per-q behaviour, collapsed to scalars.

    The per-q R² and per-q %-error plots are curves; this table-style bar chart
    aggregates each curve into the numbers you'd actually quote in a report:

      * mean per-q R²   -- average log1p-space fit quality across the q-grid
      * worst per-q R²  -- the single hardest q-point (where the baseline is weakest)
      * mean per-q %err -- average |pred-true|/true across q
      * high-q %err     -- mean %err over the top third of the q-range, where the
                           fine structural signal lives and easy I(0) scale doesn't help

    Baselines are ordered by mean per-q R² (best first). NaNs (undefined at a
    q-point) are ignored in each aggregate rather than poisoning it.
    """
    import matplotlib.pyplot as plt
    _configure_mpl()

    def _agg(r):
        r2   = np.asarray(r.r2_per_q, dtype=float)
        pct  = np.asarray(r.pct_err_per_q, dtype=float)
        r2f  = r2[np.isfinite(r2)]
        pctf = pct[np.isfinite(pct)]
        hi_slice = pct[max(0, int(len(pct) * 2 / 3)):]
        hi_slice = hi_slice[np.isfinite(hi_slice)]
        return {
            "mean_r2":  float(r2f.mean())  if r2f.size  else float("nan"),
            "worst_r2": float(r2f.min())   if r2f.size  else float("nan"),
            "mean_pct": float(pctf.mean()) if pctf.size else float("nan"),
            "highq_pct": float(hi_slice.mean()) if hi_slice.size else float("nan"),
        }

    aggs = {r.name: _agg(r) for r in results}
    ordered = sorted(results,
                     key=lambda r: aggs[r.name]["mean_r2"] if math.isfinite(aggs[r.name]["mean_r2"])
                     else float("-inf"), reverse=True)
    names  = [r.name for r in ordered]
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(ordered))]

    specs = [
        ("mean_r2",   r"mean per-$q$ $R^2$",   "{:.3f}", False),
        ("worst_r2",  r"worst per-$q$ $R^2$",  "{:.3f}", False),
        ("mean_pct",  r"mean per-$q$ \%err",   "{:.0f}", True),
        ("highq_pct", r"high-$q$ \%err",       "{:.0f}", True),
    ]

    n = len(ordered)
    fig, axes = plt.subplots(1, len(specs), figsize=(3.6 * len(specs), 0.85 * n + 1.8))
    fig.subplots_adjust(wspace=0.5)

    y = range(n)
    for ax, (key, title, fmt, clip_pct) in zip(axes, specs):
        vals = [aggs[r.name][key] for r in ordered]
        finite_vals = [v for v in vals if math.isfinite(v)]
        # clamp bar GEOMETRY so one blown-up value doesn't crush every other bar
        # into an invisible sliver; the printed label still shows the true value.
        # %err: cap at 500 (near-zero I(q) sends it to infinity), floor 0.
        # R²: floor at -1 (a catastrophically negative baseline like an
        # untrained MLP would otherwise set the axis to hundreds negative and
        # flatten the 0.9-vs-0.99 differences that actually matter).
        if clip_pct:
            cap   = min(500.0, max(finite_vals)) if finite_vals else 1.0
            floor = 0.0
        else:
            cap   = max(finite_vals) if finite_vals else 1.0
            floor = max(-1.0, min(finite_vals)) if finite_vals else 0.0
            floor = min(floor, 0.0)
        plot_vals = [min(max(v, floor), cap) if math.isfinite(v) else floor for v in vals]

        bars = ax.barh(y, plot_vals, color=colors, height=0.68, zorder=3)
        ax.set_yticks(list(y))
        ax.set_yticklabels(names if ax is axes[0] else [], color=TEXT_PRIMARY)
        ax.set_ylim(n - 0.5, -0.5)
        ax.set_title(title, color=TEXT_PRIMARY, pad=10)
        ax.tick_params(colors=TEXT_MUTED, length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        span = (cap - floor) or 1.0
        for bar, v in zip(bars, vals):
            label = "n/a" if not math.isfinite(v) else fmt.format(v)
            ax.text(bar.get_width() + 0.02 * span, bar.get_y() + bar.get_height() / 2,
                    label, va="center", ha="left", color=TEXT_PRIMARY)
        ax.set_xlim(min(0.0, floor), cap * 1.25 if cap > 0 else 1.0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_all_plots(results: list, q_grid: torch.Tensor, out_dir: str):
    """Write every plot above into ``out_dir``. Returns the list of file paths written."""
    import os
    os.makedirs(out_dir, exist_ok=True)

    written = []
    single_file_plots = {
        "summary.png":                 lambda p: plot_summary(results, p),
        "per_q_r2.png":                lambda p: plot_per_q_r2(results, q_grid, p),
        "per_q_percent_error.png":     lambda p: plot_per_q_percent_error(results, q_grid, p),
        "per_q_summary.png":           lambda p: plot_per_q_summary(results, p),
        "error_vs_atom_count.png":     lambda p: plot_error_vs_atom_count(results, p),
        "residual_vs_atom_count.png":  lambda p: plot_residual_by_atom_count(results, p),
    }
    for fname, fn in single_file_plots.items():
        full = os.path.join(out_dir, fname)
        fn(full)
        written.append(full)

    # one PNG per baseline, not one shared file
    written += plot_kratky(results, q_grid, out_dir)
    written += plot_residual_histogram(results, out_dir)
    written += plot_per_molecule_msle(results, out_dir)

    return written
