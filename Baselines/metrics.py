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
        }

    @classmethod
    def from_json(cls, data: dict) -> "EvalResult":
        """Inverse of `to_json` -- rebuilds the per-q numpy arrays."""
        return cls(
            name=data["name"], msle=data["msle"], r2_log1p=data["r2_log1p"],
            r2_raw=data["r2_raw"], us_per_atom=data["us_per_atom"],
            r2_per_q=np.array(data["r2_per_q"]),
            pct_err_per_q=np.array(data["pct_err_per_q"]),
            mean_true_log1p_per_q=np.array(data["mean_true_log1p_per_q"]),
            mean_pred_log1p_per_q=np.array(data["mean_pred_log1p_per_q"]),
            msle_by_atom_bin=data["msle_by_atom_bin"],
            n_by_atom_bin=data["n_by_atom_bin"],
        )


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

    bin_sq_errs = {label: [] for label in ATOM_BIN_LABELS}  # per-molecule MSLE, per atom-count bucket

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

        per_mol_msle = sq_err_log1p.mean(dim=1).cpu()  # (N,)
        atom_counts  = batch.padding_mask().sum(dim=1).cpu().tolist()
        for a, m in zip(atom_counts, per_mol_msle.tolist()):
            for (lo, hi), label in zip(ATOM_BINS, ATOM_BIN_LABELS):
                if lo <= a <= hi:
                    bin_sq_errs[label].append(m)
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


def plot_residual_histogram(results: list, out_dir: str) -> list:
    """Histogram of per-q-point log1p residuals (pred - true), one PNG per baseline.

    One file per baseline (``residual_histogram_<name>.png``) instead of one
    shared grid -- a name/mean annotation placed inside a small subplot could
    land on top of a bar; a full-size figure with the name and mean folded
    into the title (above the axes, never over the data) can't.

    A residual distribution centered off zero flags a systematic bias
    (baseline consistently over/under-predicts); a distribution far wider
    than the others flags high-variance, unstable predictions rather than a
    consistent offset. Built from ``mean_pred_log1p_per_q - mean_true_log1p_per_q``
    (per-q means already collected by ``evaluate``), so no extra data pass
    is needed -- this is a coarser view than per-molecule residuals, but
    cheap and sufficient as a sanity check.

    Two reference lines per plot: a muted dashed line at zero (no bias) and a
    solid colored line at that baseline's actual mean residual. When a
    baseline's residuals sit far from zero (a real, expected case -- see e.g.
    MLP on a too-small training set), the mean line is what lands inside the
    visible bars; the zero line stays as the "no bias" anchor even when it's
    off at the edge of the plot.

    Returns the list of file paths written.
    """
    import os
    import matplotlib.pyplot as plt
    _configure_mpl()

    colors = _colors(results)
    written = []

    for r in results:
        fig, ax = plt.subplots(figsize=(7, 5.5))
        residual = r.mean_pred_log1p_per_q - r.mean_true_log1p_per_q
        residual = residual[np.isfinite(residual)]
        ax.hist(residual, bins=15, color=colors[r.name], zorder=3)
        ax.axvline(0, color=TEXT_MUTED, linewidth=1, linestyle="--", zorder=1)
        title = r.name
        if len(residual) > 0:
            mean_val = float(residual.mean())
            ax.axvline(mean_val, color=TEXT_PRIMARY, linewidth=1.5, linestyle="-", zorder=2)
            title = rf"{r.name}: $\mu = {mean_val:.3f}$"
        ax.set_title(title, color=TEXT_PRIMARY, pad=10)
        ax.set_xlabel(r"$\ln(1+\mathrm{pred}) - \ln(1+\mathrm{true})$, per $q$-point",
                      color=TEXT_PRIMARY)
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
    fig, ax = plt.subplots(figsize=(9, 5.5))

    # drop buckets with zero molecules in this evaluation set entirely, rather
    # than showing an empty "n=0" tick -- a small smoke-test sample only ever
    # populates the first bucket or two, and reserving x-axis space for buckets
    # with nothing plotted just reads as confusing dead space
    counts = results[0].n_by_atom_bin if results else {}
    present_labels = [label for label in ATOM_BIN_LABELS if counts.get(label, 0) > 0]

    x = range(len(present_labels))
    for r in results:
        vals = [r.msle_by_atom_bin.get(label, float("nan")) for label in present_labels]
        ax.plot(x, vals, marker="o", color=colors[r.name], linewidth=2, label=r.name)
    xticklabels = [rf"{label} ($n={counts[label]}$)" for label in present_labels]
    ax.set_xticks(list(x))
    ax.set_xticklabels(xticklabels, color=TEXT_PRIMARY)
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


def run_all_plots(results: list, q_grid: torch.Tensor, out_dir: str):
    """Write every plot above into ``out_dir``. Returns the list of file paths written."""
    import os
    os.makedirs(out_dir, exist_ok=True)

    written = []
    single_file_plots = {
        "summary.png":             lambda p: plot_summary(results, p),
        "per_q_r2.png":            lambda p: plot_per_q_r2(results, q_grid, p),
        "per_q_percent_error.png": lambda p: plot_per_q_percent_error(results, q_grid, p),
        "error_vs_atom_count.png": lambda p: plot_error_vs_atom_count(results, p),
    }
    for fname, fn in single_file_plots.items():
        full = os.path.join(out_dir, fname)
        fn(full)
        written.append(full)

    # one PNG per baseline, not one shared file
    written += plot_kratky(results, q_grid, out_dir)
    written += plot_residual_histogram(results, out_dir)

    return written
