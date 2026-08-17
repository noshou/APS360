import glob
import json
import os
import time
from collections.abc import Callable, Iterable
from dataclasses import replace as dc_replace

import torch

from Baselines.run.baseline import Baseline
from Baselines.run.metrics import _PCT_ERR_FLOOR, run_all_plots
from Baselines.run.metrics import evaluate as _bl_evaluate
from ScatterNet import ScatterNet
from ScatterNet.batching import Batch
from ScatterNet.utils.config import RunConfig


class ScatterNetBaseline(Baseline):
    """Adapts a ScatterNet model to the `Baseline` interface.

    Lets a ScatterNet model be scored and plotted with the exact same code
    (Baselines.run.metrics.evaluate/run_all_plots) used for every baseline in
    Baselines/run/colab_baselines.ipynb.

    Optionally, when `compute_loss` is True, each forward pass also
    accumulates the composite training loss and log1p-space R2 as a side
    effect, so the single evaluation pass that drives the diagnostic plots
    ALSO yields the epoch's test loss/R2 - removing a second, redundant full
    pass over the test set. Read the accumulated result back with `loss_r2()`.

    The forward pass always runs under BF16 autocast (`torch.autocast(...,
    dtype=torch.bfloat16)`), matching Train/train.py's training-time
    precision exactly - there is no fp16/AMP toggle. ScatterNet's numerics
    guards (see message_pass.py/output_head.py) were originally tuned and
    validated against fp16-specific overflow failure modes; BF16's different
    (lower-mantissa, wider-exponent) rounding behavior has not been
    separately re-validated against those guards, so predictions from this
    path are still worth spot-checking against a fp32 reference before
    trusting a run's accuracy numbers, not just its speed.
    """

    def __init__(
        self,
        model: ScatterNet,
        device: str,
        compute_loss: bool = False,
        lambda_6: float | None = None,
        lambda_7: float | None = None,
        lambda_8: float | None = None,
        progress: bool = False,
        n_batch: int = 0,
        start_seen: int = 0,
        resume_loss_state: dict | None = None,
    ) -> None:
        """Store the model and device settings used at call time.

        Parameters
        ----------
        model : ScatterNet
            Model to wrap; called in whatever train()/eval() mode the
            caller already put it in.
        device : str
            Torch device string; autocast is only enabled on CUDA.
        compute_loss : bool, optional
            If True, accumulate the composite loss and log1p R2 over every
            batch this baseline is called on (see class docstring). If False
            (default), no loss stats are collected and `loss_r2()` returns
            NaNs.
        lambda_6 : float, optional
            Loss weight forwarded to `model.compute_loss`; required when
            `compute_loss` is True.
        lambda_7 : float, optional
            2nd-derivative roughness penalty weight forwarded to
            `model.compute_loss`; required when `compute_loss` is True.
        lambda_8 : float, optional
            Direct percent-error term weight forwarded to
            `model.compute_loss`; required when `compute_loss` is True.
        progress : bool
            Print a progress line every 20 batches.
        n_batch : int
            Total number of batches, for the "i/N" in the progress line.
        start_seen : int
            Number of molecules-batches already processed before this call
            (> 0 on a mid-pass resume), used only to label the progress
            line correctly.
        resume_loss_state : dict or None
            loss/R2 accumulator state from an earlier checkpoint (see
            `loss_state()`), folded in before this pass's own batches are
            added. None for a fresh pass.

        Returns
        -------
        None
        """
        self._model = model
        self._device = device
        self._compute_loss = compute_loss
        self._lambda_6 = lambda_6
        self._lambda_7 = lambda_7
        self._lambda_8 = lambda_8
        self._progress = progress
        self._n_batch = n_batch
        self._seen = start_seen
        self._start_seen = start_seen
        self._last_printed = -1
        self._t0 = time.time()
        # loss/R2 accumulators (only used if compute_loss given)
        self._loss_sum = 0.0
        self._mols = 0.0
        self._ss_res = 0.0
        self._sum_y = 0.0
        self._sum_y2 = 0.0
        self._n_elem = 0.0
        self._pct_err_sum = 0.0
        self._pct_err_n = 0.0
        if resume_loss_state:
            self._loss_sum = resume_loss_state["loss_sum"]
            self._mols = resume_loss_state["mols"]
            self._ss_res = resume_loss_state["ss_res"]
            self._sum_y = resume_loss_state["sum_y"]
            self._sum_y2 = resume_loss_state["sum_y2"]
            self._n_elem = resume_loss_state["n_elem"]
            self._pct_err_sum = resume_loss_state.get("pct_err_sum", 0.0)
            self._pct_err_n = resume_loss_state.get("pct_err_n", 0.0)

    def loss_state(self) -> dict:
        """Return the loss/R2 accumulator state, for mid-pass checkpointing.

        Inverse of `resume_loss_state` in `__init__` - round-trips through
        `loss_r2()`'s inputs exactly.
        """
        return {
            "loss_sum": self._loss_sum,
            "mols": self._mols,
            "ss_res": self._ss_res,
            "sum_y": self._sum_y,
            "sum_y2": self._sum_y2,
            "n_elem": self._n_elem,
            "pct_err_sum": self._pct_err_sum,
            "pct_err_n": self._pct_err_n,
        }

    def loss_r2(self) -> tuple[float, float, float]:
        """Return (mean composite loss, log1p R2, mean percent error)
        accumulated over all calls.

        Returns NaNs if `compute_loss` was False or nothing was evaluated.
        """
        if not self._compute_loss or self._mols == 0:
            return float("nan"), float("nan"), float("nan")
        mean_loss = self._loss_sum / self._mols
        ss_tot = self._sum_y2 - self._sum_y**2 / self._n_elem
        r2 = 1.0 - self._ss_res / ss_tot if ss_tot > 0 else 0.0
        pct_err = (
            self._pct_err_sum / self._pct_err_n
            if self._pct_err_n > 0
            else float("nan")
        )
        return mean_loss, r2, pct_err

    def __call__(self, batch: Batch) -> torch.Tensor:  # shape (N, Q)
        """Predict I(q) for every molecule in `batch`.

        Parameters
        ----------
        batch : Batch
            Batch of molecules to predict scattering curves for.

        Returns
        -------
        torch.Tensor
            Predicted I(q), shape (N, Q), clamped to non-negative (matching
            every other Baseline's convention - see Baselines/run/metrics.py).
        """
        # Baselines.run.metrics.evaluate() loop never moves batches to device.
        # ScatterNet needs its own explicit move, same as
        # every call site in train.py's training/eval loops.
        batch = dc_replace(
            batch,
            vocab=batch.vocab.to(self._device),
            iqval=batch.iqval.to(self._device),
            coord=batch.coord.to(self._device),
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            iq, coh, inc, fmags, sigmas = self._model(batch)
            # Composite loss/R2 as a side effect, in the same forward pass, so
            # plots eval doubles as the test loss/R2 eval. Computed inside
            # autocast + with the raw (unclamped) iq, matching
            # train.py evaluate() exactly.
            if self._compute_loss:
                lambda_6 = self._lambda_6
                lambda_7 = self._lambda_7
                lambda_8 = self._lambda_8
                assert lambda_6 is not None, (
                    "lambda_6 required when compute_loss is True"
                )
                assert lambda_7 is not None, (
                    "lambda_7 required when compute_loss is True"
                )
                assert lambda_8 is not None, (
                    "lambda_8 required when compute_loss is True"
                )
                loss = self._model.compute_loss(
                    iq,
                    coh,
                    inc,
                    fmags,
                    batch,
                    lambda_6,
                    lambda_7,
                    lambda_8,
                )
                iqf = iq.float()
                n = batch.iqval.shape[0]
                log_pred = torch.log1p(iqf)
                log_target = torch.log1p(batch.iqval)
                self._loss_sum += loss.item() * n
                self._mols += n
                self._ss_res += ((log_pred - log_target) ** 2).sum().item()
                self._sum_y += log_target.sum().item()
                self._sum_y2 += (log_target**2).sum().item()
                self._n_elem += log_target.numel()
                self._pct_err_sum += (
                    (
                        100.0
                        * (iqf - batch.iqval).abs()
                        / batch.iqval.clamp(min=_PCT_ERR_FLOOR)
                    )
                    .sum()
                    .item()
                )
                self._pct_err_n += log_target.numel()
        self._seen += 1
        if (
            self._progress
            and self._seen % 20 == 0
            and self._seen != self._last_printed
        ):
            self._last_printed = self._seen
            rate = (self._seen - self._start_seen) / (time.time() - self._t0)
            print(
                f"  [test/plots] batch {self._seen:5d}/{self._n_batch}",
                f"  {rate} batch/s",
                flush=True,
            )

        # Move the prediction back to CPU.
        return iq.float().clamp(min=0).cpu()


def save_epoch_plots(
    model: ScatterNet,
    loader: Iterable[Batch],
    q_grid: torch.Tensor,
    device: str,
    data_dir: str,
    epoch: int,
    compute_loss: bool = False,
    lambda_6: float | None = None,
    lambda_7: float | None = None,
    lambda_8: float | None = None,
    verbose: bool = False,
    start_batch: int = 0,
    resume_state: dict | None = None,
    ckpt_cb: "Callable[[dict, int], None] | None" = None,
    ckpt_interval_sec: float = 0,
) -> tuple[float, float, float]:
    """
    Evaluate `model` on `loader`, write this epoch's diagnostic plots, and
    return the test loss/R2/percent-error computed in the same single pass.

    Mirrors Baselines/run/colab_baselines.ipynb own evaluate()/run_all_plots()
    call, so training progress is comparable to the baseline notebook's
    plots. The single shared run_all_plots() drives the whole set: per-q R^2,
    per-q percent error, Kratky overlay, the per-molecule signed-residual
    histogram (with skew) and per-molecule MSLE distribution,
    error-vs-atom-count (+ scaling-slope fit) and signed-residual vs atom
    count, plus summary.json/per_q_summary.json (the aggregated summary and
    per-q summary numbers -- written as JSON, not a bar chart, since only
    ScatterNet is ever plotted here and a one-row bar chart has nothing to
    compare against; see run_all_plots(bar_chart_png=False)). Because
    evaluate() now collects the per-molecule error distributions too, every
    new baseline statistic appears in the epoch plots automatically, no
    change needed here.

    When `compute_loss` is True, this ONE pass also accumulates the
    composite loss and log1p R2 (via ScatterNetBaseline), so the caller no
    longer needs a separate `evaluate(test_loader)` pass - the test set is
    walked once, not twice. The values are numerically the same as train.py's
    `evaluate()` would produce (see ScatterNetBaseline's docstring).

    Parameters
    ----------
    model : ScatterNet
        Model to evaluate; called in whatever train()/eval() mode the
        caller already put it in.
    loader : Iterable[Batch]
        Evaluation batches (typically the test loader).
    q_grid : torch.Tensor
        q-point grid, shape (Q,).
    device : str
        Torch device string.
    data_dir : str
        Root directory; this epoch's plots are written to
        `{data_dir}/epoch_{epoch:03d}/`.
    epoch : int
        Current epoch number, used to name the per-epoch subdirectory.
    compute_loss : bool, optional
        If True, the pass also returns (test_loss, test_r2, test_pct_err).
        If False (default), the return value is (nan, nan, nan) and only
        plots are produced.
    lambda_6 : float, optional
        Loss weight, required when `compute_loss` is True.
    lambda_7 : float, optional
        2nd-derivative roughness penalty weight, required when
        `compute_loss` is True.
    lambda_8 : float, optional
        Direct percent-error term weight, required when `compute_loss`
        is True.
    verbose : bool
        Print a progress line every 20 batches.
    start_batch : int
        Global batch index of the first batch `loader` will yield (0
        normally; > 0 on a mid-pass resume, where `loader`'s sampler
        already skips the batches before this index).
    resume_state : dict or None
        Combined checkpoint state from an earlier `ckpt_cb` call: keys
        "eval" (Baselines.run.metrics.evaluate's accumulator, see its
        docstring) and "loss" (ScatterNetBaseline's loss/R2 accumulator,
        see `ScatterNetBaseline.loss_state`). None for a fresh pass.
    ckpt_cb : callable or None
        If given, called periodically as `ckpt_cb(state, batch_idx)` with
        the combined state (same shape as `resume_state`) and the last-
        processed global batch index, so the caller can write a resumable
        checkpoint for this long-running pass.
    ckpt_interval_sec : float
        Minimum wall-clock seconds between `ckpt_cb` calls. 0 (default)
        disables checkpointing even if `ckpt_cb` is given.

    Returns
    -------
    tuple of (float, float, float)
        (test_loss, test_r2) in log1p space and mean absolute percent
        error in raw I(q) space, or (nan, nan, nan) if compute_loss is
        False.
    """
    baseline = ScatterNetBaseline(
        model,
        device,
        compute_loss=compute_loss,
        lambda_6=lambda_6,
        lambda_7=lambda_7,
        lambda_8=lambda_8,
        progress=verbose,
        n_batch=start_batch + len(loader),  # type: ignore[arg-type]
        start_seen=start_batch,
        resume_loss_state=(
            resume_state["loss"] if resume_state else None
        ),
    )

    def _ckpt_cb(eval_state: dict, batch_idx: int) -> None:
        """Bundle metrics.evaluate's accumulator with the baseline's
        loss/R2 accumulator into one checkpoint payload.
        """
        assert ckpt_cb is not None
        ckpt_cb(
            {"eval": eval_state, "loss": baseline.loss_state()}, batch_idx
        )

    result = _bl_evaluate(
        baseline,
        loader,
        q_grid,
        "ScatterNet",
        start_batch=start_batch,
        resume_state=resume_state["eval"] if resume_state else None,
        ckpt_cb=_ckpt_cb if ckpt_cb is not None else None,
        ckpt_interval_sec=ckpt_interval_sec,
    )
    epoch_dir = os.path.join(data_dir, f"epoch_{epoch:03d}")
    # bar_chart_png=False: only one result (ScatterNet) is ever plotted
    # here, so a one-row bar chart has nothing to compare against --
    # summary.json/per_q_summary.json (always written regardless) carry
    # the same numbers without the pointless PNG.
    written = run_all_plots(
        [result],
        q_grid,
        epoch_dir,
        categories={"ScatterNet": "learned"},
        bar_chart_png=False,
    )
    print(
        f"  [plots] wrote {len(written)} file(s) to {epoch_dir}",
        flush=True,
    )
    return baseline.loss_r2()


def save_epoch_metrics(record: dict, data_dir: str, epoch: int) -> None:
    """Write one epoch's train/val/test loss and R2 to its own epoch dir.

    One file per epoch (`{data_dir}/epoch_{epoch:03d}/metrics.json`), so no
    epoch can clobber another's numbers and a resumed run cannot truncate the
    record of the epochs that came before it. Concatenate the files afterwards
    (`load_epoch_history`) to get the whole run. Rank 0 only.

    Parameters
    ----------
    record : dict
        This epoch's metrics: "epoch", "train_loss", "train_r2", "val_loss",
        "val_r2", "test_loss", "test_r2".
    data_dir : str
        Root plots directory.
    epoch : int
        Current epoch number, used to name the per-epoch subdirectory.

    Returns
    -------
    None
    """
    epoch_dir = os.path.join(data_dir, f"epoch_{epoch:03d}")
    os.makedirs(epoch_dir, exist_ok=True)
    out_path = os.path.join(epoch_dir, "metrics.json")
    with open(out_path, "w") as fh:
        json.dump(record, fh, indent=2)
    print(f"  [data] wrote {out_path}", flush=True)


def load_epoch_history(data_dir: str) -> list[dict]:
    """
    Rebuild the run's per-epoch history by reading every epoch's metrics.json.

    Read off disk rather than kept in memory so the history survives a resume:
    a resumed process starts with no in-memory record of the epochs it did not
    run, but their `metrics.json` files are still sitting in `data_dir` (pulled
    back from Drive with the checkpoint, if the run moved boxes).

    Ordered by the epoch number in each record, so a directory holding epochs
    from several resumes still yields one monotonic curve.

    Parameters
    ----------
    data_dir : str
        Root plots directory.

    Returns
    -------
    list of dict
        One record per epoch found, ascending by "epoch". Empty if none exist.
    """
    history = []
    for path in glob.glob(os.path.join(data_dir, "epoch_*", "metrics.json")):
        try:
            with open(path) as fh:
                history.append(json.load(fh))
        except (OSError, ValueError):
            # A run killed mid-write leaves a truncated json. Skipping it costs
            # one point on the curve; letting it raise would kill the epoch.
            print(f"  [data] skipping unreadable {path}", flush=True)
    history.sort(key=lambda h: h["epoch"])
    return history


def save_epoch_loss_plot(history: list[dict], data_dir: str) -> None:
    """Update the run-wide train/val/test loss-vs-epoch trend.

    Cheap: reads `history` (per-epoch records already on disk); no model
    call. Written ONCE to `{data_dir}/loss_per_epoch.{png,json}`, at the
    ROOT of data_dir rather than nested under this epoch's own
    subdirectory - unlike every other per-epoch artifact, this one is a
    single running trend, not a per-epoch snapshot, so nesting it meant N
    near-duplicate copies by the end of an N-epoch run. The `.json`
    sidecar holds the exact `history` the plot draws from.

    Skipped entirely (no files touched) when there's one epoch of history
    or fewer: a trend line with one point isn't a trend, and matplotlib's
    default axis autoscaling around a single x-value produces a
    degenerate range rather than a meaningful plot.

    Parameters
    ----------
    history : list of dict
        Per-epoch records, each with "epoch", "train_loss", "val_loss",
        "test_loss"; typically `load_epoch_history(data_dir)`.
    data_dir : str
        Root plots directory.

    Returns
    -------
    None
    """
    if len(history) <= 1:
        return

    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    from Baselines.run.metrics import (
        TEXT_PRIMARY,
        _configure_mpl,
        _style_axes,
        _write_json,
    )

    _configure_mpl()
    os.makedirs(data_dir, exist_ok=True)

    json_path = os.path.join(data_dir, "loss_per_epoch.json")
    _write_json(history, json_path)
    print(f"  [data] wrote {json_path}", flush=True)

    epochs = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(11, 6))
    # all-black, distinguished by linestyle + marker rather than color (see
    # _colors()'s docstring for the same reasoning applied there)
    for (key, label, marker), linestyle in zip(
        [
            ("train_loss", "train", "o"),
            ("val_loss", "val", "D"),
            ("test_loss", "test", "s"),
        ],
        ["-", "--", ":"],
    ):
        ax.plot(
            epochs,
            [h[key] for h in history],
            marker=marker,
            markersize=8.5,
            linewidth=2,
            color=TEXT_PRIMARY,
            linestyle=linestyle,
            label=label,
        )
    # LR cuts (blue) and lambda_7 ramp-step fires (red) as vertical
    # markers, so the stage-transition machinery (see Train/train.py's
    # scheduler.step() block) is visible against the loss trend. .get():
    # epochs recorded before these keys existed just don't get a line.
    lr_cut_epochs = [h["epoch"] for h in history if h.get("lr_cut")]
    lambda_7_fire_epochs = [
        h["epoch"] for h in history if h.get("lambda_7_stage_fired")
    ]
    for i, ep in enumerate(lr_cut_epochs):
        ax.axvline(
            ep,
            color="tab:blue",
            linestyle="-",
            linewidth=1.2,
            alpha=0.7,
            label="lr cut" if i == 0 else None,
        )
    for i, ep in enumerate(lambda_7_fire_epochs):
        ax.axvline(
            ep,
            color="tab:red",
            linestyle="-",
            linewidth=1.2,
            alpha=0.7,
            label="lambda_7 stage" if i == 0 else None,
        )
    ax.set_xlabel("epoch", color=TEXT_PRIMARY)
    ax.set_ylabel("loss", color=TEXT_PRIMARY)
    # epoch numbers are integers; matplotlib's default locator otherwise
    # ticks fractional epochs (1.00, 1.25, 1.50, ...), which don't mean
    # anything here
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    _style_axes(ax)
    # boxed and outside the axes (right side) so it never sits on top of
    # the loss curves themselves
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        frameon=True,
        labelcolor=TEXT_PRIMARY,
    )
    fig.tight_layout()
    out_path = os.path.join(data_dir, "loss_per_epoch.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"  [plots] wrote {out_path}", flush=True)


def save_epoch_pct_err_plot(history: list[dict], data_dir: str) -> None:
    """Update the run-wide train/val/test percent-error-vs-epoch trend.

    Same pattern as `save_epoch_loss_plot` (cheap, reads `history` off
    disk, written once at the ROOT of data_dir since it's a running trend
    not a per-epoch snapshot) - deliberately does NOT write its own JSON
    sidecar, since `history`'s records (and therefore
    `loss_per_epoch.json`, already written by `save_epoch_loss_plot`)
    already carry "train_pct_err"/"val_pct_err"/"test_pct_err" alongside
    the loss fields; a second JSON would just be a duplicate of the same
    per-epoch records.

    Parameters
    ----------
    history : list of dict
        Per-epoch records, each with "epoch", "train_pct_err",
        "val_pct_err", "test_pct_err"; typically
        `load_epoch_history(data_dir)`.
    data_dir : str
        Root plots directory.

    Returns
    -------
    None
    """
    if len(history) <= 1:
        return

    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    from Baselines.run.metrics import TEXT_PRIMARY, _configure_mpl, _style_axes

    _configure_mpl()
    os.makedirs(data_dir, exist_ok=True)

    epochs = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(11, 6))
    for (key, label, marker), linestyle in zip(
        [
            ("train_pct_err", "train", "o"),
            ("val_pct_err", "val", "D"),
            ("test_pct_err", "test", "s"),
        ],
        ["-", "--", ":"],
    ):
        ax.plot(
            epochs,
            [h.get(key, float("nan")) for h in history],
            marker=marker,
            markersize=8.5,
            linewidth=2,
            color=TEXT_PRIMARY,
            linestyle=linestyle,
            label=label,
        )
    ax.set_xlabel("epoch", color=TEXT_PRIMARY)
    ax.set_ylabel("mean percent error (%)", color=TEXT_PRIMARY)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    _style_axes(ax)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        frameon=True,
        labelcolor=TEXT_PRIMARY,
    )
    fig.tight_layout()
    out_path = os.path.join(data_dir, "pct_err_per_epoch.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"  [plots] wrote {out_path}", flush=True)


# Section headings for the config dump, in the order RunConfig declares its
# fields. Any field NOT listed here still gets dumped, under "Other" - a new
# RunConfig field must never silently vanish from the record of the run.
_CFG_SECTIONS: list[tuple[str, tuple[str, ...]]] = [
    (
        "Paths",
        (
            "hdf5",
            "encodings_sqlite3_path",
            "ckpt_best",
            "ckpt_dir",
            "resume",
        ),
    ),
    (
        "Model",
        (
            "lambda_1",
            "lambda_2",
            "lambda_3",
            "lambda_4",
            "lambda_5",
            "msg_seed",
            "atm_chunk",
            "mol_chunk",
            "compile",
            "eps_embd",
            "eps_msgp",
        ),
    ),
    (
        "Loss",
        (
            "lambda_6",
            "lambda_7",
            "lambda_8",
            "smoothing_lr_cut_trigger",
            "smoothing_n_stages",
        ),
    ),
    (
        "Training",
        (
            "lr",
            "lr_factor",
            "lr_patience",
            "lr_threshold",
            "lr_min",
            "weight_decay",
            "grad_clip",
            "adam_eps",
            "epochs",
            "batcher_seed",
            "atom_size_ceil",
            "dataset_frac",
            "num_workers",
            "max_batches",
            "ckpt_interval_sec",
            "ckpt_rclone_dest",
            "data_dir",
            "data_rclone_dest",
            "verbosity",
            "profiler",
            "prof_warmup",
            "prof_active",
            "prof_molecules",
        ),
    ),
    ("Data", ("buckets",)),
]


def _rtf_escape(text: str) -> str:
    r"""Escape a string for inclusion in an RTF body.

    RTF gives `\`, `{` and `}` structural meaning, and is a 7-bit format: a raw
    non-ASCII byte would be mis-decoded by the reader, so those characters are
    emitted as `\uN?` escapes (N = the code point, `?` = the ASCII fallback
    that readers too old to understand `\u` display instead).

    Parameters
    ----------
    text : str
        Raw text.

    Returns
    -------
    str
        RTF-safe text.
    """
    out = []
    for ch in text:
        if ch in "\\{}":
            out.append("\\" + ch)
        elif ord(ch) < 128:
            out.append(ch)
        else:
            out.append(f"\\u{ord(ch)}?")
    return "".join(out)


def save_run_config_rtf(cfg: RunConfig, data_dir: str) -> None:
    """Dump the run's full RunConfig to `{data_dir}/run_config.rtf`.

    Written once, at the start of training, so the data dir is
    self-describing: whoever reads the per-epoch metrics months later can see
    exactly which hyperparameters produced them without digging up the notebook
    cell that launched the run. Every field of the dataclass is dumped, grouped
    by `_CFG_SECTIONS`; a field missing from that table lands under "Other"
    rather than being dropped.

    Parameters
    ----------
    cfg : RunConfig
        The config the run was launched with.
    data_dir : str
        Root data directory.

    Returns
    -------
    None
    """
    from dataclasses import fields

    values = {f.name: getattr(cfg, f.name) for f in fields(cfg)}
    listed = {name for _, names in _CFG_SECTIONS for name in names}
    sections = _CFG_SECTIONS + [
        ("Other", tuple(n for n in values if n not in listed))
    ]
    key_width = max(len(n) for n in values) if values else 0

    # \f0 = proportional (headings), \f1 = monospace, \fs is in half-points.
    lines = [
        r"{\rtf1\ansi\ansicpg1252\deff0",
        r"{\fonttbl{\f0\fswiss Helvetica;}{\f1\fmodern Courier New;}}",
        r"\f0\fs32\b ScatterNet run configuration\b0\fs20\par",
        r"\i "
        + _rtf_escape(time.strftime("%Y-%m-%d %H:%M:%S %Z"))
        + r"\i0\par",
    ]
    for title, names in sections:
        names = tuple(n for n in names if n in values)
        if not names:
            continue
        lines.append(r"\par\f0\fs24\b " + _rtf_escape(title) + r"\b0\fs20\par")
        for name in names:
            row = f"{name.ljust(key_width)} = {values[name]!r}"
            lines.append(r"{\f1 " + _rtf_escape(row) + r"\par}")
    lines.append("}")

    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, "run_config.rtf")
    with open(out_path, "w", encoding="ascii") as fh:
        fh.write("\n".join(lines))
    print(f"  [data] wrote {out_path}", flush=True)
