import glob
import json
import os
import time
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import replace as dc_replace

import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped

from Baselines.baseline import Baseline
from Baselines.metrics import evaluate as _bl_evaluate
from Baselines.metrics import run_all_plots
from ScatterNet import ScatterNet
from ScatterNet.batching import Batch
from ScatterNet.utils.config import RunConfig


class ScatterNetBaseline(Baseline):
    """Adapts a ScatterNet model to the `Baseline` interface.

    Lets a ScatterNet model be scored and plotted with the exact same code
    (Baselines.metrics.evaluate/run_all_plots) used for every baseline in
    Baselines/kaggle_baselines.ipynb.

    Optionally, when `compute_loss` is True, each forward pass also
    accumulates the composite training loss and log1p-space R2 as a side
    effect, so the single evaluation pass that drives the diagnostic plots
    ALSO yields the epoch's test loss/R2 - removing a second, redundant full
    pass over the test set. Read the accumulated result back with `loss_r2()`.
    This is only correct because the plots pass forces tensor-parallel routing
    (see `_force_tp`): under TP every rank holds the full-batch output, so the
    per-batch loss is the whole-bucket loss and needs no cross-rank reduction -
    exactly the TP branch of train.py's `evaluate()`. (DP's molecule-shard
    all-reduce is unnecessary here prec1isely because TP is forced.)
    """

    def __init__(
        self,
        model: ScatterNet,
        amp: bool,
        device: str,
        compute_loss: bool = False,
        lambda_6: float | None = None,
        lambda_7: float | None = None,
        progress: bool = False,
        n_batch: int = 0,
    ) -> None:
        """Store the model and precision/device settings used at call time.

        Parameters
        ----------
        model : ScatterNet
            Model to wrap; called in whatever train()/eval() mode the
            caller already put it in.
        amp : bool
            Whether to run the forward pass under fp16 autocast.
        device : str
            Torch device string; autocast is only enabled on CUDA.
        compute_loss : bool, optional
            If True, accumulate the composite loss and log1p R2 over every
            batch this baseline is called on (see class docstring). If False
            (default), no loss stats are collected and `loss_r2()` returns
            NaNs.
        lambda_6, lambda_7 : float, optional
            Loss weights forwarded to `model.compute_loss`; required when
            `compute_loss` is True.
        progress : bool
            Print a progress line every 20 batches.
        n_batch : int
            Total number of batches, for the "i/N" in the progress line.

        Returns
        -------
        None
        """
        self._model = model
        self._amp = amp and device.startswith("cuda")
        self._device = device
        self._compute_loss = compute_loss
        self._lambda_6 = lambda_6
        self._lambda_7 = lambda_7
        self._progress = progress
        self._n_batch = n_batch
        self._seen = 0
        self._last_printed = -1
        self._t0 = time.time()
        # loss/R2 accumulators (only used if compute_loss given)
        self._loss_sum = 0.0
        self._mols = 0.0
        self._ss_res = 0.0
        self._sum_y = 0.0
        self._sum_y2 = 0.0
        self._n_elem = 0.0

    def loss_r2(self) -> tuple[float, float]:
        """Return (mean composite loss, log1p R2) accumulated over all calls.

        Returns NaNs if `compute_loss` was False or nothing was evaluated.
        Both ranks accumulate identical values under forced TP, so rank 0's
        return value is the correct global result.
        """
        if not self._compute_loss or self._mols == 0:
            return float("nan"), float("nan")
        mean_loss = self._loss_sum / self._mols
        ss_tot = self._sum_y2 - self._sum_y**2 / self._n_elem
        r2 = 1.0 - self._ss_res / ss_tot if ss_tot > 0 else 0.0
        return mean_loss, r2

    @jaxtyped(typechecker=beartype)
    def __call__(self, batch: Batch) -> Float[torch.Tensor, "N Q"]:  # noqa: F722,F821
        """Predict I(q) for every molecule in `batch`.

        Parameters
        ----------
        batch : Batch
            Batch of molecules to predict scattering curves for.

        Returns
        -------
        torch.Tensor
            Predicted I(q), shape (N, Q), clamped to non-negative (matching
            every other Baseline's convention - see Baselines/metrics.py).
        """
        # Baselines.metrics.evaluate()'s loop never moves batches to device.
        # ScatterNet needs its own explicit move, same as
        # every call site in train.py's training/eval loops.
        batch = dc_replace(
            batch,
            vocab=batch.vocab.to(self._device),
            iqval=batch.iqval.to(self._device),
            coord=batch.coord.to(self._device),
        )
        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=self._amp
        ):
            iq, fmags, sigmas, local_batch, _ = self._model(batch)
            # Composite loss/R2 as a side effect, in the same forward pass, so
            # plots eval doubles as the test loss/R2 eval. Computed inside
            # autocast + with the raw (unclamped) iq, matching
            # train.py evaluate() exactly.
            if self._compute_loss:
                lambda_6, lambda_7 = self._lambda_6, self._lambda_7
                assert lambda_6 is not None and lambda_7 is not None, (
                    "lambda_6/lambda_7 required when compute_loss is True"
                )
                loss = self._model.compute_loss(
                    iq,
                    fmags,
                    sigmas,
                    local_batch,
                    lambda_6,
                    lambda_7,
                )
                iqf = iq.float()
                n = local_batch.iqval.shape[0]
                log_pred = torch.log1p(iqf)
                log_target = torch.log1p(local_batch.iqval)
                self._loss_sum += loss.item() * n
                self._mols += n
                self._ss_res += ((log_pred - log_target) ** 2).sum().item()
                self._sum_y += log_target.sum().item()
                self._sum_y2 += (log_target**2).sum().item()
                self._n_elem += log_target.numel()
        self._seen += 1
        if (
            self._progress
            and self._seen % 20 == 0
            and self._seen != self._last_printed
        ):
            self._last_printed = self._seen
            rate = self._seen / (time.time() - self._t0)
            print(
                f"  [test/plots] batch {self._seen:5d}/{self._n_batch}",
                f"  {rate} batch/s",
                flush=True,
            )

        # Move the prediction back to CPU.
        return iq.float().clamp(min=0).cpu()


@contextmanager
def _force_tp(model: ScatterNet):
    """Temporarily disable DP routing so every prediction covers full batch.

    Baselines.metrics.evaluate() assumes a baseline's prediction shape
    matches the full batch it was given (pred.shape == batch.iqval.shape).
    A DP-routed bucket returns only a rank-local molecule shard instead,
    which would silently break that assumption (and Baselines.metrics has no
    notion of cross-rank shards to begin with - it's a plain per-process
    function, unlike train.py's own evaluate()). TP is unaffected, since
    every rank already reconstructs the full-batch output internally, so
    forcing TP here keeps every rank's independent evaluate() call
    numerically identical without touching Baselines/metrics.py.

    Parameters
    ----------
    model : ScatterNet
        Model whose `_dp_atom_threshold` is temporarily zeroed.

    Yields
    ------
    None
    """
    saved = model._dp_atom_threshold
    model._dp_atom_threshold = 0
    try:
        yield
    finally:
        model._dp_atom_threshold = saved


def save_epoch_plots(
    model: ScatterNet,
    loader: Iterable[Batch],
    q_grid: torch.Tensor,
    amp: bool,
    device: str,
    data_dir: str,
    epoch: int,
    rank: int,
    compute_loss: bool = False,
    lambda_6: float | None = None,
    lambda_7: float | None = None,
    verbose: bool = False,
) -> tuple[float, float]:
    """
    Evaluate `model` on `loader`, write this epoch's diagnostic plots, and
    return the test loss/R2 computed in the same single pass.

    Mirrors Baselines/kaggle_baselines.ipynb's own evaluate() + run_all_plots()
    call, so training progress is comparable to the baseline notebook's
    plots. The single shared run_all_plots() drives the whole set: per-q R^2,
    per-q percent error, the aggregated per-q summary, Kratky overlay, the
    per-molecule signed-residual histogram (with skew) and per-molecule MSLE
    distribution, error-vs-atom-count (+ scaling-slope fit) and signed-residual
    vs atom count, and a one-row summary bar chart. Because evaluate() now
    collects the per-molecule error distributions too, every new baseline
    statistic appears in the epoch plots automatically, no change needed here.

    When `compute_loss` is True, this ONE forced-TP pass also accumulates the
    composite loss and log1p R2 (via ScatterNetBaseline), so the caller no
    longer needs a separate `evaluate(test_loader)` pass - the test set is
    walked once, not twice. The values are numerically the same as train.py's
    `evaluate()` would produce: forcing TP just changes the parallel reduction,
    not the per-molecule output (see ScatterNetBaseline's docstring).

    Must be called on every rank in a distributed run - forward passes
    through the model (even TP-forced) need every rank's participation for
    their internal all-reduce/gather. Only rank 0 writes files; every rank
    computes identical loss/R2 under forced TP, so the return value is correct
    on every rank.

    Parameters
    ----------
    model : ScatterNet
        Model to evaluate; called in whatever train()/eval() mode the
        caller already put it in.
    loader : Iterable[Batch]
        Evaluation batches (typically the test loader).
    q_grid : torch.Tensor
        q-point grid, shape (Q,).
    amp : bool
        Whether to run the forward pass under fp16 autocast.
    device : str
        Torch device string.
    data_dir : str
        Root directory; this epoch's plots are written to
        `{data_dir}/epoch_{epoch:03d}/`.
    epoch : int
        Current epoch number, used to name the per-epoch subdirectory.
    rank : int
        This process's rank; only rank 0 writes plot files.
    compute_loss : bool, optional
        If True, the pass also returns (test_loss, test_r2). If False
        (default), the return value is (nan, nan) and only plots are
        produced.
    lambda_6, lambda_7 : float, optional
        Loss weights, required when `compute_loss` is True.
    verbose : bool
        Print a progress line every 20 batches (rank 0 only).

    Returns
    -------
    tuple of (float, float)
        (test_loss, test_r2) in log1p space, or (nan, nan) if
        compute_loss is False.
    """
    baseline = ScatterNetBaseline(
        model,
        amp,
        device,
        compute_loss=compute_loss,
        lambda_6=lambda_6,
        lambda_7=lambda_7,
        progress=verbose and rank == 0,
        n_batch=len(loader),  # type: ignore[arg-type]
    )
    with _force_tp(model):
        result = _bl_evaluate(baseline, loader, q_grid, "ScatterNet")
    if rank == 0:
        epoch_dir = os.path.join(data_dir, f"epoch_{epoch:03d}")
        written = run_all_plots([result], q_grid, epoch_dir)
        print(
            f"  [plots] wrote {len(written)} file(s) to {epoch_dir}",
            flush=True,
        )
    return baseline.loss_r2()


def save_batch_loss_plot(
    batch_losses: list[tuple[int, float]], data_dir: str, epoch: int
) -> None:
    """Plot this epoch's per-batch training loss into the epoch's plot dir.

    Cheap: `batch_losses` is data already collected in the training loop (no
    extra forward passes, no model call). On a mid-epoch resume the
    list starts at the resume point, so the recorded batch index (not the list
    position) drives the x-axis - the pre-resume stretch is left honestly blank
    rather than shifting the whole curve left.

    Parameters
    ----------
    batch_losses : list of (int, float)
        (global batch index, training loss) for each batch trained this epoch.
    data_dir : str
        Root plots directory; the file is written to
        `{data_dir}/epoch_{epoch:03d}/loss_per_batch.png`, alongside the
        epoch's diagnostic plots.
    epoch : int
        Current epoch number.

    Returns
    -------
    None
    """
    import matplotlib.pyplot as plt

    from Baselines.metrics import (
        PALETTE,
        TEXT_PRIMARY,
        _configure_mpl,
        _style_axes,
    )

    if not batch_losses:
        return
    _configure_mpl()
    epoch_dir = os.path.join(data_dir, f"epoch_{epoch:03d}")
    os.makedirs(epoch_dir, exist_ok=True)

    xs = [b for b, _ in batch_losses]
    ys = [loss for _, loss in batch_losses]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(xs, ys, color=PALETTE[0], linewidth=1.2)
    ax.set_yscale(
        "log"
    )  # loss spans well over an order of magnitude within an epoch
    ax.set_xlabel("batch", color=TEXT_PRIMARY)
    ax.set_ylabel("training loss", color=TEXT_PRIMARY)
    ax.set_title(rf"epoch {epoch}: loss per batch", color=TEXT_PRIMARY, pad=10)
    _style_axes(ax)
    fig.tight_layout()
    out_path = os.path.join(epoch_dir, "loss_per_batch.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plots] wrote {out_path}", flush=True)


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
    print(f"  [plots] wrote {out_path}", flush=True)


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
            print(f"  [plots] skipping unreadable {path}", flush=True)
    history.sort(key=lambda h: h["epoch"])
    return history


def save_epoch_loss_plot(
    history: list[dict], data_dir: str, epoch: int
) -> None:
    """Plot train/val/test loss vs epoch, up to and including `epoch`.

    Cheap: reads `history` (per-epoch records already on disk); no model call.
    Written to `{data_dir}/epoch_{epoch:03d}/loss_per_epoch.png`,
    so each epoch keeps its own snapshot of the curve instead.

    Parameters
    ----------
    history : list of dict
        Per-epoch records, each with "epoch", "train_loss", "val_loss",
        "test_loss"; typically `load_epoch_history(data_dir)`.
    data_dir : str
        Root plots directory.
    epoch : int
        Current epoch number, used to name the per-epoch subdirectory.

    Returns
    -------
    None
    """
    import matplotlib.pyplot as plt

    from Baselines.metrics import (
        PALETTE,
        TEXT_PRIMARY,
        _configure_mpl,
        _style_axes,
    )

    if not history:
        return
    _configure_mpl()
    epoch_dir = os.path.join(data_dir, f"epoch_{epoch:03d}")
    os.makedirs(epoch_dir, exist_ok=True)

    epochs = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(11, 6))
    for (key, label), color in zip(
        [("train_loss", "train"), ("val_loss", "val"), ("test_loss", "test")],
        PALETTE,
    ):
        ax.plot(
            epochs,
            [h[key] for h in history],
            marker="o",
            linewidth=2,
            color=color,
            label=label,
        )
    ax.set_xlabel("epoch", color=TEXT_PRIMARY)
    ax.set_ylabel("loss", color=TEXT_PRIMARY)
    ax.set_title(
        f"loss per epoch (through epoch {epoch})", color=TEXT_PRIMARY, pad=10
    )
    _style_axes(ax)
    ax.legend(loc="upper right", frameon=False, labelcolor=TEXT_PRIMARY)
    fig.tight_layout()
    out_path = os.path.join(epoch_dir, "loss_per_epoch.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
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
            "dp_atom_threshold",
            "compile",
            "amp",
            "amp_init_scale",
            "eps_embd",
            "eps_msgp",
        ),
    ),
    ("Loss", ("lambda_6", "lambda_7")),
    (
        "Training",
        (
            "lr",
            "weight_decay",
            "grad_clip",
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

    Written once, at the start of training (rank 0 only), so the data dir is
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
    print(f"  [plots] wrote {out_path}", flush=True)
