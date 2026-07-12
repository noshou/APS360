import os
import torch

from contextlib          import contextmanager
from dataclasses         import replace as dc_replace
from jaxtyping           import Float, jaxtyped
from beartype            import beartype
from ScatterNet.batching import Batch
from Baselines.baseline  import Baseline
from Baselines.metrics   import evaluate as _bl_evaluate, run_all_plots

class ScatterNetBaseline(Baseline):
    """Adapts a ScatterNet model to the `Baseline` interface.

    Lets a ScatterNet model be scored and plotted with the exact same code
    (Baselines.metrics.evaluate/run_all_plots) used for every baseline in
    Baselines/kaggle_baselines.ipynb.

    Optionally, when a `criterion` is supplied, each forward pass also
    accumulates the composite training loss and log1p-space R2 as a side
    effect, so the single evaluation pass that drives the diagnostic plots
    ALSO yields the epoch's test loss/R2 - removing a second, redundant full
    pass over the test set. Read the accumulated result back with `loss_r2()`.
    This is only correct because the plots pass forces tensor-parallel routing
    (see `_force_tp`): under TP every rank holds the full-batch output, so the
    per-batch loss is the whole-bucket loss and needs no cross-rank reduction -
    exactly the TP branch of train.py's `evaluate()`. (DP's molecule-shard
    all-reduce is unnecessary here precisely because TP is forced.)
    """

    def __init__(
        self,
        model,
        amp:      bool,
        device:   str,
        criterion=None,
        lambda_6: float | None = None,
        lambda_7: float | None = None,
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
        criterion : Loss, optional
            If given, accumulate the composite loss and log1p R2 over every
            batch this baseline is called on (see class docstring). If None,
            no loss stats are collected and `loss_r2()` returns NaNs.
        lambda_6, lambda_7 : float, optional
            Loss weights forwarded to `criterion.loss`; required when
            `criterion` is given.

        Returns
        -------
        None
        """
        self._model     = model
        self._amp       = amp and device.startswith("cuda")
        self._device    = device
        self._criterion = criterion
        self._lambda_6  = lambda_6
        self._lambda_7  = lambda_7
        # loss/R2 accumulators (mirrors train.py evaluate(); only used if criterion given)
        self._loss_sum = 0.0
        self._mols     = 0.0
        self._ss_res   = 0.0
        self._sum_y    = 0.0
        self._sum_y2   = 0.0
        self._n_elem   = 0.0

    def loss_r2(self) -> tuple[float, float]:
        """Return (mean composite loss, log1p R2) accumulated over all calls.

        Returns NaNs if no `criterion` was supplied or nothing was evaluated.
        Both ranks accumulate identical values under forced TP, so rank 0's
        return value is the correct global result.
        """
        if self._criterion is None or self._mols == 0:
            return float("nan"), float("nan")
        mean_loss = self._loss_sum / self._mols
        ss_tot    = self._sum_y2 - self._sum_y ** 2 / self._n_elem
        r2        = 1.0 - self._ss_res / ss_tot if ss_tot > 0 else 0.0
        return mean_loss, r2

    @jaxtyped(typechecker=beartype)
    def __call__(self, batch: Batch) -> Float[torch.Tensor, "N Q"]:
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
        # Baselines.metrics.evaluate()'s loop never moves batches to device (most
        # baselines run on CPU); ScatterNet needs its own explicit move, same as
        # every call site in train.py's training/eval loops.
        batch = dc_replace(
            batch,
            vocab=batch.vocab.to(self._device),
            iqval=batch.iqval.to(self._device),
            coord=batch.coord.to(self._device),
        )
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self._amp):
            iq, fmags, sigmas, local_batch, _ = self._model(batch)
            # Composite loss/R2 as a side effect, in the same forward pass, so the
            # plots eval doubles as the test loss/R2 eval (no second full pass).
            # Computed inside autocast + with the raw (unclamped) iq, matching
            # train.py evaluate() exactly. local_batch == batch here (TP forced).
            if self._criterion is not None:
                loss = self._criterion.loss(
                    iq, fmags, sigmas, local_batch, self._lambda_6, self._lambda_7
                )
                iqf        = iq.float()
                n          = local_batch.iqval.shape[0]
                log_pred   = torch.log1p(iqf)
                log_target = torch.log1p(local_batch.iqval)
                self._loss_sum += loss.item() * n
                self._mols     += n
                self._ss_res   += ((log_pred - log_target) ** 2).sum().item()
                self._sum_y    += log_target.sum().item()
                self._sum_y2   += (log_target ** 2).sum().item()
                self._n_elem   += log_target.numel()
        # Move the prediction back to CPU: evaluate() combines it with the loop's
        # original (unmoved, CPU) batch.iqval, and accumulates every metric on CPU
        # (see the .cpu() reductions in Baselines/metrics.py). Returning a cuda
        # tensor here would just push the same device mismatch one op downstream.
        return iq.float().clamp(min=0).cpu()


@contextmanager
def _force_tp(model):
    """Temporarily disable DP routing so every prediction covers the full input batch.

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
    model,
    loader,
    q_grid,
    amp:       bool,
    device:    str,
    plots_dir: str,
    epoch:     int,
    rank:      int,
    criterion=None,
    lambda_6:  float | None = None,
    lambda_7:  float | None = None,
 ) -> tuple[float, float]:

    """
    Evaluate `model` on `loader`, write this epoch's diagnostic plots, and
    return the test loss/R2 computed in the same single pass.

    Mirrors Baselines/kaggle_baselines.ipynb's own evaluate() + run_all_plots()
    call (per-q R^2, per-q percent error, Kratky overlay, residual histogram,
    error-vs-atom-count, and a one-row summary bar chart), so training
    progress is directly comparable to the baseline notebook's plots.

    When `criterion` is given, this ONE forced-TP pass also accumulates the
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
    plots_dir : str
        Root directory; this epoch's plots are written to
        `{plots_dir}/epoch_{epoch:03d}/`.
    epoch : int
        Current epoch number, used to name the per-epoch subdirectory.
    rank : int
        This process's rank; only rank 0 writes plot files.
    criterion : Loss, optional
        If given, the pass also returns (test_loss, test_r2). If None, the
        return value is (nan, nan) and only plots are produced.
    lambda_6, lambda_7 : float, optional
        Loss weights, required when `criterion` is given.

    Returns
    -------
    tuple of (float, float)
        (test_loss, test_r2) in log1p space, or (nan, nan) if no criterion.
    """
    baseline = ScatterNetBaseline(model, amp, device, criterion, lambda_6, lambda_7)
    with _force_tp(model):
        result = _bl_evaluate(baseline, loader, q_grid, "ScatterNet")
    if rank == 0:
        epoch_dir = os.path.join(plots_dir, f"epoch_{epoch:03d}")
        written   = run_all_plots([result], q_grid, epoch_dir)
        print(f"  [plots] wrote {len(written)} file(s) to {epoch_dir}", flush=True)
    return baseline.loss_r2()


def save_batch_loss_plot(batch_losses, plots_dir: str, epoch: int) -> None:
    """Plot this epoch's per-batch training loss into the epoch's plot dir.

    Cheap: `batch_losses` is data already collected in the training loop (no
    extra forward passes, no model call). Rank 0 only. On a mid-epoch resume the
    list starts at the resume point, so the recorded batch index (not the list
    position) drives the x-axis - the pre-resume stretch is left honestly blank
    rather than shifting the whole curve left.

    Parameters
    ----------
    batch_losses : list of (int, float)
        (global batch index, training loss) for each batch trained this epoch.
    plots_dir : str
        Root plots directory; the file is written to
        `{plots_dir}/epoch_{epoch:03d}/loss_per_batch.png`, alongside the
        epoch's diagnostic plots.
    epoch : int
        Current epoch number.

    Returns
    -------
    None
    """
    import matplotlib.pyplot as plt
    from Baselines.metrics import _configure_mpl, _style_axes, PALETTE, TEXT_PRIMARY
    if not batch_losses:
        return
    _configure_mpl()
    epoch_dir = os.path.join(plots_dir, f"epoch_{epoch:03d}")
    os.makedirs(epoch_dir, exist_ok=True)

    xs = [b for b, _ in batch_losses]
    ys = [l for _, l in batch_losses]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(xs, ys, color=PALETTE[0], linewidth=1.2)
    ax.set_yscale("log")   # loss spans well over an order of magnitude within an epoch
    ax.set_xlabel("batch", color=TEXT_PRIMARY)
    ax.set_ylabel("training loss", color=TEXT_PRIMARY)
    ax.set_title(rf"epoch {epoch}: loss per batch", color=TEXT_PRIMARY, pad=10)
    _style_axes(ax)
    fig.tight_layout()
    out_path = os.path.join(epoch_dir, "loss_per_batch.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plots] wrote {out_path}", flush=True)


def save_epoch_loss_plot(history, plots_dir: str) -> None:
    """Plot train/val/test loss vs epoch across the run so far.

    Cheap: reads `history` (already accumulated per epoch); no model call.
    Rank 0 only. Regenerated (overwritten) at every epoch boundary so the curve
    is always current and survives a mid-run crash - the final call, at the last
    epoch, is the end-of-training summary. Written to
    `{plots_dir}/loss_per_epoch.png` (run-level, not inside an epoch subdir).

    Parameters
    ----------
    history : list of dict
        Per-epoch records, each with "epoch", "train_loss", "val_loss",
        "test_loss".
    plots_dir : str
        Root plots directory.

    Returns
    -------
    None
    """
    import matplotlib.pyplot as plt
    from Baselines.metrics import _configure_mpl, _style_axes, PALETTE, TEXT_PRIMARY
    if not history:
        return
    _configure_mpl()
    os.makedirs(plots_dir, exist_ok=True)

    epochs = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(11, 6))
    for (key, label), color in zip(
        [("train_loss", "train"), ("val_loss", "val"), ("test_loss", "test")],
        PALETTE,
    ):
        ax.plot(epochs, [h[key] for h in history], marker="o", linewidth=2,
                color=color, label=label)
    ax.set_xlabel("epoch", color=TEXT_PRIMARY)
    ax.set_ylabel("loss", color=TEXT_PRIMARY)
    ax.set_title("loss per epoch", color=TEXT_PRIMARY, pad=10)
    _style_axes(ax)
    ax.legend(loc="upper right", frameon=False, labelcolor=TEXT_PRIMARY)
    fig.tight_layout()
    out_path = os.path.join(plots_dir, "loss_per_epoch.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plots] wrote {out_path}", flush=True)
