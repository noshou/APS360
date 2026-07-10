import os
import torch

from contextlib          import contextmanager
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
    """

    def __init__(self, model, amp: bool, device: str) -> None:
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

        Returns
        -------
        None
        """
        self._model  = model
        self._amp    = amp and device.startswith("cuda")

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
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self._amp):
            iq, _, _, _, _ = self._model(batch)
        return iq.float().clamp(min=0)


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
    rank:      int
 ) -> None:
    
    """
    Evaluate `model` on `loader` and write this epoch's diagnostic plots.

    Mirrors Baselines/kaggle_baselines.ipynb's own evaluate() + run_all_plots()
    call (per-q R^2, per-q percent error, Kratky overlay, residual histogram,
    error-vs-atom-count, and a one-row summary bar chart), so training
    progress is directly comparable to the baseline notebook's plots.

    Must be called on every rank in a distributed run - forward passes
    through the model (even TP-forced) need every rank's participation for
    their internal all-reduce/gather. Only rank 0 writes files.

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

    Returns
    -------
    None
    """
    with _force_tp(model):
        result = _bl_evaluate(
            ScatterNetBaseline(
                model, 
                amp, 
                device
            ), 
            loader, 
            q_grid, 
            "ScatterNet"
        )
    if rank == 0:
        epoch_dir = os.path.join(plots_dir, f"epoch_{epoch:03d}")
        written   = run_all_plots([result], q_grid, epoch_dir)
        print(f"  [plots] wrote {len(written)} file(s) to {epoch_dir}", flush=True)
