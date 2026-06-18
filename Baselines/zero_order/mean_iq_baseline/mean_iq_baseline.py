import torch

from typing          import Iterable
from jaxtyping       import Float, jaxtyped
from beartype        import beartype
from ScatterNet.batching import Batch
from Baselines.baseline  import Baseline


class MeanIQBaseline(Baseline):
    """
    Zero-order baseline: predict the training-set mean I(q) for every molecule.

    Beating this baseline proves the model is sensitive to per-molecule structure
    rather than outputting a constant curve.
    """

    _mean_iq: Float[torch.Tensor, "Q"]

    @jaxtyped(typechecker=beartype)
    def fit(self, loader: Iterable[Batch]) -> "MeanIQBaseline":
        """Compute and store the mean I(q) over all training molecules."""
        running_sum   = None
        running_count = 0

        for batch in loader:
            if running_sum is None:
                running_sum = batch.iqval.sum(dim=0)
            else:
                running_sum = running_sum + batch.iqval.sum(dim=0)
            running_count += batch.iqval.shape[0]

        if running_sum is None or running_count == 0:
            raise ValueError("loader was empty — cannot compute mean I(q)")

        self._mean_iq = running_sum / running_count
        return self

    @jaxtyped(typechecker=beartype)
    def __call__(self, batch: Batch) -> Float[torch.Tensor, "N Q"]:
        """Return the training mean I(q) repeated for each molecule in the batch."""
        N = batch.iqval.shape[0]
        return self._mean_iq.to(batch.vocab.device).unsqueeze(0).expand(N, -1)
