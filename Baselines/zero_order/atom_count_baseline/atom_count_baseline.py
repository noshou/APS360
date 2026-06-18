import torch

from typing          import Iterable
from jaxtyping       import Float, Int, jaxtyped
from beartype        import beartype
from ScatterNet.batching import Batch
from Baselines.baseline  import Baseline


class AtomCountBaseline(Baseline):
    """
    Zero-order baseline: predict the training mean I(q) bucketed by atom count M.

    Beating this baseline proves the model uses more than just molecule size.
    At inference the nearest observed bucket is used for unseen atom counts.
    """

    _mean_by_count: dict[int, torch.Tensor]

    @jaxtyped(typechecker=beartype)
    def fit(self, loader: Iterable[Batch]) -> "AtomCountBaseline":
        """Accumulate per-atom-count mean I(q) from the training loader."""
        buckets: dict[int, list[torch.Tensor]] = {}

        for batch in loader:
            counts = batch.padding_mask().sum(dim=1)  # (N,) int
            for i, c in enumerate(counts.tolist()):
                c = int(c)
                if c not in buckets:
                    buckets[c] = [batch.iqval[i].clone(), 1]
                else:
                    buckets[c][0] = buckets[c][0] + batch.iqval[i]
                    buckets[c][1] += 1

        self._mean_by_count = {
            c: s / n for c, (s, n) in buckets.items()
        }
        return self

    @jaxtyped(typechecker=beartype)
    def __call__(self, batch: Batch) -> Float[torch.Tensor, "N Q"]:
        """Return the mean I(q) for the nearest atom-count bucket per molecule."""
        counts  = batch.padding_mask().sum(dim=1).tolist()  # (N,)
        keys    = sorted(self._mean_by_count.keys())
        device  = batch.vocab.device
        rows: list[torch.Tensor] = []

        for c in counts:
            # nearest bucket by absolute distance
            closest = min(keys, key=lambda k: abs(k - c))
            rows.append(self._mean_by_count[closest].to(device))

        return torch.stack(rows, dim=0)
