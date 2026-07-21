from collections.abc import Iterable

import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped

from Baselines.baseline import Baseline
from ScatterNet.batching import Batch


class AtomCountBaseline(Baseline):
    """
    Zero-order baseline: predict the training mean I(q) bucketed by atom
    count M.

    Beating this baseline proves the model uses more than just molecule size.
    At inference the nearest observed bucket is used for unseen atom counts.
    """

    _mean_by_count: dict[int, torch.Tensor] | None

    def __init__(self) -> None:
        self._mean_by_count = None

    @jaxtyped(typechecker=beartype)
    def fit(self, loader: Iterable[Batch]) -> "AtomCountBaseline":
        """Accumulate per-atom-count mean I(q) from the training loader."""
        sums: dict[int, torch.Tensor] = {}
        counts: dict[int, int] = {}

        for batch in loader:
            atom_counts = batch.padding_mask().sum(
                dim=1
            )  # real atoms per molecule (the bucket key)
            for i, c in enumerate(atom_counts.tolist()):
                c = int(c)
                if c not in sums:
                    sums[c] = batch.iqval[i].clone()
                    counts[c] = 1
                else:
                    sums[c] = sums[c] + batch.iqval[i]
                    counts[c] += 1

        self._mean_by_count = {c: sums[c] / counts[c] for c in sums}
        return self

    @jaxtyped(typechecker=beartype)
    def __call__(self, batch: Batch) -> Float[torch.Tensor, "N Q"]:  # noqa: F722
        """Return the mean I(q) for the nearest bucket per molecule."""
        if self._mean_by_count is None:
            raise RuntimeError("AtomCountBaseline must be fit before calling")
        atom_counts = batch.padding_mask().sum(dim=1).tolist()
        keys = sorted(self._mean_by_count.keys())
        device = batch.vocab.device
        rows: list[torch.Tensor] = []

        for c in atom_counts:
            closest = min(
                keys, key=lambda k: abs(k - c)
            )  # nearest seen bucket for unseen counts
            rows.append(self._mean_by_count[closest].to(device))

        return torch.stack(rows, dim=0)
