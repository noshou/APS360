import torch

from jaxtyping           import Float, jaxtyped
from beartype            import beartype
from typing              import Iterable

from ScatterNet.batching import Batch
from Preprocess          import VOCAB
from Baselines.baseline  import Baseline


class CompositionBaseline(Baseline):
    """
    Chemistry-only baseline: predict I(q) from atom-type composition, no geometry.

    For each molecule, outputs a weighted average of per-element mean I(q) curves
    from training, weighted by the fraction of each element type in the molecule.
    Beating this baseline proves the model uses 3D coordinates, not just chemistry.
    """

    _per_elem_iq: Float[torch.Tensor, "V Q"]

    def fit(self, loader: Iterable[Batch]) -> "CompositionBaseline":
        """Build a per-element mean I(q) table from the training loader."""
        V = len(VOCAB) + 1
        sums:           torch.Tensor | None = None
        total_presence: torch.Tensor | None = None

        for batch in loader:
            N, M = batch.vocab.shape
            counts = torch.zeros(N, V).scatter_add_(
                1, batch.vocab.long(), torch.ones(N, M)
            )
            presence = (counts > 0).float()
            presence[:, 0] = 0.0

            batch_sums = presence.T @ batch.iqval.float()  # (V, Q)
            batch_presence = presence.sum(dim=0)           # (V,)

            if sums is None or total_presence is None:
                sums           = batch_sums
                total_presence = batch_presence
            else:
                sums           += batch_sums
                total_presence += batch_presence

        if sums is None or total_presence is None:
            raise RuntimeError("fit() received an empty loader")

        self._per_elem_iq = sums / total_presence.unsqueeze(-1).clamp(min=1)
        return self

    @jaxtyped(typechecker=beartype)
    def __call__(self, batch: Batch) -> Float[torch.Tensor, "N Q"]:
        """Return element-fraction-weighted mean I(q) per molecule."""
        N, M = batch.vocab.shape
        V    = len(VOCAB) + 1
        device = batch.vocab.device

        counts = torch.zeros(N, V, device=device).scatter_add_(
            1, batch.vocab.long(), torch.ones(N, M, device=device)
        )
        counts[:, 0] = 0.0

        # normalise by total real atoms per molecule to get per-element fractions
        row_totals = counts.sum(dim=1, keepdim=True).clamp(min=1)
        weights    = counts / row_totals  # (N, V)

        return weights @ self._per_elem_iq.to(device)  # (N, Q)
