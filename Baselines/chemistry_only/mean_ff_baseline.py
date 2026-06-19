import torch

from jaxtyping           import Float, jaxtyped
from beartype            import beartype

from ScatterNet.batching import Batch
from Baselines.baseline  import Baseline, build_fmag_table


class MeanFFBaseline(Baseline):
    """
    Chemistry-only baseline: predict I(q) as incoherent scattering Σ_i f_i(q)².

    Uses xraydb form factors; no geometry or training data required.
    This is the lower bound of coherent scattering — all cross terms zeroed out.
    Beating this baseline proves the model learns more than per-element form factors.
    """

    _fmag_table: Float[torch.Tensor, "V Q"]

    def __init__(
        self,
        qgrid:  Float[torch.Tensor, "Q"],
        energy: float,
    ) -> None:
        """
        Args:
            qgrid:  q-point grid in Å⁻¹ (Q,)
            energy: X-ray energy in eV for anomalous f1/f2 corrections
        """
        self._fmag_table = build_fmag_table(qgrid, energy)

    @jaxtyped(typechecker=beartype)
    def __call__(self, batch: Batch) -> Float[torch.Tensor, "N Q"]:
        """Return Σ_i f_i(q)² summed over real atoms per molecule."""
        device = batch.vocab.device
        fmag_table = self._fmag_table.to(device)

        f_mags = fmag_table[batch.vocab]                     # (N, M, Q)
        mask   = batch.padding_mask().unsqueeze(-1).float()  # (N, M, 1)

        return (f_mags ** 2 * mask).sum(dim=1)               # (N, Q)
