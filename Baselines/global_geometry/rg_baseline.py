import torch

from jaxtyping       import Float, jaxtyped
from beartype        import beartype
from ScatterNet.batching import Batch
from Baselines.baseline  import Baseline, build_fmag_table


class RgBaseline(Baseline):
    """
    Global-geometry baseline: Guinier approximation I(q) = I(0)·exp(-q²·Rg²/3).

    Rg is computed from centroid-subtracted coordinates; I(0) = (Σ f_i(0))².
    Only physically valid for q·Rg < 1.3 (Guinier region). For large molecules
    with Rg >> 1/q_max this covers only the first few q-points.
    Beating this baseline proves the model captures structure beyond global size.
    """

    _qgrid:      Float[torch.Tensor, "Q"]
    _fmag_table: Float[torch.Tensor, "V Q"]

    @jaxtyped(typechecker=beartype)
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
        self._qgrid      = qgrid
        self._fmag_table = build_fmag_table(qgrid, energy)

    @jaxtyped(typechecker=beartype)
    def __call__(self, batch: Batch) -> Float[torch.Tensor, "N Q"]:
        """Return Guinier-approximated I(q) for each molecule in the batch."""
        device = batch.vocab.device

        mask    = batch.padding_mask().float()                          # (N, M)
        n_atoms = mask.sum(dim=1, keepdim=True).clamp(min=1)           # (N, 1)

        r2  = (batch.coord ** 2).sum(dim=-1)                           # (N, M)
        rg2 = (r2 * mask).sum(dim=1, keepdim=True) / n_atoms           # (N, 1)

        f0  = self._fmag_table.to(device)[batch.vocab, 0]              # (N, M)
        i0  = ((f0 * mask).sum(dim=1, keepdim=True)) ** 2             # (N, 1)

        q2  = self._qgrid.to(device) ** 2                              # (Q,)

        return i0 * torch.exp(-q2.unsqueeze(0) * rg2 / 3)             # (N, Q)
