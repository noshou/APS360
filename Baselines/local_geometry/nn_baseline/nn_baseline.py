import torch

from jaxtyping           import Float, jaxtyped
from beartype            import beartype
from ScatterNet.batching import Batch
from Baselines.baseline  import Baseline, build_fmag_table


class NNBaseline(Baseline):
    """
    Local-geometry baseline: I(q) ≈ I(0)·sinc(q·r_nn) where r_nn is the mean
    nearest-neighbour distance across all atoms in the molecule.

    Captures the dominant bond-length scale without the full P(r) computation.
    Uses I(0) = (Σ f_i(0))². O(M²) per molecule.
    Beating this baseline proves the model uses multi-scale distance information.
    """

    _qgrid:      Float[torch.Tensor, "Q"]
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
        self._qgrid      = qgrid
        self._fmag_table = build_fmag_table(qgrid, energy)

    @jaxtyped(typechecker=beartype)
    def __call__(self, batch: Batch) -> Float[torch.Tensor, "N Q"]:
        """Return I(0)·sinc(q·r_nn) per molecule using mean nearest-neighbour distance."""
        device = batch.coord.device
        mask   = batch.padding_mask()           # (N, M)
        qgrid  = self._qgrid.to(device)
        ftable = self._fmag_table.to(device)

        preds: list[Float[torch.Tensor, "Q"]] = []

        for n in range(batch.coord.shape[0]):
            coords_n = batch.coord[n][mask[n]]  # (m, 3)
            m = coords_n.shape[0]

            if m < 2:
                preds.append(torch.zeros(qgrid.shape[0], device=device))
                continue

            diff  = coords_n.unsqueeze(0) - coords_n.unsqueeze(1)  # (m, m, 3)
            dists = diff.norm(dim=-1)                               # (m, m)
            dists.fill_diagonal_(float('inf'))
            r_nn  = dists.min(dim=1).values.mean()

            f0  = ftable[batch.vocab[n][mask[n]], 0]
            i0  = f0.sum() ** 2

            qr   = qgrid * r_nn
            sinc = torch.where(qr.abs() < 1e-8, torch.ones_like(qr), torch.sin(qr) / qr)
            preds.append(i0 * sinc)

        return torch.stack(preds)
