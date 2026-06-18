import torch

from jaxtyping           import Float, jaxtyped
from beartype            import beartype
from ScatterNet.batching import Batch
from Baselines.baseline  import Baseline, build_fmag_table


class PairPeakBaseline(Baseline):
    """
    Local-geometry baseline: I(q) ≈ I(0)·sinc(q·r*) where r* is the P(r) histogram peak.

    The peak of the pairwise distance distribution captures the dominant inter-atomic
    spacing (typically the nearest bonded distance). Uses I(0) = (Σ f_i(0))².
    Beating this baseline proves the model uses the full distance distribution,
    not just the dominant spacing.
    """

    _qgrid:      Float[torch.Tensor, "Q"]
    _fmag_table: Float[torch.Tensor, "V Q"]
    _n_bins:     int

    def __init__(
        self,
        qgrid:  Float[torch.Tensor, "Q"],
        energy: float,
        n_bins: int = 200,
    ) -> None:
        """
        Args:
            qgrid:  q-point grid in Å⁻¹ (Q,)
            energy: X-ray energy in eV for anomalous f1/f2 corrections
            n_bins: histogram resolution for P(r) peak finding
        """
        self._qgrid      = qgrid
        self._n_bins     = n_bins
        self._fmag_table = build_fmag_table(qgrid, energy)

    @jaxtyped(typechecker=beartype)
    def __call__(self, batch: Batch) -> Float[torch.Tensor, "N Q"]:
        """Return I(0)·sinc(q·r_peak) per molecule. O(M²) per molecule."""
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
            idx   = torch.triu_indices(m, m, offset=1)
            dists = dists[idx[0], idx[1]]                           # upper triangle

            hist    = torch.histc(dists, bins=self._n_bins, min=0.0, max=dists.max().item())
            r_peak  = (hist.argmax().float() + 0.5) * dists.max() / self._n_bins

            f0  = ftable[batch.vocab[n][mask[n]], 0]
            i0  = f0.sum() ** 2

            qr   = qgrid * r_peak
            sinc = torch.where(qr.abs() < 1e-8, torch.ones_like(qr), torch.sin(qr) / qr)
            preds.append(i0 * sinc)

        return torch.stack(preds)
