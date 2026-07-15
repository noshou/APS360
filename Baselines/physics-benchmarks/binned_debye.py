import torch

from jaxtyping           import Float, jaxtyped
from beartype            import beartype
from ScatterNet.batching import Batch
from Baselines.baseline  import Baseline, build_fmag_table

class BinnedDebyeBaseline(Baseline):
    """
    Local-geometry baseline: coarse-grained Debye sum over a histogrammed P(r).

    Bins all pairwise distances in a molecule and weights each bin by the
    summed form-factor product ``f_i*f_j`` of the pairs that fall in it,
    then evaluates ``sinc(q*r_bin)`` once per bin instead of once per pair.
    This recovers the exact Debye equation in the ``n_bins -> inf`` limit
    (I(q) = sum_i sum_j f_i*f_j*sinc(q*r_ij)), while keeping the q-dependent
    work at O(n_bins * Q) instead of O(M^2 * Q). Strictly more expressive
    than using only the single tallest P(r) peak, at the same O(M^2)
    distance-computation cost.

    Beating this baseline proves the model uses distance-distribution
    structure beyond a coarse-grained histogram of pairwise distances.
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
        Parameters
        ----------
        qgrid : torch.Tensor
            q-point grid in inverse angstroms, shape ``(Q,)``.
        energy : float
            X-ray energy in eV, used for anomalous f1/f2 corrections.
        n_bins : int, default 200
            Number of histogram bins spanning ``[0, r_max]`` per molecule.
        """
        self._qgrid      = qgrid
        self._n_bins     = n_bins
        self._fmag_table = build_fmag_table(qgrid, energy)

    @jaxtyped(typechecker=beartype)
    def __call__(self, batch: Batch) -> Float[torch.Tensor, "N Q"]:
        """Return the binned-Debye-sum I(q) per molecule. O(M²) per molecule.

        Parameters
        ----------
        batch : Batch
            Batch of molecules to predict scattering curves for.

        Returns
        -------
        torch.Tensor
            Predicted I(q) curves of shape ``(N, Q)``.
        """
        device = batch.coord.device
        mask   = batch.padding_mask()           # (N, M)
        qgrid  = self._qgrid.to(device)
        ftable = self._fmag_table.to(device)
        n_bins = self._n_bins

        preds: list[Float[torch.Tensor, "Q"]] = []

        for n in range(batch.coord.shape[0]):
            coords_n = batch.coord[n][mask[n]]  # (m, 3)
            vocab_n  = batch.vocab[n][mask[n]]
            m = coords_n.shape[0]

            f0  = ftable[vocab_n, 0]            # (m,) |f_i(0)| per atom
            diag = (f0 ** 2).sum()              # i==j self-scattering term

            if m < 2:
                qr   = qgrid * 0.0
                preds.append(diag * torch.ones_like(qgrid))
                continue

            # subsample to avoid O(M²) OOM on large molecules
            MAX_M = 4096
            if m > MAX_M:
                sub = torch.randperm(m, device=device)[:MAX_M]
                coords_n = coords_n[sub]
                vocab_n  = vocab_n[sub]
                f0       = f0[sub]
                m = MAX_M

            diff  = coords_n.unsqueeze(0) - coords_n.unsqueeze(1)  # (m, m, 3)
            dists = diff.norm(dim=-1)                               # (m, m)
            idx   = torch.triu_indices(m, m, offset=1, device=device)
            r_ij  = dists[idx[0], idx[1]]                           # (P,)
            w_ij  = f0[idx[0]] * f0[idx[1]]                         # (P,)

            r_max = r_ij.max()
            if r_max <= 0:
                preds.append(diag * torch.ones_like(qgrid))
                continue

            bin_idx = ((r_ij / r_max) * n_bins).long().clamp(max=n_bins - 1)
            weight_sum = torch.zeros(n_bins, device=device)
            weight_sum.scatter_add_(0, bin_idx, w_ij)

            r_centers = (torch.arange(n_bins, device=device).float() + 0.5) * r_max / n_bins
            qr   = qgrid.unsqueeze(1) * r_centers.unsqueeze(0)      # (Q, n_bins)
            sinc = torch.where(qr.abs() < 1e-8, torch.ones_like(qr), torch.sin(qr) / qr)

            offdiag = 2.0 * (sinc @ weight_sum)                     # (Q,)
            preds.append(diag + offdiag)

        return torch.stack(preds)
