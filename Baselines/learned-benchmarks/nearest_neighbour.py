import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped

from Baselines.baseline import Baseline, autocast_dtype, build_fmag_table
from ScatterNet.batching import Batch


class NNBaseline(Baseline):
    """
    Local-geometry baseline: I(q) ≈ I(0)·sinc(q·r_nn) where r_nn is the mean
    nearest-neighbour distance across all atoms in the molecule.

    Captures the dominant bond-length scale without the full P(r) computation.
    Uses I(0) = (Σ f_i(0))². O(M²) per molecule.
    Beating this baseline proves the model uses multi-scale distance
    information.
    """

    _qgrid: Float[torch.Tensor, "Q"]  # noqa: F821
    _fmag_table: Float[torch.Tensor, "V Q"]  # noqa: F722

    def __init__(
        self,
        qgrid: Float[torch.Tensor, "Q"],  # noqa: F821
        energy: float,
    ) -> None:
        """
        Args:
            qgrid:  q-point grid in Å⁻¹ (Q,)
            energy: X-ray energy in eV for anomalous f1/f2 corrections
        """
        self._qgrid = qgrid
        self._fmag_table = build_fmag_table(qgrid, energy)

    @jaxtyped(typechecker=beartype)
    def __call__(
        self, batch: Batch
    ) -> Float[torch.Tensor, "N Q"]:  # noqa: F722
        """Return I(0)*sinc(q*r_nn) per molecule using mean NN distance."""
        device = batch.coord.device
        mask = batch.padding_mask()  # (N, M)
        qgrid = self._qgrid.to(device)
        ftable = self._fmag_table.to(device)
        # BF16 on Ampere+ for the O(m^2) distance-matrix build below (native
        # tensor cores there; T4 gets None -> stays fp32, see
        # autocast_dtype's docstring).
        dtype = autocast_dtype(device)

        preds: list[Float[torch.Tensor, "Q"]] = []  # noqa: F821

        for n in range(batch.coord.shape[0]):
            coords_n = batch.coord[n][mask[n]]  # (m, 3)
            m = coords_n.shape[0]

            if m < 2:
                preds.append(torch.zeros(qgrid.shape[0], device=device))
                continue

            # subsample to avoid O(M²) OOM on large molecules
            MAX_M = 4096
            vocab_n = batch.vocab[n][mask[n]]
            if m > MAX_M:
                sub = torch.randperm(m, device=device)[:MAX_M]
                coords_n = coords_n[sub]
                vocab_n = vocab_n[sub]
                m = MAX_M

            coords_c = coords_n if dtype is None else coords_n.to(dtype)
            diff = coords_c.unsqueeze(0) - coords_c.unsqueeze(1)  # (m, m, 3)
            dists = diff.norm(dim=-1).float()  # (m, m), back to fp32
            dists.fill_diagonal_(float("inf"))  # exclude self-distance (0)
            r_nn = dists.min(
                dim=1
            ).values.mean()  # mean nearest-neighbour distance
            f0 = ftable[vocab_n, 0]  # |f_i(0)| per atom
            i0 = f0.sum() ** 2  # forward intensity I(0) = (Σ f_i(0))²
            qr = qgrid * r_nn
            sinc = torch.where(
                qr.abs() < 1e-8, torch.ones_like(qr), torch.sin(qr) / qr
            )
            preds.append(i0 * sinc)

        return torch.stack(preds)
