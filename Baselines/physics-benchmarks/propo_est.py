import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped

from Baselines.baseline import Baseline, build_fmag_table, q_chunks
from ScatterNet.batching import Batch


class PropoEstBaseline(Baseline):
    """
    Deterministic propoEst estimator of the Debye sum (port of SaxsEst
    propoEst).

    Replaces the per-pair form-factor product in the exact Debye equation
    with a single q-dependent scalar weight ``wEst(q)`` from the atom-type
    frequency table, then sums ``sinc(q r_ij)`` over all pairs. Form
    factors are treated as real magnitudes ``|f(q)|`` (imaginary part
    cancels by pair symmetry).

    Output is on the unnormalized Debye scale (rescaled by ``N^2`` to match
    ``batch.iqval`` and ``BinnedDebyeBaseline``) and clamped non-negative.
    """

    _MAX_M: int = (
        6046  # max molecule size in the dataset (top DEFAULT_BUCKETS ceiling);
    )
    # caps atoms per molecule to bound O(M^2) memory, no real subsampling

    _qgrid: Float[torch.Tensor, "Q"]  # noqa: F722,F821
    _fmag_table: Float[torch.Tensor, "V Q"]  # noqa: F722
    _E: float

    def __init__(
        self,
        qgrid: Float[torch.Tensor, "Q"],  # noqa: F722,F821
        energy: float,
        e: float = 0.380,
    ) -> None:
        """qgrid: (Q,) q-points [1/A]. energy: X-ray energy [eV].

        e: propoEst epsilon.
        """
        self._qgrid = qgrid
        self._fmag_table = build_fmag_table(qgrid, energy)
        self._E = e

    @jaxtyped(typechecker=beartype)
    def __call__(self, batch: Batch) -> Float[torch.Tensor, "N Q"]:  # noqa: F722
        """Predict I(q) per molecule, unnormalized non-negative.

        Shape (N, Q).
        """
        device = batch.coord.device
        mask = batch.padding_mask()  # (N, M)
        qgrid = self._qgrid.to(device)  # (Q,)
        ftable = self._fmag_table.to(device)  # (V, Q)
        eps = 1.0e-12

        preds: list[Float[torch.Tensor, "Q"]] = []  # noqa: F722,F821

        for n in range(batch.coord.shape[0]):
            coords_n = batch.coord[n][mask[n]]  # (m, 3)
            vocab_n = batch.vocab[n][mask[n]]  # (m,)
            m = coords_n.shape[0]

            if m == 0:
                preds.append(torch.zeros_like(qgrid))
                continue

            fatom = ftable[vocab_n]  # (m, Q) real magnitude |f_i(q)|

            # --- propEstCalc: frequency-weight estimate wEst(q) -----------
            uniq, counts = torch.unique(
                vocab_n, return_counts=True
            )  # (U,), (U,)
            w_u = ftable[uniq]  # (U, Q) per-type |f(q)|
            c_u = (counts * (counts - 1) // 2).to(
                qgrid.dtype
            )  # C(freq,2), (U,)

            a = float(m)
            s = (
                int(
                    torch.ceil(
                        torch.sqrt(torch.tensor(24.0 * a)) / self._E
                    ).item()
                )
                + 1
            )
            s_choose_2 = float(s * (s - 1) // 2)

            tot = (c_u.unsqueeze(1) / (w_u + eps)).sum(dim=0)  # (Q,)
            wEst = torch.where(
                tot.abs() < eps, torch.zeros_like(tot), s_choose_2 / tot
            )  # (Q,) real

            diag = (fatom**2).sum(dim=0)  # (Q,)  sum_i |f_i(q)|^2

            if m < 2:
                # no pairs; estimate is the diagonal, rescaled to
                # unnormalized scale
                intensity = diag * (m / s)
                preds.append(intensity.clamp_min(0.0))
                continue

            # subsample to bound O(m^2) memory (mirrors binned_debye)
            if m > self._MAX_M:
                sub = torch.randperm(m, device=device)[: self._MAX_M]
                coords_n = coords_n[sub]
                fatom = fatom[sub]
                m = self._MAX_M

            diff = coords_n.unsqueeze(0) - coords_n.unsqueeze(1)  # (m, m, 3)
            dist = diff.norm(dim=-1)  # (m, m)

            # est(q) = sum_i |f_i|^2
            #          + wEst(q) * sum_i f_i(q) * sum_{j!=i} sinc(q r_ij)
            # (f_i and wEst real, so Re(f_i * wEst) = f_i * wEst)
            # Vectorized over q in memory-bounded chunks so the (Qc, m, m)
            # intermediates never exceed the old per-q (m, m) footprint by
            # much.
            est = torch.empty_like(qgrid)
            for a, b in q_chunks(qgrid.shape[0], m):
                qd = qgrid[a:b].view(-1, 1, 1) * dist.unsqueeze(
                    0
                )  # (Qc, m, m)
                sinc = torch.where(
                    qd.abs() < 1e-8, torch.ones_like(qd), torch.sin(qd) / qd
                )
                # sum over j != i == full row sum minus j==i term (sinc(0)=1)
                s_i = sinc.sum(dim=2) - 1.0  # (Qc, m)
                fq = fatom[:, a:b].transpose(0, 1)  # (Qc, m)
                cross = wEst[a:b] * (fq * s_i).sum(dim=1)  # (Qc,)
                est[a:b] = diag[a:b] + cross

            # unnormalized (I_fortran * N^2 = est * N / s), non-negative
            intensity = (est * (m / s)).clamp_min(0.0)
            preds.append(intensity)

        return torch.stack(preds)
