import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped

from Baselines.baseline import (
    Baseline,
    autocast_dtype,
    build_fmag_table,
    q_chunks,
)
from ScatterNet.batching import Batch


class StratEstBaseline(Baseline):
    """
    Stochastic stratEst estimator of the Debye sum (port of SaxsEst stratEst).

    Stratified Hansen-Hurwitz estimator with "Heavy-Rounded Mean-Weighted
    Allocation": atoms are split into heavy/light strata by |f(q0)|, the
    sample budget ``ceil(a*m)`` is split between the strata in proportion
    to their mean |f(q0)|, and each stratum is then sampled uniformly
    with replacement.

    Only the off-diagonal (i != j) part of the Debye sum is estimated
    from the sample; the diagonal ``sum_i |f_i(q)|^2`` is computed
    exactly over all m atoms, since it is O(m) and dominates I(q) at
    high q where the off-diagonal terms cancel. This mirrors
    ``PropoEstBaseline``.

    Hansen-Hurwitz weights
    ----------------------
    Each draw in stratum h is uniform over that stratum's ``pop_h``
    atoms, so the per-draw selection probability is ``p = 1/pop_h`` and
    the HH weight of a sampled atom is ``w_h = pop_h / n_h``. Sampling
    is WITH replacement, so a sample pair ``(k, l)`` that happens to
    land on the same atom is a diagonal term, not an off-diagonal one:
    those coincident pairs are masked out and the surviving
    within-stratum pairs rescaled by ``n_h / (n_h - 1)``, which makes
    the off-diagonal estimate unbiased. Cross-stratum pairs need no
    rescaling.

    The full all-pairs Debye sum is estimated from a random subset of
    atoms rather than every pair, so the curve depends on which atoms
    were drawn (Monte Carlo sampling). The draw is seeded from each
    molecule's contents, so the result is still reproducible run to run.
    """

    _MAX_M: int = (
        6046  # max molecule size in the dataset (top DEFAULT_BUCKETS ceiling);
    )
    #   caps sampled atoms to bound O(S^2) memory, no real subsampling
    _TOL: float = 1.0e-10

    _qgrid: Float[torch.Tensor, "Q"]  # noqa: F722,F821
    _fmag_table: Float[torch.Tensor, "V Q"]  # noqa: F722
    _A: float

    def __init__(
        self,
        qgrid: Float[torch.Tensor, "Q"],  # noqa: F722,F821
        energy: float,
        a: float = 0.6,
    ) -> None:
        """qgrid: (Q,) q-points [1/A]. energy: X-ray energy [eV].

        a: sampling fraction.
        """
        self._qgrid = qgrid
        self._fmag_table = build_fmag_table(qgrid, energy)
        self._A = a

    @jaxtyped(typechecker=beartype)
    def __call__(self, batch: Batch) -> Float[torch.Tensor, "N Q"]:  # noqa: F722
        """Predict I(q) per molecule, unnormalized non-negative.

        Shape (N, Q).
        """
        device = batch.coord.device
        mask = batch.padding_mask()
        qgrid = self._qgrid.to(device)
        ftable = self._fmag_table.to(device)
        tol = self._TOL
        # BF16 on Ampere+ for the bmm below (native tensor cores there; T4
        # gets None -> stays fp32, see autocast_dtype's docstring).
        dtype = autocast_dtype(device)

        preds: list[Float[torch.Tensor, "Q"]] = []  # noqa: F722,F821

        for n in range(batch.coord.shape[0]):
            coords_n = batch.coord[n][mask[n]]  # (m, 3)
            vocab_n = batch.vocab[n][mask[n]]  # (m,)
            m = coords_n.shape[0]

            if m == 0:
                preds.append(torch.zeros_like(qgrid))
                continue
            if m < 2:
                fatom = ftable[vocab_n]  # (1, Q)
                preds.append((fatom**2).sum(dim=0).clamp_min(0.0))
                continue

            # deterministic per-molecule generator (reproducible sampling)
            seed = (int(vocab_n.sum().item()) * 1000003 + m) & 0x7FFFFFFF
            gen = torch.Generator(device="cpu")
            gen.manual_seed(seed)

            f0all = ftable[vocab_n][:, 0]  # (m,) |f_i(q0)|
            mean0 = f0all.mean()  # (1/N) sum_i |f_i(q0)|  (per-atom)

            heavy_atom = f0all - mean0 > tol  # bool (m,)
            light_atom = ~heavy_atom

            # per-stratum mean of |f(q0)| (pop weighted == per-atom mean)
            heavy_mean = (
                f0all[heavy_atom].mean()
                if heavy_atom.any()
                else torch.zeros((), device=device)
            )
            light_mean = (
                f0all[light_atom].mean()
                if light_atom.any()
                else torch.zeros((), device=device)
            )

            # cap the budget (not the drawn sample) so the HH weights
            # below stay consistent with the number of draws actually made
            total_samples = min(
                int(torch.ceil(torch.tensor(self._A * m)).item()), self._MAX_M
            )
            sum_mean = float((heavy_mean + light_mean).item())
            if sum_mean <= 0.0:
                heavy_samples = 0
            else:
                heavy_samples = int(
                    torch.ceil(
                        heavy_mean / (heavy_mean + light_mean) * total_samples
                    ).item()
                )
            heavy_samples = max(0, min(heavy_samples, total_samples))
            light_samples = total_samples - heavy_samples

            # exact diagonal over ALL m atoms: sum_i |f_i(q)|^2 (O(m), no
            # sampling)
            diag = (ftable[vocab_n] ** 2).sum(dim=0)  # (Q,)

            # A stratum drawn fewer than twice forms no within-stratum
            # pair, so its entire off-diagonal would silently drop out
            # of the estimate. On small molecules ceil(a*m) is too small
            # a budget to keep both strata above that floor, so degrade
            # to a single unstratified stratum (same weights, same
            # estimator) rather than lose one stratum's pairs altogether.
            strata = (
                (heavy_atom, heavy_samples),
                (light_atom, light_samples),
            )
            if (
                bool(heavy_atom.any())
                and bool(light_atom.any())
                and (heavy_samples < 2 or light_samples < 2)
            ):
                strata = (
                    (torch.ones_like(heavy_atom), total_samples),
                    (torch.zeros_like(light_atom), 0),
                )

            samp_idx = []
            w_h = [0.0, 0.0]  # HH weight pop_h / n_h, per stratum
            c_h = [0.0, 0.0]  # within-stratum pair rescale n_h / (n_h - 1)
            for sid, (atom_mask, n_draw) in enumerate(strata):
                if n_draw <= 0 or not bool(atom_mask.any()):
                    continue
                idx_pool = torch.nonzero(atom_mask, as_tuple=False).squeeze(
                    1
                )  # atom indices
                pop = idx_pool.shape[0]

                # Uniform draw with replacement over the stratum pool. The
                # old type-frequency inverse-CDF draw was a no-op: picking
                # a type with P = cnts[t]/pop and then a uniform atom
                # among that type's cnts[t] members gives P(atom) =
                # (cnts[t]/pop) * (1/cnts[t]) = 1/pop, so cnts[t] cancels
                # and every atom is equally likely regardless of type.
                # gen is a CPU generator (for reproducibility), so the
                # draw is made on CPU and moved onto the batch's device
                # to index there.
                pick = torch.randint(
                    pop, (n_draw,), generator=gen
                )  # (n_draw,)
                samp_idx.append(idx_pool[pick.to(device)])

                w_h[sid] = pop / n_draw  # p = 1/pop => w = pop/n
                # n_draw == 1 forms no distinct sample pair, so that
                # stratum's own off-diagonal is not estimable (only
                # reachable for tiny m); the (hh - self) term it scales
                # is identically 0 there anyway.
                c_h[sid] = n_draw / (n_draw - 1) if n_draw > 1 else 0.0

            if not samp_idx:
                preds.append(diag.clamp_min(0.0))
                continue

            # Collapse repeat draws. Sampling is with replacement, so the
            # n_h draws land on only ~(1 - e^-a) * pop_h distinct atoms;
            # folding each atom's multiplicity r_i into its amplitude
            # (G_i = r_i * g_i) is algebraically exact and shrinks the
            # dominant (Qc, U, U) kernel to ~0.56x its (S, S) size at
            # a = 0.6. Strata partition the atoms, so no unique() bucket
            # ever mixes draws from different strata.
            sel, rep = torch.unique(
                torch.cat(samp_idx), return_counts=True
            )  # (U,), (U,)
            is_h = strata[0][0][sel]  # (U,) stratum-0 member?
            wvec = torch.tensor(w_h, device=device, dtype=qgrid.dtype)  # (2,)
            cvec = torch.tensor(c_h, device=device, dtype=qgrid.dtype)  # (2,)
            w_u = torch.where(is_h, wvec[0], wvec[1])  # (U,)
            csel = coords_n[sel]  # (U, 3)

            diff = csel.unsqueeze(0) - csel.unsqueeze(1)  # (U, U, 3)
            dist = diff.norm(dim=-1)  # (U, U)

            # G_i(q) = r_i * w_i * |f_i(q)|: each q's off-diagonal estimate
            # is a quadratic form in G with the sinc kernel, split into
            # heavy/light blocks.
            G = ftable[vocab_n[sel]] * (rep.to(qgrid.dtype) * w_u).unsqueeze(
                1
            )  # (U, Q)
            U = dist.shape[0]
            m_h = is_h.to(qgrid.dtype)  # (U,) heavy selector
            m_l = 1.0 - m_h  # (U,) light selector

            est = torch.empty_like(qgrid)
            dist_c = dist if dtype is None else dist.to(dtype)
            G_c = G if dtype is None else G.to(dtype)
            m_h_c = m_h if dtype is None else m_h.to(dtype)
            m_l_c = m_l if dtype is None else m_l.to(dtype)
            # vectorize over q in memory-bounded chunks so the (Qc, U, U) sinc
            # intermediate never exceeds the old per-q (U, U) footprint by much
            for a, b in q_chunks(qgrid.shape[0], U):
                gc = G_c[:, a:b].transpose(0, 1)  # (Qc, U)
                q_ab = qgrid[a:b] if dtype is None else qgrid[a:b].to(dtype)
                qd = q_ab.view(-1, 1, 1) * dist_c.unsqueeze(0)  # (Qc, U, U)
                sinc = torch.where(
                    qd.abs() < 1e-8, torch.ones_like(qd), torch.sin(qd) / qd
                )

                # masking G per stratum lets one bmm carry both blocks, so the
                # (Qc, U, U) kernel needs no elementwise pair-weight multiply
                gh = gc * m_h_c  # (Qc, U)
                gl = gc * m_l_c  # (Qc, U)
                out = torch.bmm(
                    sinc, torch.stack((gh, gl), dim=2)
                )  # (Qc, U, 2)
                # back to fp32 for the accuracy-sensitive rescale/accumulate
                # below -- the reduced-precision block ends at the bmm.
                out = out.float()
                gh = gh.float()
                gl = gl.float()

                hh = (gh * out[..., 0]).sum(
                    dim=1
                )  # (Qc,) heavy-heavy, incl. i==j
                ll = (gl * out[..., 1]).sum(
                    dim=1
                )  # (Qc,) light-light, incl. i==j
                hl = (gh * out[..., 1]).sum(
                    dim=1
                )  # (Qc,) heavy-light (i!=j always)

                # drop each block's own i==j term (the diagonal is exact,
                # above), then rescale the within-stratum pairs;
                # cross-stratum pairs are unbiased
                off = (
                    cvec[0] * (hh - (gh**2).sum(dim=1))
                    + cvec[1] * (ll - (gl**2).sum(dim=1))
                    + 2.0 * hl
                )  # (Qc,)
                est[a:b] = diag[a:b] + off  # exact diag + est off-diag

            preds.append(est.clamp_min(0.0))

        return torch.stack(preds)
