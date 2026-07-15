import torch

from jaxtyping           import Float, jaxtyped
from beartype            import beartype
from ScatterNet.batching import Batch
from Baselines.baseline  import Baseline, build_fmag_table

class SaxsEstBaseline(Baseline):
    """
    Analytical SaxsEst estimator (pure-Python port of the SaxsEst Fortran project).

    Ports two estimators of the Debye scattering sum from
    https://github.com/noshou/SaxsEst (upstream status: "not fully validated").
    Which one ``__call__`` returns is selected by the ``method`` argument, so the
    two are registered as separate baselines ("SAXS propoEst" / "SAXS stratEst")
    and compared side by side:

    - ``propoEst`` (deterministic, ``method="propo"``), run with epsilon
      ``e = 0.380``.
    - ``stratEst`` (stochastic, Hansen-Hurwitz corrected importance sampling,
      ``method="strat"``) with the "Heavy-Rounded Mean-Weighted Allocation"
      strategy and sampling fraction ``a = 0.6``. Seeded per molecule so its
      curve is reproducible run to run despite being Monte Carlo.

    Both computations are always available as :meth:`propo_est` and
    :meth:`strat_est` regardless of the selected ``method``.

    Both estimators approximate the exact Debye equation
    ``I(q) = sum_i sum_j f_i(q) conj(f_j(q)) sinc(q r_ij)``. propoEst replaces the
    per-pair conjugate form-factor product by a single q-dependent scalar weight
    estimate ``wEst(q)`` derived from the atom-type frequency table, then still
    sums ``sinc(q r_ij)`` over all pairs. stratEst instead draws a stratified
    subset of atoms and corrects the biased sub-sum by the inverse selection
    probabilities.

    Form-factor note: SaxsEst uses complex form factors ``f = (f0 + f1) + i f2``,
    but the only per-type information the Batch and ``build_fmag_table`` expose is
    the magnitude ``|f(q)| = hypot(f0 + f1, f2)`` (exactly the SaxsEst form-factor
    magnitude). We therefore treat ``f`` as real (imaginary part dropped). SaxsEst's
    own debyeEst notes the imaginary contributions cancel by symmetry over all
    pairs, so a magnitude-only evaluation is a faithful approximation.

    Scale convention: SaxsEst normalizes its estimators (dividing by ``N^2`` or
    ``N * s``). The analog baseline ``BinnedDebyeBaseline`` and the true
    ``batch.iqval`` are on the UNNORMALIZED Debye scale, so the SaxsEst intensity
    is multiplied back by ``N^2`` (a per-molecule constant that leaves the curve
    shape unchanged). Predictions are clamped to be non-negative.

    Beating this baseline proves the model captures scattering structure beyond a
    frequency-weight approximation of the Debye sum.
    """

    _MAX_M: int   = 4096    # cap atoms per molecule to bound O(M^2) memory
    _TOL:   float = 1.0e-10

    _qgrid:      Float[torch.Tensor, "Q"]
    _fmag_table: Float[torch.Tensor, "V Q"]
    _method:     str
    _E:          float
    _A:          float

    def __init__(
        self,
        qgrid:  Float[torch.Tensor, "Q"],
        energy: float,
        method: str   = "propo",
        e:      float = 0.380,
        a:      float = 0.6,
    ) -> None:
        """
        Parameters
        ----------
        qgrid : torch.Tensor
            q-point grid in inverse angstroms, shape ``(Q,)``.
        energy : float
            X-ray energy in eV, used for anomalous f1/f2 corrections when
            building the atomic form-factor magnitude table.
        method : str
            Which estimator ``__call__`` returns: ``"propo"`` (default,
            deterministic propoEst) or ``"strat"`` (stochastic stratEst).
        e : float
            propoEst epsilon accuracy parameter (default 0.380).
        a : float
            stratEst Heavy-Rounded Mean-Weighted Allocation sampling fraction
            (default 0.6).
        """
        if method not in ("propo", "strat"):
            raise ValueError(f"method must be 'propo' or 'strat', got {method!r}")
        self._qgrid      = qgrid
        self._fmag_table = build_fmag_table(qgrid, energy)
        self._method     = method
        self._E          = e
        self._A          = a

    @jaxtyped(typechecker=beartype)
    def __call__(self, batch: Batch) -> Float[torch.Tensor, "N Q"]:
        """Predict I(q) per molecule using the estimator selected at construction.

        Dispatches to :meth:`propo_est` (``method="propo"``) or :meth:`strat_est`
        (``method="strat"``). Output is unnormalized, non-negative, shape ``(N, Q)``.
        """
        return self.propo_est(batch) if self._method == "propo" else self.strat_est(batch)

    # ------------------------------------------------------------------ #
    # propoEst (deterministic)
    # ------------------------------------------------------------------ #
    @jaxtyped(typechecker=beartype)
    def propo_est(self, batch: Batch) -> Float[torch.Tensor, "N Q"]:
        """Return the propoEst(e) I(q) per molecule, unnormalized and non-negative.

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
        mask   = batch.padding_mask()            # (N, M)
        qgrid  = self._qgrid.to(device)          # (Q,)
        ftable = self._fmag_table.to(device)     # (V, Q)
        eps    = 1.0e-12

        preds: list[Float[torch.Tensor, "Q"]] = []

        for n in range(batch.coord.shape[0]):
            coords_n = batch.coord[n][mask[n]]   # (m, 3)
            vocab_n  = batch.vocab[n][mask[n]]   # (m,)
            m = coords_n.shape[0]

            if m == 0:
                preds.append(torch.zeros_like(qgrid))
                continue

            fatom = ftable[vocab_n]              # (m, Q) real magnitude |f_i(q)|

            # --- propEstCalc: frequency-weight estimate wEst(q) ---------------
            uniq, counts = torch.unique(vocab_n, return_counts=True)   # (U,), (U,)
            w_u   = ftable[uniq]                 # (U, Q) per-type |f(q)|
            c_u   = (counts * (counts - 1) // 2).to(qgrid.dtype)       # C(freq,2), (U,)

            a     = float(m)
            s     = int(torch.ceil(torch.sqrt(torch.tensor(24.0 * a)) / self._E).item()) + 1
            s_choose_2 = float(s * (s - 1) // 2)

            tot   = (c_u.unsqueeze(1) / (w_u + eps)).sum(dim=0)        # (Q,)
            wEst  = torch.where(tot.abs() < eps,
                                torch.zeros_like(tot),
                                s_choose_2 / tot)                     # (Q,) real

            diag  = (fatom ** 2).sum(dim=0)      # (Q,)  sum_i |f_i(q)|^2

            if m < 2:
                # no pairs; estimate is the diagonal, rescaled to unnormalized scale
                intensity = diag * (m / s)
                preds.append(intensity.clamp_min(0.0))
                continue

            # subsample to bound O(m^2) memory (mirrors pair_peak)
            if m > self._MAX_M:
                sub      = torch.randperm(m, device=device)[: self._MAX_M]
                coords_n = coords_n[sub]
                fatom    = fatom[sub]
                m        = self._MAX_M

            diff = coords_n.unsqueeze(0) - coords_n.unsqueeze(1)       # (m, m, 3)
            dist = diff.norm(dim=-1)                                   # (m, m)

            # est(q) = sum_i |f_i|^2 + wEst(q) * sum_i f_i(q) * sum_{j!=i} sinc(q r_ij)
            # (f_i and wEst real, so Re(f_i * wEst) = f_i * wEst)
            est = torch.empty_like(qgrid)
            for qi in range(qgrid.shape[0]):
                qd   = qgrid[qi] * dist                                # (m, m)
                sinc = torch.where(qd.abs() < 1e-8,
                                   torch.ones_like(qd),
                                   torch.sin(qd) / qd)
                # sum over j != i  == full row sum minus the j==i term (sinc(0)=1)
                s_i  = sinc.sum(dim=1) - 1.0                           # (m,)
                cross = wEst[qi] * (fatom[:, qi] * s_i).sum()
                est[qi] = diag[qi] + cross

            # unnormalized (I_fortran * N^2 = est * N / s), non-negative
            intensity = (est * (m / s)).clamp_min(0.0)
            preds.append(intensity)

        return torch.stack(preds)

    # ------------------------------------------------------------------ #
    # stratEst (stochastic) -- Heavy-Rounded Mean-Weighted Allocation
    # ------------------------------------------------------------------ #
    @jaxtyped(typechecker=beartype)
    def strat_est(self, batch: Batch) -> Float[torch.Tensor, "N Q"]:
        """Return the stratEst(a, Heavy-Rounded Mean-Weighted) I(q) per molecule.

        Stratified importance-sampling estimator with Hansen-Hurwitz correction,
        ported faithfully from SaxsEst. Sampling is seeded from each molecule's
        content so the result is reproducible run to run despite being Monte
        Carlo. Output is unnormalized and non-negative, on the same scale as
        :meth:`propo_est`.
        """
        device = batch.coord.device
        mask   = batch.padding_mask()
        qgrid  = self._qgrid.to(device)
        ftable = self._fmag_table.to(device)
        tol    = self._TOL

        preds: list[Float[torch.Tensor, "Q"]] = []

        for n in range(batch.coord.shape[0]):
            coords_n = batch.coord[n][mask[n]]   # (m, 3)
            vocab_n  = batch.vocab[n][mask[n]]   # (m,)
            m = coords_n.shape[0]

            if m == 0:
                preds.append(torch.zeros_like(qgrid))
                continue
            if m < 2:
                fatom = ftable[vocab_n]          # (1, Q)
                preds.append((fatom ** 2).sum(dim=0).clamp_min(0.0))
                continue

            # deterministic per-molecule generator (reproducible sampling)
            seed = (int(vocab_n.sum().item()) * 1000003 + m) & 0x7FFFFFFF
            gen  = torch.Generator(device="cpu")
            gen.manual_seed(seed)

            f0all = ftable[vocab_n][:, 0]         # (m,) |f_i(q0)|
            mean0 = f0all.mean()                  # (1/N) sum_i |f_i(q0)|  (per-atom)

            heavy_atom = f0all - mean0 > tol      # bool (m,)
            light_atom = ~heavy_atom

            # per-stratum mean of |f(q0)| (population weighted == per-atom mean)
            heavy_mean = f0all[heavy_atom].mean() if heavy_atom.any() else torch.zeros((), device=device)
            light_mean = f0all[light_atom].mean() if light_atom.any() else torch.zeros((), device=device)

            total_samples = int(torch.ceil(torch.tensor(self._A * m)).item())
            sum_mean      = float((heavy_mean + light_mean).item())
            if sum_mean <= 0.0:
                heavy_samples = 0
            else:
                heavy_samples = int(torch.ceil(heavy_mean / (heavy_mean + light_mean) * total_samples).item())
            heavy_samples = max(0, min(heavy_samples, total_samples))
            light_samples = total_samples - heavy_samples

            samp_idx, samp_hh = [], []
            for atom_mask, n_draw in ((heavy_atom, heavy_samples), (light_atom, light_samples)):
                if n_draw <= 0 or not bool(atom_mask.any()):
                    continue
                idx_pool = torch.nonzero(atom_mask, as_tuple=False).squeeze(1)   # atom indices
                vocab_p  = vocab_n[idx_pool]
                # within-stratum type frequencies -> selection probability
                types, inv, cnts = torch.unique(vocab_p, return_inverse=True, return_counts=True)
                pop   = int(cnts.sum().item())
                probs = cnts.to(torch.float64) / float(pop)            # P(select type)
                # inverse-CDF draw of a type, then a uniform atom of that type
                cum   = torch.cumsum(probs, dim=0)
                u     = torch.rand(n_draw, generator=gen, dtype=torch.float64)
                t_sel = torch.searchsorted(cum, u).clamp_max(len(types) - 1)     # (n_draw,)
                for t in t_sel.tolist():
                    members = idx_pool[inv == t]
                    pick    = int(torch.randint(len(members), (1,), generator=gen).item())
                    samp_idx.append(int(members[pick].item()))
                    samp_hh.append(float(probs[t].item()))

            if len(samp_idx) < 1:
                preds.append(torch.zeros_like(qgrid))
                continue
            if len(samp_idx) > self._MAX_M:
                samp_idx = samp_idx[: self._MAX_M]
                samp_hh  = samp_hh[: self._MAX_M]

            sel   = torch.tensor(samp_idx, device=device, dtype=torch.long)
            hh    = torch.tensor(samp_hh, device=device, dtype=qgrid.dtype)      # (S,)
            fsel  = ftable[vocab_n[sel]]         # (S, Q)
            csel  = coords_n[sel]                # (S, 3)

            diff  = csel.unsqueeze(0) - csel.unsqueeze(1)   # (S, S, 3)
            dist  = diff.norm(dim=-1)                        # (S, S)
            inv_hh = 1.0 / hh                                # (S,)

            est = torch.empty_like(qgrid)
            for qi in range(qgrid.shape[0]):
                fq   = fsel[:, qi]                           # (S,)
                # diagonal: sum_i |f_i|^2 / hh_i^2
                diag = ((fq ** 2) * (inv_hh ** 2)).sum()
                qd   = qgrid[qi] * dist
                sinc = torch.where(qd.abs() < 1e-8,
                                   torch.ones_like(qd),
                                   torch.sin(qd) / qd)
                # off-diagonal (i<j, doubled): 2 f_i f_j sinc / (hh_i hh_j)
                w    = (fq * inv_hh).unsqueeze(0) * (fq * inv_hh).unsqueeze(1)   # (S,S)
                pair = (w * sinc).sum()
                off  = pair - diag                          # remove i==j (sinc=1) diagonal
                est[qi] = diag + off

            preds.append(est.clamp_min(0.0))

        return torch.stack(preds)
