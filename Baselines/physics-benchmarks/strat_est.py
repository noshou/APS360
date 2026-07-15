import torch

from jaxtyping           import Float, jaxtyped
from beartype            import beartype
from ScatterNet.batching import Batch
from Baselines.baseline  import Baseline, build_fmag_table

class StratEstBaseline(Baseline):
    
    """
    Stochastic stratEst estimator of the Debye sum (port of SaxsEst stratEst).

    Stratified importance-sampling estimator with Hansen-Hurwitz correction and
    "Heavy-Rounded Mean-Weighted Allocation": atoms split into heavy/light strata
    by |f(q0)|, sampled by within-stratum type frequency, and the biased sub-sum
    corrected by inverse selection probabilities.

    The full all-pairs Debye sum is estimated from a random subset
    of atoms rather than every pair, so the curve depends on which atoms were
    drawn (Monte Carlo sampling). The draw is seeded from each molecule's contents, 
    so the result is still reproducible run to run.
    """

    _MAX_M: int   = 6046    # max molecule size in the dataset (top DEFAULT_BUCKETS ceiling);
                            #   caps sampled atoms to bound O(S^2) memory, no real subsampling
    _TOL:   float = 1.0e-10

    _qgrid:      Float[torch.Tensor, "Q"]
    _fmag_table: Float[torch.Tensor, "V Q"]
    _A:          float

    def __init__(
        self,
        qgrid:  Float[torch.Tensor, "Q"],
        energy: float,
        a:      float = 0.6,
    ) -> None:
        """qgrid: (Q,) q-points [1/A]. energy: X-ray energy [eV]. a: sampling fraction."""
        self._qgrid      = qgrid
        self._fmag_table = build_fmag_table(qgrid, energy)
        self._A          = a

    @jaxtyped(typechecker=beartype)
    def __call__(self, batch: Batch) -> Float[torch.Tensor, "N Q"]:
        """Predict I(q) per molecule, unnormalized and non-negative, shape (N, Q)."""
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
                sinc = torch.where(qd.abs() < 1e-8, torch.ones_like(qd), torch.sin(qd) / qd)
                
                # off-diagonal (i<j, doubled): 2 f_i f_j sinc / (hh_i hh_j)
                w    = (fq * inv_hh).unsqueeze(0) * (fq * inv_hh).unsqueeze(1)   # (S,S)
                pair = (w * sinc).sum()
                off  = pair - diag                          # remove i==j (sinc=1) diagonal
                est[qi] = diag + off

            preds.append(est.clamp_min(0.0))

        return torch.stack(preds)
