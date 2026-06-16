import numpy as np
import numpy.typing as npt
from beartype import beartype
from .spherical_funcs import sphHarm, sphBess
from .formfact import FormFactors


class StuhrmannMixin:

    @beartype
    def stuhrmann(
        self,
        ff:   FormFactors,
        lMax: int,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.complex128]]:
        """Compute I(q) via the Stuhrmann decomposition.

        I(q) = 4π · Σ_{l=0}^{lMax} Σ_{m=-l}^{l} |B_lm(q)|²
        where B_lm(q) = Σ_i f_i(q) · j_l(q · r_i) · Y*_lm(θ_i, φ_i)

        Only m ≥ 0 modes are returned. For real-valued structures the negative-m
        modes are redundant: B_{l,-m} = (-1)^m · B*_{lm}. The contribution of
        the -m modes to I(q) is accounted for by doubling the m > 0 terms.

        Output index: k = l*(l+1)//2 + m  for l = 0..lMax, m = 0..l.
        Total modes: (lMax+1)*(lMax+2)//2.

        Parameters
        ----------
        ff : FormFactors
            Precomputed form factors; must contain all elements in mol.elms.
        lMax : int
            Maximum l value.

        Returns
        -------
        I_q : ndarray of float64, shape (Q,)
            Scattering intensity at each q value.
        B_lm : ndarray of complex128, shape ((lMax+1)*(lMax+2)//2, Q)
            Partial wave amplitudes for m ≥ 0 only.

        Raises
        ------
        ValueError
            If lMax is negative, or if mol.elms contains ions not in ff.
        """
        if lMax < 0:
            raise ValueError("stuhrmann: lMax must be non-negative")
        missing = set(self.elms) - set(ff.ions)
        if missing:
            raise ValueError(f"stuhrmann: ions {missing} in molecule not found in FormFactors")

        N_reduced = (lMax + 1) * (lMax + 2) // 2
        B_lm = np.zeros((N_reduced, ff.qvals.size), dtype=np.complex128)

        # Process atoms in chunks to bound peak memory.
        # Peak per chunk: Y~(lMax+1)²×chunk×16B + j~(lMax+1)×Q×chunk×8B.
        # At chunk=5000, lMax=50, Q=51 this is ~312 MB — safe for WSL.
        CHUNK = 5_000
        N = len(self.elms)
        for start in range(0, N, CHUNK):
            sl   = slice(start, min(start + CHUNK, N))
            Y    = sphHarm(lMax, self.theta[sl], self.phi[sl])   # ((lMax+1)(lMax+2)//2, chunk)
            j    = sphBess(ff.qvals, lMax, self.r[sl])           # (lMax+1, Q, chunk)
            f    = np.stack([ff.ff[ion] for ion in self.elms[sl]])  # (chunk, Q)
            Y_conj = Y.conj()                                    # (K, chunk) — once per chunk
            for l in range(lMax + 1):
                W  = f.T * j[l]                                  # (Q, chunk)
                k0 = l * (l + 1) // 2
                Y_l = Y_conj[k0 : k0 + l + 1]                   # (l+1, chunk)
                B_lm[k0 : k0 + l + 1] += Y_l @ W.T              # (l+1, Q)

        # Precompute ±m symmetry weights: m=0 → 1, m>0 → 2.
        weights = np.array([2.0 - (m == 0)
                            for l in range(lMax + 1) for m in range(l + 1)])  # (N_reduced,)
        I_q = (weights[:, None] * np.abs(B_lm) ** 2).sum(axis=0) * 4 * np.pi
        return I_q.astype(np.float64), B_lm