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
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.complex64]]:
        """Compute I(q) via the Stuhrmann decomposition.

        I(q) = (4π)² · Σ_{l=0}^{lMax} Σ_{m=-l}^{l} |B_lm(q)|²
        where B_lm(q) = Σ_i f_i(q) · j_l(q · r_i) · Y_lm(θ_i, φ_i)

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
        B_lm : ndarray of complex64, shape ((lMax+1)*(lMax+2)//2, Q)
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
        I_q  = np.zeros_like(ff.qvals, dtype=np.float64)
        B_lm = np.zeros((N_reduced, ff.qvals.size), dtype=np.complex64)

        Y = sphHarm(lMax, self.theta, self.phi)
        j = sphBess(ff.qvals, lMax, self.r)
        f = np.stack([ff.ff[ion] for ion in self.elms])

        for l in range(lMax + 1):
            j_l = j[l]
            for m in range(0, l + 1):
                k_reduced = l * (l + 1) // 2 + m
                k_full    = l * l + l + m
                Y_lm = Y[k_full].conj()
                B = (f.T * j_l * Y_lm[None, :]).sum(axis=1)
                B_lm[k_reduced] = B
                # m=0 contributes once; m>0 contributes twice (±m symmetry)
                I_q += (2.0 - (m == 0)) * np.abs(B)**2

        I_q *= (4 * np.pi)**2
        return I_q, B_lm