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

        I(q) = (4π)² · Σ_{l=0}^{lMax} Σ_{m=-l}^{l} |B_lm(q)|²
        where B_lm(q) = Σ_i f_i(q) · j_l(q · r_i) · Y_lm(θ_i, φ_i)

        The (-i)^l prefactor in the full A_lm has unit modulus and drops out of
        |A_lm|², so all arithmetic stays real.

        Parameters
        ----------
        ff : FormFactors
            Precomputed form factors; must contain all elements in mol.elms.
        lMax : int
            Maximum l value; m ranges over [-lMax, +lMax].

        Returns
        -------
        I_q : ndarray of float64, shape (Q,)
            Scattering intensity at each q value.
        B_lm : ndarray of complex128, shape ((lMax+1)², Q)
            Partial wave amplitudes.

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

        I_q  = np.zeros_like(ff.qvals, dtype=np.float64)
        B_lm = np.zeros(((lMax+1)**2, ff.qvals.size), dtype=np.complex128)

        Y = sphHarm(lMax, self.theta, self.phi)
        j = sphBess(ff.qvals, lMax, self.r)
        f = np.stack([ff.ff[ion] for ion in self.elms])

        for l in range(lMax + 1):
            j_l = j[l]
            for m in range(-l, l + 1):
                k = l * l + l + m
                Y_lm = Y[k].conj()
                B = (f.T * j_l * Y_lm[None, :]).sum(axis=1)
                B_lm[k] = B
                I_q += np.abs(B)**2

        I_q *= (4 * np.pi)**2
        return I_q, B_lm