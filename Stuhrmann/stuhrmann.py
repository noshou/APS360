from .helpers import *

@beartype
def stuhrmann(
    qvals: npt.NDArray[np.float64],     # shape (Q,) — q-grid
    elms:  list[str],                   # list of element names 
    lMax:  int,                         # maximum l value; m ranges from [-lMax,+lMax]
    theta: npt.NDArray[np.float64],     # shape (N,) — atom azimuthal angles
    phi:   npt.NDArray[np.float64],     # shape (N,) — atom polar angles
    r:     npt.NDArray[np.float64],     # shape (N,) — atom radial distances
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.complex128]]:
    
    """
    Compute I(q) via the Stuhrmann decomposition, assuming f_i = 1 (point scatterers).
    
    I(q) = (4π)² · Σ_{l=0}^{lMax} Σ_{m=-l}^{l} |B_lm(q)|²
    where B_lm(q) = Σ_i f_i(q) · j_l(q · r_i) · Y_lm(θ_i, φ_i)
    
    The (-i)^l prefactor in the full A_lm has unit modulus and drops out of |A_lm|²,
    so all arithmetic stays real.
    """    
    
    # assertion checks
    if not (theta.shape == phi.shape == r.shape) or theta.size == 0:
        raise ValueError("stuhrmann: theta, phi, r must share shape and be non-empty")
    elif len(elms) != theta.size:
        raise ValueError("functions.stuhrmann: elms must align with theta, phi, r")
    elif qvals.size == 0:
        raise ValueError("stuhrmann: qvals cannot be empty")
    elif lMax < 0:
        raise ValueError("stuhrmann: lMax must be non-negative")    
    
    # initialize; calculate harmonics, bessel funcs, f(q)
    I_q  = np.zeros_like(qvals, dtype=np.float64)   
    B_lm = np.zeros(((lMax+1)**2, qvals.size), dtype=np.complex128)
    Y = sphHarm(lMax, theta, phi)                   # (lMax+1)², N 
    j = bess(qvals, lMax, r)                        # lMax+1, Q, N 
    f = formFacts(elms, qvals)                      # N, Q

    # calculate I(q) estimate
    for l in range(lMax+1):
        j_l = j[l]
        for m in range(-l, l+1):
            k = l * l + l + m
            Y_lm = Y[k].conj()
            B = (f.T * j_l * Y_lm[None, :]).sum(axis=1)
            B_lm[k] = B
            I_q  += np.abs(B)**2
    I_q *= (4*np.pi)**2
    
    return I_q, B_lm