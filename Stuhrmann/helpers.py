import numpy as np
from beartype import beartype
from scipy.special import spherical_jn, sph_harm_y
import numpy.typing as npt
import FormFact

@beartype
def sphHarm(
    lMax:  int,
    theta: npt.NDArray[np.float64],
    phi:   npt.NDArray[np.float64],
) -> npt.NDArray[np.complex128]:
    """Compute complex spherical harmonics Y_l^m for all (l, m) with l ∈ [0, lMax], |m| ≤ l.

    Output is flattened over (l, m): row k corresponds to (l, m) where k = l² + l + m.
    Layout: row 0 = Y_0^0; rows 1, 2, 3 = Y_1^{-1}, Y_1^0, Y_1^1; rows 4..8 = Y_2^{-2}..Y_2^2; etc.
    Total (lMax+1)² rows.

    Args:
        lMax:  Maximum degree. Must be non-negative.
        theta: Azimuthal angles (0 → 2π), shape (N,).
        phi:   Polar angles (0 → π), shape (N,).

    Returns:
        Y: shape ((lMax+1)², N). Y[k, i] = Y_l^m(theta[i], phi[i]) where k = l² + l + m.

    Raises:
        ValueError: If inputs are invalid.
    """
    if lMax < 0:
        raise ValueError("functions.sphHarm: lMax must be non-negative")
    if theta.shape != phi.shape or theta.size == 0:
        raise ValueError("functions.sphHarm: theta and phi must share shape and be non-empty")

    K = (lMax + 1) ** 2
    Y = np.zeros((K, theta.size), dtype=np.complex128)

    k = 0
    for l in range(lMax + 1):
        for m in range(-l, l + 1):
            # sph_harm_y convention: (l, m, polar, azimuthal)
            Y[k] = sph_harm_y(l, m, phi, theta)
            k += 1

    return Y

@beartype
def bess(
    q:    npt.NDArray[np.float64],     # shape (Q,)
    lMax: int,
    r:    npt.NDArray[np.float64],     # shape (N,)
) -> npt.NDArray[np.float64]:          # shape (lMax+1, Q, N)
    """Compute the spherical Bessel function of the first kind for all orders 
    l = 0..lMax over a q-grid and an atom radii array.
    
    For reference, j_0(x) = sin(x)/x (the Debye kernel) and
    j_1(x) = sin(x)/x² − cos(x)/x.
    
    Args:
        q:    Scattering vector magnitudes (Å⁻¹), shape (Q,). All non-negative.
        lMax: Maximum order. Must be non-negative.
        r:    Atom radial distances (Å), shape (N,). All non-negative.
    
    Returns:
        Array of shape (lMax+1, Q, N) — entry [l, k, i] is j_l(q_k · r_i).
    
    Raises:
        ValueError: If inputs are invalid.
    """
    if r.size == 0:
        raise ValueError("functions.bess: r cannot be empty")
    if q.size == 0:
        raise ValueError("functions.bess: q cannot be empty")
    if lMax < 0:
        raise ValueError("functions.bess: lMax must be non-negative")
    if np.any(q < 0):
        raise ValueError("functions.bess: all q values must be non-negative")
    if np.any(r < 0):
        raise ValueError("functions.bess: all r values must be non-negative")
    
    l_vals = np.arange(lMax + 1)                          # shape (lMax+1,)
    qr     = q[:, None] * r[None, :]                      # shape (Q, N)
    
    # broadcast: l_vals[:, None, None] is (lMax+1, 1, 1); qr[None, :, :] is (1, Q, N)
    # result: (lMax+1, Q, N)
    return spherical_jn(l_vals[:, None, None], qr[None, :, :])

@beartype
def formFacts(
    elms:  list[str],
    qvals: npt.NDArray[np.float64],
) -> npt.NDArray[np.complex128]:
    
    """Compute the complex atomic form factors f(Q) = f0(Q) + f'(E) + i·f2(E) for a
    list of atoms over a range of scattering vectors.

    f0 is the Q-dependent Thomson scattering factor; f' and f2 are the real and
    imaginary anomalous-scattering corrections (energy-dependent, evaluated at
    12.4128 keV / 1.0 Å). f' = f1_henke - Z is derived internally so f1 does
    not double-count the nuclear contribution.

    Args:
        elms:   Element symbols for each atom, e.g. ['C', 'N', 'O'].
                Must be non-empty and recognised by the FormFact library.
        qvals:  1-D array of scattering-vector magnitudes Q = (sin θ)/λ in Å⁻¹.
                Must be non-empty.

    Returns:
        FF: Complex array of shape (n_atoms, n_qvals) where
            FF[i, j] = f(Q_j) for the i-th atom.

    Raises:
        ValueError:  If either input is empty.
        LookupError: If an element symbol or q value is not found in the FormFact library.
    """
    
    if len(elms) == 0 or qvals.size == 0:
        raise ValueError("functions.formFacts: elms and qvals must be non-empty")

    FF = np.zeros((len(elms), len(qvals)), dtype=np.complex128)
    for i, elm in enumerate(elms):
        for j, q in enumerate(qvals):
            ff_re, ff_im, iostat = FormFact.formfact.getformfactpy(q, elm)
            if iostat == -1:
                raise LookupError(
                    f"functions.formFacts: unknown element '{elm}'"
                )
            elif iostat == -2:
                raise LookupError(
                    f"functions.formFacts: Q={q} is out of range (0 to 0.5 Å⁻¹)"
                )
            FF[i, j] = complex(ff_re, ff_im)
    return FF
