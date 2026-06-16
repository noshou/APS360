import numpy as np
from beartype import beartype
from scipy.special import spherical_jn, sph_harm_y
import numpy.typing as npt

@beartype
def sphHarm(
    lMax:  int,
    theta: npt.NDArray[np.float64],
    phi:   npt.NDArray[np.float64],
) -> npt.NDArray[np.complex128]:
    """Compute complex spherical harmonics Y_l^m for m ≥ 0 only, l ∈ [0, lMax].

    Uses physics convention: theta = polar colatitude (0→π), phi = azimuthal (0→2π).
    Output is flattened over (l, m) with triangular indexing: k = l*(l+1)//2 + m.
    Layout: row 0 = Y_0^0; rows 1,2 = Y_1^0, Y_1^1; rows 3,4,5 = Y_2^0..Y_2^2; etc.
    Total (lMax+1)*(lMax+2)//2 rows.

    Negative-m modes are omitted — recover them via Y_l^{-m} = (-1)^m · conj(Y_l^m).

    Args:
        lMax:  Maximum degree. Must be non-negative.
        theta: Polar colatitudinal angles (0→π), shape (N,).
        phi:   Azimuthal angles (0→2π), shape (N,).

    Returns:
        Y: shape ((lMax+1)*(lMax+2)//2, N). Y[k, i] = Y_l^m(theta[i], phi[i])
        where k = l*(l+1)//2 + m.

    Raises:
        ValueError: If inputs are invalid.
    """
    if lMax < 0:
        raise ValueError("sphHarm: lMax must be non-negative")
    if theta.shape != phi.shape or theta.size == 0:
        raise ValueError("sphHarm: theta and phi must share shape and be non-empty")

    K = (lMax + 1) * (lMax + 2) // 2
    Y = np.empty((K, theta.size), dtype=np.complex128)
    for l in range(lMax + 1):
        ms = np.arange(0, l + 1)
        Y[l * (l + 1) // 2 : l * (l + 1) // 2 + l + 1] = sph_harm_y(l, ms[:, None], theta[None, :], phi[None, :])
    return Y

@beartype
def sphBess(
    q:    npt.NDArray[np.float64],     # shape (Q,)
    lMax: int,
    r:    npt.NDArray[np.float64],     # shape (N,)
) -> npt.NDArray[np.float64]:          # shape (lMax+1, Q, N)
    """Compute spherical Bessel functions j_l for all orders l = 0..lMax over a q-grid and radii.

    For reference, j_0(x) = sin(x)/x (the Debye kernel).

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
        raise ValueError("sphBess: r cannot be empty")
    if q.size == 0:
        raise ValueError("sphBess: q cannot be empty")
    if lMax < 0:
        raise ValueError("sphBess: lMax must be non-negative")
    if np.any(q < 0):
        raise ValueError("sphBess: all q values must be non-negative")
    if np.any(r < 0):
        raise ValueError("sphBess: all r values must be non-negative")

    l_vals = np.arange(lMax + 1)                          # shape (lMax+1,)
    qr     = q[:, None] * r[None, :]                      # shape (Q, N)

    # broadcast: l_vals[:, None, None] is (lMax+1, 1, 1); qr[None, :, :] is (1, Q, N)
    # result: (lMax+1, Q, N)
    return spherical_jn(l_vals[:, None, None], qr[None, :, :])
