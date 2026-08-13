import numpy as np
import numpy.typing as npt
from scipy.special import sph_harm_y, spherical_jn


def sphHarm(
    lMax: int,
    theta: npt.NDArray[np.float64],
    phi: npt.NDArray[np.float64],
) -> npt.NDArray[np.complex128]:
    """Compute complex spherical harmonics Y_l^m for m ≥ 0 only, l ∈ [0, lMax].

    Uses physics convention: theta = polar colatitude (0→π),
    phi = azimuthal (0→2π).

    Output is flattened over (l, m) with triangular indexing:
    k = l*(l+1)//2 + m.

    Layout: row 0 = Y_0^0; rows 1,2 = Y_1^0, Y_1^1;
    rows 3,4,5 = Y_2^0..Y_2^2; etc.
    Total (lMax+1)*(lMax+2)//2 rows.

    Negative-m modes are omitted.
    Recover them via Y_l^{-m} = (-1)^m · conj(Y_l^m).

    Parameters
    ----------
    lMax : int
        Maximum degree. Must be non-negative.
    theta : ndarray of float64, shape (N,)
        Polar colatitudinal angles (0→π).
    phi : ndarray of float64, shape (N,)
        Azimuthal angles (0→2π).

    Returns
    -------
    ndarray of complex128, shape ((lMax+1)*(lMax+2)//2, N)
        Y[k, i] = Y_l^m(theta[i], phi[i]) where k = l*(l+1)//2 + m.

    Raises
    ------
    ValueError
        If lMax is negative, or theta and phi do not share shape or are empty.
    """
    if lMax < 0:
        raise ValueError("sphHarm: lMax must be non-negative")
    if theta.shape != phi.shape or theta.size == 0:
        raise ValueError(
            "sphHarm: theta and phi must share shape and be non-empty"
        )

    K = (lMax + 1) * (lMax + 2) // 2
    Y = np.empty((K, theta.size), dtype=np.complex128)
    for l in range(lMax + 1):  # noqa: E741
        ms = np.arange(0, l + 1)
        Y[l * (l + 1) // 2 : l * (l + 1) // 2 + l + 1] = sph_harm_y(
            l, ms[:, None], theta[None, :], phi[None, :]
        )
    return Y


def sphBess(
    q: npt.NDArray[np.float64],  # shape (Q,)
    lMax: int,
    r: npt.NDArray[np.float64],  # shape (N,)
) -> npt.NDArray[np.float64]:  # shape (lMax+1, Q, N)
    """Compute spherical Bessel functions j_l for all orders
    l = 0..lMax over a q-grid and radii.

    For reference, j_0(x) = sin(x)/x (the Debye kernel).

    Parameters
    ----------
    q : ndarray of float64, shape (Q,)
        Scattering vector magnitudes (Å⁻¹). All non-negative.
    lMax : int
        Maximum order. Must be non-negative.
    r : ndarray of float64, shape (N,)
        Atom radial distances (Å). All non-negative.

    Returns
    -------
    ndarray of float64, shape (lMax+1, Q, N)
        Entry [l, k, i] is j_l(q_k · r_i).

    Raises
    ------
    ValueError
        If q or r is empty, lMax is negative,
        or q or r contains negative values.
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

    l_vals = np.arange(lMax + 1)  # shape (lMax+1,)
    qr = q[:, None] * r[None, :]  # shape (Q, N)

    # broadcast: l_vals[:, None, None] is (lMax+1, 1, 1);
    # qr[None, :, :] is (1, Q, N)
    # result: (lMax+1, Q, N)
    return spherical_jn(l_vals[:, None, None], qr[None, :, :])
