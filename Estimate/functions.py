import numpy as np
from beartype import beartype
from scipy.special import lpmv, spherical_jn
from scipy.special import factorial as fact
import numpy.typing as npt
import FormFact

@beartype
def sphHarm(
    l: int,
    m: int,
    theta: npt.NDArray[np.float64],
    phi:   npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
        
    """Compute spherical harmonic Y_l^m on a grid.

    Args:
        l: Degree (0, 1, 2, ...). Controls overall shape complexity.
        m: Order (-l ≤ m ≤ l). Controls angular structure.
        theta: Polar coordinates
        phi:   Polar coordinates
    Returns:
        Y: Y_l^m 
    """
    
    # assertion checks
    if (l < 0):
        raise ValueError("functions.sphHarm: l is out of bound")
    elif ((m < -l) or (m > l)):
        raise ValueError("functions.sphHarm: m is out of bounds")
    elif (theta.shape != phi.shape or theta.size == 0):
        raise ValueError("functions.sphHarm: Invalid grid size")
    
    # calculate associated legendre polynomial
    P = lpmv(abs(m), l, np.cos(phi))
    
    # calculate azimuthal 
    if m == 0:
        azimuthal = 1.0
    elif m > 0:
        azimuthal = np.sqrt(2) * np.cos(m * theta)
    else:  
        azimuthal = np.sqrt(2) * np.sin(abs(m) * theta)    
    
    # calculate normilization constant
    N = np.sqrt((2*l+1)/(4*np.pi) * (fact(l-abs(m),True)/(fact(l+abs(m),True))))
    
    # calculate the value of Y
    Y = N * P * azimuthal
    
    return Y

@beartype 
def bess (
    q: int|float, 
    l: int,
    r: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
    
    """
    Compute the spherical Bessel function of the first kind, order l: j_l(q·r).
    For reference, j_0(x) = sin(x)/x (the kernel of the Debye formula) and
    j_1(x) = sin(x)/x² - cos(x)/x.

    Args:
        l:  The order
        q:  Magnitude of the scattering vector (Å⁻¹ in typical SAXS units).
            Must be non-negative.
        r:  Radial distance, e.g. an atom's distance from the origin (Å).
            Must be non-negative.
    Returns:
        The values of j_l evaluated at the product q·r.

    Raises:
        ValueError: If q or r is negative.
    """    # assertion checks
    
    if (r.size == 0):
        raise ValueError("functions.bess: r cannot be empty")
    elif (l < 0):
        raise ValueError("functions.bess: l must be non-negative")
    elif (q < 0):
        raise ValueError("functions.bess: q must be positive")
    
    # caluclate the spherical bessel function of the first kind
    return spherical_jn(l, q*r)

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