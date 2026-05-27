import numpy as np
from beartype import beartype
import numpy.typing as npt
import FormFact as ff

@beartype
class Molecule:
    """Immutable representation of a molecule in spherical coordinates.
    
    Attributes are read-only and computed from Cartesian coordinates (x, y, z)
    provided at initialization.
    """
    _r:      npt.NDArray[np.float64]   # radial distances of atoms
    _theta:  npt.NDArray[np.float64]   # azimuthal angles (radians)
    _phi:    npt.NDArray[np.float64]   # polar angles (radians)
    _elms:   list[str]                 # element symbols for each atom
    _qvals:  npt.NDArray[np.float64]   # scattering vector magnitudes
    
    def __init__(
        self,
        x:    npt.NDArray[np.float64],
        y:    npt.NDArray[np.float64],
        z:    npt.NDArray[np.float64],
        elms: list[str]
    ):
        """Initialize a Molecule from Cartesian coordinates and Q values.
        
        Args:
            x: x-coordinates of atoms (1D array)
            y: y-coordinates of atoms (1D array)
            z: z-coordinates of atoms (1D array)
            elms: element symbols for each atom (list of strings)
        
        Raises:
            ValueError: If arrays have mismatched shapes, are empty,
                        or any atom lies at the origin (r = 0).
        """
        if not (x.shape == y.shape == z.shape == len(elms)) or x.size == 0:
            raise ValueError("Molecule: arrays must share shape and be non-empty")
        
        r = np.sqrt(x**2 + y**2 + z**2)
        if np.any(r == 0):
            raise ValueError("Molecule: atom at origin (r = 0) not allowed")
        
        # Molecule is only defined on a subset of SAXS, so get q vals
        object.__setattr__(self, "_qvals", ff.getqvalues())
        
        # Use object.__setattr__ to bypass potential __setattr__ restrictions
        object.__setattr__(self, "_r",     r)
        object.__setattr__(self, "_theta", np.arctan2(y, x))
        object.__setattr__(self, "_phi",   np.arccos(z / r))
        object.__setattr__(self, "_elms",  elms)
        object.__setattr__(self, "_qvals", q)

    @property
    def r(self: Molecule) -> npt.NDArray[np.float64]:
        """Radial distances from origin (Å). Read-only."""
        return self._r
    
    @property
    def theta(self: Molecule) -> npt.NDArray[np.float64]:
        """Azimuthal angles in radians (arctan2(y, x)). Read-only."""
        return self._theta
    
    @property
    def phi(self: Molecule) -> npt.NDArray[np.float64]:
        """Polar angles in radians (arccos(z / r)). Read-only."""
        return self._phi
    
    @property
    def elms(self: Molecule) -> list[str]:
        """Element symbols for each atom. Read-only."""
        return self._elms        
    
    @property
    def qVals(self: Molecule) -> npt.NDArray[np.float64]:
        """Scattering vector magnitudes Q = (sinθ)/λ (Å⁻¹). Read-only."""
        return self._qvals