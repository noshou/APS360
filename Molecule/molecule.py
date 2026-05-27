import numpy as np
from beartype import beartype
import numpy.typing as npt
import FormFact as ff

@beartype
class Molecule:
    
    """Immutable representation of a molecule.

    Cartesian coordinates are stored as a (n, 3) array; angular coordinates
    (theta, phi) as a (n, 2) array; radial distances as a (n,) array.
    """
    
    _coords: npt.NDArray[np.float64]  # (n, 3); x, y, z in Å
    _angles: npt.NDArray[np.float64]  # (n, 2); theta (azimuthal), phi (polar) in radians
    _r:      npt.NDArray[np.float64]  # (n,)  ; radial distances in Å
    _elms:   list[str]                # element symbols for each atom
    _qvals:  npt.NDArray[np.float64]  # scattering vector magnitudes (Å⁻¹)
    _name:   str                      # molecule name
    _size:   int                      # number of atoms

    def __init__(
        self,
        x:    npt.NDArray[np.float64],
        y:    npt.NDArray[np.float64],
        z:    npt.NDArray[np.float64],
        elms: list[str],
        name: str,
        size: int
    ):
        
        """Initialize a Molecule from Cartesian coordinates.

        Args:
            x:    x-coordinates of atoms in Å (1D array, length = size).
            y:    y-coordinates of atoms in Å (1D array, length = size).
            z:    z-coordinates of atoms in Å (1D array, length = size).
            elms: Element symbols for each atom, e.g. ['C', 'N', 'O'].
            name: Human-readable name of the molecule.
            size: Number of atoms. Must equal len(elms) and the length of x/y/z.

        Raises:
            ValueError: If x, y, z and elms are not all the same length, if
                        size does not match, if the arrays are empty, or if any
                        atom lies at the origin (r = 0).
        """
        
        n = len(elms)
        if size != n or x.size != n or y.size != n or z.size != n or n == 0:
            raise ValueError("Molecule: x, y, z, elms, and size must all be consistent and non-empty")

        r = np.sqrt(x**2 + y**2 + z**2)
        if np.any(r == 0):
            raise ValueError("Molecule: atom at origin (r = 0) not allowed")

        object.__setattr__(self, "_coords", np.column_stack((x, y, z)))
        object.__setattr__(self, "_angles", np.column_stack((np.arctan2(y, x), np.arccos(z / r))))
        object.__setattr__(self, "_r",      r)
        object.__setattr__(self, "_elms",   elms)
        object.__setattr__(self, "_size",   size)
        object.__setattr__(self, "_name",   name)
        object.__setattr__(self, "_qvals",  ff.f0factor.getqvals())

    @classmethod
    def fromXYZ(cls, xyz_fp: str) -> 'Molecule':
        
        """Load a Molecule from an XYZ file.

        XYZ format:
            Line 1: number of atoms (integer)
            Line 2: molecule name (string)
            Lines 3+: one atom per line; element x y z

        Useage:
            M = Molecule.fromXYZ("your_xyz_file.xyz")

        Args:
            xyz_fp: Path to the .xyz file.

        Returns:
            A new Molecule instance.

        Raises:
            FileNotFoundError: If xyz_fp does not exist.
            ValueError: If the file is malformed (bad atom count, wrong number
                        of columns, or non-numeric coordinates).
        """
        try:
            with open(xyz_fp) as f:
                try:
                    size = int(f.readline())
                except ValueError:
                    raise ValueError(f"Molecule.fromXYZ: first line of '{xyz_fp}' must be an integer atom count")
                name = f.readline().strip()
                el = []
                xs, ys, zs = [], [], []

                for lineno, line in enumerate(f, start=3):
                    parts = line.split()
                    if len(parts) != 4:
                        raise ValueError(f"Molecule.fromXYZ: expected 4 columns on line {lineno}, got {len(parts)}")
                    elm, x, y, z = parts
                    try:
                        xs.append(float(x))
                        ys.append(float(y))
                        zs.append(float(z))
                    except ValueError:
                        raise ValueError(f"Molecule.fromXYZ: non-numeric coordinate on line {lineno}")
                    el.append(elm)
        except FileNotFoundError:
            raise FileNotFoundError(f"Molecule.fromXYZ: file '{xyz_fp}' not found")

        xs = np.array(xs, dtype=np.float64)
        ys = np.array(ys, dtype=np.float64)
        zs = np.array(zs, dtype=np.float64)
        return cls(xs, ys, zs, el, name, size)

    @property
    def coords(self) -> npt.NDArray[np.float64]:
        """Cartesian coordinates as (n, 3) array; columns are x, y, z in Å. Read-only."""
        return self._coords

    @property
    def x(self) -> npt.NDArray[np.float64]:
        """x-coordinates in Å. Read-only."""
        return self._coords[:, 0]

    @property
    def y(self) -> npt.NDArray[np.float64]:
        """y-coordinates in Å. Read-only."""
        return self._coords[:, 1]

    @property
    def z(self) -> npt.NDArray[np.float64]:
        """z-coordinates in Å. Read-only."""
        return self._coords[:, 2]

    @property
    def angles(self) -> npt.NDArray[np.float64]:
        """Angular coordinates as (n, 2) array; columns are theta, phi in radians. Read-only."""
        return self._angles

    @property
    def theta(self) -> npt.NDArray[np.float64]:
        """Azimuthal angles in radians (arctan2(y, x)). Read-only."""
        return self._angles[:, 0]

    @property
    def phi(self) -> npt.NDArray[np.float64]:
        """Polar angles in radians (arccos(z / r)). Read-only."""
        return self._angles[:, 1]

    @property
    def r(self) -> npt.NDArray[np.float64]:
        """Radial distances from origin in Å. Read-only."""
        return self._r

    @property
    def elms(self) -> list[str]:
        """Element symbols for each atom. Read-only."""
        return self._elms

    @property
    def qVals(self) -> npt.NDArray[np.float64]:
        """Scattering vector magnitudes Q = (sinθ)/λ in Å⁻¹. Read-only."""
        return self._qvals

    @property
    def name(self) -> str:
        """Human-readable name of the molecule. Read-only."""
        return self._name

    @property
    def size(self) -> int:
        """Number of atoms in the molecule. Read-only."""
        return self._size
