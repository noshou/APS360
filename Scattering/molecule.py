import numpy as np
from beartype import beartype
import numpy.typing as npt
from .stuhrmann import StuhrmannMixin

@beartype
class Molecule(StuhrmannMixin):

    """
    Immutable representation of a molecule.

    Cartesian coordinates are stored as a (n, 3) array; angular coordinates
    (theta, phi) as a (n, 2) array using physics convention (theta = polar
    colatitude from z-axis, phi = azimuthal angle in xy-plane); radial
    distances as a (n,) array.

    Methods
    -------
    Molecule.fromXYZ(xyz_fp) : classmethod
        Load a Molecule from an XYZ file.
    Molecule.stuhrmann(ff, lMax) : StuhrmannMixin
        Compute I(q) and B_lm via the Stuhrmann decomposition.
    """

    _coords: npt.NDArray[np.float64]  # (n, 3); x, y, z in Å
    _angles: npt.NDArray[np.float64]  # (n, 2); theta (polar, 0→π), phi (azimuthal, 0→2π)
    _r:      npt.NDArray[np.float64]  # (n,)  ; radial distances in Å
    _elms:   list[str]                # element symbols for each atom
    _name:   str                      # molecule name

    def __init__(
        self,
        x:    npt.NDArray[np.float64],
        y:    npt.NDArray[np.float64],
        z:    npt.NDArray[np.float64],
        elms: list[str],
        name: str,
    ):

        """
        Initialize a Molecule from Cartesian coordinates.

        Args:
            x:    x-coordinates of atoms in Å (1D array).
            y:    y-coordinates of atoms in Å (1D array).
            z:    z-coordinates of atoms in Å (1D array).
            elms: Element symbols for each atom, e.g. ['C', 'N', 'O'].
            name: Human-readable name of the molecule.

        Raises:
            ValueError: If x, y, z and elms are not all the same length, if
                        the arrays are empty, or if any atom lies at the origin (r = 0).
        """

        n = len(elms)
        if x.size != n or y.size != n or z.size != n or n == 0:
            raise ValueError("Molecule: x, y, z, and elms must all be the same length and non-empty")

        r = np.sqrt(x**2 + y**2 + z**2)
        coords = np.column_stack((x, y, z))
        # Physics convention: theta = polar colatitude (arccos(z/r), 0→π),
        #                     phi   = azimuthal angle  (arctan2(y,x), 0→2π)
        # r=0 only occurs for single-atom molecules (atom is its own centroid).
        # j_l(0)=0 for l>0 so the angle is irrelevant; use r_safe to avoid NaN.
        r_safe = np.where(r > 0, r, 1.0)
        angles = np.column_stack((
            np.arccos(np.clip(z / r_safe, -1.0, 1.0)),
            np.arctan2(y, x),
        ))
        coords.flags.writeable = False
        angles.flags.writeable = False
        r.flags.writeable      = False

        object.__setattr__(self, "_coords", coords)
        object.__setattr__(self, "_angles", angles)
        object.__setattr__(self, "_r",      r)
        object.__setattr__(self, "_elms",   elms)
        object.__setattr__(self, "_name",   name)


    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("Molecule is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("Molecule is immutable")

    @classmethod
    def fromXYZ(cls, xyz_fp: str) -> 'Molecule':

        """Load a Molecule from an XYZ file.

        XYZ format (strict):
            Line 1: number of atoms (integer)
            Line 2: molecule name (string)
            Lines 3+: one atom per line; exactly 4 columns: element x y z

        Usage:
            M = Molecule.fromXYZ("your_xyz_file.xyz")

        Args:
            xyz_fp: Path to the .xyz file.

        Returns:
            A new Molecule instance.

        Raises:
            FileNotFoundError: If xyz_fp does not exist.
            ValueError: If the file is malformed (bad atom count, wrong number
                        of columns, non-numeric coordinates, or atom count mismatch).
        """
        try:
            with open(xyz_fp) as f:
                line1 = f.readline()
                line2 = f.readline()
                try:
                    # Variant A: atom count first, then name
                    declared = int(line1)
                    name = line2.strip()
                except ValueError:
                    # Variant B: name first, then atom count (cluster datasets)
                    try:
                        declared = int(line2)
                        name = line1.strip()
                    except ValueError:
                        raise ValueError(f"Molecule.fromXYZ: could not find atom count in first two lines of '{xyz_fp}'")
                if declared == 0:
                    raise ValueError(f"Molecule.fromXYZ: file '{xyz_fp}' declares 0 atoms")
                el = []
                xs, ys, zs = [], [], []

                for lineno, line in enumerate(f, start=3):
                    parts = line.split()
                    if len(parts) < 4:
                        raise ValueError(f"Molecule.fromXYZ: expected at least 4 columns on line {lineno}, got {len(parts)}")
                    elm = parts[0]
                    try:
                        xs.append(float(parts[1]))
                        ys.append(float(parts[2]))
                        zs.append(float(parts[3]))
                    except ValueError:
                        raise ValueError(f"Molecule.fromXYZ: non-numeric coordinate on line {lineno}")
                    el.append(elm)
        except FileNotFoundError:
            raise FileNotFoundError(f"Molecule.fromXYZ: file '{xyz_fp}' not found")

        if len(el) != declared:
            raise ValueError(
                f"Molecule.fromXYZ: declared {declared} atoms but found {len(el)} in '{xyz_fp}'"
            )

        xs = np.array(xs, dtype=np.float64)
        ys = np.array(ys, dtype=np.float64)
        zs = np.array(zs, dtype=np.float64)
        if not (np.isfinite(xs).all() and np.isfinite(ys).all() and np.isfinite(zs).all()):
            raise ValueError(f"Molecule.fromXYZ: non-finite coordinate (inf or nan) in '{xyz_fp}'")
        # Centre at geometric centroid so the Stuhrmann expansion is origin-independent.
        xs -= xs.mean()
        ys -= ys.mean()
        zs -= zs.mean()
        return cls(xs, ys, zs, el, name)

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
        """Angular coordinates as (n, 2) array; columns are theta (polar), phi (azimuthal) in radians. Read-only."""
        return self._angles

    @property
    def theta(self) -> npt.NDArray[np.float64]:
        """Polar colatitudinal angles in radians (arccos(z / r), 0→π). Read-only."""
        return self._angles[:, 0]

    @property
    def phi(self) -> npt.NDArray[np.float64]:
        """Azimuthal angles in radians (arctan2(y, x), 0→2π). Read-only."""
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
    def name(self) -> str:
        """Human-readable name of the molecule. Read-only."""
        return self._name

    @property
    def size(self) -> int:
        """Number of atoms in the molecule. Read-only."""
        return len(self._elms)

    def toHDF5(self, group: object) -> None:
        """Serialize structural data into an open h5py Group.

        Writes: coords (float64, (n,3)), angles (float64, (n,2)), r (float64, (n,))
        all ZFP lossless; elms (str, (n,)) uncompressed; name as a group attribute.
        I(q) is not stored here.
        """
        import h5py       # lazy — keeps Scattering independent of storage deps
        import hdf5plugin # type: ignore[import]
        _zfp = hdf5plugin.Zfp(reversible=True)  # type: ignore[attr-defined]
        group.attrs['name'] = self._name                                        # type: ignore[union-attr]
        group.create_dataset('coords', data=self._coords, chunks=True, **_zfp)  # type: ignore[union-attr]
        group.create_dataset('angles', data=self._angles, chunks=True, **_zfp)  # type: ignore[union-attr]
        group.create_dataset('r',      data=self._r,      chunks=True, **_zfp)  # type: ignore[union-attr]
        group.create_dataset('elms',   data=np.array(self._elms, dtype=h5py.string_dtype()))  # type: ignore[union-attr]

    @classmethod
    def fromHDF5(cls, group: object) -> 'Molecule':
        """Reconstruct a Molecule from an h5py Group written by toHDF5.

        Loads coords, angles, and r directly from the file rather than recomputing.
        """
        coords = group['coords'][:]                                             # type: ignore[index]
        angles = group['angles'][:]                                             # type: ignore[index]
        r      = group['r'][:]                                                  # type: ignore[index]
        elms   = [e.decode() if isinstance(e, bytes) else e
                  for e in group['elms'][:]]                                   # type: ignore[index]
        name   = str(group.attrs['name'])                                       # type: ignore[union-attr]
        mol = cls.__new__(cls)
        coords.flags.writeable = False
        angles.flags.writeable = False
        r.flags.writeable      = False
        object.__setattr__(mol, '_coords', coords)
        object.__setattr__(mol, '_angles', angles)
        object.__setattr__(mol, '_r',      r)
        object.__setattr__(mol, '_elms',   elms)
        object.__setattr__(mol, '_name',   name)
        return mol
