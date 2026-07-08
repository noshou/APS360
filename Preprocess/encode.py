import re
import sqlite3
import h5py
import hdf5plugin  # noqa: F401

from beartype        import beartype
from beartype.typing import List, Tuple
from .vocab          import VOCAB
import os

class Encoding:

    """Singleton that builds and queries the molecule encoding database.

    On first construction, reads every molecule from an HDF5 file, maps each
    atom's element string to an integer VOCAB index, and persists the results
    to a SQLite database.  Subsequent constructions with the same ``db_name``
    reuse the existing file without re-parsing.

    The database is the primary input to ``Batcher``: call
    ``get_in_range(min, max)`` to retrieve all molecules whose atom count falls
    in a size bucket, then let ``Batcher`` accumulate rows until the total
    atom count would exceed ``atom_size_ceil`` and split there.

    Examples
    --------
    >>> from Preprocess import encode
    >>> encode("my_dataset", "I(q)@L=50.h5")
    # writes my_dataset-ENCODING.sqlite3
    """

    _CHARGE_RE = re.compile(r'[0-9]*[+\-]+$')

    _ENCODING_SCHEMA = """
    CREATE TABLE IF NOT EXISTS items (
        stem      TEXT    NOT NULL,
        grp       TEXT    NOT NULL,
        atoms     INTEGER NOT NULL,
        VOCAB_idx BLOB    NOT NULL,
        PRIMARY KEY (stem, grp)
    );
    """

    _INDEX_SCHEMA = "CREATE INDEX IF NOT EXISTS idx_atoms ON items(atoms);"

    _CHUNK = 10_000
    _path: str
    _max:  int

    _initialized = False
    _instance = None

    def __new__(cls, *args, **kwargs):
        """Return the shared singleton instance, creating it on first call.

        Parameters
        ----------
        *args : tuple
            Positional arguments forwarded to `__init__` (unused here).
        **kwargs : dict
            Keyword arguments forwarded to `__init__` (unused here).

        Returns
        -------
        Encoding
            The single shared `Encoding` instance.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @beartype
    def __init__(self, db_name: str, hdf5_path: str):
        """Build or reuse the encoding database.

        Parameters
        ----------
        db_name : str
            Base name for the output file; produces ``<db_name>-ENCODING.sqlite3``.
        hdf5_path : str
            Path to the source HDF5 file produced by the preprocessing pipeline.
        """

        _DBPATH = f"{db_name}-ENCODING.sqlite3"

        if os.path.exists(_DBPATH):
            print(f"Found existing database: '{_DBPATH}'. Skipping HDF5 parsing loop.")
            self._path = _DBPATH
            self._max  = self.max_atom_count()
            self._initialized = True
            return

        if self._initialized:
            return

        sqlite3.register_converter("BLOB", self._convert_array)

        conn = sqlite3.connect(_DBPATH, detect_types=sqlite3.PARSE_DECLTYPES)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA wal_autocheckpoint=0")  # checkpoint once at close, not every 4 MB
        cursor = conn.cursor()
        cursor.executescript(self._ENCODING_SCHEMA)

        _ENCODING_ADD = """
        INSERT OR IGNORE INTO items (stem, grp, atoms, VOCAB_idx)
        VALUES (?,?,?,?)
        """

        try:
            with h5py.File(hdf5_path, "r") as f:
                buf: List[Tuple] = []
                for group_name, group_obj in f.items():
                    if not isinstance(group_obj, h5py.Group):
                        continue
                    for stem, mol_obj in group_obj.items():
                        if stem.startswith("__tmp__") or not isinstance(mol_obj, h5py.Group):
                            continue
                        try:
                            raw = mol_obj["elms"][:].tolist()  # type: ignore
                            elms = [e.decode() if isinstance(e, bytes) else e for e in raw]
                            enc = self._encode_ions(elms)
                        except (LookupError, ValueError):
                            continue

                        buf.append((stem, group_name, len(enc), bytes(enc)))
                        if len(buf) >= self._CHUNK:
                            cursor.executemany(_ENCODING_ADD, buf)
                            conn.commit()
                            buf.clear()
                if buf:
                    cursor.executemany(_ENCODING_ADD, buf)
                    conn.commit()

            # Build index after all data inserted — single-pass B-tree, no fragmentation
            cursor.executescript(self._INDEX_SCHEMA)
            conn.commit()
        finally:
            cursor.execute("SELECT MAX(atoms) FROM items")
            max_ = cursor.fetchone()
            self._max = max_[0] if max_[0] is not None else 0
            conn.close()

        self._path = _DBPATH
        self._initialized = True

    @beartype
    def _bare(self, _ion: str) -> str:
        """Strip charge suffixes and normalize special-case ion aliases.

        Parameters
        ----------
        _ion : str
            Raw ion symbol, possibly with a trailing charge (e.g. 'Fe2+').

        Returns
        -------
        str
            Lowercase bare element symbol, with 'cval'/'siva' remapped to
            'c'/'si' and any charge suffix removed.
        """
        ion = _ion.strip()
        if ion.lower() == 'cval':
            return 'c'
        elif ion.lower() == 'siva':
            return 'si'
        else:
            return self._CHARGE_RE.sub('', ion).lower()

    def _adapt_array(self, lst: list) -> bytes:
        return bytes(lst)

    def _convert_array(self, blob: bytes) -> list:
        return list(blob)

    @beartype
    def _encode_ions(self, ions: List[str]) -> List[int]:
        """Map a list of ion symbols to their 1-based VOCAB indices.

        Parameters
        ----------
        ions : list of str
            Ion symbols for each atom in a molecule, in atom order.

        Returns
        -------
        list of int
            VOCAB index for each ion, in the same order as `ions`.

        Raises
        ------
        LookupError
            If an ion symbol (and its bare form) is not found in VOCAB.
        ValueError
            If `ions` is empty.
        """
        enc: List[int] = []
        for ion in ions:
            key = ion.lower().strip()
            if key not in VOCAB:
                key = self._bare(ion)
            if key not in VOCAB:
                raise LookupError(f"{ion!r} not found in xraydb vocabulary")
            enc.append(VOCAB[key])
        if len(enc) == 0:
            raise ValueError("ion list is empty")
        return enc

    @beartype
    def get_in_range(self, min_atoms: int, max_atoms: int) -> List[Tuple[str, str, int, List[int]]]:
        """Return all molecules whose atom count falls in [min_atoms, max_atoms].

        Results are ordered by atom count ascending so a caller can greedily
        accumulate rows and split when a running total exceeds a ceiling.

        Parameters
        ----------
        min_atoms : int
            Lower bound (inclusive).
        max_atoms : int
            Upper bound (inclusive).

        Returns
        -------
        list of tuple of (str, str, int, list of int)
            ``(grp, stem, atoms, VOCAB_idx)`` tuples.  ``grp`` and
            ``stem`` are the HDF5 keys needed to load tensor data via
            ``f[grp][stem]``.  ``VOCAB_idx`` is the list of integer VOCAB
            indices for each atom in the molecule.
        """

        _QUERY = """
            SELECT grp, stem, atoms, VOCAB_idx
            FROM items
            WHERE atoms BETWEEN ? AND ?
            ORDER BY atoms ASC
        """

        sqlite3.register_converter("BLOB", self._convert_array)
        conn = sqlite3.connect(self._path, detect_types=sqlite3.PARSE_DECLTYPES)
        try:
            cursor = conn.cursor()
            cursor.execute(_QUERY, (min_atoms, max_atoms))
            results = cursor.fetchall()
        finally:
            conn.close()
        return results

    @beartype
    def get_meta_in_range(self, min_atoms: int, max_atoms: int) -> List[Tuple[str, str, int]]:
        """Like get_in_range but omits VOCAB_idx, for lazy loading pipelines.

        Parameters
        ----------
        min_atoms : int
            Lower bound (inclusive).
        max_atoms : int
            Upper bound (inclusive).

        Returns
        -------
        list of tuple of (str, str, int)
            ``(grp, stem, atoms)`` tuples ordered by atom count ascending.
        """
        _QUERY = """
            SELECT grp, stem, atoms
            FROM items
            WHERE atoms BETWEEN ? AND ?
            ORDER BY atoms ASC
        """
        conn = sqlite3.connect(self._path)
        try:
            cursor = conn.cursor()
            cursor.execute(_QUERY, (min_atoms, max_atoms))
            results = cursor.fetchall()
        finally:
            conn.close()
        return results

    def max_atom_count(self) -> int:
        """Return the largest atom count across all molecules in the database.

        Returns
        -------
        int
            Maximum value of the ``atoms`` column, cached after first computation.
        """
        if hasattr(self, "_max") and self._max is not None:
            return self._max
        conn = sqlite3.connect(self._path)
        try:
            row = conn.execute("SELECT MAX(atoms) FROM items").fetchone()
            self._max = row[0] if row[0] is not None else 0
        finally:
            conn.close()
        return self._max

    def count(self) -> int:
        """Return the total number of molecules in the dataset.

        Returns
        -------
        int
            Row count of the ``items`` table.
        """
        conn = sqlite3.connect(self._path)
        try:
            return conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]
        finally:
            conn.close()
