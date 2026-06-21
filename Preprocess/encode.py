import re
import json
import sqlite3
import h5py
import hdf5plugin  # noqa: F401
from beartype import beartype
from beartype.typing import List, Tuple
from .vocab import VOCAB
import os

class Encoding:
    
    """
    Singleton that builds and queries the molecule encoding database.

    On first construction, reads every molecule from an HDF5 file, maps each
    atom's element string to an integer VOCAB index, and persists the results
    to a SQLite database.  Subsequent constructions with the same ``db_name``
    reuse the existing file without re-parsing.

    The database is the primary input to ``Batcher``: call
    ``get_in_range(min, max)`` to retrieve all molecules whose atom count falls
    in a size bucket, then let ``Batcher`` accumulate rows until the total
    atom count would exceed ``atom_size_ceil`` and split there.
    
    Usage:
    ``` from Preprocess import encode
        encode("my_dataset", "I(q)@L=50.h5")
        # writes my_dataset-ENCODING.sqlite3
    ```
    """

    _CHARGE_RE = re.compile(r'[0-9]*[+\-]+$')
    
    _ENCODING_SCHEMA = """
    CREATE TABLE IF NOT EXISTS items (
        stem      TEXT       NOT NULL,
        grp       TEXT       NOT NULL,
        atoms     INTEGER    NOT NULL,
        VOCAB_idx JSON_ARRAY NOT NULL,
        PRIMARY KEY (stem, grp)
    );
    CREATE INDEX IF NOT EXISTS idx_atoms ON items(atoms);
    """

    _CHUNK = 10_000
    _path: str
    _max:  int
    
    _initialized = False
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @beartype
    def __init__(self, db_name: str, hdf5_path: str):
        """Build or reuse the encoding database.

        Args:
            db_name:    base name for the output file; produces ``<db_name>-ENCODING.sqlite3``.
            hdf5_path:  path to the source HDF5 file produced by the preprocessing pipeline.
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

        # Register adapters before opening the connection context
        sqlite3.register_adapter(list, self._adapt_array)
        sqlite3.register_converter("JSON_ARRAY", self._convert_array)

        conn = sqlite3.connect(_DBPATH, detect_types=sqlite3.PARSE_DECLTYPES)
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

                        buf.append((stem, group_name, len(enc), enc))
                        if len(buf) >= self._CHUNK:
                            cursor.executemany(_ENCODING_ADD, buf)
                            buf.clear()
                if buf:
                    cursor.executemany(_ENCODING_ADD, buf)

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
        ion = _ion.strip()
        if ion.lower() == 'cval':
            return 'c'
        elif ion.lower() == 'siva':
            return 'si'
        else:
            return self._CHARGE_RE.sub('', ion).lower()

    @beartype
    def _adapt_array(self, lst: list) -> str:
        return json.dumps(lst)

    @beartype
    def _convert_array(self, text: bytes) -> list:
        return json.loads(text.decode("utf-8"))
    
    @beartype
    def _encode_ions(self, ions: List[str]) -> List[int]:
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

        Args:
            min_atoms:  lower bound (inclusive).
            max_atoms:  upper bound (inclusive).

        Returns:
            List of ``(grp, stem, atoms, VOCAB_idx)`` tuples.  ``grp`` and
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
        """Like get_in_range but omits VOCAB_idx, for lazy loading pipelines."""
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
        """Return the largest atom count across all molecules in the database."""
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
        """Return the total number of molecules in dataset"""
        conn = sqlite3.connect(self._path)
        try:
            return conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]
        finally:
            conn.close()