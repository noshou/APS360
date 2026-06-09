import bisect
import torch
from torch.utils.data import Dataset
from Preprocess import Encoding
from .batch import Batch
from beartype.typing import List, Tuple
import h5py

Row       = Tuple[str, str, int, List[int]]
_RawBatch = List[Row]


class Batcher(Dataset):
    
    """
    Dataset of size-bucketed molecule sub-batches.

    On construction, queries SQLite for each (min_atoms, max_atoms) bucket and
    splits any bucket whose total atom count exceeds ``atom_size_ceil`` into
    balanced sub-batches via recursive binary search on a prefix-sum array.
    The resulting sub-batches are stored in ``_batches``; each index maps to
    one sub-batch of molecules whose combined atom count is ≤ ``atom_size_ceil``.
    
    Each ``Row`` is ``(grp, stem, atoms, VOCAB_idx)``, the HDF5 keys and
    encoded atom identities for one molecule.  Each ``_RawBatch`` is a list of
    rows whose total atom count is ≤ ``atom_size_ceil``.
    """

    _batches: List[_RawBatch]
    _db_path: str

    def __init__(
        self,
        hdf5_db: str,
        enc: Encoding,
        batches: list[tuple[int,int]],
        atom_size_ceil: int = -1
    ): 
        
        """
        Args:
            hdf5_db:        path to raw HDF5 data.
            enc:            Encoding instance for SQLite queries.
            batches:        list of (min_atoms, max_atoms) size buckets to load.
            atom_size_ceil: maximum total atoms per sub-batch; buckets exceeding
                            this are recursively split via binary search. If <=0, 
                            set to double of largest atom size.
        """
        
        try:
            with h5py.File(hdf5_db, "r"):
                pass
        except FileNotFoundError:
            raise FileNotFoundError(f"HDF5 file not found: '{hdf5_db}'")
        except PermissionError:
            raise PermissionError(f"No permission to read: '{hdf5_db}'")
        except OSError as e:
            raise OSError(f"Could not open HDF5 file: {e}") from e

        self._db_path = hdf5_db
        self._batches = []

        if atom_size_ceil <= 0:
            atom_size_ceil = 2 * enc.max_atom_count()
        
        def split_batch(
            result: List[_RawBatch], 
            lo: int, 
            hi: int, 
            query: _RawBatch, 
            ceil: int
        ) -> None:
            sub = query[lo:hi+1]
            count = sum(row[2] for row in sub)
            if count <= ceil or hi <= lo:
                result.append(sub)
                return
            prefix: List[int] = []
            running = 0
            for row in sub:
                running += row[2]
                prefix.append(running)
            mid = bisect.bisect_left(prefix, count // 2)
            mid = max(0, min(mid, len(sub) - 2))
            split_batch(result, lo,           lo + mid, query, ceil)
            split_batch(result, lo + mid + 1, hi,       query, ceil)

        for i, (min_a, max_a) in enumerate(batches):
            if min_a > max_a:
                print(f"Invalid bound size in batch {i}: skipping...")
                continue
            query = enc.get_in_range(min_a, max_a)
            count = sum(row[2] for row in query)
            if count > atom_size_ceil:
                split_batch(self._batches, 0, len(query) - 1, query, atom_size_ceil)
            else:
                self._batches.append(query)

    def __len__(self) -> int:
        return len(self._batches)

    def __getitem__(self, i: int) -> Batch:
        
        """Load and return sub-batch ``i`` as a typed, shape-validated ``Batch``."""
        
        rows  = self._batches[i]
        vocab = [torch.tensor(row[3]) for row in rows]
        iqval: List = []
        radii: List = []
        angle: List = []

        with h5py.File(self._db_path, "r") as db:
            for grp, stem, _, _ in rows:
                iqval.append(torch.tensor(db[grp][stem]["I_q"][:]))    # type: ignore
                radii.append(torch.tensor(db[grp][stem]["r"][:]))      # type: ignore
                angle.append(torch.tensor(db[grp][stem]["angles"][:])) # type: ignore

        return Batch.from_lists(vocab, iqval, radii, angle)
