import bisect
from Preprocess import Encoding
from beartype.typing import List, Tuple
import h5py
from .batch_set import BatchSet
import numpy as np
from numpy.random import default_rng

_TRAIN = 0.7
_VAL   = 0.15
_TEST  = 0.15
class Batcher:
    
    """
    Factory that builds train, validation, and test BatchSets from an HDF5 dataset.

    Call ``get_sets()`` to retrieve ``(train, val, test)`` as ``BatchSet`` objects
    ready for use with ``DataLoader(batch_size=1, collate_fn=lambda x: x[0])``.

    Construction proceeds in two steps:

    1. ``_batches_init``: for each (min_atoms, max_atoms) size bucket, queries
        the HDF5 metadata for matching molecules sorted by atom count, then recursively
        splits any bucket whose total atom count exceeds ``atom_size_ceil`` into balanced
        sub-lists via binary search on a prefix-sum array. Produces a flat list of
        batches, each a list of (grp, stem, atoms) rows whose combined atom count
        is <= ``atom_size_ceil``.

    2. ``_batches_stratify``: collapses any batch with fewer than 3 molecules into
        its nearest neighbour (by median atom count, ties go down), then splits each
        batch independently (default: 70/15/15) at the molecule level using a seeded RNG.
        Wraps the three resulting lists in ``BatchSet`` instances.
    """
    
    _train: "BatchSet"
    _val:   "BatchSet"
    _test:  "BatchSet"
    
    def __init__(
        self,
        hdf5_db:        str,
        enc:            Encoding,
        batches:        list[tuple[int,int]],
        seed:           int,
        atom_size_ceil: int   = -1,
        train_percent:  float = _TRAIN,
        val_percent:    float = _VAL,
        test_percent:   float = _TEST
    ): 
        
        """
        Args:
            hdf5_db:        path to raw HDF5 data.
            enc:            Encoding instance for SQLite queries.
            batches:        list of (min_atoms, max_atoms) size buckets to load.
            atom_size_ceil: maximum total atoms per batch; exceeded buckets are
                            split via binary search. If <=0, set to 3× max atom count.
            seed:           RNG seed for reproducible train/val/test splits.
            train_percent:  fraction of molecules for training.
            val_percent:    fraction for validation.
            test_percent:   fraction for testing. The larger of
                            val/test receives the floor-rounding remainder; ties go to test.
        """
        
        if (
            train_percent <= 0 or 
            val_percent   <= 0 or 
            test_percent  <= 0 or 
            (train_percent + val_percent + test_percent) != 1.
            ):
            raise ValueError("Invalid values for train, validation, testing, batch sizes")
        elif (seed < 0):
            raise ValueError("Invalid seed; must be >= 0")
        
        if atom_size_ceil <= 0:
            atom_size_ceil = 3 * enc.max_atom_count()
        
        all_batches = self._batches_init(hdf5_db, enc, batches, atom_size_ceil)
        self._batches_stratify(seed, train_percent, val_percent, test_percent, hdf5_db, enc, all_batches)

    
    def _batches_init(
        self,
        hdf5_db: str,
        enc: Encoding,
        batches: list[tuple[int,int]],
        atom_size_ceil: int,
    ) -> list:
        
        """
        Build the flat list of batches across all size buckets.

        Args:
            hdf5_db:        path to raw HDF5 data.
            enc:            Encoding instance for SQLite queries.
            batches:        list of (min_atoms, max_atoms) size buckets.
            atom_size_ceil: maximum total atoms per batch.

        Returns:
            Flat list of batches, each a list of (grp, stem, atoms) rows.
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

        result: list = []

        def split_batch(
            lo: int,
            hi: int,
            query: list,
        ) -> None:
            sub = query[lo:hi+1]
            count = sum(row[2] for row in sub)
            if count <= atom_size_ceil or hi <= lo:
                result.append(sub)
                return
            prefix: List[int] = []
            running = 0
            for row in sub:
                running += row[2]
                prefix.append(running)
            mid = bisect.bisect_left(prefix, count // 2)
            mid = max(0, min(mid, len(sub) - 2))
            split_batch(lo,           lo + mid, query)
            split_batch(lo + mid + 1, hi,       query)

        for i, (min_a, max_a) in enumerate(batches):
            if min_a > max_a:
                print(f"Invalid bucket {i} ({min_a}, {max_a}): skipping...")
                continue
            query = enc.get_meta_in_range(min_a, max_a)
            count = sum(row[2] for row in query)
            if count > atom_size_ceil:
                split_batch(0, len(query) - 1, query)
            else:
                result.append(query)

        return result
    
    def _batches_stratify(
        self,
        seed:         int,
        trn_p:        float,
        val_p:        float,
        tst_p:        float,
        hdf5_db:      str,
        enc:          Encoding,
        batches_flat: list,
    ):
        # Collapse batches with fewer than 3 molecules into an adjacent batch.
        # Merge direction: toward the neighbour whose median atom count is closer.
        # Tie or edge case: merge down (into previous).
        i = 0
        while i < len(batches_flat):
            if len(batches_flat[i]) < 3:
                if len(batches_flat) == 1:
                    raise ValueError("Only one batch and it has fewer than 3 molecules.")
                if i == 0:
                    batches_flat[1] = batches_flat[0] + batches_flat[1]
                    batches_flat.pop(0)
                    # don't advance i; recheck index 0 (the merged batch)
                elif i == len(batches_flat) - 1:
                    batches_flat[i - 1] = batches_flat[i - 1] + batches_flat[i]
                    batches_flat.pop(i)
                    i -= 1
                else:
                    sparse_med = np.median([r[2] for r in batches_flat[i]])
                    prev_med   = np.median([r[2] for r in batches_flat[i - 1]])
                    next_med   = np.median([r[2] for r in batches_flat[i + 1]])
                    if abs(sparse_med - prev_med) <= abs(sparse_med - next_med):
                        batches_flat[i - 1] = batches_flat[i - 1] + batches_flat[i]
                        batches_flat.pop(i)
                        i -= 1  # recheck the merged batch
                    else:
                        batches_flat[i + 1] = batches_flat[i] + batches_flat[i + 1]
                        batches_flat.pop(i)
            else:
                i += 1

        rng = default_rng(seed)

        train_batches: list = []
        val_batches:   list = []
        test_batches:  list = []

        for batch in batches_flat:
            idx = rng.permutation(len(batch))
            n   = len(batch)

            n_trn = max(1, int(np.floor(trn_p * n)))
            # remainder goes to whichever of val/test is larger; tie → test
            if val_p > tst_p:
                n_tst = max(1, int(np.floor(tst_p * n)))
                n_val = max(1, n - n_trn - n_tst)
            else:
                n_val = max(1, int(np.floor(val_p * n)))
                n_tst = max(1, n - n_trn - n_val)

            train_batches.append([batch[j] for j in idx[:n_trn]])
            val_batches  .append([batch[j] for j in idx[n_trn:n_trn + n_val]])
            test_batches .append([batch[j] for j in idx[n_trn + n_val:n_trn + n_val + n_tst]])

        self._train = BatchSet(hdf5_db, enc, [b for b in train_batches if b])
        self._val   = BatchSet(hdf5_db, enc, [b for b in val_batches   if b])
        self._test  = BatchSet(hdf5_db, enc, [b for b in test_batches  if b])
    
    def get_sets(self) -> Tuple[BatchSet, BatchSet, BatchSet]:
        return self._train, self._val, self._test 
    
        