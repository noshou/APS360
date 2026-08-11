import bisect

import h5py
import numpy as np
from beartype.typing import List, Tuple
from numpy.random import default_rng

from Preprocess import Encoding

from .batch_set import BatchSet

_TRAIN = 0.7
_VAL = 0.15
_TEST = 0.15


class Batcher:
    """Builds train, validation, and test BatchSets from an HDF5 dataset.

    Call ``get_sets()`` to retrieve ``(train, val, test)``
    as ``BatchSet`` objects ready for use with
    ``DataLoader(batch_size=1, collate_fn=lambda x: x[0])``.

    Construction proceeds in two steps:

    1. ``_batches_init``: for each (min_atoms, max_atoms) size bucket, queries
        the HDF5 metadata for matching molecules sorted by atom count, then
        splits any bucket whose total atom count exceeds ``atom_size_ceil`` to
        sub-lists via binary search on a prefix-sum array. Produces a list of
        batches, each a list of (grp, stem, atoms) rows whose combined atom
        count is <= ``atom_size_ceil``.

    2. ``_batches_stratify``: collapses any batch with <= 3 molecules into
        its nearest neighbour (by median atom count, ties go down), then splits
        batch independently (default: 70/15/15) at the molecule level.
        Wraps the three resulting lists in ``BatchSet`` instances.
    """

    __train: "BatchSet"
    __val: "BatchSet"
    __test: "BatchSet"

    def __init__(
        self,
        hdf5_db: str,
        enc: Encoding,
        batches: list[tuple[int, int]],
        seed: int,
        atom_size_ceil: int = -1,
        train_percent: float = _TRAIN,
        val_percent: float = _VAL,
        test_percent: float = _TEST,
    ):
        """Construct train, validation, and test BatchSets from an HDF5 dataset

        Parameters
        ----------
        hdf5_db : str
            Path to raw HDF5 data.
        enc : Encoding
            Encoding instance for SQLite queries.
        batches : list of tuple of int
            List of (min_atoms, max_atoms) size buckets to load.
        seed : int
            RNG seed for reproducible train/val/test splits. Must be >= 0.
        atom_size_ceil : int, optional
            Maximum total atoms per batch; exceeded buckets are split via
            binary search. If <= 0, set to 3x the max atom count (default -1).
        train_percent : float, optional
            Fraction of molecules for training (default 0.7).
        val_percent : float, optional
            Fraction of molecules for validation (default 0.15).
        test_percent : float, optional
            Fraction of molecules for testing (default 0.15). The larger of
            val/test receives the floor-rounding remainder; ties go to test.

        Raises
        ------
        ValueError
            If any of ``train_percent``, ``val_percent``, ``test_percent`` is
            <= 0, if they do not sum to 1, or if ``seed`` is negative.
        """

        if (
            train_percent <= 0
            or val_percent <= 0
            or test_percent <= 0
            or (train_percent + val_percent + test_percent) != 1.0
        ):
            raise ValueError(
                "Invalid values for train, validation, testing, batch sizes"
            )
        elif seed < 0:
            raise ValueError("Invalid seed; must be >= 0")

        if atom_size_ceil <= 0:
            atom_size_ceil = 3 * enc.max_atom_count()

        all_batches = self._batches_init(hdf5_db, enc, batches, atom_size_ceil)
        self._batches_stratify(
            seed,
            train_percent,
            val_percent,
            test_percent,
            hdf5_db,
            enc,
            all_batches,
        )

    def _batches_init(
        self,
        hdf5_db: str,
        enc: Encoding,
        batches: list[tuple[int, int]],
        atom_size_ceil: int,
    ) -> list:
        """Build the flat list of batches across all size buckets.

        Parameters
        ----------
        hdf5_db : str
            Path to raw HDF5 data.
        enc : Encoding
            Encoding instance for SQLite queries.
        batches : list of tuple of int
            List of (min_atoms, max_atoms) size buckets.
        atom_size_ceil : int
            Maximum total atoms per batch.

        Returns
        -------
        list
            Flat list of batches, each a list of (grp, stem, atoms) rows.

        Raises
        ------
        FileNotFoundError
            If ``hdf5_db`` does not exist.
        PermissionError
            If ``hdf5_db`` cannot be read due to insufficient permissions.
        OSError
            If the HDF5 file cannot otherwise be opened.
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
            """Recursively bisect ``query[lo:hi+1]``
            until each part fits under the atom ceiling.

            Parameters
            ----------
            lo : int
                Inclusive start index into ``query``.
            hi : int
                Inclusive end index into ``query``.
            query : list
                Full list of (grp, stem, atoms) rows being split; only the
                ``[lo, hi]`` slice is considered on this call.
            """
            sub = query[lo : hi + 1]
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
            split_batch(lo, lo + mid, query)
            split_batch(lo + mid + 1, hi, query)

        for i, (min_a, max_a) in enumerate(batches):
            if min_a > max_a:
                print(f"Invalid bucket {i} ({min_a}, {max_a}): skipping...")
                continue
            query = enc.get_in_range(min_a, max_a)
            count = sum(row[2] for row in query)
            if count > atom_size_ceil:
                split_batch(0, len(query) - 1, query)
            else:
                result.append(query)

        return result

    def _batches_stratify(
        self,
        seed: int,
        trn_p: float,
        val_p: float,
        tst_p: float,
        hdf5_db: str,
        enc: Encoding,
        batches_flat: list,
    ):
        """Collapse tiny batches, split each into train/val/test,
        and store the resulting BatchSets.

        Batches with fewer than 3 molecules are merged into a neighbouring
        batch (by nearest median atom count, ties merge down), then each
        remaining batch is independently split at the molecule level using
        a seeded RNG. The results are wrapped into ``BatchSet`` instances and
        assigned to ``self.__train``, ``self.__val``, and ``self.__test``.

        Parameters
        ----------
        seed : int
            RNG seed for reproducible splits.
        trn_p : float
            Fraction of molecules per batch for training.
        val_p : float
            Fraction of molecules per batch for validation.
        tst_p : float
            Fraction of molecules per batch for testing.
        hdf5_db : str
            Path to raw HDF5 data, forwarded to ``BatchSet``.
        enc : Encoding
            Encoding instance for lazy per-batch VOCAB_idx lookups, forwarded
            to ``BatchSet``.
        batches_flat : list
            Flat list of batches (each a list of (grp, stem, atoms) rows)
            produced by ``_batches_init``. Mutated in place while merging
            undersized batches.

        Raises
        ------
        ValueError
            If only one batch remains and it has fewer than 3 molecules.
        """
        # Collapse batches with fewer than 3 molecules into an adjacent batch.
        # Merge towards neighbour whose median atom count is closer.
        # Tie or edge case: merge down (into previous).
        i = 0
        while i < len(batches_flat):
            if len(batches_flat[i]) < 3:
                if len(batches_flat) == 1:
                    raise ValueError(
                        "Only one batch and it has fewer than 3 molecules."
                    )
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
                    prev_med = np.median([r[2] for r in batches_flat[i - 1]])
                    next_med = np.median([r[2] for r in batches_flat[i + 1]])
                    if abs(sparse_med - prev_med) <= abs(
                        sparse_med - next_med
                    ):
                        batches_flat[i - 1] = (
                            batches_flat[i - 1] + batches_flat[i]
                        )
                        batches_flat.pop(i)
                        i -= 1  # recheck the merged batch
                    else:
                        batches_flat[i + 1] = (
                            batches_flat[i] + batches_flat[i + 1]
                        )
                        batches_flat.pop(i)
            else:
                i += 1

        rng = default_rng(seed)

        train_batches: list = []
        val_batches: list = []
        test_batches: list = []

        for batch in batches_flat:
            idx = rng.permutation(len(batch))
            n = len(batch)

            n_trn = max(1, int(np.floor(trn_p * n)))
            # remainder goes to whichever of val/test is larger; tie → test
            if val_p > tst_p:
                n_tst = max(1, int(np.floor(tst_p * n)))
                n_val = max(1, n - n_trn - n_tst)
            else:
                n_val = max(1, int(np.floor(val_p * n)))
                n_tst = max(1, n - n_trn - n_val)

            train_batches.append([batch[j] for j in idx[:n_trn]])
            val_batches.append([batch[j] for j in idx[n_trn : n_trn + n_val]])
            test_batches.append(
                [batch[j] for j in idx[n_trn + n_val : n_trn + n_val + n_tst]]
            )

        self.__train = BatchSet(hdf5_db, enc, [b for b in train_batches if b])
        self.__val = BatchSet(hdf5_db, enc, [b for b in val_batches if b])
        self.__test = BatchSet(hdf5_db, enc, [b for b in test_batches if b])

    def get_sets(self) -> Tuple[BatchSet, BatchSet, BatchSet]:
        """Return the pre-built train, validation, and test BatchSets.

        Returns
        -------
        tuple of BatchSet
            A ``(train, val, test)`` tuple of ``BatchSet``, used with
            ``DataLoader(batch_size=1, collate_fn=lambda x: x[0])``.
        """
        return self.__train, self.__val, self.__test
