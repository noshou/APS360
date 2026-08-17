import bisect
import hashlib
import re
from typing import Dict, List, Tuple

import h5py
import numpy as np
from numpy.random import default_rng

from Preprocess import Encoding

from .batch_set import BatchSet

_TRAIN = 0.7
_VAL = 0.15
_TEST = 0.15

# Groups where a numeric suffix in the stem is a genuine family axis (shell
# size, cluster size, isomer index, stoichiometry) rather than an arbitrary
# catalog id. _stem_family_prefix() (stem text before the first digit) is
# only safe to apply here - on an id/catalog group like COD or rcsb (e.g.
# "COD_0", "COD_1", ...) the same rule would collapse the ENTIRE group into
# one family, since the catalog number sits right after a fixed prefix.
_FAMILY_PREFIX_GROUPS = frozenset(
    {
        "hydration_shells",
        "ar_ne_clusters",
        "(NaCl)_nCl-",
        "si_ge_clusters",
        "binary_clusters",
        "(Na,Co,Ag,Pb,Mo,Fe)_monoatomic_clusters",
    }
)

_FIRST_DIGIT_RE = re.compile(r"\d")


class _UnionFind:
    """Disjoint-set over an arbitrary hashable universe, path-compressed."""

    def __init__(self, items) -> None:  # noqa: ANN001
        self._parent = {x: x for x in items}

    def find(self, x):  # noqa: ANN001, ANN201
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[x] != root:
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, a, b) -> None:  # noqa: ANN001
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[ra] = rb


def _stem_family_prefix(stem: str) -> str:
    """Stem text before its first digit; the whole stem if it has none."""
    m = _FIRST_DIGIT_RE.search(stem)
    return stem if m is None else stem[: m.start()]


class Batcher:
    """Builds train, validation, and test BatchSets from an HDF5 dataset.

    Call ``get_sets()`` to retrieve ``(train, val, test)``
    as ``BatchSet`` objects ready for use with
    ``DataLoader(batch_size=1, collate_fn=lambda x: x[0])``.

    Construction proceeds in three steps:

    1. ``_query_buckets``: for each (min_atoms, max_atoms) size bucket,
        queries the HDF5 metadata for matching molecules sorted by atom
        count.

    2. ``_family_split_assignment``: groups molecules that would leak
        across train/val/test if split independently - exact-composition
        duplicates (same element sequence in the same source group, e.g.
        re-deposited/redetermined structures) via a DB lookup, plus
        (for a fixed allowlist of parametric-family datasets only, see
        ``_FAMILY_PREFIX_GROUPS``) same stem-prefix-before-first-digit
        series (e.g. varying cluster/shell size) - then assigns each
        whole family to exactly one of train/val/test via a seeded
        shuffle + greedy deficit fill, so no family straddles a split.

    3. Per bucket, per split: rows are filtered to that split via the
        family assignment, then chunked to <= ``atom_size_ceil`` total
        atoms per batch (binary search on a prefix-sum array, same as
        before). Batches with <= 3 molecules are collapsed into a
        neighbouring batch (nearest median atom count, ties merge down)
        WITHIN that split's own batch list, so a family's single-split
        guarantee is never broken by merging.
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

        bucket_queries = self._query_buckets(hdf5_db, enc, batches)

        # Dedupe across buckets (a molecule can match >1 configured bucket
        # if their ranges overlap) before computing family/split
        # assignment - the assignment is per (grp, stem), not per query hit.
        unique_rows: Dict[Tuple[str, str], int] = {}
        for query in bucket_queries:
            for grp, stem, atoms in query:
                unique_rows[(grp, stem)] = atoms

        split_of = self._family_split_assignment(
            enc, unique_rows, seed, train_percent, val_percent, test_percent
        )

        train_flat: list = []
        val_flat: list = []
        test_flat: list = []
        for query in bucket_queries:
            per_split: Dict[str, list] = {"train": [], "val": [], "test": []}
            for row in query:
                per_split[split_of[(row[0], row[1])]].append(row)
            for label, dest in (
                ("train", train_flat),
                ("val", val_flat),
                ("test", test_flat),
            ):
                rows = per_split[label]
                if not rows:
                    continue
                sub_result: list = []
                self._split_batch(
                    0, len(rows) - 1, rows, atom_size_ceil, sub_result
                )
                dest.extend(sub_result)

        self._collapse_tiny_batches(train_flat)
        self._collapse_tiny_batches(val_flat)
        self._collapse_tiny_batches(test_flat)

        self.__train = BatchSet(hdf5_db, enc, train_flat)
        self.__val = BatchSet(hdf5_db, enc, val_flat)
        self.__test = BatchSet(hdf5_db, enc, test_flat)

    def _query_buckets(
        self,
        hdf5_db: str,
        enc: Encoding,
        batches: list[tuple[int, int]],
    ) -> list:
        """Query each configured size bucket, unsplit by atom_size_ceil.

        Parameters
        ----------
        hdf5_db : str
            Path to raw HDF5 data.
        enc : Encoding
            Encoding instance for SQLite queries.
        batches : list of tuple of int
            List of (min_atoms, max_atoms) size buckets.

        Returns
        -------
        list
            One list per configured bucket, each a list of (grp, stem,
            atoms) rows ordered by atom count ascending.

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
        for i, (min_a, max_a) in enumerate(batches):
            if min_a > max_a:
                print(f"Invalid bucket {i} ({min_a}, {max_a}): skipping...")
                continue
            result.append(enc.get_in_range(min_a, max_a))
        return result

    @staticmethod
    def _split_batch(
        lo: int,
        hi: int,
        query: list,
        atom_size_ceil: int,
        result: list,
    ) -> None:
        """Recursively bisect ``query[lo:hi+1]`` under the atom ceiling.

        Parameters
        ----------
        lo : int
            Inclusive start index into ``query``.
        hi : int
            Inclusive end index into ``query``.
        query : list
            Full list of (grp, stem, atoms) rows being split; only the
            ``[lo, hi]`` slice is considered on this call.
        atom_size_ceil : int
            Maximum total atoms per resulting batch.
        result : list
            Appended to in place with each resulting sub-list.
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
        Batcher._split_batch(lo, lo + mid, query, atom_size_ceil, result)
        Batcher._split_batch(lo + mid + 1, hi, query, atom_size_ceil, result)

    @staticmethod
    def _collapse_tiny_batches(batches_flat: list) -> None:
        """Merge any batch with <= 3 molecules into a neighbour, in place.

        Merge towards whichever neighbour's median atom count is closer;
        tie or edge case merges down (into the previous batch).

        Parameters
        ----------
        batches_flat : list
            Flat list of batches (each a list of (grp, stem, atoms) rows),
            all belonging to the SAME split. Mutated in place.

        Raises
        ------
        ValueError
            If only one batch remains and it has fewer than 3 molecules.
        """
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

    def _family_split_assignment(
        self,
        enc: Encoding,
        rows: Dict[Tuple[str, str], int],
        seed: int,
        trn_p: float,
        val_p: float,
        tst_p: float,
    ) -> Dict[Tuple[str, str], str]:
        """Assign every (grp, stem) to train/val/test, one split per family.

        A "family" is a connected component under two union signals:
        exact-composition duplicates (same element sequence, same source
        group - e.g. a structure re-deposited/redetermined under a
        different id) and, for ``_FAMILY_PREFIX_GROUPS`` only, same
        stem-prefix-before-first-digit (a parametric series: shell size,
        cluster size, isomer, stoichiometry). Families are then shuffled
        with a seeded RNG and greedily filled toward the target train/
        val/test proportions (by molecule count), so no family straddles
        a split - fixing the leak where a duplicate/family member the
        model has effectively already seen in train inflates val/test
        pct_err's apparent generalization.

        Parameters
        ----------
        enc : Encoding
            Encoding instance for the composition-duplicate DB lookup.
        rows : dict
            Mapping (grp, stem) -> atoms, covering every molecule to be
            split (deduped across overlapping buckets).
        seed : int
            RNG seed for the family shuffle.
        trn_p, val_p, tst_p : float
            Target train/val/test proportions by molecule count.

        Returns
        -------
        dict
            Mapping (grp, stem) -> "train" | "val" | "test".
        """
        keys = list(rows.keys())
        uf = _UnionFind(keys)

        # Signal 1: exact-composition duplicates. Only candidates - rows
        # sharing (grp, atoms) with >= 2 members - can possibly match, so
        # VOCAB_idx is fetched only for those, not the whole dataset.
        by_grp_atoms: Dict[Tuple[str, int], list] = {}
        for grp, stem in keys:
            by_grp_atoms.setdefault((grp, rows[(grp, stem)]), []).append(
                (grp, stem)
            )
        candidates = [
            k for grp_atoms in by_grp_atoms.values() for k in grp_atoms
            if len(grp_atoms) >= 2
        ]
        # Processed and hashed one chunk at a time, never holding more than
        # _CHUNK molecules' raw VOCAB_idx at once: on this dataset, atom
        # count coincidentally collides within a group for almost the
        # WHOLE dataset (candidates ~= all rows), so materializing every
        # candidate's full element-sequence list simultaneously is
        # multiple GB (SUM(atoms) ~= 680M ints) - a 16-byte digest per row
        # is what actually stays resident.
        if candidates:
            by_composition: Dict[Tuple[str, bytes], Tuple[str, str]] = {}
            _CHUNK = 2000
            for i in range(0, len(candidates), _CHUNK):
                chunk_vocab = enc.get_vocab_for_keys(
                    candidates[i : i + _CHUNK]
                )
                for (grp, stem), v in chunk_vocab.items():
                    digest = hashlib.blake2b(
                        bytes(sorted(v)), digest_size=16
                    ).digest()
                    sig = (grp, digest)
                    first = by_composition.get(sig)
                    if first is None:
                        by_composition[sig] = (grp, stem)
                    else:
                        uf.union(first, (grp, stem))
                del chunk_vocab

        # Signal 2: parametric-family stem series, allowlisted groups only.
        by_prefix: Dict[Tuple[str, str], Tuple[str, str]] = {}
        for grp, stem in keys:
            if grp not in _FAMILY_PREFIX_GROUPS:
                continue
            sig = (grp, _stem_family_prefix(stem))
            first = by_prefix.get(sig)
            if first is None:
                by_prefix[sig] = (grp, stem)
            else:
                uf.union(first, (grp, stem))

        families: Dict[Tuple[str, str], list] = {}
        for key in keys:
            families.setdefault(uf.find(key), []).append(key)
        family_list = list(families.values())

        rng = default_rng(seed)
        order = rng.permutation(len(family_list))

        n_total = len(keys)
        target = {
            "train": trn_p * n_total,
            "val": val_p * n_total,
            "test": tst_p * n_total,
        }
        count = {"train": 0, "val": 0, "test": 0}
        split_of: Dict[Tuple[str, str], str] = {}
        for idx in order:
            family = family_list[idx]
            # Largest-remaining-deficit bin packing: send the whole family
            # to whichever split is furthest below its target share.
            label = max(count, key=lambda s: target[s] - count[s])
            for key in family:
                split_of[key] = label
            count[label] += len(family)

        n_families_gt1 = sum(1 for f in family_list if len(f) > 1)
        print(
            f"  split: {n_total} molecules, {len(family_list)} families "
            f"({n_families_gt1} multi-member) -> "
            f"train {count['train']} val {count['val']} test {count['test']}"
        )
        return split_of

    def get_sets(self) -> Tuple[BatchSet, BatchSet, BatchSet]:
        """Return the pre-built train, validation, and test BatchSets.

        Returns
        -------
        tuple of BatchSet
            A ``(train, val, test)`` tuple of ``BatchSet``, used with
            ``DataLoader(batch_size=1, collate_fn=lambda x: x[0])``.
        """
        return self.__train, self.__val, self.__test
