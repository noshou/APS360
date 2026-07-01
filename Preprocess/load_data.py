import numpy as np
import pandas as pd
import h5py, os
import hdf5plugin  # pip install hdf5plugin
from beartype import beartype
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from itertools import islice
import sys
from tqdm import tqdm
from Scattering.formfact import FormFactors
from Scattering.molecule import Molecule

# Per-worker globals set once by _worker_init - avoids pickling ff/lMax per task.
_ff:   FormFactors | None = None
_lMax: int | None         = None

def _worker_init(ff: FormFactors, lMax: int) -> None:
    global _ff, _lMax
    _ff   = ff
    _lMax = lMax

def _process_mol(xyz_path: str) -> tuple[str, np.ndarray, Molecule] | None:
    assert _ff is not None and _lMax is not None
    try:
        mol = Molecule.fromXYZ(xyz_path)
        I_q, _ = mol.stuhrmann(_ff, _lMax)
    except Exception:
        return None
    return os.path.basename(xyz_path)[:-4], I_q, mol

@beartype
def loadFormFact(
    makeup_tsv: str,
    energy: int|float    = 1.25e4,
    qMax: int|float      = 0.5,
    qMin: int|float      = 0,
    step: float          = 0.01,
    col_name : str       = 'atom',
    log_path: str | None = None,
    ) -> FormFactors:

    """
    Builds a FormFactors object from a makeup TSV of ion symbols.

    Args:
        makeup_tsv: Path to a tab-separated file with a column of ion symbols (e.g. 'C', 'Fe2+').
        energy:     X-ray energy in eV. Default 12.5 keV.
        qMax:       Maximum momentum transfer in Å⁻¹.
        qMin:       Minimum momentum transfer in Å⁻¹.
        step:       q-grid step size in Å⁻¹.
        col_name:   Name of the column in makeup_tsv containing ion symbols.
        log_path:   Optional path for logging skipped ions; passed through to FormFactors.fromIons.

    Returns:
        FormFactors with precomputed complex form factors over the q grid.
    """

    df = pd.read_csv(makeup_tsv, sep='\t')
    ions = df[col_name].tolist()
    return FormFactors.fromIons(ions, float(energy), qMin, qMax, step, log_path=log_path)

def buildDB(
    groups: dict[str, str],
    ff: FormFactors,
    lMax: int,
    out_path: str,
    sources_tsv: str | None = None,
    makeup_tsv: str | None  = None,
    workers: int | None     = None,
    ) -> None:

    """
    Compute Stuhrmann decompositions for all XYZ files and write to an HDF5 database.

    Molecule computation is parallelized across `workers` processes; HDF5 writes
    are serialized on the main process (h5py is not concurrency-safe for writes).

    B_lm coefficients are computed internally by the Stuhrmann decomposition but
    discarded - the NN predicts I(q) directly, using B_lm as a physics-structured
    intermediate learned during training, not as a supervised target.

    HDF5 layout:
        /attrs:                      lMax (int), energy (float)
        /q_grid:                     float64 (Q,)   - momentum transfer grid in Å⁻¹
        /sources_tsv:                uint8   (N,)   - raw bytes of sources_tsv, if provided
        /makeup_tsv:                 uint8   (M,)   - raw bytes of makeup_tsv, if provided
        /<group>/<stem>.attrs[name]: str            - molecule name from XYZ header
        /<group>/<stem>/I_q:         float32 (Q,)   - orientationally-averaged scattering intensity
        /<group>/<stem>/coords:      float64 (n, 3) - centroid-subtracted Cartesian coordinates in Å
        /<group>/<stem>/angles:      float64 (n, 2) - spherical angles: col 0 = theta (0→π), col 1 = phi (0→2π)
        /<group>/<stem>/r:           float64 (n,)   - radial distances from centroid in Å
        /<group>/<stem>/elms:        str     (n,)   - element symbol per atom (variable-length UTF-8)

    All floating-point datasets are compressed with ZFP lossless compression; the
    uint8 TSV blobs use Bitshuffle+Zstd (ZFP is float-only). `elms` is stored
    uncompressed as a variable-length string dataset.

    Args:
        groups:      Dict mapping group name to directory of .xyz files.
        ff:          Precomputed FormFactors; must contain all ions present in the XYZ files.
        lMax:        Maximum spherical harmonic degree. Runtime scales as O((lMax+1)²) per molecule.
        out_path:    Output path for the HDF5 file.
        sources_tsv: Optional path to a provenance TSV to embed verbatim in the database.
        makeup_tsv:  Optional path to the ion makeup TSV to embed verbatim in the database.
        workers:     Number of worker processes. Defaults to os.cpu_count().
    """

    # ZFP lossless for floating-point data; bitshuffle+Zstd for metadata (uint8 not ZFP-compatible).
    data_compress = hdf5plugin.Zfp(reversible=True)                # type: ignore[attr-defined]
    meta_compress = hdf5plugin.Bitshuffle(cname='zstd', clevel=22) # type: ignore[attr-defined]

    with h5py.File(out_path, 'a', libver='latest') as hf:

        hf.attrs['lMax']   = lMax
        hf.attrs['energy'] = ff.energy

        if 'q_grid' not in hf:
            hf.create_dataset('q_grid', data=ff.qvals, chunks=True, **data_compress)

        if sources_tsv is not None and 'sources_tsv' not in hf:
            with open(sources_tsv, 'rb') as f:
                hf.create_dataset(
                    'sources_tsv',
                    data=np.frombuffer(f.read(), dtype=np.uint8),
                    chunks=True,
                    **meta_compress
                    )

        if makeup_tsv is not None and 'makeup_tsv' not in hf:
            with open(makeup_tsv, 'rb') as f:
                hf.create_dataset(
                    'makeup_tsv',
                    data=np.frombuffer(f.read(), dtype=np.uint8),
                    chunks=True,
                    **meta_compress
                    )

        all_xyz = {
            gname: [e.path for e in os.scandir(gdir) if e.name.endswith('.xyz')]
            for gname, gdir in groups.items()
        }
        total = sum(len(v) for v in all_xyz.values())

        _TMP       = '__tmp__'
        _Q         = len(ff.qvals)
        _written   = 0
        _LOG_EVERY = 10_000

        _n_workers = workers or os.cpu_count() or 1
        _inflight  = _n_workers * 4

        with ProcessPoolExecutor(
            max_workers=_n_workers,
            initializer=_worker_init,
            initargs=(ff, lMax),
        ) as pool, tqdm(
            total=total, unit='mol', file=sys.stderr, dynamic_ncols=True,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}, {rate_fmt}]{postfix}',
            miniters=100, mininterval=2.0,
        ) as pbar:
            for group_name, xyz_paths in all_xyz.items():
                hf_group = hf.require_group(group_name)

                # Drop half-written tmp entries and any corrupt committed entries.
                already_done = set()
                for key in list(hf_group.keys()):
                    if key.startswith(_TMP):
                        del hf_group[key]
                        continue
                    try:
                        mol_grp = hf_group[key]
                        for ds in ('I_q', 'coords', 'angles', 'r', 'elms'):
                            mol_grp[ds][()]
                        already_done.add(key)
                    except Exception:
                        del hf_group[key]

                pending = [p for p in xyz_paths
                            if os.path.basename(p)[:-4] not in already_done]
                pbar.update(len(xyz_paths) - len(pending))

                # Sliding window: keep at most _inflight tasks queued at once
                # to avoid queuing all 1.3M futures and their results in memory.
                it = iter(pending)
                futures = {pool.submit(_process_mol, p): p for p in islice(it, _inflight)}

                while futures:
                    # Block until at least one future finishes, then drain ALL
                    # currently-finished futures before refilling. Using wait()
                    # with FIRST_COMPLETED avoids rebuilding the waiter set on
                    # every single completion (the O(n²) trap of repeatedly
                    # calling next(as_completed(futures))).
                    done, _ = wait(futures, return_when=FIRST_COMPLETED)

                    for fut in done:
                        del futures[fut]
                        nxt = next(it, None)
                        if nxt is not None:
                            futures[pool.submit(_process_mol, nxt)] = nxt

                        result = fut.result()
                        if result is None:
                            pbar.update()
                            continue
                        stem, I_q, mol = result
                        pbar.set_postfix(g=group_name[:20], m=stem[:18], refresh=False)
                        pbar.update()

                        tmp_name = f'{_TMP}{stem}'
                        try:
                            tmp = hf_group.create_group(tmp_name)
                            tmp.create_dataset('I_q', data=I_q, chunks=True, **data_compress)
                            mol.toHDF5(tmp)
                            # Read back chunk data to verify it landed before committing.
                            _n = tmp['coords'].shape[0]
                            assert (tmp['I_q'][()].shape    == (_Q,)
                                and tmp['coords'][()].shape == (_n, 3)
                                and tmp['angles'][()].shape == (_n, 2)
                                and tmp['r'][()].shape      == (_n,)
                                and len(tmp['elms'][()])    == _n)
                            hf.move(f'{group_name}/{tmp_name}', f'{group_name}/{stem}')
                        except Exception:
                            if tmp_name in hf_group:
                                del hf_group[tmp_name]
                            raise

                        _written += 1
                        if _written % _LOG_EVERY == 0:
                            hf.flush()
                            size_gb = os.path.getsize(out_path) / 1e9
                            kb_per_mol = size_gb * 1e6 / _written
                            est_gb = kb_per_mol * total / 1e6
                            pbar.write(
                                f"[disk] {_written:,} written | "
                                f"{size_gb:.2f} GB on disk | "
                                f"{kb_per_mol:.1f} KB/mol | "
                                f"est. total {est_gb:.0f} GB"
                            )