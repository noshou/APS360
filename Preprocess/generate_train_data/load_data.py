import os
import sys
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from itertools import islice

import h5py
import hdf5plugin  # pip install hdf5plugin
import numpy as np
from tqdm import tqdm

from Preprocess.generate_train_data.formfact import FormFactors
from Preprocess.generate_train_data.molecule import Molecule

# Per-worker globals set once by _worker_init: avoids pickling ff/lMax
# per task.
_ff: FormFactors | None = None
_lMax: int | None = None


def _worker_init(ff: FormFactors, lMax: int) -> None:
    """Initialize per-worker globals so `ff`/`lMax` aren't repickled per task.

    Parameters
    ----------
    ff : FormFactors
        Precomputed form factors, shared by all tasks in this worker process.
    lMax : int
        Maximum spherical harmonic degree used by `_process_mol`.

    Returns
    -------
    None
    """
    global _ff, _lMax
    _ff = ff
    _lMax = lMax


def _process_mol(xyz_path: str) -> tuple[str, np.ndarray, Molecule] | None:
    """Load one XYZ file and compute its Stuhrmann-decomposed intensity.

    Parameters
    ----------
    xyz_path : str
        Path to the molecule's .xyz file.

    Returns
    -------
    tuple of (str, numpy.ndarray, Molecule) or None
        ``(stem, I_q, mol)`` where `stem` is the filename without extension
        and directory, `I_q` is the orientationally-averaged scattering
        intensity, and `mol` is the parsed `Molecule`. Returns None if
        parsing or the Stuhrmann computation raises any exception.
    """
    try:
        assert _ff is not None and _lMax is not None
        mol = Molecule.fromXYZ(xyz_path)
        I_q, _ = mol.stuhrmann(_ff, _lMax)
    except Exception as _e:
        # Log first occurrence of each unique error type to stderr for
        # diagnosis.
        _key = type(_e).__name__ + ":" + str(_e)[:120]
        if not hasattr(_process_mol, "_seen"):
            _process_mol._seen = set()  # type: ignore[attr-defined]
        if _key not in _process_mol._seen:  # type: ignore
            _process_mol._seen.add(_key)  # type: ignore
            import traceback as _tb

            print(
                f"[_process_mol] FAIL {os.path.basename(xyz_path)}: {_key}",
                flush=True,
            )
            _tb.print_exc()
        return None
    return os.path.basename(xyz_path)[:-4], I_q, mol


def buildDB(
    groups: dict[str, str],
    ff: FormFactors,
    lMax: int,
    out_path: str,
    sources_tsv: str | None = None,
    makeup_tsv: str | None = None,
    workers: int | None = None,
    atom_range: tuple[int, int] | None = None,
) -> None:
    """Compute Stuhrmann decompositions for all XYZ files, write to HDF5.

    Molecule computation is parallelized across `workers` processes;
    HDF5 writes are serialized on the main process (h5py is not
    concurrency-safe for writes).

    B_lm coefficients are computed internally by the Stuhrmann
    decomposition but discarded. The NN predicts I(q) directly, using
    B_lm as a physics-structured intermediate learned during training,
    not as a supervised target.

    HDF5 layout:
        /attrs:                 lMax (int),     energy (float)
        /q_grid:                float64 (Q,)    momentum transfer grid, Å⁻¹
        /sources_tsv:           uint8   (N,)    raw bytes of sources_tsv
        /makeup_tsv:            uint8   (M,)    raw bytes of makeup_tsv
        /<g>/<s>.attrs[name]:   str             molecule name from XYZ header
        /<g>/<s>/I_q:           float32 (Q,)    orientationally-avg. intensity
        /<g>/<s>/coords:        float64 (n, 3)  centroid-subtracted coords, Å
        /<g>/<s>/angles:        float64 (n, 2)  spherical angles: col 0 =
                                                theta (0→π), col 1 = phi (0→2π)
        /<g>/<s>/r:             float64 (n,)    radial dist from centroid, Å
        /<g>/<s>/elms:          str     (n,)    element symbol per atom
                                                (variable-length UTF-8)

    All floating-point datasets are compressed with ZFP lossless
    compression; the uint8 TSV blobs use Bitshuffle+Zstd (ZFP is
    float-only). `elms` is stored uncompressed as a variable-length
    string dataset.

    Parameters
    ----------
    groups : dict of str to str
        Dict mapping group name to directory of .xyz files.
    ff : FormFactors
        Precomputed FormFactors; must contain all ions present in the
        XYZ files.
    lMax : int
        Maximum spherical harmonic degree. Runtime scales as
        O((lMax+1)^2) per molecule.
    out_path : str
        Output path for the HDF5 file.
    sources_tsv : str or None, optional
        Optional path to a provenance TSV to embed verbatim in the database.
    makeup_tsv : str or None, optional
        Optional path to the ion makeup TSV to embed verbatim in the database.
    workers : int or None, optional
        Number of worker processes. Defaults to os.cpu_count().

    Returns
    -------
    None
    """

    # ZFP lossless for floating-point data; bitshuffle+Zstd for metadata
    # (uint8 not ZFP-compatible).
    data_compress = hdf5plugin.Zfp(reversible=True)  # type: ignore[attr-defined]
    meta_compress = hdf5plugin.Bitshuffle(  # type: ignore[attr-defined]
        cname="zstd", clevel=22
    )

    # Integrity check: if the file exists but is corrupt, auto-recover by
    # copying readable groups to a new file before proceeding.
    if os.path.exists(out_path):
        try:
            with h5py.File(out_path, "r") as _chk:
                _ = list(_chk.keys())
                # Also probe group-level keys to catch metadata corruption
                # that only surfaces when iterating below root level.
                for _gname in groups:
                    if _gname in _chk:
                        _ = list(_chk[_gname].keys())  # type: ignore
        except Exception as _e:
            # "already open for write" means HDF5 consistency flag was
            # left set by an abrupt kill. h5clear fixes it without
            # touching data.
            if any(
                s in str(_e)
                for s in (
                    "already open for write",
                    "file consistency",
                    "bad object header version",
                )
            ):
                import platform
                import subprocess

                print(
                    "[buildDB] Clearing HDF5 consistency flags on "
                    f"{out_path}..."
                )
                if platform.system() == "Windows":
                    # h5clear is not available natively on Windows; run
                    # via WSL.
                    wsl_path = out_path.replace("\\", "/").replace(
                        "C:", "/mnt/c"
                    )
                    subprocess.run(
                        [
                            "wsl",
                            "-d",
                            "Ubuntu-24.04",
                            "--",
                            "h5clear",
                            "-s",
                            wsl_path,
                        ],
                        check=True,
                    )
                else:
                    subprocess.run(["h5clear", "-s", out_path], check=True)
                print("[buildDB] Flags cleared, proceeding normally.")
            else:
                print(
                    f"[buildDB] WARNING: {out_path} is corrupt ({_e}). "
                    "Auto-recovering..."
                )
                _tmp = out_path + ".recovering.h5"
                _bad = out_path + ".corrupt.h5"
                try:
                    with (
                        h5py.File(out_path, "r") as _src,
                        h5py.File(_tmp, "w") as _dst,
                    ):
                        for _key in ("q_grid", "sources_tsv", "makeup_tsv"):
                            try:
                                _src.copy(_key, _dst)
                            except Exception:
                                pass
                        for _attr in ("lMax", "energy"):
                            if _attr in _src.attrs:
                                _dst.attrs[_attr] = _src.attrs[_attr]
                        for _gname in groups:
                            _dst.create_group(_gname)
                            _ok = 0
                            try:
                                for _mol in _src[_gname]:  # type: ignore
                                    try:
                                        _src.copy(
                                            f"{_gname}/{_mol}", _dst[_gname]
                                        )
                                        _ok += 1
                                    except Exception:
                                        pass
                            except Exception:
                                pass
                            print(f"[buildDB]   recovered {_gname}: {_ok}")
                except Exception as _e2:
                    print(
                        f"[buildDB]   recovery failed ({_e2}), starting fresh"
                    )
                    if os.path.exists(_tmp):
                        os.remove(_tmp)
                    _tmp = None
                if _tmp and os.path.exists(_tmp):
                    os.rename(out_path, _bad)
                    os.rename(_tmp, out_path)
                    print(f"[buildDB]   corrupt file saved as {_bad}")
                else:
                    os.rename(out_path, _bad)
                    print(
                        "[buildDB]   starting fresh "
                        f"(corrupt file saved as {_bad})"
                    )

    with h5py.File(out_path, "a", libver="latest") as hf:
        hf.attrs["lMax"] = lMax
        hf.attrs["energy"] = ff.energy

        if "q_grid" not in hf:
            hf.create_dataset(
                "q_grid", data=ff.qvals, chunks=True, **data_compress
            )

        if sources_tsv is not None and "sources_tsv" not in hf:
            with open(sources_tsv, "rb") as f:
                hf.create_dataset(
                    "sources_tsv",
                    data=np.frombuffer(f.read(), dtype=np.uint8),
                    chunks=True,
                    **meta_compress,
                )

        if makeup_tsv is not None and "makeup_tsv" not in hf:
            with open(makeup_tsv, "rb") as f:
                hf.create_dataset(
                    "makeup_tsv",
                    data=np.frombuffer(f.read(), dtype=np.uint8),
                    chunks=True,
                    **meta_compress,
                )

        def _atom_count(path: str) -> int | None:
            try:
                with open(path) as _f:
                    return int(_f.readline())
            except (ValueError, OSError):
                return None

        all_xyz = {}
        for gname, gdir in groups.items():
            paths = [
                e.path for e in os.scandir(gdir) if e.name.endswith(".xyz")
            ]
            if atom_range is not None:
                lo, hi = atom_range
                paths = [
                    p
                    for p in paths
                    if (n := _atom_count(p)) is not None and lo <= n <= hi
                ]
            all_xyz[gname] = paths
        total = sum(len(v) for v in all_xyz.values())

        _TMP = "__tmp__"
        _Q = len(ff.qvals)
        _written = 0
        _LOG_EVERY = 10_000

        _n_workers = workers or os.cpu_count() or 1
        _inflight = _n_workers * 4

        def _make_pool():
            return ProcessPoolExecutor(
                max_workers=_n_workers,
                initializer=_worker_init,
                initargs=(ff, lMax),
            )

        pool = _make_pool()
        try:
            with tqdm(
                total=total,
                unit="mol",
                file=sys.stderr,
                dynamic_ncols=True,
                bar_format=(
                    "{l_bar}{bar}| {n_fmt}/{total_fmt} "
                    "[{elapsed}, {rate_fmt}]{postfix}"
                ),
                miniters=100,
                mininterval=2.0,
            ) as pbar:
                for group_name, xyz_paths in all_xyz.items():
                    hf_group = hf.require_group(group_name)

                    # Drop half-written tmp entries and any corrupt
                    # committed entries.
                    already_done = set()
                    for key in list(hf_group.keys()):
                        if key.startswith(_TMP):
                            del hf_group[key]
                            continue
                        try:
                            mol_grp = hf_group[key]
                            for ds in ("I_q", "coords", "angles", "r", "elms"):
                                mol_grp[ds][()]  # type: ignore
                            already_done.add(key)
                        except Exception:
                            del hf_group[key]

                    pending = [
                        p
                        for p in xyz_paths
                        if os.path.basename(p)[:-4] not in already_done
                    ]
                    pbar.update(len(xyz_paths) - len(pending))

                    # Sliding window: keep at most _inflight tasks queued
                    # at once, to avoid queuing all 1.3M futures and
                    # their results in memory.
                    it = iter(pending)
                    futures = {
                        pool.submit(_process_mol, p): p
                        for p in islice(it, _inflight)
                    }

                    while futures:
                        # Block until at least one future finishes, then
                        # drain ALL currently-finished futures before
                        # refilling. Using wait() with FIRST_COMPLETED
                        # avoids rebuilding the waiter set on every
                        # single completion (the O(n²) trap of
                        # repeatedly calling next(as_completed(futures))).
                        try:
                            done, _ = wait(
                                futures, return_when=FIRST_COMPLETED
                            )
                        except Exception:
                            done = set()

                        for fut in done:
                            path = futures.pop(fut)
                            try:
                                nxt = next(it, None)
                                if nxt is not None:
                                    futures[pool.submit(_process_mol, nxt)] = (
                                        nxt
                                    )
                            except BrokenProcessPool:
                                pbar.write(
                                    f"[buildDB] pool broken after {path}, "
                                    "restarting..."
                                )
                                hf.flush()
                                pool.shutdown(wait=False)
                                pool = _make_pool()
                                # Re-queue remaining inflight paths
                                # (excluding the crashed one)
                                remaining = list(futures.values())
                                futures = {}
                                for p in remaining:
                                    futures[pool.submit(_process_mol, p)] = p
                                if nxt is not None: # pyright: ignore[reportPossiblyUnboundVariable]
                                    futures[pool.submit(_process_mol, nxt)] = ( # type: ignore
                                        nxt
                                    ) # type: ignore
                                break  # restart while loop with new pool

                            try:
                                result = fut.result()
                            except BrokenProcessPool:
                                pbar.write(
                                    "[buildDB] pool broken getting "
                                    "result, restarting..."
                                )
                                hf.flush()
                                pool.shutdown(wait=False)
                                pool = _make_pool()
                                remaining = list(futures.values())
                                futures = {}
                                for p in remaining:
                                    futures[pool.submit(_process_mol, p)] = p
                                break
                            except Exception:
                                pbar.update()
                                continue

                            if result is None:
                                pbar.update()
                                continue
                            stem, I_q, mol = result
                            pbar.set_postfix(
                                g=group_name[:20], m=stem[:18], refresh=False
                            )
                            pbar.update()

                            tmp_name = f"{_TMP}{stem}"
                            try:
                                tmp = hf_group.create_group(tmp_name)
                                tmp.create_dataset(
                                    "I_q",
                                    data=I_q,
                                    chunks=True,
                                    **data_compress,
                                )
                                mol.toHDF5(tmp)
                                # Read back chunk data to verify it
                                # landed before committing.
                                _n = tmp["coords"].shape[0]  # type: ignore
                                assert (
                                    tmp["I_q"][()].shape == (_Q,)  # type: ignore
                                    and tmp["coords"][()].shape == (_n, 3)  # type: ignore
                                    and tmp["angles"][()].shape == (_n, 2)  # type: ignore
                                    and tmp["r"][()].shape == (_n,)  # type: ignore
                                    and len(tmp["elms"][()]) == _n # type: ignore
                                )  # type: ignore
                                hf.move(
                                    f"{group_name}/{tmp_name}",
                                    f"{group_name}/{stem}",
                                )
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
        finally:
            pool.shutdown(wait=False)
