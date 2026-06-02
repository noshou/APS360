import numpy as np
import numpy.typing as npt
import pandas as pd
import h5py, os
import hdf5plugin  # pip install hdf5plugin
from beartype import beartype
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from Scattering.formfact import FormFactors
from Scattering.molecule import Molecule

# Per-worker globals set once by _worker_init — avoids pickling ff/lMax per task.
_ff:   FormFactors | None = None
_lMax: int | None         = None

def _worker_init(ff: FormFactors, lMax: int) -> None:
    global _ff, _lMax
    _ff   = ff
    _lMax = lMax

def _process_mol(xyz_path: str) -> tuple[str, npt.NDArray[np.float64], npt.NDArray[np.complex64]] | None:
    assert _ff is not None and _lMax is not None
    mol = Molecule.fromXYZ(xyz_path)
    try:
        I_q, B_lm = mol.stuhrmann(_ff, _lMax)
    except ValueError:
        return None
    return os.path.basename(xyz_path)[:-4], I_q, B_lm

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
    workers: int | None     = None,
    ) -> None:
    
    """
    Compute Stuhrmann decompositions for all XYZ files and write to an HDF5 database.

    Molecule computation is parallelized across `workers` processes; HDF5 writes
    are serialized on the main process (h5py is not concurrency-safe for writes).

    HDF5 layout:
        /attrs:          lMax (int), energy (float)
        /q_grid:         float64 (Q,)  — momentum transfer grid in Å⁻¹
        /sources_tsv:    uint8 raw bytes of sources_tsv, if provided
        /<group>/<stem>/I_q:   float64    (Q,)           — scattering intensity
        /<group>/<stem>/B_lm:  complex128 ((lMax+1)², Q) — partial wave amplitudes

    All datasets are compressed using Zstd level 22 (maximum).

    Args:
        groups:      Dict mapping group name to directory of .xyz files.
        ff:          Precomputed FormFactors; must contain all ions present in the XYZ files.
        lMax:        Maximum spherical harmonic degree. Runtime scales as O((lMax+1)²) per molecule.
        out_path:    Output path for the HDF5 file.
        sources_tsv: Optional path to a provenance TSV to embed verbatim in the database.
        workers:     Number of worker processes. Defaults to os.cpu_count().
    """
    
    # ZFP lossless for floating-point data; bitshuffle+Zstd for metadata (uint8 not ZFP-compatible).
    data_compress = hdf5plugin.Zfp(reversible=True)               # type: ignore[attr-defined]
    meta_compress = hdf5plugin.Bitshuffle(cname='zstd', clevel=22) # type: ignore[attr-defined]

    with h5py.File(out_path, 'a') as hf:

        hf.attrs['lMax']              = lMax
        hf.attrs['energy']            = ff.energy
        hf.attrs['conjugate_symmetry'] = True

        if 'q_grid' not in hf:
            hf.create_dataset(
                'q_grid',
                data=ff.qvals,
                chunks=True,
                **data_compress
                )

        if sources_tsv is not None and 'sources_tsv' not in hf:
            with open(sources_tsv, 'rb') as f:
                hf.create_dataset(
                    'sources_tsv',
                    data=np.frombuffer(f.read(), dtype=np.uint8),
                    chunks=True,
                    **meta_compress
                    )

        all_xyz = {
            gname: [e.path for e in os.scandir(gdir) if e.name.endswith('.xyz')]
            for gname, gdir in groups.items()
        }
        total = sum(len(v) for v in all_xyz.values())

        _TMP      = '__tmp__'
        _Q        = len(ff.qvals)
        _LM       = (lMax + 1) * (lMax + 2) // 2  # reduced modes (m ≥ 0 only)
        _chunk    = (_LM, _Q)

        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_worker_init,
            initargs=(ff, lMax),
        ) as pool, tqdm(total=total, unit='mol') as pbar:
            for group_name, xyz_paths in all_xyz.items():
                hf_group = hf.require_group(group_name)

                # Drop any half-written entries left by a previous crash.
                for key in list(hf_group.keys()):
                    if key.startswith(_TMP):
                        del hf_group[key]

                already_done = set(hf_group.keys())
                pending = [p for p in xyz_paths
                           if os.path.basename(p)[:-4] not in already_done]
                pbar.update(len(xyz_paths) - len(pending))
                futures = {pool.submit(_process_mol, p): p for p in pending}

                for fut in as_completed(futures):
                    result = fut.result()
                    if result is None:
                        pbar.update()
                        continue
                    stem, I_q, B_lm = result
                    pbar.set_postfix(group=group_name, mol=stem, refresh=False)
                    pbar.update()

                    tmp_name = f'{_TMP}{stem}'
                    try:
                        tmp = hf_group.create_group(tmp_name)
                        tmp.create_dataset(
                            'I_q',
                            data=I_q,
                            chunks=True,
                            **data_compress
                        )
                        tmp.create_dataset(
                            'B_lm_r',
                            data=B_lm.real,
                            chunks=_chunk,
                            **data_compress
                        )
                        tmp.create_dataset(
                            'B_lm_i',
                            data=B_lm.imag,
                            chunks=_chunk,
                            **data_compress
                        )
                        # Verify shapes written cleanly before committing.
                        iq_ds = tmp['I_q']
                        br_ds = tmp['B_lm_r']
                        bi_ds = tmp['B_lm_i']
                        assert isinstance(iq_ds, h5py.Dataset) and iq_ds.shape == (_Q,)
                        assert isinstance(br_ds, h5py.Dataset) and br_ds.shape == (_LM, _Q)
                        assert isinstance(bi_ds, h5py.Dataset) and bi_ds.shape == (_LM, _Q)
                        hf.move(f'{group_name}/{tmp_name}',
                                f'{group_name}/{stem}')
                    except Exception:
                        if tmp_name in hf_group:
                            del hf_group[tmp_name]
                        raise

