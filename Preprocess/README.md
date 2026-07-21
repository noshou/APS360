# I(q)@L=50 Database

## Parameters


| Parameter | Value            |
| ----------- | ------------------ |
| energy    | 12 500 eV        |
| qMin      | 0 angstrom^-1    |
| qMax      | 0.5 angstrom^-1  |
| step      | 0.01 angstrom^-1 |
| Q         | 51 points        |
| lMax      | 50               |

## Files


| File                            | Size    | Description                                                                                                                                           |
| --------------------------------- | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `I(q)@L=50.h5`                  | ~66 GB  | HDF5 database of I(q) curves and molecular data                                                                                                       |
| `iq_train_set-ENCODING.sqlite3` | ~860 MB | Encoding index: maps every molecule to its atom count and VOCAB indices, so the data pipeline never needs to scan the 66 GB HDF5 file during training |
| `xyz_coordinate_files.7z`       | ~6.5 GB | Source XYZ geometry files for all molecule groups (LZMA2, max compression). Only needed to re-run the build pipeline from scratch.                    |

## Retrieving the dataset

The HDF5 files are hosted on **[HuggingFace (noshou/iq_train_set)](https://huggingface.co/datasets/noshou/iq_train_set)** and **[Kaggle (noso0s0n/iql50)](https://www.kaggle.com/datasets/noso0s0n/iql50)**. The training code (ScatterNet model, preprocessing pipeline, baselines) lives in the **[noshou/APS360](https://github.com/noshou/APS360)** GitHub repository; the `Preprocess/` directory contains the encoding and data pipeline code.

Download both the HDF5 file and the encoding DB with the HuggingFace CLI (recommended - resumes interrupted downloads):

```bash
pip install huggingface_hub
hf download noshou/iq_train_set "I(q)@L=50.h5" "iq_train_set-ENCODING.sqlite3" \
    --repo-type dataset --local-dir Preprocess/
```

Or in Python:

```python
from huggingface_hub import hf_hub_download
for filename in ["I(q)@L=50.h5", "iq_train_set-ENCODING.sqlite3"]:
    hf_hub_download(
        repo_id   = "noshou/iq_train_set",
        filename  = filename,
        repo_type = "dataset",
        local_dir = "Preprocess/",
    )
```

Both files are also available on the **[Kaggle dataset](https://www.kaggle.com/datasets/noso0s0n/iql50)** and are mounted directly as notebook inputs when using `kaggle_train.ipynb` / `kaggle_baselines.ipynb` -- no download step needed there.

Place the downloaded files at `Preprocess/I(q)@L=50.h5` and `Preprocess/iq_train_set-ENCODING.sqlite3` (the paths all pipeline scripts expect).

## Running training

### Local (CLI)

Edit `Train/train.yaml` to set paths, then:

```bash
python Train/train.py --config Train/train.yaml
```

Key paths in `train.yaml`:

```yaml
hdf5:                    Preprocess/I(q)@L=50.h5                          # downloaded above
encodings_sqlite3_path:  Preprocess/iq_train_set-ENCODING.sqlite3         # downloaded above
```

### Kaggle (notebook)

Open `Baselines/kaggle_baselines.ipynb`. Set `NOTEBOOK_NAME` to your Kaggle notebook slug at the top of the setup cell, and attach the [`noso0s0n/iql50`](https://www.kaggle.com/datasets/noso0s0n/iql50) dataset as a notebook input -- it provides both `I(q)@L=50.h5` and `iq_train_set-ENCODING.sqlite3` pre-mounted under `/kaggle/input/datasets/noso0s0n/iql50/`, no download or build step needed. The notebook clones the repo, installs dependencies, and runs all baselines.

---

Produced by `buildDB()` in `load_data.py`. The file is opened in append mode (`'a'`), so existing entries are skipped on resume.

## Root attributes


| Attribute | Type  | Description                            |
| ----------- | ------- | ---------------------------------------- |
| `lMax`    | int   | Maximum spherical harmonic degree used |
| `energy`  | float | X-ray energy in eV (e.g.`12500.0`)     |

## Root datasets


| Path           | dtype   | Shape  | Compression          | Description                                            |
| ---------------- | --------- | -------- | ---------------------- | -------------------------------------------------------- |
| `/q_grid`      | float64 | `(Q,)` | ZFP lossless         | Momentum transfer grid in angstrom^-1;`Q = len(qvals)` |
| `/sources_tsv` | uint8   | `(N,)` | Bitshuffle + Zstd-22 | Raw bytes of provenance TSV (optional)                 |
| `/makeup_tsv`  | uint8   | `(M,)` | Bitshuffle + Zstd-22 | Raw bytes of ion makeup TSV (optional)                 |

Both TSV datasets are written once and never overwritten on subsequent runs.

## Molecule data -- `/<group>/<stem>/`

Each `.xyz` file produces one HDF5 group nested two levels deep.

```
/<group_name>/
    <stem>.attrs['name']    str
    <stem>/
        I_q      float32  (Q,)
        coords   float64  (n, 3)
        angles   float64  (n, 2)
        r        float64  (n,)
        elms     str      (n,)
```


| Level     | Key            | Description                                                                                                        |
| ----------- | ---------------- | -------------------------------------------------------------------------------------------------------------------- |
| group     | `<group_name>` | Arbitrary label supplied via the`groups` dict argument                                                             |
| subgroup  | `<stem>`       | Filename without`.xyz` extension                                                                                   |
| attribute | `name`         | Molecule name string (from XYZ line 2)                                                                             |
| dataset   | `I_q`          | Orientationally-averaged scattering intensity, float32`(Q,)`, ZFP lossless                                         |
| dataset   | `coords`       | Centroid-subtracted Cartesian coordinates, float64`(n, 3)`, ZFP lossless                                           |
| dataset   | `angles`       | Spherical angles, float64`(n, 2)`: col 0 = theta (polar, 0 to pi), col 1 = phi (azimuthal, 0 to 2pi), ZFP lossless |
| dataset   | `r`            | Radial distances from centroid in angstroms, float64`(n,)`, ZFP lossless                                           |
| dataset   | `elms`         | Element symbol per atom, variable-length UTF-8 string`(n,)`, uncompressed                                          |

`Q` is the number of points in `/q_grid` and is fixed for the whole file. `n` varies per molecule.

Coordinates are centroid-subtracted (shifted to geometric centroid before storage). `angles` and `r` are stored pre-computed for fast loading; they are consistent with `coords` via:

```
r[i]     = norm(coords[i])
theta[i] = arccos(z[i] / r[i])   (0 if r = 0)
phi[i]   = arctan2(y[i], x[i])
```

Form factors are **not** stored -- they are recomputed from `xraydb`.

## Groups

The `groups` argument maps each group name to a directory of `.xyz` files. Every group becomes a top-level HDF5 group containing one subgroup per molecule.


| Group                                   |     Molecules | Atom range  | Description                   |
| ----------------------------------------- | --------------: | ------------- | ------------------------------- |
| COD                                     |       532,302 | 1-6,032     | Crystallography Open Database |
| QM9                                     |       133,844 | 3-29        | Small organic molecules       |
| tmQM                                    |       108,541 | 7-569       | Transition metal complexes    |
| rcsb_sml                                |        96,158 | 28-6,036    | PDB small structures          |
| viro3D                                  |        60,488 | 173-6,046   | Viral protein structures      |
| hydration_shells                        |        48,571 | 3-147       | Water solvation shells        |
| rcsb_med                                |        31,749 | 2,996-6,046 | PDB medium structures         |
| mofs                                    |        30,863 | 10-5,760    | Metal-organic frameworks      |
| (Na,Co,Ag,Pb,Mo,Fe)_monoatomic_clusters |         1,282 | 2-380       | Monoatomic clusters           |
| binary_clusters                         |           371 | 2-1,482     | Binary alloy clusters         |
| si_ge_clusters                          |           217 | 4-60        | Silicon/germanium clusters    |
| ar_ne_clusters                          |           127 | 2-55        | Noble gas clusters            |
| (NaCl)_nCl-                             |            70 | 3-71        | Sodium chloride clusters      |
| **TOTAL**                               | **1,044,583** | **1-6,046** |                               |

## Compression codecs


| Codec                            | Used for                                 | Notes                            |
| ---------------------------------- | ------------------------------------------ | ---------------------------------- |
| ZFP lossless (`reversible=True`) | `q_grid`, `I_q`, `coords`, `angles`, `r` | Floating-point; exact round-trip |
| Bitshuffle + Zstd level 22       | `sources_tsv`, `makeup_tsv`              | uint8 blobs; ZFP incompatible    |

`elms` is a variable-length UTF-8 string dataset and is stored uncompressed.

## B-tree corruption recovery (rcsb_med, June 2026)

The `rcsb_med` group B-tree was corrupted mid-build (at roughly 40% completion, ~40k of 101,989 entries written). Standard h5py operations on it (`del`, `keys()`) raised checksum errors. Recovery procedure:

### Step 1 -- OHDR binary scan

Scan the raw file with `mmap.find(b'OHDR')`, skip non-v2 headers (version byte != 2), then call `H5Oopen_by_addr` via ctypes on h5py's bundled libhdf5 to open each candidate object directly by byte offset, bypassing the corrupted B-tree. Each call is wrapped in a `signal.SIGALRM` timeout (1 s) to prevent infinite hangs on pathological corrupted objects. Valid molecule groups are written incrementally to a recovery file (checkpoint every 200 molecules for resume safety).

Result: 37 GB file, 4.7 M OHDR signatures, ~26 min, 25 timeouts.

**Warning -- zombie objects**: OHDR scan finds ALL HDF5 objects ever written to the file, including orphaned objects from previous build runs that were logically deleted but not physically zeroed. After recovery, cross-check every recovered key against the source XYZ directory and delete any key with no matching `<stem>.xyz`. In this run: 107,616 raw hits, 67,616 were garbage (old unprefixed hydration_shells orphans from a previous naming convention), leaving 40,000 legitimate rcsb_med entries.

### Step 2 -- Fresh file rebuild

`del hf['rcsb_med']` also fails with checksum errors on a corrupted group. Solution: build a new file from scratch using `h5py.File.copy()` (H5Ocopy -- raw chunk copy, no decompression) to transfer all intact top-level groups/datasets from the original, then copy rcsb_med from the recovery file. Rename rebuilt file over original.

Result: ~12 min to rebuild.

### Step 3 -- Resume build_db

With the recovered 40,000 entries in place, `build_db.py` resumes normally: it opens the file in append mode, skips entries that already exist, and fills in the remaining 61,989 rcsb_med entries plus all subsequent groups (rcsb_sml, si_ge_clusters, tmQM, viro3D).

### Key tools

- `h5clear -s <file>`: reset write-open flags left by an interrupted write
- `H5Oopen_by_addr` (ctypes): open HDF5 objects by raw byte offset, bypassing B-trees
- `signal.SIGALRM`: bound hanging C-library calls to a fixed timeout

## Crash safety

Entries are written under a temporary name `__tmp__<stem>` and atomically moved to `<stem>` only after shape assertions pass. Any `__tmp__*` keys found at startup are cleaned up before processing resumes.
