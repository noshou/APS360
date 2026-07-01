# I(q)@L=50 Database

## Retrieving the dataset

The HDF5 files are hosted on HuggingFace at **[noshou/iq_train_set](https://huggingface.co/datasets/noshou/iq_train_set)**.

Download with the HuggingFace CLI (recommended - resumes interrupted downloads):

```bash
pip install huggingface_hub
huggingface-cli download noshou/iq_train_set "I(q)@L=50.h5" \
    --repo-type dataset --local-dir Preprocess/
```

Or in Python:

```python
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id   = "noshou/iq_train_set",
    filename  = "I(q)@L=50.h5",
    repo_type = "dataset",
    local_dir = "Preprocess/",
)
```

Place the downloaded file at `Preprocess/I(q)@L=50.h5` (the path all pipeline scripts expect).

---

Two files are maintained:

- **`I(q)@L=50.h5`** - full database including COD (41.6 GB)
- **`I(q)@L=50_train.h5`** - training subset with COD excluded (23.4 GB); this is the file used for model training

Produced by `buildDB()` in `load_data.py`. The file is opened in append mode (`'a'`), so existing entries are skipped on resume.

## Root attributes

| Attribute | Type  | Description                            |
| --------- | ----- | -------------------------------------- |
| `lMax`    | int   | Maximum spherical harmonic degree used |
| `energy`  | float | X-ray energy in eV (e.g. `12500.0`)    |

## Root datasets

| Path           | dtype   | Shape  | Compression          | Description                                             |
| -------------- | ------- | ------ | -------------------- | ------------------------------------------------------- |
| `/q_grid`      | float64 | `(Q,)` | ZFP lossless         | Momentum transfer grid in angstrom^-1; `Q = len(qvals)` |
| `/sources_tsv` | uint8   | `(N,)` | Bitshuffle + Zstd-22 | Raw bytes of provenance TSV (optional)                  |
| `/makeup_tsv`  | uint8   | `(M,)` | Bitshuffle + Zstd-22 | Raw bytes of ion makeup TSV (optional)                  |

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

| Level     | Key            | Description                                                                                                         |
| --------- | -------------- | ------------------------------------------------------------------------------------------------------------------- |
| group     | `<group_name>` | Arbitrary label supplied via the `groups` dict argument                                                             |
| subgroup  | `<stem>`       | Filename without `.xyz` extension                                                                                   |
| attribute | `name`         | Molecule name string (from XYZ line 2)                                                                              |
| dataset   | `I_q`          | Orientationally-averaged scattering intensity, float32 `(Q,)`, ZFP lossless                                         |
| dataset   | `coords`       | Centroid-subtracted Cartesian coordinates, float64 `(n, 3)`, ZFP lossless                                           |
| dataset   | `angles`       | Spherical angles, float64 `(n, 2)`: col 0 = theta (polar, 0 to pi), col 1 = phi (azimuthal, 0 to 2pi), ZFP lossless |
| dataset   | `r`            | Radial distances from centroid in angstroms, float64 `(n,)`, ZFP lossless                                           |
| dataset   | `elms`         | Element symbol per atom, variable-length UTF-8 string `(n,)`, uncompressed                                          |

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

### Training file groups (`I(q)@L=50_train.h5`)

| Group                                   | Number of Molecules | Molecule Size Range | Description |
| --------------------------------------- | ------------------- | ------------------- | ----------- |
| QM9                                     | 133,844             | 3-29                |             |
| tmQM                                    | 108,541             | 7-569               |             |
| rcsb_med                                | 46,495              | 3,000-18,222        |             |
| hydration_shells                        | 48,571              | 3-147               |             |
| mofs                                    | 30,868              | 10-24,948           |             |
| (Na,Co,Ag,Pb,Mo,Fe)_monoatomic_clusters | 1,282               | 2-380               |             |
| binary_clusters                         | 371                 | 2-55*               |             |
| si_ge_clusters                          | 217                 | 4-60                |             |
| ar_ne_clusters                          | 127                 | 2-55                |             |
| (NaCl)_nCl-                             | 70                  | 3-71                |             |

**one outlier (CuAu55_CuAu38.xyz) is at 1,482*

**TRAINING TOTAL: 370,386 molecules**

### Full database additional group (`I(q)@L=50.h5` only)

| Group | Number of Molecules | Molecule Size Range | Description |
| ----- | ------------------- | ------------------- | ----------- |
| COD   | 532,612             | 1-78,818            | Excluded from training file; crystallographic structures with high redundancy |

**FULL DATABASE TOTAL: 902,998 molecules**

### Planned future groups (logged in `sources_tsv`, not yet computed)

| Group    | Planned Molecules | Molecule Size Range | Description |
| -------- | ----------------- | ------------------- | ----------- |
| rcsb_sml | 96,184            | 28-6,121            | Smaller RCSB PDB structures; requires additional compute |
| viro3D   | 85,063            | 180-64,938          | Virus protein structure predictions                      |

`sources_tsv` already contains provenance entries for all planned groups. Molecule data will be added via `build_db.py` when compute is available.

## Compression codecs

| Codec                            | Used for                                 | Notes                            |
| -------------------------------- | ---------------------------------------- | -------------------------------- |
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

## Typical parameters (default run)

| Parameter | Value            |
| --------- | ---------------- |
| energy    | 12 500 eV        |
| qMin      | 0 angstrom^-1    |
| qMax      | 0.5 angstrom^-1  |
| step      | 0.01 angstrom^-1 |
| Q         | 51 points        |
| lMax      | set at call      |
