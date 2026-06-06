# I(q)@L=50 Database

Produced by `buildDB()` in `load_data.py`. The file is opened in append mode (`'a'`), so existing entries are skipped on resume.

## Root attributes

| Attribute | Type  | Description                            |
| --------- | ----- | -------------------------------------- |
| `lMax`    | int   | Maximum spherical harmonic degree used |
| `energy`  | float | X-ray energy in eV (e.g. `12500.0`)    |

## Root datasets

| Path           | dtype   | Shape  | Compression          | Description                                     |
| -------------- | ------- | ------ | -------------------- | ----------------------------------------------- |
| `/q_grid`      | float64 | `(Q,)` | ZFP lossless         | Momentum transfer grid in Å⁻¹; `Q = len(qvals)` |
| `/sources_tsv` | uint8   | `(N,)` | Bitshuffle + Zstd-22 | Raw bytes of provenance TSV (optional)          |
| `/makeup_tsv`  | uint8   | `(M,)` | Bitshuffle + Zstd-22 | Raw bytes of ion makeup TSV (optional)          |

Both TSV datasets are written once and never overwritten on subsequent runs.

## Molecule data — `/<group>/<stem>/`

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

| Level     | Key            | Description                                                                                                 |
| --------- | -------------- | ----------------------------------------------------------------------------------------------------------- |
| group     | `<group_name>` | Arbitrary label supplied via the `groups` dict argument                                                     |
| subgroup  | `<stem>`       | Filename without `.xyz` extension                                                                           |
| attribute | `name`         | Molecule name string (from XYZ line 2)                                                                      |
| dataset   | `I_q`          | Orientationally-averaged scattering intensity, float32 `(Q,)`, ZFP lossless                                 |
| dataset   | `coords`       | Centroid-subtracted Cartesian coordinates, float64 `(n, 3)`, ZFP lossless                                   |
| dataset   | `angles`       | Spherical angles, float64 `(n, 2)`: col 0 = theta (polar, 0→π), col 1 = phi (azimuthal, 0→2π), ZFP lossless |
| dataset   | `r`            | Radial distances from centroid in Å, float64 `(n,)`, ZFP lossless                                           |
| dataset   | `elms`         | Element symbol per atom, variable-length UTF-8 string `(n,)`, uncompressed                                  |

`Q` is the number of points in `/q_grid` and is fixed for the whole file. `n` varies per molecule.

Coordinates are centroid-subtracted (shifted to geometric centroid before storage). `angles` and `r` are stored pre-computed for fast loading; they are consistent with `coords` via:

```
r[i]     = norm(coords[i])
theta[i] = arccos(z[i] / r[i])   (0 if r = 0)
phi[i]   = arctan2(y[i], x[i])
```

Form factors are **not** stored — they are recomputed from `xraydb`.

## Groups

The `groups` argument maps each group name to a directory of `.xyz` files. Every group becomes a top-level HDF5 group containing one subgroup per molecule.

| Group                                   | Number of Molecules | Molecule Size Range | Descriptions |
| --------------------------------------- | ------------------- | ------------------- | ------------ |
| COD                                     | 532,612             | 1-78,818            |              |
| QM9                                     | 133,844             | 3-29                |              |
| tmQM                                    | 108,540             | 7-569               |              |
| rcsb_med                                | 101,989             | 3,000 - 18,222      |              |
| rcsb_sml                                | 96,184              | 28-6,121            |              |
| viro3D                                  | 85,063              | 180-64,938          |              |
| hydration_shells                        | 48,571              | 3-147               |              |
| mofs                                    | 30,871              | 10-24,948           |              |
| (Na,Co,Ag,Pb,Mo,Fe)_monoatomic_clusters | 1,282               | 2-380               |              |
| binary_clusters                         | 317                 | 2-55*               |              |
| si_ge_clusters                          | 217                 | 4-60                |              |
| ar_ne_clusters                          | 127                 | 2-55                |              |
| (NaCl)_nCl-                             | 70                  | 3-71                |              |

**one outlier (CuAu55_CuAu38.xyz) is at 1,482*

**TOTAL MOLECULES: 1,139,743**

## Compression codecs

| Codec                            | Used for                                 | Notes                            |
| -------------------------------- | ---------------------------------------- | -------------------------------- |
| ZFP lossless (`reversible=True`) | `q_grid`, `I_q`, `coords`, `angles`, `r` | Floating-point; exact round-trip |
| Bitshuffle + Zstd level 22       | `sources_tsv`, `makeup_tsv`              | uint8 blobs; ZFP incompatible    |

`elms` is a variable-length UTF-8 string dataset and is stored uncompressed.

## Crash safety

Entries are written under a temporary name `__tmp__<stem>` and atomically moved to `<stem>` only after shape assertions pass. Any `__tmp__*` keys found at startup are cleaned up before processing resumes.

## Typical parameters (default run)

| Parameter | Value       |
| --------- | ----------- |
| energy    | 12 500 eV   |
| qMin      | 0 Å⁻¹       |
| qMax      | 0.5 Å⁻¹     |
| step      | 0.01 Å⁻¹    |
| Q         | 51 points   |
| lMax      | set at call |
