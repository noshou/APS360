# FormFact Library

## Overview

A Fortran library for X-ray scattering calculations providing atomic form factors (f₀) and anomalous  
scattering corrections (f₁, f₂), compiled as a Python extension via f2py.

- **f₀(Q)**: Scattering amplitude as a function of Q = (sin θ)/λ — decreases with angle due to destructive interference from the electron cloud
- **f₁, f₂**: Real and imaginary anomalous scattering corrections — energy-dependent, evaluated at 12.4128 keV (1.0 Å)

Data source: International Tables for Crystallography Vol. C (DOI: 10.1107/97809553602060000600)

---

## Building

```bash
cd APS630
mkdir -p build && cd build
cmake ..
make
```

This does three things in order:

1. Compiles the OCaml parsers (`F0Parser`, `F1F2Parser`)
2. Runs the parsers to regenerate `F0Factor.f90` and `F1F2Factors.f90` from source data
3. Compiles the Fortran into a Python extension via f2py

The compiled extension (`FormFact*.so`) is placed in the project root.

---

## Supported Elements

| Symbol | Element       | Notes |
| ------ | ------------- | ----- |
| `h`    | Hydrogen      |       |
| `he`   | Helium        |       |
| `li`   | Lithium       |       |
| `be`   | Beryllium     |       |
| `b`    | Boron         |       |
| `c`    | Carbon        |       |
| `n`    | Nitrogen      |       |
| `o`    | Oxygen        |       |
| `f`    | Fluorine      |       |
| `ne`   | Neon          |       |
| `na`   | Sodium        |       |
| `mg`   | Magnesium     |       |
| `mg2+` | Magnesium ion |       |
| `al`   | Aluminium     |       |
| `si`   | Silicon       |       |
| `p`    | Phosphorus    |       |
| `s`    | Sulfur        |       |
| `cl`   | Chlorine      |       |
| `ar`   | Argon         |       |
| `k`    | Potassium     |       |
| `ca`   | Calcium       |       |
| `mn2+` | Manganese ion |       |
| `ni`   | Nickel        |       |
| `cu2+` | Copper ion    |       |
| `se`   | Selenium      |       |

Element symbols are case-insensitive. Ionic forms (e.g. `Zn2+`) strip the charge suffix for f₁/f₂ lookup but use the full symbol for f₀. Elements not in this table will cause a runtime error — see **Adding a New Element** below.

---

## Adding a New Element

### f₀ data

Add a column to `data/f0_factors/f0.csv`. The header row is element symbols (lowercase), rows are q values from 0.00 to 0.50 in steps of 0.02 Å⁻¹. Example for neutral zinc:

```
q,  ..., zn
0.00, ..., 30.0
0.02, ..., 29.8
...
```

### f₁/f₂ data

Add an entry to `data/f1_f2_factors/(12412.8)eV_f1_f2.json`:

```json
"zn": [[f1_value, f2_value]]
```

The TSV files in `data/f1_f2_factors/` are the raw Henke tables — use them to find the f₁/f₂ values at 12412.8 eV.

### Rebuild

```bash
cd build && make
```

CMake tracks dependencies: editing the CSV or JSON automatically triggers parser re-runs and recompilation.

---

## Dependencies

| Tool                                   | Purpose              |
| -------------------------------------- | -------------------- |
| gfortran                               | Fortran compiler     |
| Python + numpy                         | f2py extension build |
| OCaml (≥ 4.08)                         | Parser compilation   |
| ocamlfind                              | OCaml package finder |
| OCaml packages: `csv`, `str`, `yojson` | Used by parsers      |

Install OCaml dependencies via opam:

```bash
opam install csv str yojson
```
