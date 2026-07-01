# Baselines

A suite of 8 non-neural reference models for SAXS I(q) prediction. All share the  
same interface as ScatterNet so they can be dropped into the same evaluation loop.

## Interface

```python
class Baseline(ABC):
    def fit(self, loader: Iterable[Batch]) -> "Baseline": ...  # no-op default
    def __call__(self, batch: Batch) -> Float[Tensor, "N Q"]: ...
```

Baselines that require training statistics implement `fit(loader)`. Those that  
compute directly from coordinates or physics need only `__init__`.

```python
# typical usage
mean_iq = MeanIQBaseline().fit(train_loader)
rg      = RgBaseline(qgrid, energy)          # no fit needed

for batch in test_loader:
    pred = mean_iq(batch)   # (N, Q) - same shape as ScatterNet output
```

---

## Baselines

### Zero-order: no molecule information

| Class               | `__init__` | `fit`    | What it uses                       |
| ------------------- | ---------- | -------- | ---------------------------------- |
| `MeanIQBaseline`    | `()`       | required | global mean I(q) over training set |
| `AtomCountBaseline` | `()`       | required | per-atom-count-bucket mean I(q)    |

**Beating these proves** the model is sensitive to individual molecule structure.

---

### Chemistry-only: no 3D geometry

| Class                 | `__init__`        | `fit`      | What it uses                                |
| --------------------- | ----------------- | ---------- | ------------------------------------------- |
| `CompositionBaseline` | `()`              | required   | element-type fractions → weighted mean I(q) |
| `MeanFFBaseline`      | `(qgrid, energy)` | not needed | incoherent scattering Σ f_i(q)² from xraydb |

**Beating these proves** the model uses 3D coordinates, not just chemical formula.

---

### Global geometry: single characteristic length scale

| Class              | `__init__`        | `fit`      | What it uses                                 |
| ------------------ | ----------------- | ---------- | -------------------------------------------- |
| `RgBaseline`       | `(qgrid, energy)` | not needed | Guinier: I(q) = I(0)·exp(-q²·Rg²/3)          |
| `PowerLawBaseline` | `(qgrid)`         | required   | log-log least-squares fit: I(q) = n·A·q^(-α) |

**Beating these proves** the model captures structure beyond global size or a simple power law.

**Caveats:**

- `RgBaseline` is only valid for q·Rg < 1.3 (Guinier region). For large molecules  
(Rg >> 1/q_max) this covers only the first few q-points. Best used on small-molecule  
datasets (COD, QM9) where the full q range is informative.
- `PowerLawBaseline` fits a **single global A and α** via least-squares regression in  
log-log space. A per-class fit (separate A, α for COD / QM9 / hydration shells) would  
be a stronger baseline; this is left as a future improvement.  
Most meaningful at high q; with q_max = 0.5 Å⁻¹ the Porod regime is not fully entered.

---

### Local geometry: multi-scale distance information

| Class              | `__init__`                    | `fit`      | What it uses                                               |
| ------------------ | ----------------------------- | ---------- | ---------------------------------------------------------- |
| `PairPeakBaseline` | `(qgrid, energy, n_bins=200)` | not needed | P(r) histogram peak → I(q) ≈ I(0)·sinc(q·r*)               |
| `NNBaseline`       | `(qgrid, energy)`             | not needed | mean nearest-neighbour distance → I(q) ≈ I(0)·sinc(q·r_nn) |

**Beating these proves** the model uses the full pairwise distance distribution, not  
just a single dominant length scale.

**Caveat:** Both are O(M²) per molecule. Fine for small molecules (M < 80 in COD/QM9),  
slow for large proteins.

---

## What beating each baseline proves

| Baseline              | Proves                                      |
| --------------------- | ------------------------------------------- |
| `MeanIQBaseline`      | model output is molecule-sensitive          |
| `AtomCountBaseline`   | model uses more than molecule size          |
| `CompositionBaseline` | model uses 3D geometry, not just chemistry  |
| `MeanFFBaseline`      | model learns per-q form factor weighting    |
| `RgBaseline`          | model captures structure beyond global size |
| `PowerLawBaseline`    | model works beyond a simple power law       |
| `PairPeakBaseline`    | model uses the full distance distribution   |
| `NNBaseline`          | model uses multi-scale geometry             |

---

## Recommended workflow

```
1. fit baselines on train set
2. evaluate baselines on test set  → establishes the comparison floor
3. train ScatterNet on train set
4. evaluate ScatterNet on test set → compare against step 2
```

Baselines are **fit on train, evaluated on test**. Evaluating baselines on the training set and comparing against ScatterNet's test performance would be an unfair comparison. Step 1 can be done before any ScatterNet training starts, giving an early sense of how much headroom the model has to improve over each baseline tier.

---

## Parameters

Baselines that take `qgrid` and `energy` must match the HDF5 dataset exactly:

```python
qgrid  = torch.linspace(0, 0.5, 51)  # Q=51, qMin=0, qMax=0.5 Å⁻¹
energy = 12500.0                       # eV, L=50 run
```

---

## Directory structure

```
Baselines/
    baseline.py                  # abstract Baseline + build_fmag_table utility
    zero_order/
        mean_iq_baseline/
        atom_count_baseline/
    chemistry_only/
        composition_baseline/
        mean_ff_baseline/
    global_geometry/
        radius_of_gyration_baseline/
        power_law_baseline/
    local_geometry/
        pr_baseline/
        nn_baseline/
    ablations/                   # ScatterNet ablations (fixed-sigma, no ff-penalty)
```
