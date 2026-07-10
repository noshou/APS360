# ScatterNet: Design Reference

GNN that predicts X-ray powder scattering curves I(q) from atomic coordinates and species, using Random Fourier Features for O(M·λ₅) all-pairs kernel aggregation, and tensor/data parallelism across GPUs (routed per batch, see §7).

---

## Table of Contents

- [Notation](#notation)
- [1. Vocabulary and Atom Tokens](#1-vocabulary-and-atom-tokens)
- [2. Data Pipeline](#2-data-pipeline)
  - [2.1 Batch](#21-batch)
  - [2.2 Bucketing](#22-bucketing-batcher)
  - [2.3 Loading](#23-loading-batchset)
- [3. Embed](#3-embed)
- [4. LayerHead](#4-layerhead)
- [5. MessagePass](#5-messagepass)
  - [Mathematical Formulation](#mathematical-formulation)
  - [Pass 1: Accumulate Global Context](#pass-1-accumulate-global-context)
  - [Pass 2: Per-Atom Update](#pass-2-per-atom-update)
  - [MishGLU Gate](#mishglu-gate)
  - [Sigma Update](#sigma-update)
  - [bmm vs einsum](#bmm-vs-einsum)
  - [Checkpointing Strategy](#checkpointing-strategy)
  - [Distributed AllReduce](#distributed-allreduce)
- [6. OutputHead](#6-outputhead)
- [7. ScatterNet](#7-scatternet)
  - [Tensor Parallel Forward](#tensor-parallel-forward)
  - [Custom Autograd Functions](#custom-autograd-functions)
- [8. Loss](#8-loss)
- [9. Training Loop](#9-training-loop)
- [10. Hyperparameter Reference](#10-hyperparameter-reference)
- [11. End-to-End Data Flow](#11-end-to-end-data-flow)
- [12. Profiling and Optimization on Kaggle](#12-profiling-and-optimization-on-kaggle)
- [Appendix](#appendix)
  - [A1. mol_chunk and atm_chunk optimizations](#a1-mol_chunk-and-atm_chunk-optimizations)
  - [A2. dp_atom_threshold optimizations](#a2-dp_atom_threshold-optimizations)

---

## Notation

| Symbol | Meaning                                                  |
| ------ | -------------------------------------------------------- |
| N      | molecules in a batch                                     |
| M      | atoms per molecule (padded to the longest in the batch)  |
| Q      | number of q-points in the scattering grid                |
| λ₁     | atom embedding dimension (`lambda_1`, default 128)       |
| λ₂     | message-passing rounds (`lambda_2`, default 5)           |
| λ₃     | OutputHead MLP starting width (`lambda_3`, default 128)  |
| λ₄     | OutputHead MLP halving steps (`lambda_4`, default 4)     |
| λ₅     | Random Fourier Features count (`lambda_5`, default 256)  |
| λ₆     | form-factor penalty weight (`lambda_6`, default 0.1)     |
| λ₇     | sigma L2 penalty weight (`lambda_7`, default 0.1)        |
| Nc     | molecules per N-chunk (`mol_chunk`)                      |
| mc     | atoms per M-chunk (`atm_chunk`)                          |
| V      | VOCAB size = len(VOCAB) + 1 (row 0 is padding)           |
| ε_e    | `eps_embd` numerical floor in Embed (default 1e-8)       |
| ε_m    | `eps_msgp` numerical floor in MessagePass (default 1e-3) |
| r_m    | Cartesian coordinates of atom m, shape (3,) in Å         |
| e_m    | embedding vector of atom m, shape (λ₁,)                  |
| f_m(q) | form factor magnitude of atom m at q-point q, scalar     |
| σ_m(q) | RFF kernel bandwidth of atom m at q-point q, scalar      |
| φ_m(q) | RFF feature vector of atom m at q-point q, shape (λ₅,)   |

---

## 1. Vocabulary and Atom Tokens

### `Preprocess.VOCAB`

VOCAB is a module-level singleton built at import time from `xraydb.get_xraydb().f0_ions()`, which returns all ions and elements with a tabulated Cromer-Mann atomic form factor f0 in the NIST xraydb database.

Atom indices are 1-based. Index 0 is reserved as the padding sentinel (`batch.vocab == 0` means no atom). `nn.Embedding` is constructed with `len(VOCAB)+1` rows and `padding_idx=0`.

### Anomalous Correction Handling

| Case                                        | Treatment                                                                      |
| ------------------------------------------- | ------------------------------------------------------------------------------ |
| Transuranics (Np, Pu, Am, Cm, Bk, Cf, ions) | `f0` only; Chantler f1/f2 unavailable or unreliable                            |
| Special cases (`siva->si`, `cval->c`, etc.) | Remapped to canonical base element before f1/f2 lookup                         |
| Normal                                      | Charge suffix stripped for f1/f2 (`fe2+` -> `fe`); full ion string kept for f0 |

---

## Dataset

The training dataset (`noshou/iq_train_set` on HuggingFace) contains 1,044,583 molecules with precomputed I(q) curves. Full schema, file descriptions, download instructions, and group statistics are in [`Preprocess/README.md`](Preprocess/README.md).

`Preprocess/iq_train_set-ENCODING.sqlite3` (tracked via Git LFS, ~860 MB) ships with the repo and is the primary training index. It maps every molecule to its atom count and VOCAB indices so the data pipeline never needs to scan the 66 GB HDF5 file during training.

## 2. Data Pipeline

### 2.1 `Batch`

`Batch` is a frozen dataclass with runtime shape checking via jaxtyping + beartype.

| Field   | Shape     | Dtype   | Contents                                              |
| ------- | --------- | ------- | ----------------------------------------------------- |
| `vocab` | (N, M)    | int64   | VOCAB index per atom. 0 = padding.                    |
| `iqval` | (N, Q)    | float32 | Ground-truth I(q) from simulation.                    |
| `coord` | (N, M, 3) | float32 | Cartesian x/y/z in Å. Padded positions are (0, 0, 0). |

`batch.padding_mask()` returns (N, M) bool, True = real atom, derived as `vocab != 0`. Coordinates and intensities can legitimately be zero, so vocab is the only reliable mask source.

`Batch.from_lists(vocabs, iqvals, coords)` pads each list with `pad_sequence(..., batch_first=True, padding_value=0)`.

### 2.2 Bucketing (`Batcher`)

Molecules span 2 to 78,819 atoms. Without bucketing, a fixed-size batch would mix wildly different molecule sizes and waste most of each batch to padding.

**Step 1 (`_batches_init`):** for each `(min_atoms, max_atoms)` range in `DEFAULT_BUCKETS` (60 entries), molecules are sorted by atom count and their total atom count is summed. If the total is <= `atom_size_ceil` (default: 3x largest molecule), the group becomes one batch. If it exceeds the ceiling, the group is split recursively at the median atom-count boundary via binary search on prefix sums.

**Step 2 (`_batches_stratify`):** batches with < 3 molecules are merged into their nearest neighbour by median atom count. Molecules within each batch are shuffled by `batcher_seed`, then split 70/15/15 (train/val/test) with a floor of 1 molecule per split. Each bucket contributes one sub-batch to each split, so `len(train_loader) == len(val_loader) == len(test_loader)` always; the 70/15/15 is at the molecule level.

### 2.3 Loading (`BatchSet`)

`BatchSet` is a `torch.utils.data.Dataset` where each item is a pre-built, fully-padded `Batch`. DataLoader is used with `batch_size=1` and a passthrough `collate_fn`.

`__getitem__(i)` opens the HDF5 file and reads all molecules in batch i:

| HDF5 dataset | Contents                          | Conversion                                          |
| ------------ | --------------------------------- | --------------------------------------------------- |
| `elms`       | byte strings of element/ion names | decoded ->`enc._encode_ions()` -> int tensor (M_i,) |
| `I_q`        | simulated scattering curve        | float tensor (Q,)                                   |
| `coords`     | Cartesian positions               | float tensor (M_i, 3)                               |

The file is opened and closed per call. With `num_workers > 0`, each DataLoader worker maintains its own file handle.

---

## 3. `Embed`

### Design Decisions

Two-part form factor: `_f0f1` and `_f2` are separate linear layers for real and imaginary parts, mirroring Cromer-Mann notation (`f = (f0+f1) + i*f2`). Only the magnitude `|f(q)|` is used downstream, consistent with powder-averaged X-ray scattering where phases average out over random orientations.

Bilinear sigma: `NoTrilinBilin(λ₁, Q, Q)` (an `nn.Bilinear`-equivalent that avoids the `aten::_trilinear` kernel and its `torch.compile` graph break - see [§5](#5-messagepass)) captures the interaction between identity (embed) and scattering strength (f_mag). A linear layer over concatenation misses cross-terms; bilinear is the minimal model that captures multiplicative coupling.

PReLU: channel-wise PReLU allows each embedding channel to independently learn its negative slope, giving more expressivity in the first non-linearity than a fixed activation.

Asymmetric masking: `f_mags` and `sigmas` are multiplied by the padding mask; `embeds` is not. Zeroing f_mags is sufficient to exclude padding atoms since intensity contributions are gated by f_mag^2.

ε is passed at forward time rather than fixed at construction, allowing the numerical floor to be tuned without changing the model.

Converts each atom's VOCAB index into three tensors that feed the rest of the pipeline:

| Output   | Shape         | Meaning                          |
| -------- | ------------- | -------------------------------- |
| `embeds` | (N, M, 1, λ₁) | Learned per-atom identity vector |
| `f_mags` | (N, M, Q, 1)  | Estimated form factor magnitude  |
| `sigmas` | (N, M, Q, 1)  | RFF kernel bandwidth per q-point |

### `Embed` Layers

1. **Learnable Parameters**

  | Parameter       | Shape      | Notes                                                     |
  | --------------- | ---------- | --------------------------------------------------------- |
  | `_mbd.weight`   | (V, λ₁)    | Embedding table. Row 0 frozen at zero via`padding_idx=0`. |
  | `_f0f1.weight`  | (Q, λ₁)    | Linear: embedding -> real part of form factor.            |
  | `_f0f1.bias`    | (Q,)       |                                                           |
  | `_f2.weight`    | (Q, λ₁)    | Linear: embedding -> imaginary part of form factor.       |
  | `_f2.bias`      | (Q,)       |                                                           |
  | `_prelu.weight` | (λ₁,)      | One learned negative slope per embedding channel.         |
  | `_sigma.weight` | (Q, λ₁, Q) | Bilinear weight: embedding x f_mag -> sigma logit.        |
  | `_sigma.bias`   | (Q,)       |                                                           |

2. **Forward Pass**
  ```{python}
  batch.vocab                           (N, M)

  # embedding with channel-wise PReLU
  _mbd(vocab)                           (N, M, λ₁)   lookup
    .transpose(-1,-2)                   (N, λ₁, M)   channels to dim=1 for PReLU
  _prelu(...)                           (N, λ₁, M)   f(x) = x if x>=0 else a*x (a learned)
    .transpose(-1,-2)                   (N, M, λ₁)   restore
  -> embed                              (N, M, λ₁)

  # complex form factor magnitude
  _f0f1(embed)                          (N, M, Q)    real part (f0+f1)
  _f2(embed)                            (N, M, Q)    imaginary part (f2)
  hypot(f_rel, f_img) + ε_e             (N, M, Q)    |f(q)| = sqrt(f_rel^2 + f_img^2)
  -> f_mag                              (N, M, Q)

  # sigma: per-atom per-q RFF bandwidth
  _sigma(embed, f_mag)                  (N, M, Q)    bilinear(λ₁, Q) -> Q
  softplus(...) + ε_e                   (N, M, Q)    strictly positive
    .unsqueeze(-1) * mask.unsqueeze(-1) (N, M, Q, 1) zero padding atoms
  -> sigma                              (N, M, Q, 1)

  # pack into LayerHead
  embed.unsqueeze(-2)  -> embeds        (N, M, 1, λ₁)  Q=1, broadcastable
  f_mag.unsqueeze(-1) * mask -> f_mags  (N, M, Q, 1)
  sigma                -> sigmas        (N, M, Q, 1)
  ```
3. **Activations**

  | Activation      | Location               | Behaviour                                                                        |
  | --------------- | ---------------------- | -------------------------------------------------------------------------------- |
  | `nn.PReLU(λ₁)`* | After embedding lookup | Learnable per-channel negative slope. Acts on dim=1, hence the double-transpose. |
  | `F.softplus`    | On sigma logits        | `log(1 + exp(x))`, smooth positive-enforcing.                                    |

**Comparison of *`PReLU`*  to other activation functions:*

  | Activation | λ₁  | Val loss | Val R² | Test R² |
  | ---------- | --- | -------- | ------ | ------- |
  | LeakyReLU  | 64  | 0.77     | 0.71   | 0.69    |
  | PReLU      | 128 | 0.67     | 0.74   | 0.73    |
  | PReLU      | 64  | 0.93     | 0.62   | 0.61    |
  | LeakyReLU  | 128 | NaN      | NaN    | NaN     |
  | ELU        | 64  | 2.44     | -0.05  | -0.01   |
  | Mish       | 64  | NaN      | NaN    | NaN     |

---

## 4. `LayerHead`

`LayerHead` is a `NamedTuple` (immutable, typed) passed between Embed, MessagePass, and OutputHead.

| Field    | Shape from Embed | Shape after MessagePass | Meaning                                                                 |
| -------- | ---------------- | ----------------------- | ----------------------------------------------------------------------- |
| `embeds` | (N, M, 1, λ₁)    | (N, M, Q, λ₁)           | Per-atom identity vector. Q=1 on construction, expanded by MessagePass. |
| `f_mags` | (N, M, Q, 1)     | unchanged               | Form factor magnitude. Trailing 1 broadcasts over λ₁.                   |
| `sigmas` | (N, M, Q, 1)     | updated per round       | RFF bandwidth. Trailing 1 broadcasts over λ₁.                           |

The `*` wildcard in the jaxtyping annotation for `embeds` ("N M * λ₁") allows Q to be 1 or Q without a type error. `NamedTuple._replace(...)` produces modified copies without mutation.

---

## 5. `MessagePass`

Runs λ₂ rounds of kernel-weighted neighbourhood aggregation. Instead of computing all M^2 pairwise distances, Random Fourier Features factor the kernel so that the aggregate for atom i is a dot product against a global summary tensor, reducing complexity from O(M^2) to O(M·λ₅).

### Mathematical Formulation

Atom coordinates are divided by the per-atom bandwidth at each q-point:

```{Latex}
r~_m(q) = r_m / σ_m(q)
```

Large σ: scaled coordinates are small, so the RBF kernel sees atoms as close regardless of physical distance (long-range aggregation). Small σ: narrow kernel, short-range aggregation. The model learns σ.

The RBF kernel being approximated:

```{Latex}
k(r~_i, r~_j) = exp(-||r~_i - r~_j||^2 / 2)
```

Rahimi & Recht (2007): draw Ω in R^{λ₅x3} from N(0,I), draw b in R^{λ₅} from Uniform(0, 2π), define:

```{Latex}
φ_m(q) = sqrt(2/λ₅) * cos(Ω * r~_m(q) + b)     in R^{λ₅}
```

Then `E[φ_i(q) · φ_j(q)] = k(r~_i(q), r~_j(q))`. In code: Ω is `_omegafrq` (fixed buffer, seeded from `msg_seed`); b is `_biasterm` (learned `nn.Parameter`, phases can shift during training).

Per molecule, define global context tensors:

```{Latex}
features[q, d]    = sum_m φ_m(q, d)                 (Q, λ₅)      kernel weight sum
chem_env[q, d, l] = sum_m φ_m(q, d) * e_m(q, l)     (Q, λ₅, λ₁)  kernel-weighted embedding sum
```

Then for atom i:

```{Latex}
locality_i[q, l] ≈ sum_m k(r~_i(q), r~_m(q)) * e_m(q, l)   via dot(φ_i, chem_env)
weights_i[q]     ≈ sum_m k(r~_i(q), r~_m(q))                via dot(φ_i, features)
agg_i             = RMSNorm(locality_i / weights_i)
```

### bmm vs einsum

The two heavy contractions use bmm on 3D-reshaped tensors rather than einsum. The naive einsum:

```
  einsum('nmqd,nmql->nqdl', zrff, emb_slice)
```

would broadcast to `(Nc, mc, Q, λ₅, λ₁)` before contracting over m. At typical values `(Nc=4, mc=64, Q=256, λ₅=128, λ₁=128)` that is ~4 GB per chunk. The bmm reformulation:

```{python}
  zb = zrff.permute(0,2,3,1).reshape(Nc*Q, λ₅, mc)
  eb = emb.permute(0,2,1,3).reshape(Nc*Q, mc, λ₁)
  bmm(zb, eb)   # (Nc*Q, λ₅, λ₁)  ~16 MB
```

The `weights` einsum (`'nmqd,nqd->nmq'`) is safe because there is no λ₁ factor.

### Checkpointing Strategy

Two levels with opposite `use_reentrant` settings:

N-chunk level (`use_reentrant=True`): the entire `_n_chunk_round` (pass 1 -> AllReduce -> pass 2) runs under `torch.no_grad()` in the forward pass, so `chem_env (Nc, Q, λ₅, λ₁)` is created and freed per N-chunk. During backward, the chunk is fully re-executed. `use_reentrant=False` would fail here because `_pass_2._step` is a closure over `cont.chem_env`, which would keep all N-chunks' tensors alive simultaneously.

M-chunk level (`use_reentrant=False`): `_step` inside `_pass_1` and `_pass_2` are pure functions with no closures over large mutable tensors, so the non-reentrant API is safe and preferred. Their purpose is to avoid materialising `(Nc, M, Q, λ₅)` across all atoms.

### Distributed AllReduce

`MessagePass._AllReduce` is a custom `torch.autograd.Function`. Forward: `dist.all_reduce(SUM)` on `features` and `chem_env`. Backward: `dist.all_reduce(SUM)` again, because each rank's gradient for its partial atom sum is genuinely partial and must be summed across contributors.

### Mean-Normalization (fp16 safety)

After the AllReduce, `features` and `chem_env` are divided by the per-molecule real-atom count (a global count: summed across ranks under TP via `_count_all_reduce`, local under DP). They are sums over all atoms, so O(M) ≈ 1e3-1e4 for large molecules; in fp16 (`amp`) that overflows the `_step2` contractions (`locality`, `weights`) and their backward, producing NaN gradients that `GradScaler` cannot recover (the overflow is set by activation magnitude, not loss scale, and Turing/T4 has no bf16 to absorb the range). Dividing by the atom count turns both into **means** (O(1)). Because `_step2` forms `locality / weights` and feeds it into `RMSNorm` - which is invariant to any positive per-atom scale - the aggregate is algebraically unchanged (verified identical to ~2e-6 post-RMSNorm), while every fp16 intermediate stays in range. Mathematically it is "sum then contract then divide" rewritten as "mean then contract", exact up to fp32 rounding, and a no-op in the fp32 path beyond that.

### `MessagePass` Layers

1. **Learnable Parameters**

  | Parameter          | Shape      | Notes                                       |
  | ------------------ | ---------- | ------------------------------------------- |
  | `_proj_agg.weight` | (2λ₁, λ₁)  | MishGLU projection: agg -> [p1, p2]         |
  | `_proj_agg.bias`   | (2λ₁,)     |                                             |
  | `_biasterm`        | (λ₅,)      | Learnable RFF phase offsets b               |
  | `_sigbilin.weight` | (1, λ₁, Q) | Bilinear: (embedding, f_mag) -> sigma delta |
  | `_sigbilin.bias`   | (1,)       |                                             |
  | `_rms_norm.weight` | (λ₁,)      | RMSNorm scale                               |

  Buffers (fixed):

  | Buffer      | Shape   | Notes                                         |
  | ----------- | ------- | --------------------------------------------- |
  | `_omegafrq` | (λ₅, 3) | RFF frequency matrix Ω, seeded from`msg_seed` |

2. **Message Passing**
  i. **Pass 1: Accumulate Global Context**  
    `_pass_1` iterates over M-chunks and accumulates `features` and `chem_env` for the current N-chunk. Each `_step` is gradient-checkpointed (`use_reentrant=False`) to avoid holding the full `(Nc, M, Q, λ₅)` RFF tensor in memory.
    ```{Latex}
    # per M-chunk inputs:
    emb_slice    (Nc, mc, Q, λ₁)
    crd_slice    (Nc, mc, 3)
    sig_slice    (Nc, mc, Q, 1)
    msk_slice    (Nc, mc)            bool, True = real atom

    # r~_m = r_m / σ_m(q)
    scaled_coords  (Nc, mc, Q, 3)   crd_slice.unsqueeze(-2) / sig_slice.clamp(min=ε_m)

    # φ_m = sqrt(2/λ₅) * cos(Ω*r~_m + b)
    proj           (Nc, mc, Q, λ₅)  scaled_coords @ Ω.T + b
    zrff           (Nc, mc, Q, λ₅)  sqrt(2/λ₅) * cos(proj), zeroed at padding

    # partial sum_m φ_m
    step_features  (Nc, Q, λ₅)      zrff.sum(dim=1)

    # partial sum_m φ_m x e_m, via bmm to avoid (Nc,mc,Q,λ₅,λ₁) intermediate
    zb             (Nc*Q, λ₅, mc)   zrff.permute(0,2,3,1).reshape(...)
    eb             (Nc*Q, mc, λ₁)   emb_slice.permute(0,2,1,3).reshape(...)
    step_chem_env  (Nc, Q, λ₅, λ₁)  bmm(zb, eb).reshape(Nc, Q, λ₅, λ₁)

    # accumulate across M-chunks:
    features  (Nc, Q, λ₅)       += step_features
    chem_env  (Nc, Q, λ₅, λ₁)   += step_chem_env
    ```
  After the M-chunk loop, `_AllReduce` sums `features` and `chem_env` across ranks so every rank holds global sums over all atoms.  
  ii. **Pass 2: Per-Atom Update**  
  `_pass_2` recomputes φ_m per M-chunk and uses the globally complete `chem_env` to update embeddings and sigmas.
  ```{rtf}
  # recompute φ_m (intermediate freed during forward by checkpointing)
  scaled_coords  (Nc, mc, Q, 3)
  proj           (Nc, mc, Q, λ₅)
  zrff           (Nc, mc, Q, λ₅)  zeroed at padding

  # locality_m ≈ sum_{m'} k(r~_m, r~_{m'}) * e_{m'}, via bmm
  zb             (Nc*Q, mc, λ₅)   zrff.permute(0,2,1,3).reshape(...)
  cb             (Nc*Q, λ₅, λ₁)   chem_env.reshape(...)
  locality       (Nc, mc, Q, λ₁)  bmm(zb, cb).reshape(Nc, Q, mc, λ₁).permute(0,2,1,3)

  # weights_m ≈ sum_{m'} k(r~_m, r~_{m'})  [einsum safe: no λ₁ factor]
  weights        (Nc, mc, Q)       einsum('nmqd,nqd->nmq', zrff, features).abs()

  # normalised aggregate
  agg            (Nc, mc, Q, λ₁)  rms_norm(locality / weights.unsqueeze(-1).clamp(min=ε_m))

  # MishGLU gate
  [p1, p2]       each (Nc, mc, Q, λ₁)  proj_agg(agg).chunk(2, dim=-1)
  gate           (Nc, mc, Q, λ₁)  p1 * Mish(p2) * mask

  # residual embedding update
  new_emb        (Nc, mc, Q, λ₁)  emb_slice + gate

  # sigma update
  f_in           (Nc, mc, Q, 1)   ffs_slice expanded to Q
  delta          (Nc, mc, Q, 1)   tanhshrink(sigbilin(new_emb, f_in))
  new_sig        (Nc, mc, Q, 1)   softplus(sig_slice + delta)
  ```
3. **Activations**

  | Activation         | Location                | Behaviour                                                                                          |
  | ------------------ | ----------------------- | -------------------------------------------------------------------------------------------------- |
  | `cos`              | RFF feature computation | Core of the RFF kernel approximation.                                                              |
  | `F.mish` (p2 path) | MishGLU gate            | `x*tanh(softplus(x))`. Near zero when p2 << 0 (gate closed); near-linear when p2 >> 0 (gate open). |
  | `nn.RMSNorm`       | After locality/weights  | Normalises aggregate magnitude, preventing residual stream from compounding across rounds.         |
  | `F.tanhshrink`     | Sigma delta             | `x - tanh(x)`. Near zero, output ≈ 0 (sticky region). For large                                    |
  | `F.softplus`       | Sigma output            | Ensures σ > 0 always.                                                                              |

  - MishGLU Gate  
  `_proj_agg` is `nn.Linear(λ₁, 2λ₁)`. The output is split into p1 (value path) and p2 (gate path). `gate = p1 * Mish(p2)` is added to the atom embedding as a residual. The final `* mask` zeroes contributions from padding atoms.  
  The GLU pattern lets the network decide per-channel and per-q-point whether to incorporate the neighbourhood context, rather than always adding the full aggregate.
  - Sigma Update
    ```{rtf}
    σ_new = softplus( σ_old + tanhshrink( bilinear(e_updated, f_mag) ) )
    ```
    `_sigbilin` is `nn.Bilinear(λ₁, Q, 1)`. The bilinear coupling `e^T W f` makes the sigma delta depend on the interaction between the atom's learned representation and its scattering strength.
    Tanhshrink is unbounded by design: for large bilinear outputs sigma can shift significantly. The sigma L2 penalty in the loss (λ₇ · σ²) is the actual blowup prevention mechanism; its gradient grows with σ and pulls it back down. The two reach equilibrium where the bilinear delta just balances the MSLE gradient against the penalty. The maximum cumulative sigma change is also bounded by λ₂ rounds being small (default 5).
    The sticky property near zero (tanhshrink ≈ 0, gradient = `1 - sech²(x)` = 0 at x=0) is a beneficial side effect: early in training when the bilinear has weak outputs, sigmas stay stable rather than wandering, then move more freely as the bilinear gains signal.
    Softplus wraps the whole update to maintain σ > 0, since division by σ in the RFF step cannot encounter zero.

---

## 6. `OutputHead`

Collapses per-atom representations into a predicted I(q) curve per molecule. Each atom's contribution is weighted by f_mag² before summing, mirroring the diagonal terms of the Debye equation.  In the full Debye equation `I(q) = sum_j sum_k f_j f_k sinc(q r_jk)`, diagonal terms (j=k) contribute `sum_j f_j²`. The model learns a per-atom correction `contribs_m(q)` weighted by the form factor, so each atom contributes `contribs_m(q) * f_m(q)²`. Summing over atoms gives a Debye-inspired I(q) without the O(M²) pair sum.

### `OutputHead` Layers

1. **Learnable Parameters**

  | Parameter               | Shape       | Notes                                              |
  | ----------------------- | ----------- | -------------------------------------------------- |
  | `_bilinear.weight`      | (λ₃, λ₁, 1) | Bilinear: (embedding, f_mag scalar) -> λ₃ features |
  | `_bilinear.bias`        | (λ₃,)       |                                                    |
  | `_mlp / layer_i.weight` | varies      | MLP linear layers (halving pyramid)                |
  | `_mlp / layer_i.bias`   | varies      |                                                    |

  MLP with `lambda_3=128, lambda_4=4`:
  ```{rtf}
  Linear(128->64) -> Mish -> Linear(64->32) -> Mish -> Linear(32->16) -> Mish -> Linear(16->8) -> Mish -> Linear(8->1)
  ```
  No Mish after the final linear; `softplus` is applied to the MLP output.
2. **Forward Pass**
  ```{rtf}
  # inputs from MessagePass:
  msg_head.embeds  (N, M, Q, λ₁)
  msg_head.f_mags  (N, M, Q, 1)
  msg_head.sigmas  (N, M, Q, 1)

  mask             (N, M, 1)    padding_mask().unsqueeze(-1), float
  iq_accum         (N, Q)       zeros

  # per M-chunk:
  emb_c    (N, mc, Q, λ₁)
  fmag_c   (N, mc, Q, 1)
  mask_c   (N, mc, 1)

  bilinear(emb_c, fmag_c)      (N, mc, Q, λ₃)  λ₁-vec x 1-scalar -> λ₃ features
  F.mish(...)                  (N, mc, Q, λ₃)
  _mlp(...)                    (N, mc, Q, 1)   halving pyramid
  F.softplus(...)              (N, mc, Q, 1)   positivity constraint
    .squeeze(-1)               (N, mc, Q)      per-atom scattering scalar (contribs)

  contribs * fmag_c.squeeze(-1)^2 * mask_c  (N, mc, Q)  Debye-weighted contribution
  iq_accum += (...).sum(dim=1)               (N, Q)

  # return:
  iq_accum           (N, Q)    predicted I(q)
  f_mags.squeeze(-1) (N, M, Q)
  sigmas.squeeze(-1) (N, M, Q)
  ```
3. **Activations**

  | Activation   | Location           | Behaviour                                                |
  | ------------ | ------------------ | -------------------------------------------------------- |
  | `F.mish`     | After bilinear     | Applied to bilinear features before the MLP.             |
  | `nn.Mish`    | Between MLP layers | Between each pair of linear layers except the last.      |
  | `F.softplus` | After full MLP     | Ensures per-atom scattering scalar is strictly positive. |

---

## 7. `ScatterNet`

Top-level module. Wraps Embed, MessagePass, and OutputHead, and routes each batch to one of two parallelism strategies across GPUs: atom-dimension tensor parallelism (TP), or, for small-atom-count batches during training, molecule-dimension data parallelism (DP).

### Module Registry

| Submodule | Type          |
| --------- | ------------- |
| `_emb`    | `Embed`       |
| `_msg`    | `MessagePass` |
| `_out`    | `OutputHead`  |

`_eps_embd` and `_eps_msgp` are plain Python floats (not parameters or buffers); they are not moved by `.to(device)`.

`forward` returns a 5-tuple: `(iq, f_mags, sigmas, local_batch, loss_scale)`. `local_batch` and `loss_scale` are only meaningful when DP-routing (below); otherwise `local_batch is batch` and `loss_scale == 1.0`. Always pass `local_batch` (not the original `batch`) to the loss, and multiply the loss by `loss_scale` before `backward()` - see [Training Loop](#9-training-loop).

### Single-GPU Forward

```{rtf}
batch -> Embed(batch, ε_e)              LayerHead: (N,M,1,λ₁), (N,M,Q,1), (N,M,Q,1)
      -> MessagePass(batch, head, ε_m)  LayerHead: (N,M,Q,λ₁), (N,M,Q,1), (N,M,Q,1)
      -> OutputHead(batch, head)        (N,Q), (N,M,Q), (N,M,Q)
```

### Routing: TP vs DP

With a process group active, each batch picks a strategy from `M` (padded atoms/molecule), `N` (molecule count), `mol_chunk`, and `dp_atom_threshold`:

```{python}
route_dp = model.training and dp_atom_threshold > 0 and M < dp_atom_threshold and N >= 2*mol_chunk
```

`dp_atom_threshold = 0` (default) always uses TP - matches pre-DP behaviour exactly. `evaluate()` runs with `model.eval()`, so `model.training` is `False` and eval/test always use TP regardless of the threshold (needed since `evaluate()` assumes both ranks see identical full-batch outputs).

TP shards atoms of the *same* molecules across ranks and needs an all-reduce mid-forward to reconstruct each atom's full neighbourhood (see `MessagePass._AllReduce`). For a bucket with very few atoms per molecule (e.g. `max_atoms=3`), that all-reduce's fixed latency cost dwarfs the tiny amount of per-rank compute it buys - DP routes those buckets by molecule instead, with **no in-model communication at all**.

**Why `N >= 2*mol_chunk` is required, not optional:** DP halves the *outer* N-chunk loop (`ceil(N/2/mol_chunk)` vs `ceil(N/mol_chunk)`), but unlike TP it does **not** halve `M` before MessagePass's own `atm_chunk` loop runs over it (TP shards `M` first; DP keeps the full `M` per molecule). So a DP-routed bucket runs roughly **2x the inner M-chunk-loop launches** TP would've had on the same bucket - that only pays for itself if halving `N` actually shrinks the outer loop. If a bucket's `N` already fits in one N-chunk (`N < 2*mol_chunk`, common for large-`M`/small-`N` buckets, since `atom_size_ceil` caps total atoms per batch), DP buys zero outer-loop reduction while still eating the un-halved-`M` cost - pure overhead, and it stays on TP. Without this guard, setting `dp_atom_threshold` too high routes exactly these buckets into DP and measurably slows training down instead of speeding it up.

**Which buckets are safe to route (the memory upper bound):** the "DP keeps the full `M` per molecule" property also bounds `dp_atom_threshold` from *above*. TP shards `M` across ranks, so its per-rank activation memory is ~`M/world_size`; DP holds the full `M`per rank, so a DP-routed bucket's peak memory scales with its full`M`. Routing a large-`M` bucket to DP therefore *raises* peak memory (toward the ~16 GB T4 OOM ceiling) while buying almost no speed - a large-`M`bucket necessarily has modest`N` (`atom_size_ceil` caps total atoms/batch), so its TP all-reduce fires few times and was never the bottleneck. The buckets that are both **safe and worth routing are high-`N`, low-`M`**: thousands of tiny molecules, where TP's all-reduce fires thousands of times and the full-`M`memory cost is negligible. So`dp_atom_threshold` has a safe *band* - high enough to catch the many-molecule buckets, low enough to leave the large-`M` buckets on TP.

**Why chunking stays even for DP buckets:** DP-routed buckets are *not* run whole - the outer N-chunk loop is load-bearing regardless of routing, because it bounds `chem_env`, shape `(N, Q, λ5, λ1)` (~3.3 MB/molecule at `Q=51, λ5=λ1=128`). A full `mols=15385` bucket would need ~51 GB for that tensor alone, so chunking the `N` loop is mandatory; DP only halves `N` per rank, it does not remove the loop.

### Data Parallel Forward

Molecule dimension N is split across `ws` ranks (`shard = ceil(N/ws)`, `n0 = rank*shard`), each rank keeping the *full* `M` atoms of its molecules:

```{rtf}
Step 1: Embed on full batch (identical on all ranks)
Step 2: Slice N dimension -> local_batch (n1-n0 molecules, full M), local_head
Step 3: MessagePass + OutputHead on local_batch, single-GPU path, no collectives
Step 4: return iq, f_mags, sigmas (shape (n1-n0, ...)), local_batch, loss_scale=(n1-n0)/N
```

Since molecules don't interact, each rank's local outputs are already final - nothing to gather. `loss_scale` rescales the rank's local-mean loss so that, after the usual grad SUM all-reduce (below), the reconstructed gradient equals the true global-mean-loss gradient rather than double-counting it.

### Tensor Parallel Forward

Atom dimension M is sharded across `ws` ranks (`shard = ceil(M/ws)`, `m0 = rank*shard`):

```{rtf}
Step 1: Embed on full batch (identical on all ranks)
        embed_head: (N, M, 1, λ₁), (N, M, Q, 1), (N, M, Q, 1)

Step 2: Shard M dimension
        shard_batch.vocab  (N, m1-m0)
        shard_head.embeds  (N, m1-m0, 1, λ₁)
        shard_head.f_mags  (N, m1-m0, Q, 1)
        shard_head.sigmas  (N, m1-m0, Q, 1)

Step 3: MessagePass on shard (each rank processes its atoms)
        msg_head: embeds (N, m1-m0, Q, λ₁), sigmas updated

Step 4: OutputHead on shard
        iq_partial    (N, Q)       I(q) sum over this rank's atoms only
        f_mags_shard  (N, m1-m0, Q)
        sigmas_shard  (N, m1-m0, Q)

Step 5: Gather
        _DistributedSum(iq_partial)  -> iq      (N, Q)     global sum
        _AllGatherDim1(f_mags_shard) -> f_mags  (N, M, Q)  cat along dim=1
        _AllGatherDim1(sigmas_shard) -> sigmas  (N, M, Q)
```

### Custom Autograd Functions

`_DistributedSum` (outer, for I(q)):

- Forward: `all_reduce(SUM)` - partial I(q) from each rank summed globally.
- Backward: identity, no communication. All ranks computed the same scalar loss from the same all-reduced I(q), so the gradient is already correct.

`_AllGatherDim1` (outer, for f_mags and sigmas):

- Forward: pad last rank's shard to `ceil(M/ws)` for uniform buffer sizes, gather along dim=1, trim to M.
- Backward: slice `grad[:, m0:m0+M_local]`. No communication.

`MessagePass._AllReduce` (inner, for features and chem_env):

- Forward: `all_reduce(SUM)`.
- Backward: `all_reduce(SUM)` again. Required here (unlike `_DistributedSum`) because each rank's gradient for its partial atom sum is genuinely partial.

---

## 8. `Loss`

`Loss` is an `nn.Module` with two registered buffers and no learnable parameters.

### Buffers

| Buffer        | Shape  | Contents                                                               |
| ------------- | ------ | ---------------------------------------------------------------------- |
| `_fmag_table` | (V, Q) | Reference form factor magnitudes from xraydb. Row 0 = zeros (padding). |
| `_q_weights_` | (1, Q) | Kratky weights`(1 + q²)` per q-point.                                  |

Form factor table construction: `q -> s = q/(4π)` converts to crystallographic s (sinθ/λ used by xraydb), then `|f(q)| = hypot(f0 + f1_chantler, f2_chantler)`. Transuranics: f0 only.

### Loss Terms

**Term 1: Kratky-weighted MSLE** (`_kratky_MSLE`):

```{Latex}
L_kratky(n, q) = (1 + q²) * (log1p(Î(q)) - log1p(I(q)))²
```

`log1p` handles the multi-decade dynamic range of I(q). The `(1+q²)` Kratky weight emphasises high-q structure; without it the Guinier region (low-q, high intensity) would dominate all gradients.

**Term 2: Form-factor penalty** (`_ff_penalty`):

```{Latex}
L_ff(n, q) = λ₆ * (1/n_atoms) * sum_m mask * (log1p(f_hat_m(q)) - log1p(f_ref_m(q)))²
```

Anchors predicted per-atom form factors to xraydb reference values, preventing the model from learning arbitrary f_mags that fit I(q) via cancellation. Atom-count normalisation makes the penalty size-independent.

**Term 3: Sigma L2 penalty** (`_sg_penalty`):

```{Latex}
L_sigma(n, q) = λ₇ * (1/n_atoms) * sum_m mask * σ_m(q)²
```

Primary mechanism preventing sigma blowup. The penalty gradient grows with σ, pulling large bandwidths back down. Combined with tanhshrink's unbounded delta in MessagePass, the system reaches equilibrium where the bilinear delta balances the MSLE gradient against the penalty.

**Total:**

```{Latex}
L_total = mean_{n,q}[ L_kratky(n,q) + L_ff(n,q) + L_sigma(n,q) ]
```

`.mean()` averages over all N molecules and all Q q-points. Per-molecule normalisation inside L_ff and L_sigma prevents large molecules from dominating.

---

## 9. Training Loop

### Optimizer

`torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)`

Standard Adam (not AdamW; weight decay is applied inside the gradient update, not decoupled). Single parameter group, no learning rate schedule.

### Per-Batch Step

```
1. Move batch to device
2. optimizer.zero_grad(set_to_none=True)
3. [autocast fp16 if amp] iq, fmags, sigmas, local_batch, loss_scale = model(batch)
4. [autocast fp16 if amp] loss = criterion.loss(iq, fmags, sigmas, local_batch, λ₆, λ₇) * loss_scale
5. scaler.scale(loss).backward()
6. [distributed] dist.all_reduce(SUM) on every param.grad   (grads still scaled)
7. scaler.unscale_(optimizer)
8. clip_grad_norm_(model.parameters(), grad_clip)
9. scaler.step(optimizer);  scaler.update()
```

Step 6 is explicit because the model uses tensor/data parallelism (see [ScatterNet §7](#7-scatternet)), not DDP. In TP mode, each rank's `param.grad` after backward is a partial sum over its atom shard, so a SUM all_reduce is required (DDP would average). In DP mode, `loss_scale = local_N / global_N` (step 4) makes the same SUM reconstruct the correct global-mean gradient from each rank's rescaled local-mean loss - one all-reduce rule serves both routing modes.

**Mixed precision (`amp`).** With `amp=True` the forward and loss run under fp16 `autocast` and gradients are scaled by a `GradScaler` (CUDA only; `amp=False` makes every scaler call a pass-through, so the fp32 path is unchanged). The RFF projection in `MessagePass` (`coord/σ @ Ω`) is forced back to fp32 via a nested `autocast(enabled=False)`: `σ` clamps at `eps_msgp≈1e-3`, so the scaled coordinate can reach ~1e5 and overflow fp16's 65504 into `cos(inf)=NaN`; that projection is a tiny inner-dim-3 contraction, so fp32 there is nearly free while the heavy `bmm`s stay in fp16 tensor cores. **Ordering matters for the manual all-reduce:** the SUM in step 6 runs on the *scaled* grads and *before* `unscale_` (step 7), so both ranks unscale the identical summed gradient, make the same inf/nan skip decision, and evolve the scale factor in lockstep - a per-rank scale divergence would corrupt the scaled-grad sum. `grad_norm` (step 8) goes to `inf` on an fp16 overflow batch; `scaler.step` then skips the update and halves the scale, and (under `verbosity="diagnostic"`) that batch is printed.

### Epoch Metrics

After all training batches, `evaluate()` runs over `val_loader` and `test_loader` with `torch.no_grad()`. Both ranks use identical loaders (same seed, no shuffle) so the TP all_reduce inside the model works correctly. Only rank 0 uses the returned `(mean_loss, R²)` for logging and checkpointing.

R² is computed in the log1p domain: `1 - SS_res / SS_tot`, where `SS_tot = sum(y²) - (sum(y))²/n` (online, one pass).

Evaluation is done once per epoch for both val and test. Val is used for checkpointing (best model selection); test is strictly a held-out report and does not influence any decision.

### Checkpoint and Resume

| File          | Contents                                | Saved when                                        |
| ------------- | --------------------------------------- | ------------------------------------------------- |
| `ckpt_best`   | model weights only                      | val_loss improves                                 |
| `ckpt_resume` | weights + optimizer + epoch + batch_idx | every`ckpt_interval_sec` seconds and at epoch end |

Mid-epoch resume: `torch.manual_seed(batcher_seed + epoch)` re-seeds the shuffle identically, then the loop fast-forwards over `batch_idx` batches via `continue`. Both checkpoints are pushed to a rclone remote (`ckpt_rclone_dest`) for Kaggle session crash durability.

---

## 10. Hyperparameter Reference

| Name                | Default | What it controls                                                                                                                           |
| ------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `lambda_1`          | 128     | Atom embedding dimension. Width of the per-atom vector throughout MessagePass.                                                             |
| `lambda_2`          | 5       | Message-passing rounds. Also bounds the maximum cumulative sigma change.                                                                   |
| `lambda_3`          | 128     | OutputHead MLP starting width. Must satisfy`lambda_3 >= 2^lambda_4`.                                                                       |
| `lambda_4`          | 4       | OutputHead halving steps. With defaults: 128->64->32->16->8->1.                                                                            |
| `lambda_5`          | 256     | RFF count. More = tighter kernel approximation, higher memory cost.                                                                        |
| `lambda_6`          | 0.1     | Form-factor penalty weight.                                                                                                                |
| `lambda_7`          | 0.1     | Sigma L2 penalty weight. Primary blowup prevention mechanism.                                                                              |
| `msg_seed`          | 42      | Seed for fixed RFF frequency matrix Ω.                                                                                                     |
| `atm_chunk`         | 1024    | Atoms per M-chunk. Reduce to lower VRAM.                                                                                                   |
| `mol_chunk`         | 32      | Molecules per N-chunk. Reduce to lower VRAM on large molecules.                                                                            |
| `dp_atom_threshold` | 101     | Batches with padded atom count`M` below this **and** molecule count `N >= 2*mol_chunk` route through DP instead of TP                      |
| `compile`           | False   | torch.compile Embed/MessagePass/OutputHead's checkpointed step functions (fullgraph=True, dynamic=True).                                   |
| `eps_embd`          | 1e-8    | Numerical floor in Embed (softplus, hypot).                                                                                                |
| `eps_msgp`          | 1e-3    | Numerical floor in MessagePass (sigma clamp, aggregate denominator).                                                                       |
| `lr`                | 3e-4    | Adam learning rate.                                                                                                                        |
| `weight_decay`      | 1e-5    | Adam L2 weight decay.                                                                                                                      |
| `grad_clip`         | 1.0     | Max gradient L2 norm before clipping.                                                                                                      |
| `epochs`            | 50      | Training epochs.                                                                                                                           |
| `batcher_seed`      | 0       | Seed for train/val/test split and per-epoch shuffle.                                                                                       |
| `atom_size_ceil`    | -1      | Max total atoms per batch (-1 = 3x largest molecule).                                                                                      |
| `num_workers`       | 4       | DataLoader worker processes.                                                                                                               |
| `ckpt_interval_sec` | 600     | Seconds between mid-epoch resume checkpoints.                                                                                              |
| `profiler`          | False   | Diagnostic run: per-rank torch.profiler + per-section wall-clock timers, then stop. Traces written per rank to`./profiler_trace/rank<r>/`. |
| `prof_warmup`       | 1       | Profiler warmup batches (profiled, discarded).                                                                                             |
| `prof_active`       | 3       | Profiler active batches (recorded). Loop runs`1 + prof_warmup + prof_active` batches; raise `prof_active` for more representative stats.   |

---

## 11. End-to-End Data Flow

```
HDF5 file
  └─ BatchSet.__getitem__
       └─ elms -> enc._encode_ions() -> vocab (N, M)
       └─ I_q                        -> iqval (N, Q)
       └─ coords                     -> coord (N, M, 3)
       └─ pad_sequence               -> Batch

Batch
  └─ Embed
       └─ _mbd(vocab)                 -> (N, M, λ₁)
       └─ PReLU                       -> (N, M, λ₁)
       └─ _f0f1, _f2, hypot          -> f_mag (N, M, Q)
       └─ _sigma bilinear, softplus  -> sigma (N, M, Q, 1)
       └─ LayerHead: embeds (N,M,1,λ₁), f_mags (N,M,Q,1), sigmas (N,M,Q,1)

  [Distributed: shard M across ranks]

  └─ MessagePass x λ₂ rounds
       └─ embeds.expand Q dim        -> (N, M, Q, λ₁)
       └─ [per round]
            └─ _pass_1: accumulate features (Nc,Q,λ₅), chem_env (Nc,Q,λ₅,λ₁)
            └─ _AllReduce (features, chem_env)
            └─ _pass_2: locality, weights, agg, MishGLU gate -> new embeds
                        bilinear + tanhshrink + softplus      -> new sigmas
       └─ LayerHead: embeds (N,M,Q,λ₁), sigmas updated, f_mags unchanged

  └─ OutputHead
       └─ [per M-chunk]
            └─ bilinear(embeds, f_mags) -> (N,mc,Q,λ₃)
            └─ Mish -> MLP -> softplus  -> (N,mc,Q)
            └─ * f_mags² * mask -> sum over atoms
       └─ iq_accum (N, Q)

  [Distributed: _DistributedSum(iq), _AllGatherDim1(f_mags, sigmas)]

Loss
  └─ _kratky_MSLE:  (1+q²)*(log1p(Î)-log1p(I))²    (N,Q)
  └─ _ff_penalty:   λ₆*(log1p(f̂)-log1p(f_ref))²/n  (N,Q)
  └─ _sg_penalty:   λ₇*σ²/n_atoms                   (N,Q)
  └─ .mean() -> scalar

Optimizer: Adam(SUM-reduced grads, clip at grad_clip) -> parameter update
```

---

## 12. Profiling and Optimization on Kaggle

### Running the Profiler

Set `profiler: true` in your YAML config or pass `--profiler` on the CLI. This runs a short **diagnostic** instead of normal training: the loop runs `1 + prof_warmup + prof_active` batches (defaults `1 + 1 + 3 = 5`; tune with `--prof_warmup`/`--prof_active`), then stops - no eval or checkpointing. Adjust `prof_active` higher (e.g. 20–50) to average over many buckets.

### Which Buckets Get Profiled

The `1 + prof_warmup + prof_active` budget is split three ways across bucket **metadata** (atom counts only - no tensors loaded), not drawn randomly from the shuffled train loader:

1. **Heaviest by `N*M_shard**` (`shard = ceil(max_atoms/world_size)`) - the compute-time proxy. Dominated by huge-`N`/tiny-`M` buckets, since TP's all-reduce cost scales with the *number* of N-chunks (`N/mol_chunk × λ2` rounds), not with `M` - a bucket with thousands of tiny molecules pays that fixed per-N-chunk cost thousands of times over.
2. **Heaviest by raw `M**` - the memory-risk proxy. A large-`M`/small-`N` bucket can rank low on `N*M_shard` (small `N` keeps the product down) while still having the largest per-chunk RFF tensors (`atm_chunk`-sized chunks are actually full when `M` is large) - invisible to the first ranking alone.
3. **A band around the median `N*M_shard**` - a "regular" batch baseline, so the two worst-case groups have something typical to compare against instead of only ever showing outliers.

Each group is deduplicated against the ones before it. The single heaviest bucket by `N*M_shard` stays first (`bi=0`, the profiler's "wait" step) so it gets a clean `peak_alloc` reading with no torch-trace overhead. The startup log line reports one example bucket from each group (`mols x atoms`) so you can sanity-check what got selected.

**The section-timer report breaks these three groups out separately** (`---- per-group breakdown ----`, printed after the combined summary), each with its own batch count, mean `compute`/`forward`/`backward`/`grad_allreduce` (ms/batch), and peak `peak_alloc`. This matters because the combined numbers blend three structurally different populations: comparing a profiler run's *combined* average against an older run (or against a run with different `dp_atom_threshold`/group sizes) isn't apples-to-apples, since the mix of bucket types changed, not just the routing logic. Compare **within the same group** across runs instead (e.g. `heaviest N*M_shard`'s `grad_allreduce` before vs. after changing `dp_atom_threshold`) to isolate the effect you actually changed.

Two decoupled layers of profiling run on **every rank**:

1. **Section timers** - a CUDA-synced wall-clock breakdown printed at the end, over the **full** `prof_active` window (so averages are representative). Each rank prints time spent in `data_wait` / `h2d` / `forward` / `loss` / `backward` / `grad_allreduce` / `clip` / `step`, plus the heaviest batches by data-wait and by compute (with molecule count, max atoms, and real atoms). These cost ~no extra memory. Comparing the same section **across ranks** localizes tensor-parallel skew: a rank fast in compute but slow in `grad_allreduce` is *waiting* on a slower peer - the usual cause of the NCCL `ALLREDUCE` watchdog timeout.
2. **torch.profiler** - a CPU+CUDA TensorBoard trace per rank at `./profiler_trace/rank<r>/` for kernel-level drill-down. This buffers every op in host RAM and materializes them at export, so it is memory-heavy: it samples only `min(prof_active, 3)` steady-state steps regardless of `prof_active`, and runs with `with_stack`/`profile_memory` **off** (a long active window or `with_stack` triggers the host OOM-killer → worker `SIGKILL`). Raising `prof_active` lengthens the cheap section-timer window, not the heavy trace.

On Kaggle, use the dedicated profiler cell (which sets `profiler=True`, `prof_warmup`, `prof_active` in `RunConfig`), then run the TensorBoard cell immediately after:

```python
%load_ext tensorboard
%tensorboard --logdir ./profiler_trace
```

On the CLI, view the trace with:

```
tensorboard --logdir ./profiler_trace
```

Or load the `.pt.trace.json` file directly at `chrome://tracing`.

### Interpreting the Profiler

**Start with the section timers** (printed to stdout) before opening the trace - they tell you which bucket of time to chase. High `data_wait` ⇒ the DataLoader is starving the GPU (see below); high `forward`/`backward` ⇒ go into the trace. A large `grad_allreduce` (or `backward`, which contains the in-model `_AllReduce`) on one rank but not the other ⇒ tensor-parallel skew, not a slow collective.

**High `data_wait**`: CPU/IO is the bottleneck. Each `BatchSet.__getitem__` re-opens the HDF5 file and runs a Python encode loop per atom, so heavy buckets stall. Raise `num_workers`; if `data_wait` stays high and tracks the heavy buckets in the report, cache `Batch` objects as `.pt` files or hoist the HDF5 handle out of `__getitem__`.

**GPU idle gaps** between CUDA kernels in the trace: CPU is the bottleneck. Causes are usually DataLoader (HDF5 reads), Python overhead between chunks, or the per-parameter `all_reduce` loop.

**Short CUDA bars, lots of idle**: chunks are too small and kernel launch overhead dominates. Increase `atm_chunk`.

**OOM**: chunks are too large. Reduce `atm_chunk` or `mol_chunk`.

**Memory fragmentation**: `torch.cuda.memory_reserved()` greatly exceeds `torch.cuda.memory_allocated()`. If OOM despite low `memory_allocated`, reduce `atm_chunk` to create less fragmentation.

**Gradient all_reduce**: the per-parameter `dist.all_reduce` loop is serial. If it shows up as a significant bottleneck, flatten all gradients into one buffer, all_reduce once, then copy back. Only worth the complexity if the profiler confirms it.

### Quick Optimization Checklist (T4 x2)

| Setting       | Recommendation                                       |
| ------------- | ---------------------------------------------------- |
| `num_workers` | 2-4 (0 = serial, >4 = diminishing returns)           |
| `pin_memory`  | True (already set in train.py for GPU)               |
| `atm_chunk`   | Start at 1024; raise to 2048 if GPU is underutilised |
| `mol_chunk`   | Start at 16; drop to 8 if OOM on large molecules     |
| `verbosity`   | `"batch"` for loss + memory stats every 20 batches   |
| `max_batches` | Set to 20 for a quick smoke test before a full run   |

---

## Appendix

All runs on 2x T4, λ₁=λ₅=128, λ₂=5.

---

### A1. mol_chunk and atm_chunk optimizations

Takeaway: compute is roughly invariant to chunk shape; the choice trades kernel-launch count against peak memory. `56 x 100` was chosen for the fewest launches. The true worst-bucket peak under the current profiler is 15.12G on a 16G T4, which the old shuffle-window sampler understated at 14.98G.

Note: this predates the current profiler and is **not directly comparable to the table above**. It was taken with the older shuffle-window sampler (no `heavy_nm`/`heavy_m`/`median` stratification, so the true worst bucket was usually missed and peaks are understated), and before the conditional-checkpointing episode. Treat these as within-table comparisons only.

| `mol_chunk` | `atm_chunk` | product | result          | notes                                                       |
| ----------- | ----------- | ------- | --------------- | ----------------------------------------------------------- |
| 64          | 64          | 4096    | ok (baseline)   | ~15.3 / 18.2 s/batch (r0/r1)                                |
| 128         | 16          | 2048    | ok              | atm too small, weak matmul contraction                      |
| 104         | 46          | 4784    | ok              | more launches (69,507)                                      |
| **56**      | **100**     | 5600    | **ok (chosen)** | fewest launches (58,889), balanced; atm_chunk reduced to 80 |
| 32          | 200         | 6400    | ok, at ceiling  | peak 14.98G; ~14.5 / 17.3 s/batch                           |
| 64          | 128         | 8192    | OOM             |                                                             |
| 32          | 512         | 16384   | OOM             | looked stable in the profiler window, OOM'd mid-run         |
| 256         | 256         | 65536   | OOM             |                                                             |

---

### A2. dp_atom_threshold optimizations

- TP (tensor parallel)
  - split one molecule's atoms across both GPUs. Each rank holds part of every molecule, so it must all-reduce mid-forward to reconcile the shared context. Good for few, large molecules; costly when molecules are tiny (thousands of little collectives).
- DP (data parallel)
  - split the molecules across both GPUs. Each rank owns whole, disjoint molecules and needs no mid-forward communication, just the one gradient all-reduce at the end of the step. Good for many small molecules.

The model routes each batch to whichever fits (dp_atom_threshold, see §7): high-N/low-M buckets go DP, large-M buckets stay TP.

Worst-rank = the epoch-limiting rank (always rank 1 here, see the TP shard-imbalance note in §12). "in-model all-reduce" is the NCCL collective fired inside the TP forward. Two transitions matter; the **speed cliff is between 10 and 63**: at 10 only the `M=3` bucket routes DP (a degenerate DP, most buckets still pay the TP all-reduce), so it stays slow; by 63 every high-`N`/low-`M` bucket routes DP and the all-reduce (68s, 47% of CUDA) disappears, giving a 1.8x speedup. The **memory step is between 101 and 102**: at 102 the `mols=550, M=101` bucket flips to DP, which holds the full `M=101` per rank instead of sharding it, pushing peak from 13.55G to 14.51G for no speed gain. `dp_atom_threshold = 101` is chosen because the strict `M < threshold` gate keeps that bucket on TP: full speedup at the lower peak (~2.4G headroom). See §7 for the safe-band reasoning.

| `dp_atom_threshold` | worst-rank s/batch | peak CUDA mem | Self CUDA total | Small molecule all-reduce time |
| ------------------- | ------------------ | ------------- | --------------- | ------------------------------ |
| 0 (all TP)          | 40.5               | 13.55G        | 145s            | 68s (47%)                      |
| 10                  | 33.7               | 13.55G        | 145s            | 68s (47%)                      |
| 63                  | 22.5               | 13.55G        | 62s             | eliminated                     |
| 100                 | 22.5               | 13.55G        | 62s             | eliminated                     |
| **101 (chosen)**    | **22.6**           | **13.55G**    | **62s**         | **eliminated**                 |
| 102                 | 22.5               | 14.51G        | 62s             | eliminated                     |
| 125                 | 22.5               | 14.51G        | 62s             | eliminated                     |
| 250                 | 22.4               | 14.51G        | 62s             | eliminated                     |
| 500                 | 22.4               | 14.51G        | 62s             | eliminated                     |
| 1000                | 22.5               | 14.51G        | 62s             | eliminated                     |
| 2000                | 22.2               | 14.51G        | 62s             | eliminated                     |
| 3000                | 22.4               | 14.51G        | 62s             | eliminated                     |
| 4000                | 22.4               | 14.51G        | 62s             | eliminated                     |

---
