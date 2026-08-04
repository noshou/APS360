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
  - [Low-q coherent limit](#low-q-coherent-limit)

---

## Notation


| Symbol  | Meaning                                                                    |
| --------- | ---------------------------------------------------------------------------- |
| N       | molecules in a batch                                                       |
| M       | atoms per molecule (padded to the longest in the batch)                    |
| Q       | number of q-points in the scattering grid                                  |
| λ₁    | atom embedding dimension (`lambda_1`, default 128)                         |
| λ₂    | message-passing rounds (`lambda_2`, default 4)                             |
| λ₃    | OutputHead MLP starting width (`lambda_3`, default 64)                     |
| λ₄    | OutputHead MLP halving steps (`lambda_4`, default 4; ladder ends at width 2) |
| λ₅    | Random Fourier Features count (`lambda_5`, default 128)                    |
| λ₆    | form-factor penalty weight (`lambda_6`, default 1.0)                       |
| Nc      | molecules per N-chunk (`mol_chunk`)                                        |
| mc      | atoms per M-chunk (`atm_chunk`)                                            |
| V       | VOCAB size = len(VOCAB) + 1 (row 0 is padding)                             |
| ε_e    | `eps_embd` numerical floor in Embed (default 1e-8)                         |
| ε_m    | `eps_msgp` numerical floor in MessagePass (default 1e-3)                   |
| r_m     | Cartesian coordinates of atom m, shape (3,) in Å                          |
| e_m     | embedding vector of atom m, shape (λ₁,)                                  |
| f_m(q)  | form factor magnitude of atom m at q-point q, scalar                       |
| σ_m(q) | RFF kernel bandwidth of atom m at q-point q, scalar                        |
| φ_m(q) | RFF feature vector of atom m at q-point q, shape (λ₅,)                   |

---

## 1. Vocabulary and Atom Tokens

### `Preprocess.VOCAB`

VOCAB is a module-level singleton built at import time from `xraydb.get_xraydb().f0_ions()`, which returns all ions and elements with a tabulated Cromer-Mann atomic form factor f0 in the NIST xraydb database.

Atom indices are 1-based. Index 0 is reserved as the padding sentinel (`batch.vocab == 0` means no atom). `nn.Embedding` is constructed with `len(VOCAB)+1` rows and `padding_idx=0`.

### Anomalous Correction Handling


| Case                                        | Treatment                                                                      |
| --------------------------------------------- | -------------------------------------------------------------------------------- |
| Transuranics (Np, Pu, Am, Cm, Bk, Cf, ions) | `f0` only; Chantler f1/f2 unavailable or unreliable                            |
| Special cases (`siva->si`, `cval->c`, etc.) | Remapped to canonical base element before f1/f2 lookup                         |
| Normal                                      | Charge suffix stripped for f1/f2 (`fe2+` -> `fe`); full ion string kept for f0 |

---

## Dataset

The training dataset (`noshou/iq_train_set` on HuggingFace) contains 1,044,583 molecules with precomputed I(q) curves. Full schema, file descriptions, download instructions, and group statistics are in `[Preprocess/README.md](Preprocess/README.md)`.

`iq_train_set-ENCODING.sqlite3` (~860 MB) is the primary training index. It maps every molecule to its atom count and VOCAB indices so the data pipeline never needs to scan the 66 GB HDF5 file during training. It is hosted alongside the HDF5 file on HuggingFace and Kaggle (not tracked in this git repo) -- see [Preprocess/README.md](Preprocess/README.md) for download instructions.

## 2. Data Pipeline

### 2.1 `Batch`

`Batch` is a frozen dataclass with runtime shape checking via jaxtyping + beartype.


| Field   | Shape     | Dtype   | Contents                                               |
| --------- | ----------- | --------- | -------------------------------------------------------- |
| `vocab` | (N, M)    | int64   | VOCAB index per atom. 0 = padding.                     |
| `iqval` | (N, Q)    | float32 | Ground-truth I(q) from simulation.                     |
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
| -------------- | ----------------------------------- | ----------------------------------------------------- |
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

Asymmetric masking: `f_mags` and `sigmas` are multiplied by the padding mask; `embeds` is not. Zeroing f_mags is sufficient to exclude padding atoms since every intensity contribution carries a factor of f_mag (f_mag in the coherent term, f_mag^2 in the incoherent one).

ε is passed at forward time rather than fixed at construction, allowing the numerical floor to be tuned without changing the model.

Converts each atom's VOCAB index into three tensors that feed the rest of the pipeline:


| Output   | Shape           | Meaning                          |
| ---------- | ----------------- | ---------------------------------- |
| `embeds` | (N, M, 1, λ₁) | Learned per-atom identity vector |
| `f_mags` | (N, M, Q, 1)    | Estimated form factor magnitude  |
| `sigmas` | (N, M, Q, 1)    | RFF kernel bandwidth per q-point |

### `Embed` Layers

1. **Learnable Parameters**


| Parameter       | Shape        | Notes                                                     |
| ----------------- | -------------- | ----------------------------------------------------------- |
| `_mbd.weight`   | (V, λ₁)    | Embedding table. Row 0 frozen at zero via`padding_idx=0`. |
| `_f0f1.weight`  | (Q, λ₁)    | Linear: embedding -> real part of form factor.            |
| `_f0f1.bias`    | (Q,)         |                                                           |
| `_f2.weight`    | (Q, λ₁)    | Linear: embedding -> imaginary part of form factor.       |
| `_f2.bias`      | (Q,)         |                                                           |
| `_prelu.weight` | (λ₁,)      | One learned negative slope per embedding channel.         |
| `_sigma.weight` | (Q, λ₁, Q) | Bilinear weight: embedding x f_mag -> sigma logit.        |
| `_sigma.bias`   | (Q,)         |                                                           |

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
  arctan window(... + log_env)          (N, M, Q)    log σ into (log floor, log max)
  exp(...)                              (N, M, Q)    strictly positive
    .unsqueeze(-1) * mask.unsqueeze(-1) (N, M, Q, 1) zero padding atoms
  -> sigma                              (N, M, Q, 1)

  # pack into LayerHead
  embed.unsqueeze(-2)  -> embeds        (N, M, 1, λ₁)  Q=1, broadcastable
  f_mag.unsqueeze(-1) * mask -> f_mags  (N, M, Q, 1)
  sigma                -> sigmas        (N, M, Q, 1)
```

3. **Activations**


| Activation        | Location               | Behaviour                                                                        |
| ------------------- | ------------------------ | ---------------------------------------------------------------------------------- |
| `nn.PReLU(λ₁)`* | After embedding lookup | Learnable per-channel negative slope. Acts on dim=1, hence the double-transpose. |
| `arctan_log_sigma_window` | On sigma logits | Squashes log σ into `(log sigma_floor, log sigma_max)`; `exp` then makes it positive. Shared with MessagePass. |

**Comparison of `PReLU`  to other activation functions:*


| Activation | λ₁ | Val loss | Val R² | Test R² |
| ------------ | ------ | ---------- | --------- | ---------- |
| LeakyReLU  | 64   | 0.77     | 0.71    | 0.69     |
| PReLU      | 128  | 0.67     | 0.74    | 0.73     |
| PReLU      | 64   | 0.93     | 0.62    | 0.61     |
| LeakyReLU  | 128  | NaN      | NaN     | NaN      |
| ELU        | 64   | 2.44     | -0.05   | -0.01    |
| Mish       | 64   | NaN      | NaN     | NaN      |

---

## 4. `LayerHead`

`LayerHead` is a `NamedTuple` (immutable, typed) passed between Embed, MessagePass, and OutputHead.


| Field    | Shape from Embed | Shape after MessagePass | Meaning                                                                 |
| ---------- | ------------------ | ------------------------- | ------------------------------------------------------------------------- |
| `embeds` | (N, M, 1, λ₁)  | (N, M, Q, λ₁)         | Per-atom identity vector. Q=1 on construction, expanded by MessagePass. |
| `f_mags` | (N, M, Q, 1)     | unchanged               | Form factor magnitude. Trailing 1 broadcasts over λ₁.                 |
| `sigmas` | (N, M, Q, 1)     | updated per round       | RFF bandwidth. Trailing 1 broadcasts over λ₁.                         |

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

  `_proj_agg`, `_sigbilin`, and `_rms_norm` are each an `nn.ModuleList` of length λ₂: every message-passing round gets its own instance (`_proj_agg[i]`, `_sigbilin[i]`, `_rms_norm[i]`), not a single set of weights reused across rounds.


| Parameter             | Shape         | Notes                                                |
| ----------------------- | --------------- | ------------------------------------------------------ |
| `_proj_agg[i].weight` | (2λ₁, λ₁) | MishGLU projection: agg -> [p1, p2]                  |
| `_proj_agg[i].bias`   | (2λ₁,)      |                                                      |
| `_biasterm`           | (λ₅,)       | Learnable RFF phase offsets b (shared across rounds) |
| `_sigbilin[i].weight` | (1, λ₁, Q)  | Bilinear: (embedding, f_mag) -> sigma delta          |
| `_sigbilin[i].bias`   | (1,)          |                                                      |
| `_rms_norm[i].weight` | (λ₁,)       | RMSNorm scale                                        |

  Buffers (fixed):


| Buffer      | Shape     | Notes                                          |
| ------------- | ----------- | ------------------------------------------------ |
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

  # sigma update (rounds 0 .. λ₂-2 only; the last round has no sigma head)
  f_in           (Nc, mc, Q)      ffs_slice.squeeze(-1), no expand needed
  delta          (Nc, mc, Q, 1)   ptanhshrink(qdiag(RMSNorm(new_emb), f_in))
  new_sig        (Nc, mc, Q, 1)   exp(window(log sig_slice + delta)) * mask
```

3. **Activations**


| Activation         | Location                | Behaviour                                                                                          |
| -------------------- | ------------------------- | ---------------------------------------------------------------------------------------------------- |
| `cos`              | RFF feature computation | Core of the RFF kernel approximation.                                                              |
| `F.mish` (p2 path) | MishGLU gate            | `x*tanh(softplus(x))`. Near zero when p2 << 0 (gate closed); near-linear when p2 >> 0 (gate open). |
| `nn.RMSNorm`       | After locality/weights  | Normalises aggregate magnitude, preventing residual stream from compounding across rounds.         |
| `PTanhShrink`      | Sigma delta             | `y - c*tanh(y/c)` with a bounded learned width `c`. Cubic near 0 (sticky region); `f' = tanh²(y/c)`. |
| `arctan_log_sigma_window` | Sigma output     | Squashes log σ back into `(sigma_floor, sigma_max)` every round. Same function Embed uses.        |

- MishGLU Gate
  `_proj_agg` is `nn.Linear(λ₁, 2λ₁)`. The output is split into p1 (value path) and p2 (gate path). `gate = p1 * Mish(p2)` is added to the atom embedding as a residual. The final `* mask` zeroes contributions from padding atoms.
  The GLU pattern lets the network decide per-channel and per-q-point whether to incorporate the neighbourhood context, rather than always adding the full aggregate.
- Sigma Update
  ```{rtf}
  σ_new = exp( window( log σ_old + ptanhshrink( qdiag(RMSNorm(e_updated), f_mag) ) ) )
  ```

  `_sigbilin[r]` is a `PTanhShrink` wrapping a `QDiagBilin(λ₁, Q, q_points=Q)`: one weight matrix **per q-point**, so the sigma delta at q depends on that q's embedding coupled against the atom's whole form-factor spectrum. That is `Q * λ₁ * Q` weights per round rather than `λ₁ * Q`, a 51x capacity increase in the sigma path. It costs nothing in memory because the output index and the input's q axis are the same index: contracting λ₁ first gives `(Nc, mc, Q, Q)` and never materializes the `(Nc, mc, Q, Q, Q)` a generic bilinear would (1.012 GiB vs 20.3 MiB at `(4, 512, 51)`).

  The update is additive in **log** space and is squashed back into `(sigma_floor, sigma_max)` by the same `arctan_log_sigma_window` Embed uses, so the window now binds the σ that actually reaches the kernel and the loss, not just Embed's round-0 σ. This is load-bearing: σ is the RBF bandwidth and the kernel forms `r/σ`, so `d(kernel)/dσ` carries a `σ^-2` pole. The previous `squareplus(σ + tanhshrink(...))` was unbounded below (it asymptotes to `b/(4|x|) -> 0⁺`), which discarded the window on round 1: measured `min σ = 0.0107` with **41% of entries under Embed's 0.5 floor**, and `|dL/dσ|` reaching 1.5e8 at σ = 0.002 against 2.4e3 at the floor, an exact `σ^-2` scaling. Under a global-norm clip that rescaled every other parameter's gradient toward zero, which is what made the σ pathway untrainable.

  The head reads `RMSNorm(new_emb)`, not the raw residual. It was the only unnormalized linear map in the block, with a pre-activation std of `sqrt(Q/3)*std(e)*f_rms` ≈ 25-90, which in log-σ units saturates the window immediately.

  The sticky property near zero is retained and is now explicit: `PTanhShrink` is cubic near the origin (`f(y) ≈ y³/3c²`, `f' = tanh²(y/c)`), so early in training when the bilinear has weak outputs sigmas stay stable rather than wandering. Its width `c` is **bounded** rather than free, because `dg/dc` has the same sign for every `y`: a free `c` has a strictly monotone incentive to grow (measured 1 -> 30 in 600 SGD steps, still climbing), which both closes the layer and makes `dg/dlog_c -> -c` unbounded.

  The last round (`r = λ₂-1`) has **no** sigma head at all. σ feeds the *next* round's kernel and `OutputHead` ignores σ entirely, so that round's σ weights had no path to the loss and measured `grad = None`.

---

## 6. `OutputHead`

Collapses per-atom representations into a predicted I(q) curve per molecule, as a sum of the two limits of the Debye equation:

```{rtf}
I(q) = (sum_j w_j(q) f_j(q))²  +  sum_j c_j(q) f_j(q)²
        \-- coherent --/           \-- incoherent --/
```

The MLP emits two per-atom, per-q channels (w, c) instead of one. In the full Debye equation `I(q) = sum_j sum_k f_j f_k sinc(q r_jk)`, the two terms above are its endpoints. At q -> 0 every `sinc(q r_jk) -> 1`, so the double sum factorizes to `(sum_j f_j)²`, which scales like M². At high q the off-diagonal pairs oscillate away, leaving the diagonal `sum_j f_j²`, which scales like M. Squaring a sum over atoms *is* the pairwise double sum, so the coherent term is recovered from one O(M) pass rather than an O(M²) pair sum.

Both channels stay O(1) at every q; the M² scaling comes from the squaring op, not from learned magnitude. This matters because a single linear sum-of-f² head (the previous design) cannot reach `(sum f)²` at any parameter setting. It could only approximate low q by inflating its per-atom weights to ~M, which from standard init left high q roughly correct immediately while low q started M-fold low and converged far slower. See [Appendix: low-q coherent limit](#low-q-coherent-limit) for the full diagnosis.

### `OutputHead` Layers

1. **Learnable Parameters**


| Parameter               | Shape           | Notes                                                |
| ------------------------- | ----------------- | ------------------------------------------------------ |
| `_bilinear.weight`      | (λ₃, λ₁, 1) | Bilinear: (embedding, f_mag scalar) -> λ₃ features |
| `_bilinear.bias`        | (λ₃,)         |                                                      |
| `_mlp / layer_i.weight` | varies          | MLP linear layers (halving pyramid, terminal width 2) |
| `_mlp / layer_i.bias`   | varies          |                                                      |

  MLP with `lambda_3=64, lambda_4=4`:

```{rtf}
  Linear(64->32) -> Mish -> Linear(32->16) -> Mish -> Linear(16->8) -> Mish -> Linear(8->4) -> Mish -> Linear(4->2)
```

  The ladder always terminates at width 2, one channel each for w and c. No Mish after the final linear; the two channels then take **different** activations, because the two Debye limits have different sign requirements (see [Why w is signed](#why-w-is-signed)):

  - `c` (incoherent) goes through `SqrP`, square-plus, `(x + sqrt(x² + b))/2` with a learned `b`. It weights `sum_j c_j f_j²`, a sum of squares, so it must stay positive.
  - `w` (coherent) goes through `PBId`, a bent identity, and is **signed**. `coh` is squared afterwards, so `I(q) >= 0` holds either way.

  The final layer's **weight is scaled by 0.05 and its bias zeroed**, and each channel is given a constant offset solving `PBId(x) = 1` and `SqrP(x) = 1`, so both start at unity at step 0: `I(0) = (sum f)²` is correct before any training. At the default `b = 4` the incoherent offset is exactly `x = 0`, where `SqrP'(0) = 1/2` is at its maximum.

  The offsets are **constants**, and in particular contain no `M`. An intermediate version targeted `w = c = 1/sqrt(M)` instead; that makes `coh = sqrt(M)*f` and so `I(0) = M*f²`, a factor of M *below* the true `(sum f)² = M²*f²` (measured `I_pred/I_true` = 1.7e-2 at M=64 down to 3.0e-4 at M=3426), which in a log-space loss is a multi-nat residual that does not shrink with training. It also computed `M` from the **chunk** mask rather than the molecule's, so identical weights on the same batch gave I values spanning 82x across `atm_chunk` in {64 … 1024}. With the constant offsets, `I_pred/I_true = 1.09` at init and chunk-invariance is exact to fp32 accumulation order (5e-7 relative).

  Note this init is correct at `q -> 0` **only**. With `w = c = 1` at every q the head emits `(sum f)² + sum f²` across the whole grid, so the high-q limit starts too large by a factor of ~M (measured on real geometries: 351% error at 28 atoms, 33 047% at 3 426, 96 985% at 5 436). Reaching `sum f²` requires `w -> 1/sqrt(M)` at high q, which is what the signed channel exists to make reachable.
2. **Forward Pass**

```{rtf}
  # inputs from MessagePass:
  msg_head.embeds  (N, M, Q, λ₁)
  msg_head.f_mags  (N, M, Q, 1)
  msg_head.sigmas  (N, M, Q, 1)

  mask             (N, M, 1)    padding_mask().unsqueeze(-1), float
  coh_accum        (N, Q)       zeros
  inc_accum        (N, Q)       zeros

  # per M-chunk:
  emb_c    (N, mc, Q, λ₁)
  fmag_c   (N, mc, Q, 1)
  mask_c   (N, mc, 1)

  bilinear(emb_c, fmag_c)      (N, mc, Q, λ₃)  λ₁-vec x 1-scalar -> λ₃ features
  F.mish(...)                  (N, mc, Q, λ₃)
  _mlp(...)                    (N, mc, Q, 2)   halving pyramid
  _pbid([..., :1]).squeeze(-1) (N, mc, Q)      w, coherent   - SIGNED
  SqrP([..., 1] + inc_offset)  (N, mc, Q)      c, incoherent - positive

  w * fmag_c.squeeze(-1)   * mask_c  (N, mc, Q)  coherent, f¹ (NOT squared here)
  c * fmag_c.squeeze(-1)^2 * mask_c  (N, mc, Q)  incoherent, f²
  coh_accum += (...).sum(dim=1)       (N, Q)
  inc_accum += (...).sum(dim=1)       (N, Q)

  # return (coh_accum is NOT squared here - see note below):
  coh_accum          (N, Q)    coherent partial, sum(w * f)
  inc_accum          (N, Q)    incoherent partial, sum(c * f²)
  f_mags.squeeze(-1) (N, M, Q)
  sigmas.squeeze(-1) (N, M, Q)

  # the caller finishes it, once the sums cover every atom:
  OutputHead.combine(coh, inc) -> coh**2 + inc   (N, Q)   predicted I(q)
```

  **Why `forward` returns the coherent sum unsquared.** Squaring a *partial* sum computes `sum_parts (sum_j w_j f_j)²` where the correct quantity is `(sum_parts sum_j w_j f_j)²`. The two differ by every cross-part pair term, so every atom pair whose members land in different parts is silently dropped from the coherent sum, and the head degrades toward the incoherent-only behaviour it was written to replace.
  The atom dimension gets split at two independent levels, and the invariant has to hold at both. Inside `forward`, the `out_chunk` loop splits it, handled by accumulating over all chunks before returning. One level up, the tensor-parallel path (see [§7](#tensor-parallel-forward)) shards it across ranks, so `forward` cannot square at all: it hands back `coh_accum` and `inc_accum` raw, and each caller in `ScatterNet.forward` calls `OutputHead.combine` only once its sums cover every atom of the molecule. The single-GPU and DP paths combine immediately (both hold whole molecules); the TP path stacks the two partials, all-reduces them in one collective, and combines after.
  Nothing in a normal run surfaces a violation. The truncated model is still a smooth function of the parameters, so it still trains and the loss still goes down; it just converges to a different (chunk- or rank-dependent) function than intended, and no loss curve, R², or per-q plot distinguishes the two. The only reliable checks are explicit invariance tests: forward the same batch at `out_chunk = M` (no cross-chunk pairs to lose) and at `out_chunk = 1` (every pair is cross-chunk), and separately at world sizes 1 and 2, asserting I(q) matches in both. Measured, squaring before the rank reduction gives a 48% error at world size 2 while passing every other diagnostic in the repo.
3. **Activations**


| Activation   | Location           | Behaviour                                                |
| -------------- | -------------------- | ---------------------------------------------------------- |
| `F.mish`     | After bilinear     | Applied to bilinear features before the MLP.             |
| `nn.Mish`    | Between MLP layers | Between each pair of linear layers except the last.      |
| `SqrP` | On MLP channel 1   | Square-plus `(x + sqrt(x² + b))/2`, learned `b`. Keeps the incoherent channel `c` strictly positive; it weights a sum of squares. Note its negative tail decays as `b/(4|x|)`, not exponentially, so it cannot switch a channel fully off. |
| `PBId`       | On MLP channel 0   | Bent identity, `w·(sqrt(x²+1)-1)/2 + x + b`. Leaves the coherent channel `w` **signed**. |

#### Why w is signed

`c` weights `sum_j c_j f_j²`, the diagonal term, so it has to be positive. `w` does not: `coh = sum_j w_j f_j` is squared before it reaches `I(q)`, so non-negativity is guaranteed by the architecture rather than by the activation.

Forcing `w > 0` costs the head the only cheap route to the high-q limit. A sum of M strictly-positive terms satisfies `coh >= M · min_j(w_j f_j)`, so it can shrink only if *every* `w_j` shrinks together, toward the molecule-size-dependent value `1/sqrt(M)` (0.577 at M=3, 0.080 at M=156, 0.013 at M=6046). Under `softplus` those sit at pre-activations of −0.25, −2.48 and −4.35, where the derivative has fallen to 0.44, 0.077 and 0.0128 — so the head must learn a size-dependent envelope while its gradient is being suppressed by up to 34x.

A signed `w` reaches the same place by **destructive interference**: the `sqrt(M)` reduction falls out of random-phase cancellation with no size envelope to learn. That is also the physical mechanism. The coherent amplitude is `sum_j f_j exp(i q · r_j)`, and it decays at high q because the phases cancel, not because the atomic amplitudes shrink, so `w_j(q)` acts as a per-atom stand-in for the `cos(q · r_j)` phase factor.

`PBId` rather than a bare identity, `PReLU`, or `Mish`: all four are signed, but the channel's job is to *cross zero cleanly*, and only the first two do. `Mish` has its minimum at `x = −1.1924` where the derivative is 8.3e-6, is non-monotonic on the negatives (each value is reachable from two pre-activations), and clamps them at −0.309 while leaving the positives unbounded. `PReLU`'s kink sits exactly at the crossing point. `PBId` is monotone with `f'(0) = 1` and `f' ∈ (1 − |w|/2, 1 + |w|/2)`, and its learned bend `w` interpolates between the exact identity (`w = 0`) and the standard bent identity (`w = 1`). It is left unconstrained; past `|w| = 2` it loses monotonicity and folds, so bound it with `2·tanh` if a run drives it there.

---

## 7. `ScatterNet`

Top-level module. Wraps Embed, MessagePass, and OutputHead, and routes each batch to one of two parallelism strategies across GPUs: atom-dimension tensor parallelism (TP), or, for small-atom-count batches during training, molecule-dimension data parallelism (DP).

### Module Registry


| Submodule | Type          |
| ----------- | --------------- |
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

`dp_atom_threshold = 0` (default) always uses TP - matches pre-DP behaviour exactly.

`forward()` applies the same routing rule in eval as in train (it does **not** branch on `model.training` — feeding the compiled step functions eval-only shapes caused a recompile storm). `evaluate()` all-reduces its six metric accumulators for a DP-routed bucket, so both routings give identical numbers. In practice eval lands on TP for nearly every bucket anyway: a val/test batch holds only ~15% of its bucket's molecules, and that small `N` fails `route_dp`'s `N >= 2*mol_chunk`. The plots pass forces TP outright (`eval_plots._force_tp`), since `Baselines/run/metrics.py` has no notion of cross-rank shards.

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
        coh_partial   (N, Q)       sum(w*f)  over this rank's atoms only
        inc_partial   (N, Q)       sum(c*f²) over this rank's atoms only
        f_mags_shard  (N, m1-m0, Q)
        sigmas_shard  (N, m1-m0, Q)

Step 5: Gather
        stack((coh_partial, inc_partial), -1)      (N, Q, 2)  one collective
        _DistributedSum(...)         -> parts     (N, Q, 2)  global sums
        OutputHead.combine(parts[...,0], parts[...,1]) -> iq (N, Q)
        _AllGatherDim1(f_mags_shard) -> f_mags    (N, M, Q)  cat along dim=1
        _AllGatherDim1(sigmas_shard) -> sigmas    (N, M, Q)
```

The reduce must precede the square. `OutputHead.forward` therefore returns both partials unsquared and never calls `combine` itself; only `ScatterNet.forward` does, once the sums span every atom. Squaring per rank would compute `sum_ranks (sum_j w_j f_j)²` and drop every atom pair straddling a shard boundary, measured at a 48% error on 2 ranks. See [§6](#6-outputhead) for the full argument. The two partials are stacked so this stays a single collective rather than two.

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
| --------------- | -------- | ------------------------------------------------------------------------ |
| `_fmag_table` | (V, Q) | Reference form factor magnitudes from xraydb. Row 0 = zeros (padding). |
| `_q_weights_` | (1, Q) | Kratky weights`(1 + q²)` per q-point.                                 |

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

#### The σ ratchet (2026-08-02)

Embed originally bounded σ with `clamp(z + log env, log floor, log max)`, where `z` is the log-space deviation of the learned bandwidth from the 1/q envelope, `z = log σ - log env`, `env(q) = min(1/q, sigma_max)`. A hard clamp makes saturation an **absorbing state**: `clamp` has exactly zero gradient outside its range, so MSLE's gradient reaches `_emb._sigma` through a derivative of zero for precisely the entries that already left the window. Nothing can pull a saturated entry back, so the pinned fraction can only rise. Measured on checkpoints from the 2026-08-01 run:

| | `_sigma.weight` std | \|z\| mean | \|z\| p99 | pinned low | pinned high | **live** |
|---|---|---|---|---|---|---|
| step 207    | 0.00522 (1.02x init) | 0.379 | 1.58 | 0.34% | 3.71% | **95.9%** |
| step 10661  | 0.00995 (1.95x init) | 2.358 | 7.78 | 16.9% | 21.5% | **61.5%** |
| prev run, 213k steps | 0.02560 (5.02x init) | 56.97 | 298.8 | 38.7% | 57.2% | **4.2%** |

`_sigma.weight` is the fastest-growing parameter in the model (‖Δ‖/‖·‖ = 1.709 over that span; next fastest is `_f0f1.weight` at 0.299). The endpoint is σ reduced to a binary floor-or-max switch. Because the 1/q envelope spans only `log(100) − log(2) = 3.9` nats across the grid, a |z| p99 of 7.78 means **z has drowned the envelope**: measured per-q σ went from tracking 1/q at step 207 (95.2, 15.7, 26.6, 19.7, 9.3, 4.6, 2.8 Å) to non-monotone noise by step 10661 (15.7, 8.6, 15.7, 5.1, 21.5, 5.0, 4.8 Å). Since σ sets the RFF kernel radius per q, that turns into grid-frequency oscillation in the predicted I(q).

**`clamp` → scaled `arctan`** is the fix:

```
u = z + log env
log σ = mid + half·(2/π)·arctan( (π/2)·(u − mid)/half )
```

The `2/π` and `π/2` make `d(log σ)/du = 1` at the midpoint, so the map is the identity where it does not need to bend, and it squashes into the **open** interval so no entry ever lands exactly on a boundary. Keeping `d(σ)/dz` alive is what lets MSLE pull a saturated entry back.

##### Choosing the saturation

This first shipped as a scaled `tanh`, which is not a nonzero-gradient guarantee: `tanh` hits exactly 1.0 in fp32 once `|u − mid|/half > ~8.7`, and the previous run's |z| mean of 57 is far past that. Sweeping the candidates, all normalised to odd, range (−1, 1), slope 1 at the midpoint so they are directly comparable. Top row is `d(log σ)/du`, bottom row is how much of the window the map has consumed:

| | z=0 | z=2.36 | z=7.78 | z=15 | z=57 |
|---|---|---|---|---|---|
| **erf** (e^−x²) | 1.00 | 5.4e-01 | 1.1e-03 | 1.2e-11 | 1.2e-158 |
| | 0% | 74% | 100% | 100% | 100% |
| **tanh** (e^−2x) | 1.00 | 4.9e-01 | 1.1e-02 | 4.8e-05 | **0.0** |
| | 0% | 71% | 99% | 100% | 100% |
| **gudermannian** (e^−1.57x) | 1.00 | 4.7e-01 | 2.0e-02 | 2.7e-04 | 4.2e-15 |
| | 0% | 69% | 99% | 100% | 100% |
| **x/√(1+x²)** (1/x³) | 1.00 | 4.2e-01 | 3.3e-02 | 5.3e-03 | 1.0e-04 |
| | 0% | 67% | 95% | 99% | 100% |
| **arctan** (1/x²) | 1.00 | 3.4e-01 | 4.5e-02 | 1.2e-02 | 8.7e-04 |
| | 0% | 61% | 86% | 93% | 98% |
| **softsign** x/(1+\|x\|) (1/x²) | 1.00 | 2.8e-01 | 6.5e-02 | 2.3e-02 | 2.0e-03 |
| | 0% | 47% | 75% | 85% | 96% |
| **algebraic p=1/2** (1/x^1.5) | **nan** | 1.4e-01 | 5.0e-02 | 2.6e-02 | 5.6e-03 |
| | 0% | 24% | 40% | 50% | 68% |
| **algebraic p=1/4** (1/x^1.25) | **nan** | 3.4e-02 | 1.5e-02 | 9.4e-03 | 3.2e-03 |
| | 0% | 6% | 10% | 14% | 22% |

Three conclusions.

**It is one spectrum.** Everything above is the algebraic family `x/(1+|x|^p)^(1/p)` plus the exponential ones: `p=2` gives 1/x³, `p=1` (softsign) gives 1/x², `p<1` is fatter. `tanh`, `erf` and the Gudermannian are all exponential and all strictly worse than `arctan` past z≈5. Note also that for any *bounded* map `∫f' dx` must converge, so `f'` has to decay faster than 1/x; `arctan`'s 1/x² is near the practical limit.

**`p<1` is disqualified.** `|u|^0.5` has infinite slope at the origin, so the composite's derivative is NaN at exactly `u = 0`, the window midpoint where the map should be the identity and where most entries sit.

**Algebraic identity is not numerical identity.** The Gudermannian has four standard forms that agree on value and on gradient inside the window and diverge completely outside it in fp32:

| form | z=57 | z=150 | z=298 |
|---|---|---|---|
| `arctan(sinh x)` | 4.2e-15 | 0.0 | **nan** |
| `2·arctan(tanh(x/2))` | 0.0 | 0.0 | 0.0 |
| `arcsin(tanh x)` | **nan** | **nan** | **nan** |
| `2·arctan(eˣ) − π/2` | 4.2e-15 | **nan** | **nan** |

`sinh` and `exp` overflow fp32 at argument ≈89, and the backward then evaluates `inf/inf`; `arcsin(tanh x)` divides by `√(1−y²) = 0` once `tanh` rounds to 1.0. A zero gradient stalls one entry, but a NaN gradient propagates through `clip_grad_norm_` and poisons **every** parameter on that step. This is not hypothetical: the previous run's |z| p99 was 298.

Plain `torch.atan` avoids all of it. Its backward is the single op `1/(1+x²)`, which cannot overflow or emit NaN for any finite input:

| | z=298 | z=1e4 | z=1e8 | z=3e38 |
|---|---|---|---|---|
| arctan | 3.2e-05 | 2.8e-08 | 2.8e-16 | 0.0 |
| softsign | 7.8e-05 | 7.0e-08 | 0.0 | 0.0 |
| tanh | 0.0 | 0.0 | 0.0 | 0.0 |

The cost of `arctan` over `tanh` is interior resolution: it approaches its asymptote more slowly, so at the run's mean |z| = 2.36 it has consumed 61% of the window against tanh's 71%. A given |z| therefore reaches a less extreme σ, which `_sigma`'s weights absorb. Softsign is the reasonable alternative if a fatter tail and a cheaper op (one division, no transcendental) are worth dropping to 47%.

**Total:**

```{Latex}
L_total = mean_{n,q}[ L_kratky(n,q) + L_ff(n,q) ]
```

`.mean()` averages over all N molecules and all Q q-points. Per-molecule normalisation inside L_ff prevents large molecules from dominating.

---

## 9. Training Loop

### Optimizer

`torch.optim.Adam(..., lr=lr, decoupled_weight_decay=True)` over **two** parameter groups:

| group | members | weight decay | lr | betas |
|---|---|---|---|---|
| decay    | weight matrices                                              | `weight_decay` | `lr` | (0.9, 0.999) |
| no-decay | biases, `rms_norm`, `prelu`, `biasterm`, `_mbd`               | 0.0 | `lr` | (0.9, 0.999) |

**Why `_mbd` left the decay group.** The embedding table is indexed, so a vocab entry absent from a batch receives no gradient, while decoupled decay applies every step regardless. The 2026-08-01 run had **52 of 211 rows with `exp_avg_sq == 0`** (never once seen in the data) shrinking monotonically toward zero. Decay is only meaningful for a parameter the loss pushes back on.

> **Resume compatibility.** Removing the third group (three groups back to two) changes the optimizer `state_dict` layout again, and the per-param `state` is keyed by index into the flattened group order, so it cannot be remapped. Pre-2026-08-02 checkpoints raise a `RuntimeError` naming the mismatch rather than silently attaching the wrong Adam moments.

### Learning Rate Schedule

`torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=lr_factor, patience=lr_patience, threshold=lr_threshold, threshold_mode="rel", min_lr=lr_min)`, stepped once per epoch (not per batch) on the **validation** loss. An epoch counts as progress only if `val_loss < best * (1 - lr_threshold)`; after `lr_patience` consecutive non-improving epochs the LR is multiplied by `lr_factor`, floored at `lr_min`. `lr_factor=1.0` disables decay.

Validation, not train loss (which is averaged over the epoch and so lags the current weights) and not test (which would leak the test split into the training procedure).

**AMP is off, and should stay off on T4.** In the previous run `GradScaler` was built with `init_scale=1024` but collapsed inside epoch 1 and sat at 0.0078-0.125 for all 11 epochs. A loss scale below 1 divides gradients rather than multiplying them, so anything under ~9.5e-7 flushed to exactly zero in the fp16 backward. Measured: `exp_avg` was bit-zero for all of `_msg.*`, `_emb._sigma.weight` and `_out._bilinear`, so the entire geometry pathway received no gradient for 11 epochs and stayed at init. The cliff lands where the arithmetic predicts, with `_out._mlp.layer_0` at 6.7e-7 and everything behind it at 0.

It kept backing off because something overflows fp16 even at scale 2⁻⁴. Unconfirmed candidates: the backward of `OutputHead`'s two accumulators, `c * fmc**2` (incoherent, ~1e4 amplification) and `w * fmc` (coherent), and `locality / weights.clamp(min=eps_msgp)` in `MessagePass` (a `1/weights²` backward term). The two-channel head made this **worse**, not better: unlike the per-atom quantities, `coh_accum = sum_j w_j f_j` is an M-scale sum, and it is then squared, so the forward reaches ~M²·f² and the backward through the square carries a `2·coh_accum` factor that is itself M-scale. The old single-channel head had no operation anywhere that squared a sum over atoms, so it never produced either. The root issue is that this model's gradient range is wider than fp16's exponent, so some regions overflow at scale 1 while others underflow below it and no single scale works. T4 has no bf16, so fp32 is the only correct option; `amp=False` sidesteps the overflow rather than fixing it.

**Why plateau rather than exponential.** `ExponentialLR(gamma)` fires unconditionally, regardless of how training is going, and it assumes an epoch is a fixed amount of work. `dataset_frac` breaks that assumption: at `dataset_frac=0.15` an epoch is ~1/7 the optimiser steps, so the old `gamma=0.7`/epoch decayed ~7x faster *per step* than the same number implied at `frac=1.0`. It annealed to `lr=5.2e-6` by epoch 11 whether or not the model was still improving, and in that run it was not improving for an unrelated reason (MessagePass received zero gradient throughout under fp16, see above), so `best_val` freezing at 2.1816 from epoch 4 said nothing about whether the LR was right. Keying the cut off measured validation progress fixes the unconditional part: the LR now never decays while the model is still improving, and the schedule needs no known horizon, which matters because `cfg.epochs` is a per-invocation count rather than the run's total.

It is **not** fully scale-free, though, and it is worth being precise about that. `lr_patience` and `lr_threshold` are counted in *epochs*, so shrinking `dataset_frac` shortens each epoch, puts less progress in it, and makes a fixed relative-improvement threshold proportionally harder to clear, cutting the LR earlier in step-terms. Going from `frac=0.15` to `frac=0.05` triples that effect, which is why the notebook loosens `lr_patience` to 3 and `lr_threshold` to 5e-4 rather than using the `RunConfig` defaults. At `frac=0.05` with `epochs=20` the entire run is roughly one pass over the full dataset, so per-epoch plateau detection is inherently noisy regardless.

This is also why a horizon-based schedule (cosine, step-to-a-target) is not usable here: a resume runs `cfg.epochs` MORE epochs from wherever it left off, so there is no known endpoint to anneal toward.

A reactive mid-epoch cut (windowed train-loss plateau detection, `plateau_window`/`plateau_patience`/`plateau_factor`) previously ran alongside this. It was removed: on a live run it kept firing well past the point where per-parameter gradient magnitudes (Adam's `exp_avg`/`exp_avg_sq`) had stopped shrinking, cutting lr ~150x below its starting value over two epochs while gradients were still very much alive - the windowed detector was reacting to batch-to-batch noise, not real stalls, and throttled training far more aggressively than the per-epoch decay alone would have.

**Rank lockstep.** `ReduceLROnPlateau.step(metric)` makes the LR a function of the metric rather than of the epoch count, so a rank-*local* metric here would silently desync the LR across ranks mid-run. It is safe because `val_loss` is rank-identical: `evaluate()` all-reduces its six stat accumulators on DP-split batches (`Train/train.py`), and on the TP path every rank sees the same batch and the same all-reduced I(q). Anything fed to `scheduler.step()` must preserve that property.

**Resume**: unlike `ExponentialLR`, this schedule is **path-dependent**. `best`, `num_bad_epochs`, and `cooldown_counter` cannot be reconstructed from the epoch number, so the checkpoint's `"scheduler"` state is load-bearing rather than an optimisation. The LR itself still rides in the optimizer: `ReduceLROnPlateau` mutates `optimizer.param_groups["lr"]` in place, so a checkpoint already has the true current lr baked into `optimizer.state_dict()`. On resume, `optimizer.load_state_dict(ckpt["optimizer"])` runs **before** `scheduler.load_state_dict(ckpt["scheduler"])`, and the latter only restores the scheduler's own bookkeeping, never touching `optimizer.param_groups`. So the lr used after a resume always comes from the checkpoint, never from `RunConfig.lr`; `cfg.lr` only matters for a fresh run. The checkpoint also saves the current lr as an explicit top-level `lr` field (redundant with `optimizer.state_dict()`, but avoids digging into optimizer state to inspect it).

The scheduler's `state_dict()` and the AMP `GradScaler`'s `state_dict()` are both saved in every resume checkpoint too (`"scheduler"`, `"scaler"` keys) and restored on resume. Neither is needed to reconstruct the correct lr (see above), but skipping the `GradScaler` restore meant every resume reset its adaptive loss scale back to `amp_init_scale`, which could cause a burst of skipped/inf-grad steps right after a resume; saving it makes resumes bit-for-bit resume the scaler's adapted state instead. Old checkpoints saved before this change lack both keys - the resume code guards on `"scheduler" in ckpt` / `"scaler" in ckpt` so loading them still works, it just starts the scheduler/scaler fresh.

### Per-Batch Step

```
1. Move batch to device
2. optimizer.zero_grad(set_to_none=True)
3. [autocast fp16 if amp] iq, fmags, sigmas, local_batch, loss_scale = model(batch)
4. [autocast fp16 if amp] loss = criterion.loss(iq, fmags, local_batch, λ₆) * loss_scale
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

| File                                             | Contents                                                                                                                                                               | Saved when                                                   |
| -------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------- |
| `ckpt_best`                                      | model weights only                                                                                                                                                     | val_loss improves                                            |
| `ckpt_dir/checkpoint_<epoch>_<batch>.pt` (train) / `checkpoint_<epoch>_<phase>_<batch>.pt` (val/test) | weights + optimizer + scheduler + GradScaler + epoch + phase + batch_idx + `best_val` + this epoch's already-known train/val/test scalars + phase's partial accumulator | every `ckpt_interval_sec` seconds and at every phase boundary |

Each epoch runs three phases in order - `train`, `val`, `test` (test also covers the diagnostic-plots pass) - and every phase gets the same mid-phase checkpointing, not just training: val and test are thousands of batches long too (walking the full val/test set once per epoch), so losing a whole pass to a late Kaggle-session timeout used to be as costly as losing the training tail. `phase` records which of the three the checkpoint belongs to; `batch_idx` is -1 at a phase boundary (that phase is done, the run moves on to the next one) or the last-processed batch mid-phase. A mid-val/test checkpoint also carries `eval_state` - that phase's accumulator (running sums, or for the test/plots pass also the growing per-molecule error-distribution lists) - and this epoch's `train_loss`/`train_r2`/`val_loss`/`val_r2` scalars already computed by earlier phases, so a resume into `val` or `test` never has to redo an earlier phase just to reconstruct them.

`best_val` is the best end-of-epoch validation loss seen so far, not a per-checkpoint value (it's only updated once, after test completes - see the trailer block in `Train/train.py`). Checkpoints written before this field was renamed still use the key `val_loss` for the same quantity; the resume loader falls back to it.

Resume checkpoints are numbered, not overwritten: each save under `ckpt_dir` gets its own file. `train` keeps the original two-part name (`checkpoint_<epoch>_<batch>.pt` mid-phase, `checkpoint_<epoch>_final.pt` at the phase boundary); `val`/`test` - added later - get a phase segment so they don't collide with train's (`checkpoint_<epoch>_<phase>_<batch>.pt` mid-phase, `checkpoint_<epoch>_<phase>_final.pt` at the phase boundary). The full history survives for later comparison (e.g. debugging a training collapse) instead of only the latest save.

Resume walks the phase order (`train` -> `val` -> `test`) from wherever the checkpoint left off: a mid-phase checkpoint (`batch_idx >= 0`) redoes that exact phase from `batch_idx + 1`, using the saved `eval_state` to pick the running accumulator back up instead of restarting it at zero; a phase-boundary checkpoint (`batch_idx == -1`) skips straight to the next phase (or, for `test`, to the next epoch). `torch.manual_seed(batcher_seed + epoch)` re-seeds the training shuffle identically on a mid-train resume; val/test are unshuffled, so a mid-phase resume there just skips the first `batch_idx + 1` indices via a resumable sequential sampler (`_ResumableSequentialSampler`) - same "skipped batches are never even handed to the DataLoader" trick as the training sampler, so resuming late into a long val/test pass costs nothing extra. Both `ckpt_best` and each numbered resume checkpoint are pushed to a rclone remote (`ckpt_rclone_dest`) for Kaggle session crash durability.

### Run Data (metrics + diagnostic plots)

When `data_dir` is set, everything the run records lands under it (via `Train/eval_plots.py`, reusing `Baselines/run/metrics.py` so the plots are directly comparable to the baseline notebook):


| File                                 | Scope     | Cost                                                                                                                      |
| -------------------------------------- | ----------- | --------------------------------------------------------------------------------------------------------------------------- |
| `epoch_NNN/` (per-q R², Kratky, …) | per epoch | full test-set pass (expensive; see note)                                                                                  |
| `epoch_NNN/metrics.json`             | per epoch | free — that epoch's train/val/test loss and R²                                                                          |
| `epoch_NNN/loss_per_epoch.png`       | per epoch | free — train/val/test loss vs epoch, through this epoch; rebuilt by reading every earlier epoch's`metrics.json` off disk |
| `run_config.rtf`                     | run-level | free — the full`RunConfig`, dumped once when training starts                                                             |

`epoch_NNN` is the epoch number zero-padded to 3 digits (`epoch_001`, …). Every metric file is scoped to one epoch on purpose: a single run-level metrics file would be rewritten from the in-memory history at every epoch boundary, and since a resumed process starts with no memory of the epochs it did not run, resuming would silently truncate the record back to the resume point (and push the truncated copy to Drive). Reading the history back off disk instead means a resumed run extends the curve rather than restarting it. Concatenate the per-epoch `metrics.json` files after the run for the whole history.

If `data_rclone_dest` is set, the whole `data_dir` is pushed to that rclone remote after every epoch.

---

## 10. Hyperparameter Reference


| Name                | Default | What it controls                                                                                                                                                                                                                         |
| --------------------- | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `lambda_1`          | 128     | Atom embedding dimension. Width of the per-atom vector throughout MessagePass.                                                                                                                                                           |
| `lambda_2`          | 4       | Message-passing rounds. Also bounds the maximum cumulative sigma change.                                                                                                                                                                 |
| `lambda_3`          | 64      | OutputHead MLP starting width (bilinear output width, MLP input). Must satisfy`lambda_3 >= 2^lambda_4`.                                                                                                                                   |
| `lambda_4`          | 4       | OutputHead halving steps. The ladder always terminates at width 2 (the w and c channels), appending a step if the halvings do not land there. With defaults: 64->32->16->8->4->2.                                                         |
| `lambda_5`          | 128     | RFF count. More = tighter kernel approximation, higher memory cost.                                                                                                                                                                      |
| `lambda_6`          | 1.0     | Form-factor penalty weight.                                                                                                                                                                                                              |
| `msg_seed`          | 42      | Seed for fixed RFF frequency matrix Ω.                                                                                                                                                                                                  |
| `atm_chunk`         | 512     | Atoms per M-chunk. Reduce to lower VRAM.                                                                                                                                                                                                 |
| `mol_chunk`         | 256     | Molecules per N-chunk. Reduce to lower VRAM on large molecules.                                                                                                                                                                          |
| `dp_atom_threshold` | 101     | Batches with padded atom count`M` below this **and** molecule count `N >= 2*mol_chunk` route through DP instead of TP                                                                                                                    |
| `compile`           | True    | torch.compile Embed/MessagePass/OutputHead's checkpointed step functions (fullgraph=True, dynamic=True).                                                                                                                                 |
| `amp`               | False   | fp16 autocast + GradScaler (CUDA only). **Leave off.** The GradScaler collapsed below 1.0 in the previous run and silently zeroed every gradient upstream of `_out._bilinear`; see the AMP note below.                                   |
| `amp_init_scale`    | 1024    | GradScaler starting loss scale when`amp` is on. Lower than torch's 65536 because activations are ~O(1) after mean-normalization.                                                                                                         |
| `eps_embd`          | 1e-8    | Numerical floor in Embed (hypot, form factors).                                                                                                                                                                                              |
| `eps_msgp`          | 1e-3    | Numerical floor in MessagePass (sigma clamp, aggregate denominator).                                                                                                                                                                     |
| `lr`                | 1.3e-4  | Adam learning rate (starting value; see`lr_factor`).                                                                                                                                                                                     |
| `lr_factor`         | 0.5     | ReduceLROnPlateau decay factor, applied when val loss stops improving. 1.0 = no decay.                                                                                                                                                   |
| `lr_patience`       | 2       | Consecutive non-improving epochs tolerated before the LR is cut.                                                                                                                                                                         |
| `lr_threshold`      | 1e-3    | Relative improvement needed to count as progress:`val_loss < best * (1 - lr_threshold)`.                                                                                                                                                 |
| `lr_min`            | 1e-6    | Floor the LR is never reduced below.                                                                                                                                                                                                     |
| `weight_decay`      | 0.1     | AdamW decoupled decay. Applied to weight matrices only: not `_mbd`, biases, norm/gain params, or the RFF phases.                                                                                                                          |
| `grad_clip`         | 1.0     | Max gradient L2 norm before clipping.                                                                                                                                                                                                    |
| `epochs`            | 20      | Training epochs.                                                                                                                                                                                                                         |
| `batcher_seed`      | 0       | Seed for train/val/test split and per-epoch shuffle.                                                                                                                                                                                     |
| `dataset_frac`      | 1.0     | Fraction of each split's batches to use, (0.0, 1.0]. Applies to**train, val and test** - eval costs on the order of a train epoch, so thinning train alone just lets eval dominate. Deterministic off `batcher_seed`, fixed for the run. |
| `atom_size_ceil`    | -1      | Max total atoms per batch (-1 = 3x largest molecule).                                                                                                                                                                                    |
| `num_workers`       | 4       | DataLoader worker processes.                                                                                                                                                                                                             |
| `ckpt_interval_sec` | 600     | Seconds between mid-epoch resume checkpoints.                                                                                                                                                                                            |
| `profiler`          | False   | Diagnostic run: per-rank torch.profiler + per-section wall-clock timers, then stop. Traces written per rank to`./profiler_trace/rank<r>/`.                                                                                               |
| `prof_warmup`       | 1       | Profiler warmup batches (profiled, discarded).                                                                                                                                                                                           |
| `prof_active`       | 3       | Profiler active batches (recorded). Loop runs`1 + prof_warmup + prof_active` batches; raise `prof_active` for more representative stats.                                                                                                 |

### Validated 2xT4 config

The mixed-precision config trained and validated on 2x T4 (16 GiB each), fp16 loss falling cleanly from ~85 with peak ~5.8 GiB:

```python
lambda_2 = 4, lambda_5 = 128,         # message-passing rounds / RFF count
atm_chunk = 512, mol_chunk = 256,     # see Appendix A1 - 1024/512 OOM'd in real training on 2xT4
compile = True, amp = True,           # amp_init_scale defaults to 1024
dp_atom_threshold = 101,
```

`atm_chunk=1024, mol_chunk=512` was tried first and OOM'd on real hardware (GPU already at 12.13/14.56G, +3.19G alloc failed) - see Appendix A1 for the full chunk-size sweep and why bigger chunks don't help. `lambda_2`/`lambda_5` are the speed/capacity dials (`lambda_2` is linear in wall-clock).

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
       └─ _sigma bilinear, exp + 1/q envelope -> sigma (N, M, Q, 1)
       └─ LayerHead: embeds (N,M,1,λ₁), f_mags (N,M,Q,1), sigmas (N,M,Q,1)

  [Distributed: shard M across ranks]

  └─ MessagePass x λ₂ rounds
       └─ embeds.expand Q dim        -> (N, M, Q, λ₁)
       └─ [per round]
            └─ _pass_1: accumulate features (Nc,Q,λ₅), chem_env (Nc,Q,λ₅,λ₁)
            └─ _AllReduce (features, chem_env)
            └─ mean-normalise features/chem_env by per-molecule atom count (fp16 safety)
            └─ _pass_2: locality, weights, agg, MishGLU gate -> new embeds
                        qdiag + ptanhshrink + arctan window   -> new sigmas
       └─ LayerHead: embeds (N,M,Q,λ₁), sigmas updated, f_mags unchanged

  └─ OutputHead
       └─ [per M-chunk]
            └─ bilinear(embeds, f_mags) -> (N,mc,Q,λ₃)
            └─ Mish -> MLP -> (N,mc,Q,2) -> PBId -> w (signed)
                                         └─ SqrP     -> c (positive)
            └─ w * f_mags  * mask -> sum over atoms -> coh_accum (N,Q)
            └─ c * f_mags² * mask -> sum over atoms -> inc_accum (N,Q)
       └─ returns coh_accum, inc_accum UNSQUARED  (N,Q) each

  [Distributed: _DistributedSum(stack(coh, inc)), _AllGatherDim1(f_mags, sigmas)]

  └─ OutputHead.combine(coh, inc) = coh² + inc  (N, Q)
       [square happens HERE, after every chunk AND every rank is summed in]

Loss
  └─ _kratky_MSLE:  (1+q²)*(log1p(Î)-log1p(I))²    (N,Q)
  └─ _ff_penalty:   λ₆*(log1p(f̂)-log1p(f_ref))²/n  (N,Q)
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


| Setting       | Recommendation                                                                                                                                    |
| --------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `num_workers` | 2-4 (0 = serial, >4 = diminishing returns)                                                                                                        |
| `pin_memory`  | True (already set in train.py for GPU)                                                                                                            |
| `atm_chunk`   | Start at 512; raising it buys no steady-state speed (compute is chunk-invariant, Appendix A1) and only costs memory + bigger torch.compile stalls |
| `mol_chunk`   | Start at 256; drop to 128 if OOM on large molecules                                                                                               |
| `verbosity`   | `"batch"` for loss + memory stats every 20 batches                                                                                                |
| `max_batches` | Set to 20 for a quick smoke test before a full run                                                                                                |

---

## Appendix

> **Superseded (fp32 era, pre-2026-07-10).** Older sweep methodology notes below A1/A2's current tables predate the fp16 migration and the current profiler's `heavy_nm`/`heavy_m`/`median` stratification; treat those as historical context for the profiling approach, not comparable numbers. A1's current table (below) is the up-to-date fp16-path sweep: validated config is `λ₂=3, λ₅=64, atm_chunk=512, mol_chunk=256, amp=True`, peak ~5.8G.

---

### A1. mol_chunk and atm_chunk optimizations

**Current sweep (2026-07-10, fp16 path)**: 2x T4, `λ₁=128, λ₂=3, λ₃=128, λ₄=4, λ₅=64`, `amp=True`, `compile=True`, `dp_atom_threshold=101`, profiler over 54 stratified batches (1 heaviest-`N*M_shard` + 2 more heavy-data/heavy-compute groups, see §12). "Steady-state" excludes the handful of torch.compile recompile-stall batches (see notes) to isolate actual per-batch compute; "profiled total" is the raw 54-batch sum including those stalls.


| `atm_chunk` | `mol_chunk` | result                      | peak CUDA mem | steady-state ms/batch | profiled total (54 batches)   | notes                                                                                                                                                                                                                                                                              |
| ------------- | ------------- | ----------------------------- | --------------- | ----------------------- | ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1024        | 512         | **OOM (real training run)** | n/a           | n/a                   | n/a                           | GPU already at 12.13/14.56G in use, +3.19G alloc failed. Not a profiler run - this crashed a real (non-profiler) training run and is what started this investigation.                                                                                                              |
| **512**     | **256**     | **ok (chosen)**             | **~5.8G**     | **~880**              | **~78.7s (r1) / ~72.9s (r0)** | Profiled before the`.contiguous()` recompile fix (see below), so its compile-stall batches include extra stride/storage_offset-driven recompiles on top of the legitimate chunk-tier ones.                                                                                         |
| 800         | 400         | ok, but worse               | ~8.9G         | ~880                  | ~130.8s (r1) / ~125.0s (r0)   | Profiled after the`.contiguous()` fix - only 4 clean recompiles/rank (vs. ~14 noisy ones at 512/256), but the fewer stalls each compile a bigger graph and take longer overall (47.4s/22.0s/16.5s per rank vs. 512/256's 20.5s/6.9s/6.4s). Steady-state is unchanged from 512/256. |

Takeaway: **steady-state compute is invariant to chunk size** (~880 ms/batch at both 512/256 and 800/400) - this reconfirms the same finding from the older fp32-era sweeps below. Raising chunk size only buys fewer torch.compile recompiles at the cost of each one being slower to compile (bigger graph) and more peak memory - a net loss here, not a win. `512/256` is chosen: same steady-state speed as `800/400`, ~40% less peak memory, and a smaller worst-case compile stall. Don't raise further without re-profiling; don't lower without a memory-pressure reason, since it buys nothing on speed either.

Two separate bugs were found and fixed in the course of this sweep (both in `ScatterNet/model/message_pass.py` and related files, see git history for exact diffs):

- **RMSNorm/autocast dtype mismatch**: `MessagePass._rms_norm` ran under fp16 autocast against its fp32 weight, forcing an unfused fallback kernel (PyTorch warning: `Mismatch dtype ... Cannot dispatch to fused implementation`). Fixed by forcing that call to fp32 (matching the weight) via `torch.autocast(..., enabled=False)`, same pattern already used for the nearby RFF block.
- **Non-contiguous chunk-slice views causing extra torch.compile recompiles**: `crd_slice`/`sig_slice`/`emb_slice`/`msk_slice`/`ffs_slice` (MessagePass), `embeds`/`f_mags`/`mask` chunks (OutputHead), and `output_head`/`f_mag_pred`/`sigma_pred` (Loss) are all views into larger padded tensors; their stride/storage_offset vary by chunk index, and torch.compile guards on that independently of shape - multiplying recompiles well past the intended `_CHUNK_TIERS` axis. Fixed by adding `.contiguous()` at each of these call sites. Confirmed via `TORCH_LOGS=recompiles`: dropped from ~14 distinct guard failures/rank (mostly stride/storage_offset noise) to 4 clean ones (2 legitimate chunk-tier shape growths, 1 Inductor fusion-size heuristic, 1 `None`-vs-tensor branch in the loss).

---

**Older fp32-era sweep** (predates the fp16 migration, mean-normalization, and the NoTrilinBilin swap; kept for historical profiling-methodology context only - not comparable to the table above). All runs on 2x T4, λ₁=λ₅=128, λ₂=5, fp32 (no amp). Also predates the current profiler's stratified sampling (used an older shuffle-window sampler that usually missed the true worst bucket, understating peaks).


| `mol_chunk` | `atm_chunk` | product | result         | notes                                                       |
| ------------- | ------------- | --------- | ---------------- | ------------------------------------------------------------- |
| 64          | 64          | 4096    | ok (baseline)  | ~15.3 / 18.2 s/batch (r0/r1)                                |
| 128         | 16          | 2048    | ok             | atm too small, weak matmul contraction                      |
| 104         | 46          | 4784    | ok             | more launches (69,507)                                      |
| 56          | 100         | 5600    | ok             | fewest launches (58,889), balanced; atm_chunk reduced to 80 |
| 32          | 200         | 6400    | ok, at ceiling | peak 14.98G; ~14.5 / 17.3 s/batch                           |
| 64          | 128         | 8192    | OOM            |                                                             |
| 32          | 512         | 16384   | OOM            | looked stable in the profiler window, OOM'd mid-run         |
| 256         | 256         | 65536   | OOM            |                                                             |

---

### A2. dp_atom_threshold optimizations

- TP (tensor parallel)
  - split one molecule's atoms across both GPUs. Each rank holds part of every molecule, so it must all-reduce mid-forward to reconcile the shared context. Good for few, large molecules; costly when molecules are tiny (thousands of little collectives).
- DP (data parallel)
  - split the molecules across both GPUs. Each rank owns whole, disjoint molecules and needs no mid-forward communication, just the one gradient all-reduce at the end of the step. Good for many small molecules.

The model routes each batch to whichever fits (dp_atom_threshold, see §7): high-N/low-M buckets go DP, large-M buckets stay TP.

Worst-rank = the epoch-limiting rank (always rank 1 here, see the TP shard-imbalance note in §12). "in-model all-reduce" is the NCCL collective fired inside the TP forward. Two transitions matter; the **speed cliff is between 10 and 63**: at 10 only the `M=3` bucket routes DP (a degenerate DP, most buckets still pay the TP all-reduce), so it stays slow; by 63 every high-`N`/low-`M` bucket routes DP and the all-reduce (68s, 47% of CUDA) disappears, giving a 1.8x speedup. The **memory step is between 101 and 102**: at 102 the `mols=550, M=101` bucket flips to DP, which holds the full `M=101` per rank instead of sharding it, pushing peak from 13.55G to 14.51G for no speed gain. `dp_atom_threshold = 101` is chosen because the strict `M < threshold` gate keeps that bucket on TP: full speedup at the lower peak (~2.4G headroom). See §7 for the safe-band reasoning.


| `dp_atom_threshold` | worst-rank s/batch | peak CUDA mem | Self CUDA total | Small molecule all-reduce time |
| --------------------- | -------------------- | --------------- | ----------------- | -------------------------------- |
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

### Low-q coherent limit

**Symptom**: percent error at low q climbing off the chart epoch after epoch while high q converged normally.

**Correction (2026-08-04)**: an earlier version of this section blamed the "Kratky" diagnostic plot, on the grounds that `q²·I(q)` multiplies the low-q region by ~0 and suppresses the band where the amplitude was broken. That is wrong, and it is wrong about code that does not exist: `plot_kratky` (`Baselines/run/metrics.py:977`) applies no `q²` at all. Its own docstring is accurate ("mean log1p(I(q)) vs q"); only the function name, the filename, and this document called it Kratky.

The plot did hide the failure, but by a different mechanism, and that mechanism is still uncorrected. It draws `mean_j ln(1+I_true(q))` against `mean_j ln(1+I_pred(q))`, averaging the two curves over molecules **separately** (`metrics.py:354-355, 446-447`). That difference is the mean *signed* log residual, so a model over-predicting half the test set by +3 log units and under-predicting the other half by −3 draws two coincident curves. There is no band, no percentile, and no `n` on the figure. `ln(1+I) ≈ 2 ln M + const`, so the mean is also dominated by the largest molecules. The per-molecule residual histogram (`metrics.py:1102`) is the honest counterpart and does not have this defect.

Two related caps also participated: `metrics.py:906-907` clamps the per-q R² axis at `-3.0` and `metrics.py:963-964` clamps per-q percent error at `500%`, both silently, with no off-scale marker. "Climbing off the chart" was literal.

**Root cause**: the old `OutputHead` could not represent the low-q limit at any parameter setting. Ground truth is the Debye sum `I(q) = sum_i sum_j f_i f_j sinc(q r_ij)`, and its two endpoints behave very differently in atom count M:

- At `q -> 0` every `sinc(q r_ij) -> 1`, so the double sum factorizes to `(sum_j f_j)²`. Both the diagonal and all M² - M off-diagonal pairs contribute, so `I(0)` scales like **M²**.
- At high q the off-diagonal pairs oscillate with `r_ij` and average away, leaving only the M diagonal terms `sum_j f_j²`, which scales like **M**.

The old head computed `I(q) = sum_j contribs_j(q) · f_j(q)²`: a single sum over atoms, linear in the per-atom contributions, with no operation anywhere that squares a sum over atoms. That is structurally the incoherent (high-q) limit and nothing else. `(sum f)²` was not in its function class for *any* weights, since a linear functional of the per-atom terms cannot produce the M² - M cross terms. The only way it could approach the right magnitude at low q was to inflate `contribs_j` itself to ~M, which is a fundamentally different (and molecule-size-dependent) thing for the MLP to learn than a well-conditioned O(1) per-atom weight.

That explains the asymmetry in the observed convergence exactly. From standard init the per-atom contributions sit at O(1), which is already right for the high-q limit, so high q was roughly correct immediately. Low q started M-fold low and could only climb by driving the MLP output up by a factor that grows with molecule size, which is slow, size-dependent, and fights the form-factor penalty. Nothing about the optimiser, the learning rate, or the loss weighting was going to fix a term that is absent from the model's function class.

**Fix**: the two-channel coherent + incoherent head, `I(q) = (sum_j w_j f_j)² + sum_j c_j f_j²` (see [§6](#6-outputhead)). Squaring a sum over atoms *is* the pairwise double sum, so the M² term is recovered from one O(M) pass, both channels stay O(1) at every q, and the M-scaling comes from the architecture rather than from learned magnitude.

**Retracted**: an earlier version of this document blamed the `(1 + q²)` Kratky loss weight for starving low-q gradient, with a secondary story about σ being crushed at low q. Both are wrong and are withdrawn. The q-grid here is `q ∈ [0, 0.5] Å⁻¹`, so `1 + q²` spans 1.00 at `q = 0` to 1.25 at `q = 0.5`: a 25% spread across the entire grid. A weight that varies by a quarter cannot starve anything, let alone produce an error gap of orders of magnitude, and the arithmetic should have ruled it out before it was written down.

---

### The σ⁻² pole (2026-08-03)

**Symptom**: gradients exploding and the clip firing on every step, with the σ pathway simultaneously frozen. At `grad_clip = 1.0` against measured norms of 10-18 the clip was gradient *normalisation*, not a safety valve, and because buckets are size-sorted the clip factor varied with molecule size, systematically downweighting large molecules.

**Root cause**: σ is the RBF bandwidth and the kernel forms `r/σ`, whose backward carries `-r/σ²`. So `∂z/∂σ` has a `σ⁻²` pole and the condition number of the σ → z map is `‖r‖/σ²`. Embed bounds σ into `[0.5, 100]` with an arctan squash precisely to cap that term at ~4. MessagePass then discarded the window on round 1: its `squareplus(σ + shrink(...))` update is unbounded below, asymptoting to `b/(4|x|) → 0⁺`. Measured after one round at init: `min σ = 0.0107`, **41% of entries below Embed's floor**, and `|dL/dσ|` of 1.5e8 at σ = 0.002 against 2.4e3 at the floor, scaling exactly as `σ⁻²`. A global-norm clip against a 1e6-1e8 norm rescales every other parameter's gradient to ~1e-8, which is why the σ and RFF parameters measured grad RMS ~1e-13 and, against Adam's default `eps = 1e-8`, took updates of `lr·g/eps` instead of `lr`: 59% of the model was effectively frozen.

**Fix**: the σ update is now additive in log space and re-squashed by the *same* `arctan_log_sigma_window` Embed uses, every round, so the bound holds on the σ that reaches the kernel and the loss. The head also reads `RMSNorm(new_emb)` rather than the raw residual (it was the only unnormalized linear map in the block, pre-activation std ≈ 25-90), `NoTrilinBilin` inits from the true `in1*in2` fan-in rather than `in1`, `adam_eps` drops to 1e-12, and `grad_clip` rises to 5.0. Measured after: σ ∈ [1.4, 29.2] with 0% below the floor, no parameter with a zero or non-finite gradient, and a global grad norm of 1.1 against 1062 before.

The sigma *penalty* in the loss is gone with it. It was measured identically 0 at init (σ ∈ [1.11, 40.9] against a `4·env` band of [8.0, 400.0]) which made it the only, and dead, gradient path to the σ parameters. The window now bounds σ structurally, so there is nothing left for a penalty to charge for.

### Form-factor init (2026-08-03)

Untrained `Embed` emits `f_mag ~ 0.5` where the physical value is ~7. Since both head channels are homogeneous of degree 2 in `f`, `d log I / d log f_mag = 2` exactly, so that is a ~200x error in `I(q)`, and it measured as **99.5% of the entire gradient norm at init** (`_f0f1` and `_f2` at |g| = 96 and 102, against 0.10 for the largest MessagePass weight). `ScatterNet` now passes a physical `f_init` curve into `Embed`, which biases `_f0f1` onto it and shrinks both weight matrices by `_F_INIT_WEIGHT_GAIN`.

`f_init` averages the **organic** elements (H/C/N/O/P/S), not the whole vocabulary. The vocab is 211 xraydb ions with mean and median `|f|(0)` both ≈ 46.7, but the structures trained on are proteins and viral capsids: `|f|(0)` is 6.0 for C, 7.0 for N, 8.0 for O. Using the vocab mean overshoots by ~6.7x, and since `I ~ f²` that is a ~45x error pointing the wrong way. Measured `I_pred/I_true` at init: 30.3 with the vocab mean, **1.09** with the organic mean.
