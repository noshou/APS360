# ScatterNet: Design Reference

GNN that predicts X-ray powder scattering curves I(q) from atomic coordinates and species, using Random Fourier Features for O(M·λ₅) all-pairs kernel aggregation. Single-GPU, BF16 autocast throughout (see §8).

For scripting on cloud proviers please refer to `_shell_scrpts/README.md` .

## Table of Contents

- [Notation](#notation)
- [1. Vocabulary and Atom Tokens](#1-vocabulary-and-atom-tokens)
- [2. Data Pipeline](#2-data-pipeline)
  - [2.1 Batch](#21-batch)
  - [2.2 Bucketing](#22-bucketing-batcher)
  - [2.3 Loading](#23-loading-batchset)
- [3. Embed](#3-embed)
  - [Inter-stage tensors](#inter-stage-tensors)
- [4. MessagePass](#4-messagepass)
  - [Mathematical Formulation](#mathematical-formulation)
  - [Pass 1: Accumulate Global Context](#pass-1-accumulate-global-context)
  - [Pass 2: Per-Atom Update](#pass-2-per-atom-update)
  - [MishGLU Gate](#mishglu-gate)
  - [Sigma Update](#sigma-update)
  - [bmm vs einsum](#bmm-vs-einsum)
  - [Checkpointing Strategy](#checkpointing-strategy)
  - [Mean-Normalization](#mean-normalization)
- [5. OutputHead](#5-outputhead)
- [6. LambdaHead](#6-lambdahead)
- [7. SplineSmooth](#7-splinesmooth)
- [8. ScatterNet](#8-scatternet)
- [9. Loss](#9-loss)
- [10. Training Loop](#10-training-loop)
- [11. Hyperparameter Reference](#11-hyperparameter-reference)
- [12. End-to-End Data Flow](#12-end-to-end-data-flow)
- [13. Profiling and Optimization](#13-profiling-and-optimization)
- [Appendix](#appendix)
  - [A1. mol_chunk and atm_chunk optimizations](#a1-mol_chunk-and-atm_chunk-optimizations)
  - [Low-q coherent limit](#low-q-coherent-limit)

---

## Notation


| Symbol  | Meaning                                                                      |
| --------- | ------------------------------------------------------------------------------ |
| N       | molecules in a batch                                                         |
| M       | atoms per molecule (padded to the longest in the batch)                      |
| Q       | number of q-points in the scattering grid                                    |
| λ₁    | atom embedding dimension (`lambda_1`, default 128)                           |
| λ₂    | message-passing rounds (`lambda_2`, default 4)                               |
| λ₃    | OutputHead MLP starting width (`lambda_3`, default 256)                      |
| λ₄    | OutputHead MLP halving steps (`lambda_4`, default 4; ladder ends at width 2) |
| λ₅    | Random Fourier Features count (`lambda_5`, default 128)                      |
| λ₆    | form-factor penalty weight (`lambda_6`, default 1.0)                         |
| λ₇    | 2nd-derivative smoothness penalty weight (`lambda_7`, default 0.25)          |
| Nc      | molecules per N-chunk (`mol_chunk`)                                          |
| mc      | atoms per M-chunk (`atm_chunk`)                                              |
| V       | VOCAB size = len(VOCAB) + 1 (row 0 is padding)                               |
| ε_e    | `eps_embd` numerical floor in Embed (default 1e-8)                           |
| ε_m    | `eps_msgp` numerical floor in MessagePass (default 1e-3)                     |
| r_m     | Cartesian coordinates of atom m, shape (3,) in Å                            |
| e_m     | embedding vector of atom m, shape (λ₁,)                                    |
| f_m(q)  | form factor magnitude of atom m at q-point q, scalar                         |
| σ_m(q) | RFF kernel bandwidth of atom m at q-point q, scalar                          |
| φ_m(q) | RFF feature vector of atom m at q-point q, shape (λ₅,)                     |

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

`Batch` is a frozen dataclass.


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

- **Two-part form factor**: `_f0f1` and `_f2` are separate linear layers for real and imaginary parts, mirroring Cromer-Mann notation (`f = (f0+f1) + i*f2`). Only the magnitude `|f(q)|` is used downstream, since phases average out over random orientations in powder-averaged scattering.
- **Bilinear sigma**: `NoTrilinBilin(λ₁, Q, Q)` (an `nn.Bilinear`-equivalent that avoids the `aten::_trilinear` kernel and its `torch.compile` graph break - see [§4](#4-messagepass)) captures the interaction between identity (embed) and scattering strength (f_mag); a linear layer over concatenation would miss the cross-term.
- **PReLU**: channel-wise, so each embedding channel independently learns its negative slope.
- **Asymmetric masking**: `f_mags` and `sigmas` are multiplied by the padding mask; `embeds` is not. Zeroing f_mags is sufficient to exclude padding atoms since every intensity contribution carries a factor of f_mag (f_mag in the coherent term, f_mag² in the incoherent one).
- **ε passed at forward time** rather than fixed at construction, so the numerical floor can be tuned without changing the model.

Converts each atom's VOCAB index into three tensors that feed the rest of the pipeline:


| Output   | Shape           | Meaning                          |
| ---------- | ----------------- | ---------------------------------- |
| `embeds` | (N, M, 1, λ₁) | Learned per-atom identity vector |
| `f_mags` | (N, M, Q, 1)    | Estimated form factor magnitude  |
| `sigmas` | (N, M, Q, 1)    | RFF kernel bandwidth per q-point |

### `Embed` Layers

1. **Learnable Parameters**


| Parameter        | Shape        | Notes                                                     |
| ------------------ | -------------- | ----------------------------------------------------------- |
| `__mbd.weight`   | (V, λ₁)    | Embedding table. Row 0 frozen at zero via`padding_idx=0`. |
| `__f0f1.weight`  | (Q, λ₁)    | Linear: embedding -> real part of form factor.            |
| `__f0f1.bias`    | (Q,)         |                                                           |
| `__f2.weight`    | (Q, λ₁)    | Linear: embedding -> imaginary part of form factor.       |
| `__f2.bias`      | (Q,)         |                                                           |
| `__prelu.weight` | (λ₁,)      | One learned negative slope per embedding channel.         |
| `__sigma.weight` | (Q, λ₁, Q) | Bilinear weight: embedding x f_mag -> sigma logit.        |
| `__sigma.bias`   | (Q,)         |                                                           |

(Fields are name-mangled, e.g. `__mbd` is stored as `_Embed__mbd`; table names omit the class prefix.)

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

  # pack into (embeds, f_mags, sigmas) tuple
  embed.unsqueeze(-2)  -> embeds        (N, M, 1, λ₁)  Q=1, broadcastable
  f_mag.unsqueeze(-1) * mask -> f_mags  (N, M, Q, 1)
  sigma                -> sigmas        (N, M, Q, 1)
```

3. **Activations**


| Activation                | Location               | Behaviour                                                                                                      |
| --------------------------- | ------------------------ | ---------------------------------------------------------------------------------------------------------------- |
| `nn.PReLU(λ₁)`*         | After embedding lookup | Learnable per-channel negative slope. Acts on dim=1, hence the double-transpose.                               |
| `arctan_log_sigma_window` | On sigma logits        | Squashes log σ into`(log sigma_floor, log sigma_max)`; `exp` then makes it positive. Shared with MessagePass. |

**`PReLU` vs. alternatives at the embedding activation:*


| Activation | λ₁ | Val loss | Val R² | Test R² |
| ------------ | ------ | ---------- | --------- | ---------- |
| LeakyReLU  | 64   | 0.77     | 0.71    | 0.69     |
| PReLU      | 128  | 0.67     | 0.74    | 0.73     |
| PReLU      | 64   | 0.93     | 0.62    | 0.61     |
| LeakyReLU  | 128  | NaN      | NaN     | NaN      |
| ELU        | 64   | 2.44     | -0.05   | -0.01    |
| Mish       | 64   | NaN      | NaN     | NaN      |

---

### Inter-stage tensors

Embed, MessagePass, and OutputHead pass a plain `(embeds, f_mags, sigmas)` tuple between them (no container class; each stage unpacks positionally).


| Field    | Shape from Embed | Shape after MessagePass | Meaning                                                                 |
| ---------- | ------------------ | ------------------------- | ------------------------------------------------------------------------- |
| `embeds` | (N, M, 1, λ₁)  | (N, M, Q, λ₁)         | Per-atom identity vector. Q=1 on construction, expanded by MessagePass. |
| `f_mags` | (N, M, Q, 1)     | unchanged               | Form factor magnitude. Trailing 1 broadcasts over λ₁.                 |
| `sigmas` | (N, M, Q, 1)     | updated per round       | RFF bandwidth. Trailing 1 broadcasts over λ₁.                         |

MessagePass returns `embeds` and `sigmas` updated and `f_mags` unchanged (pass-through). OutputHead only consumes `embeds` and `f_mags`; `sigmas` is not part of the Debye-limit accumulation.

---

## 4. `MessagePass`

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

N-chunk level (`use_reentrant=True`): the entire `_n_chunk_round` (pass 1 -> mean-normalize -> pass 2) runs under `torch.no_grad()` in the forward pass, so `glob_ctx (Nc, Q, λ₅, λ₁)` is created and freed per N-chunk. During backward, the chunk is fully re-executed. `use_reentrant=False` would fail here because `_pass_2`'s step is a closure over `cont.glob_ctx`, which would keep all N-chunks' tensors alive simultaneously.

M-chunk level (`use_reentrant=False`): `_step1`/`_step2` are pure functions with no closures over large mutable tensors, so the non-reentrant API is safe and preferred. Their purpose is to avoid materialising `(Nc, M, Q, λ₅)` across all atoms.

### Mean-Normalization

Between passes, `features` and `glob_ctx` (sums over every atom in the N-chunk) are divided by the per-molecule real-atom count, turning the sums into means. `_step2` immediately forms `atmenv / weights` and feeds it into `RMSNorm`, which is invariant to any shared positive scale on its input, so the division is algebraically a no-op (verified identical to ~2e-6 post-RMSNorm) - "sum then contract then divide" rewritten as "mean then contract."

It still earns its keep numerically. The sums are O(M) ≈ 1e3-1e4 for large molecules, and the heavy bmm contractions run under BF16 autocast, which trades fp16's narrow exponent range for a narrow ~8-bit mantissa. Keeping `features`/`glob_ctx` at O(1) rather than O(M) before the contraction keeps that mantissa spent on signal rather than magnitude, the same reasoning that motivated the divide originally under fp16 overflow, just for precision instead of range.

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
   `_pass_1` iterates over M-chunks and accumulates `features` and `glob_ctx` for the current N-chunk. Each M-chunk step is gradient-checkpointed (`use_reentrant=False`) to avoid holding the full `(Nc, M, Q, λ₅)` RFF tensor in memory.
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
   step_glob_ctx  (Nc, Q, λ₅, λ₁)  bmm(zb, eb).reshape(Nc, Q, λ₅, λ₁)

   # accumulate across M-chunks:
   features  (Nc, Q, λ₅)       += step_features
   glob_ctx  (Nc, Q, λ₅, λ₁)   += step_glob_ctx
   ```

  After the M-chunk loop, `features`/`glob_ctx` are mean-normalised by the per-molecule atom count (see [Mean-Normalization](#mean-normalization) above).
  ii. **Pass 2: Per-Atom Update**
  `_pass_2` recomputes φ_m per M-chunk and uses the fully-accumulated `glob_ctx` to update embeddings and sigmas.

```{rtf}
  # recompute φ_m (intermediate freed during forward by checkpointing)
  scaled_coords  (Nc, mc, Q, 3)
  proj           (Nc, mc, Q, λ₅)
  zrff           (Nc, mc, Q, λ₅)  zeroed at padding

  # locality_m ≈ sum_{m'} k(r~_m, r~_{m'}) * e_{m'}, via bmm
  zb             (Nc*Q, mc, λ₅)   zrff.permute(0,2,1,3).reshape(...)
  cb             (Nc*Q, λ₅, λ₁)   glob_ctx.reshape(...)
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


| Activation                | Location                | Behaviour                                                                                             |
| --------------------------- | ------------------------- | ------------------------------------------------------------------------------------------------------- |
| `cos`                     | RFF feature computation | Core of the RFF kernel approximation.                                                                 |
| `F.mish` (p2 path)        | MishGLU gate            | `x*tanh(softplus(x))`. Near zero when p2 << 0 (gate closed); near-linear when p2 >> 0 (gate open).    |
| `nn.RMSNorm`              | After locality/weights  | Normalises aggregate magnitude, preventing residual stream from compounding across rounds.            |
| `PTanhShrink`             | Sigma delta             | `y - c*tanh(y/c)` with a bounded learned width `c`. Cubic near 0 (sticky region); `f' = tanh²(y/c)`. |
| `arctan_log_sigma_window` | Sigma output            | Squashes log σ back into`(sigma_floor, sigma_max)` every round. Same function Embed uses.            |

- MishGLU Gate
  `_proj_agg` is `nn.Linear(λ₁, 2λ₁)`. The output is split into p1 (value path) and p2 (gate path). `gate = p1 * Mish(p2)` is added to the atom embedding as a residual. The final `* mask` zeroes contributions from padding atoms.
  The GLU pattern lets the network decide per-channel and per-q-point whether to incorporate the neighbourhood context, rather than always adding the full aggregate.
- Sigma Update

  ```{rtf}
  σ_new = exp( window( log σ_old + ptanhshrink( qdiag(RMSNorm(e_updated), f_mag) ) ) )
  ```

  `_sigbilin[r]` is a `PTanhShrink` wrapping a `QDiagBilin(λ₁, Q, q_points=Q)`: one weight matrix **per q-point**, so the sigma delta at q depends on that q's embedding coupled against the atom's whole form-factor spectrum. That is `Q * λ₁ * Q` weights per round rather than `λ₁ * Q`, a 51x capacity increase in the sigma path. It costs nothing in memory because the output index and the input's q axis are the same index: contracting λ₁ first gives `(Nc, mc, Q, Q)` and never materializes the `(Nc, mc, Q, Q, Q)` a generic bilinear would (1.012 GiB vs 20.3 MiB at `(4, 512, 51)`).

  The update is additive in **log** space and squashed back into `(sigma_floor, sigma_max)` by the same `arctan_log_sigma_window` Embed uses, so the window binds the σ that actually reaches the kernel and the loss, not just Embed's round-0 σ. This is load-bearing: σ is the RBF bandwidth, the kernel forms `r/σ`, and `d(kernel)/dσ` carries a `σ^-2` pole. An earlier unbounded update (`squareplus(σ + tanhshrink(...))`, asymptoting to `b/(4|x|) -> 0⁺`) let σ collapse well under Embed's floor by round 1, and the resulting `σ^-2` gradient blowup dominated the global-norm clip, starving every other parameter.

  The head reads `RMSNorm(new_emb)`, not the raw residual, since the raw residual's pre-activation std (`sqrt(Q/3)*std(e)*f_rms` ≈ 25-90) would saturate the log-σ window immediately.

  `PTanhShrink` is cubic near the origin (`f(y) ≈ y³/3c²`, `f' = tanh²(y/c)`), so early in training, while the bilinear output is small, σ stays sticky rather than wandering. Its width `c` is **bounded** rather than free: `dg/dc` has the same sign for every `y`, so a free `c` climbs monotonically (measured 1 -> 30 in 600 steps, still rising), closing the layer and blowing up `dg/dlog_c`.

  The last round (`r = λ₂-1`) has **no** sigma head at all. σ feeds the *next* round's kernel and `OutputHead` ignores σ entirely, so that round's σ weights had no path to the loss and measured `grad = None`.

---

## 5. `OutputHead`

Collapses per-atom representations into a predicted I(q) curve per molecule, as a sum of the two limits of the Debye equation:

```{rtf}
I(q) = (sum_j w_j(q) f_j(q))²  +  sum_j c_j(q) f_j(q)²
        \-- coherent --/           \-- incoherent --/
```

The MLP emits two per-atom, per-q channels (w, c) instead of one. In the full Debye equation `I(q) = sum_j sum_k f_j f_k sinc(q r_jk)`, the two terms above are its endpoints. At q -> 0 every `sinc(q r_jk) -> 1`, so the double sum factorizes to `(sum_j f_j)²`, which scales like M². At high q the off-diagonal pairs oscillate away, leaving the diagonal `sum_j f_j²`, which scales like M. Squaring a sum over atoms *is* the pairwise double sum, so the coherent term is recovered from one O(M) pass rather than an O(M²) pair sum.

Both channels stay O(1) at every q; the M² scaling comes from the squaring op, not from learned magnitude. This matters because a single linear sum-of-f² head (the previous design) cannot reach `(sum f)²` at any parameter setting. It could only approximate low q by inflating its per-atom weights to ~M, which from standard init left high q roughly correct immediately while low q started M-fold low and converged far slower. See [Appendix: low-q coherent limit](#low-q-coherent-limit) for the full diagnosis.

### `OutputHead` Layers

1. **Learnable Parameters**


| Parameter                | Shape           | Notes                                                 |
| -------------------------- | ----------------- | ------------------------------------------------------- |
| `__bilinear.weight`      | (λ₃, λ₁, 1) | Bilinear: (embedding, f_mag scalar) -> λ₃ features  |
| `__bilinear.bias`        | (λ₃,)         |                                                       |
| `__mlp / layer_i.weight` | varies          | MLP linear layers (halving pyramid, terminal width 2) |
| `__mlp / layer_i.bias`   | varies          |                                                       |

(Name-mangled, as in Embed - see [§3](#3-embed).)

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

  # f_mags/sigmas are no longer routed through OutputHead - ScatterNet.forward
  # reads them directly off MessagePass's (embeds, f_mags, sigmas) tuple.

  # the caller finishes it, once the sums cover every atom:
  # SplineSmooth(sigmas, f_mags, coh, inc, batch) -> smoothed, combined I(q)
  # (N, Q). See the SplineSmooth section below.
```

  **Why `forward` returns the coherent sum unsquared.** Squaring a *partial* sum computes `sum_parts (sum_j w_j f_j)²` where the correct quantity is `(sum_parts sum_j w_j f_j)²`. The two differ by every cross-part pair term, so every atom pair whose members land in different parts is silently dropped from the coherent sum, and the head degrades toward the incoherent-only behaviour it was written to replace.
  `forward`'s own `out_chunk` loop splits the atom dimension, so it cannot square until every chunk is accumulated: it hands back `coh_accum` and `inc_accum` raw once its sums cover every atom of the (whole, single-GPU) molecule, and `SplineSmooth` does the squaring as its final step (see below).
  Nothing in a normal run surfaces a violation of this if it were broken. The truncated model would still be a smooth function of the parameters, so it would still train and the loss would still go down, converging to a chunk-dependent function instead of the intended one, with no loss curve, R², or per-q plot distinguishing the two. The only reliable check is an explicit invariance test: forward the same batch at `out_chunk = M` (no cross-chunk pairs to lose) and at `out_chunk = 1` (every pair is cross-chunk), asserting I(q) matches.
3. **Activations**


| Activation | Location           | Behaviour                                                                                                                                                                |
| ------------ | -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `F.mish`   | After bilinear     | Applied to bilinear features before the MLP.                                                                                                                             |
| `nn.Mish`  | Between MLP layers | Between each pair of linear layers except the last.                                                                                                                      |
| `SqrP`     | On MLP channel 1   | Square-plus`(x + sqrt(x² + b))/2`, learned `b`. Keeps the incoherent channel `c` strictly positive; it weights a sum of squares. Note its negative tail decays as `b/(4 |
| `PBId`     | On MLP channel 0   | Bent identity,`w·(sqrt(x²+1)-1)/2 + x + b`. Leaves the coherent channel `w` **signed**.                                                                                |

#### Why w is signed

`c` weights `sum_j c_j f_j²`, the diagonal term, so it has to be positive. `w` does not: `coh = sum_j w_j f_j` is squared before it reaches `I(q)`, so non-negativity is guaranteed by the architecture rather than by the activation.

Forcing `w > 0` costs the head the only cheap route to the high-q limit. A sum of M strictly-positive terms satisfies `coh >= M · min_j(w_j f_j)`, so it can shrink only if *every* `w_j` shrinks together, toward the molecule-size-dependent value `1/sqrt(M)` (0.577 at M=3, 0.080 at M=156, 0.013 at M=6046). Under `softplus` those sit at pre-activations of −0.25, −2.48 and −4.35, where the derivative has fallen to 0.44, 0.077 and 0.0128 — so the head must learn a size-dependent envelope while its gradient is being suppressed by up to 34x.

A signed `w` reaches the same place by **destructive interference**: the `sqrt(M)` reduction falls out of random-phase cancellation with no size envelope to learn. That is also the physical mechanism. The coherent amplitude is `sum_j f_j exp(i q · r_j)`, and it decays at high q because the phases cancel, not because the atomic amplitudes shrink, so `w_j(q)` acts as a per-atom stand-in for the `cos(q · r_j)` phase factor.

`PBId` rather than a bare identity, `PReLU`, or `Mish`: all four are signed, but the channel's job is to *cross zero cleanly*, and only the first two do. `Mish` has its minimum at `x = −1.1924` where the derivative is 8.3e-6, is non-monotonic on the negatives (each value is reachable from two pre-activations), and clamps them at −0.309 while leaving the positives unbounded. `PReLU`'s kink sits exactly at the crossing point. `PBId` is monotone with `f'(0) = 1` and `f' ∈ (1 − |w|/2, 1 + |w|/2)`, and its learned bend `w` interpolates between the exact identity (`w = 0`) and the standard bent identity (`w = 1`). It is left unconstrained; past `|w| = 2` it loses monotonicity and folds, so bound it with `2·tanh` if a run drives it there.

---

## 6. `LambdaHead`

Small per-q MLP that predicts a non-negative smoothing penalty Λ(q) from pooled per-molecule features. Two independent instances feed `SplineSmooth` below, one per channel (coherent, incoherent).

Λ must be able to reach exactly (near-)zero, so a molecule the model already fits well gets no smoothing correction downstream. The terminal layer is bias-initialized to `-10` so every run starts there (`softplus(-10) ~= 4.5e-5`), matching this codebase's "start near the identity" convention elsewhere (`OutputHead`, etc.). The terminal weight is left at its default random init rather than zeroed — zeroing it would kill gradient into the hidden layer structurally (`d(output)/d(hidden)` scales with that weight regardless of the activation), not just at that one unit.

`softplus` was chosen over `relu` and `SqrP`/square-plus for the same non-negativity job elsewhere in the model. `relu` is dead (zero gradient) once the pre-activation goes negative, which is exactly the regime `Λ ~ 0` lives in. `SqrP` stays live everywhere, but its negative tail decays as `~1/|x|`, so reaching a small Λ needs an impractically large negative pre-activation (`~1e6`); `softplus`'s tail decays exponentially, so a modest bias (`-10`) already gets Λ small at init.

### `LambdaHead` Layers

1. **Learnable Parameters**


| Parameter           | Shape          | Notes                                                       |
| --------------------- | ---------------- | ------------------------------------------------------------- |
| `net.0.weight,bias` | (16, 2), (16,) | `Linear(2, hidden)`, input is `(pooled_sigma, pooled_fmag)` |
| `net.2.weight,bias` | (1, 16), (1,)  | Terminal`Linear(hidden, 1)`; bias initialized to `-10`      |

  `SplineSmooth` holds two independent instances (`_lmb_coh`, `_lmb_inc`), one per channel, each with its own weights.

2. **Forward Pass**

```{rtf}
  # inputs: pooled_sigma, pooled_fmag (N, Q) per-molecule, per-q

  x = stack(pooled_sigma, pooled_fmag, dim=-1)   (N, Q, 2)
  Linear(2, 16)(x)                                (N, Q, 16)
  Mish(...)                                       (N, Q, 16)
  Linear(16, 1)(...)                              (N, Q, 1)
  softplus(...).squeeze(-1)                       (N, Q)   >= 0
```

3. **Activations**


| Activation | Location     | Behaviour                                                                                                                                                     |
| ------------ | -------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `nn.Mish`  | Hidden layer | Between the two linear layers.                                                                                                                                |
| `softplus` | Output layer | Keeps Λ non-negative (required for`PPSpline`'s `(I + DᵀΛD)` solve to stay positive-definite); exponential tail lets Λ reach near-zero from a modest bias. |

---

## 7. `SplineSmooth`

Pools `OutputHead`'s per-atom `sigmas`/`f_mags` to per-molecule, per-q features, feeds them to a `LambdaHead` per channel to get Λ(q), and passes `coh`/`inc` through a `PPSpline` (Whittaker/P-spline smoother, `Y_s = (I + DᵀΛD)⁻¹Y_r`) using that Λ before combining: `coh_smooth**2 + inc_smooth`.

Smoothing happens on `coh` **before** squaring, with its own `PPSpline` instance, not on `coh**2` after. Squaring entangles curvature with slope (`d²/dq²[u²] = 2(u')² + 2u·u''`), so `smooth(coh)² != smooth(coh²)`; the two channels also need independent Λ since they have different noise/curvature profiles.

### `SplineSmooth` Layers

1. **Learnable Parameters**

  None directly — `SplineSmooth` owns two `LambdaHead` instances (`_lmb_coh`, `_lmb_inc`, see [§6](#6-lambdahead)) and two `PPSpline` instances (`_pps_coh`, `_pps_inc`). `PPSpline` itself has no learnable parameters: `Λ` is supplied per call by the matching `LambdaHead`, and the difference operator `D`/identity `I` are fixed buffers baked from `q_points`/`delta_q` at construction.

2. **Forward Pass**

```{rtf}
  # inputs: sigmas, f_mags (N, M, Q) per-atom; coh, inc (N, Q) raw, unsquared

  mask          (N, M)    batch.padding_mask()
  n_atms        (N, 1)    mask.sum(dim=1), clamped >= 1

  sigmas_pooled (N, Q)    masked-mean of sigmas over atoms
  f_mags_pooled (N, Q)    masked-mean of f_mags over atoms

  lam_coh = lmb_coh(sigmas_pooled, f_mags_pooled)   (N, Q)  softplus, >= 0
  lam_inc = lmb_inc(sigmas_pooled, f_mags_pooled)   (N, Q)  independent MLP

  coh_smooth = pps_coh(coh, lam_coh)   (N, Q)   (I + DᵀΛ_coh D)⁻¹ coh
  inc_smooth = pps_inc(inc, lam_inc)   (N, Q)   (I + DᵀΛ_inc D)⁻¹ inc

  return coh_smooth**2 + inc_smooth    (N, Q)   squared AFTER smoothing
```

  Inside `PPSpline.forward`: `Λ`'s boundary values (`q=0`, `q=Q-1`) are dropped (`lam[:, 1:-1]`), since the second-difference stencil `D` has no row for the two edge grid points; `DᵀΛD` is then formed per-molecule and the linear system solved with `torch.linalg.solve`.

3. **Activations**

  None directly in `SplineSmooth` — see [§6 `LambdaHead`](#6-lambdahead) for the activations in its inputs.

---

## 8. `ScatterNet`

Top-level module, single-GPU only: no process groups, no sharding, no routing of any kind. Wraps `Embed`, `MessagePass`, `OutputHead`, and `SplineSmooth`, and owns `compute_loss` (see [§9](#9-loss)).

### Module Registry


| Submodule  | Type              | Role                                                     |
| ------------ | ------------------- | ---------------------------------------------------------- |
| `__emb`    | `Embed`           | per-atom embeddings, form factors, sigmas                |
| `__msg`    | `MessagePass`     | λ₂ rounds of RFF message passing                       |
| `__out`    | `OutputHead`      | accumulates raw coherent/incoherent Debye sums           |
| `__spline` | `SplineSmooth`    | smooths + combines the two channels into I(q)            |
| `__iq2d`   | `IQ2ndDerivative` | central-difference 2nd derivative, used by`compute_loss` |

`__eps_embd` and `__eps_msgp` are plain Python floats (not parameters or buffers); they are not moved by `.to(device)`. `__fmag_table` (V, Q) and `__q_weights_` (1, Q) are non-persistent buffers built at construction time and used by `compute_loss`; see [§9](#9-loss).

### Forward Pass

`forward(batch: Batch)` returns a 5-tuple `(iq, coh, inc, f_mags, sigmas)`:

```{rtf}
batch -> Embed(batch, ε_e)               (embeds, f_mags, sigmas): (N,M,1,λ₁), (N,M,Q,1), (N,M,Q,1)
      -> MessagePass(batch, head, ε_m)   (embeds, f_mags, sigmas): (N,M,Q,λ₁), (N,M,Q,1), (N,M,Q,1)
      -> OutputHead(batch, msg_head)     (coh, inc): (N,Q), (N,Q)              [unsquared, unsmoothed]
      -> SplineSmooth(sigmas, f_mags, coh, inc, batch) -> iq: (N,Q)
```

`coh` and `inc` are the raw accumulated Debye sums straight off `OutputHead`, before `SplineSmooth` touches them; they are returned alongside `iq` because `compute_loss`'s 2nd-derivative penalty (`lambda_7`) is applied to them directly, pre-smoothing. `f_mags` and `sigmas` come back squeezed to `(N, M, Q)` for the form-factor penalty and `SplineSmooth`'s per-molecule pooling respectively. The whole molecule lives on one device, so `coh`/`inc` already cover every atom by the time `OutputHead` returns - no cross-rank reduction of any kind.

---

## 9. `Loss`

There is no standalone `Loss` module. Loss computation is `ScatterNet.compute_loss(output_head, coh, inc, f_mag_pred, batch, lambda_6, lambda_7)`, a method on `ScatterNet` itself, using the `__fmag_table` / `__q_weights_` buffers described in [§8](#8-scatternet).

Form factor table construction: `q -> s = q/(4π)` converts to crystallographic s (sinθ/λ used by xraydb), then `|f(q)| = hypot(f0 + f1_chantler, f2_chantler)`. Transuranics: f0 only.

### Loss Terms

**Term 1: Kratky-weighted MSLE**, on the final (post-smoothing) `iq`:

```{Latex}
L_kratky(n, q) = (1 + q²) * (log1p(Î(q)) - log1p(I(q)))²
```

`log1p` handles the multi-decade dynamic range of I(q). The `(1+q²)` Kratky weight emphasises high-q structure; without it the Guinier region (low-q, high intensity) would dominate all gradients.

**Term 2: Form-factor penalty**, weight `lambda_6`:

```{Latex}
L_ff(n, q) = λ₆ * (1/n_atoms) * sum_m mask * (log1p(f_hat_m(q)) - log1p(f_ref_m(q)))²
```

Anchors predicted per-atom form factors to xraydb reference values, preventing the model from learning arbitrary f_mags that fit I(q) via cancellation. Atom-count normalisation makes the penalty size-independent.

**Term 3: 2nd-derivative smoothness penalty**, weight `lambda_7`, applied to `coh`/`inc` *before* `SplineSmooth`:

```{Latex}
L_smooth(n, q) = λ₇ * (coh''(q)² + inc''(q)²)
```

`coh''`/`inc''` come from `IQ2ndDerivative`, a central-difference second derivative (`(f[q+Δq] - 2f[q] + f[q-Δq]) / Δq²`, second-order one-sided at the grid boundaries). This penalizes `OutputHead`'s raw curves directly rather than the post-`SplineSmooth` output, so the term nudges `OutputHead` itself toward an already-reasonable curve instead of letting `SplineSmooth`'s learned Λ compensate for a rough raw prediction. Default `lambda_7 = 0.25`.

**Total:**

```{Latex}
L_total = mean_{n,q}[ L_kratky(n,q) + L_ff(n,q) + L_smooth(n,q) ]
```

`.mean()` averages over all N molecules and all Q q-points. Per-molecule normalisation inside `L_ff` prevents large molecules from dominating. Sigma needs no explicit penalty term: `MessagePass` bounds it structurally with `Embed`'s arctan sigma window on every round (see `ScatterNet/utils/sigma_window.py`).

---

## 10. Training Loop

### Optimizer

`torch.optim.Adam(..., lr=lr, decoupled_weight_decay=True)` over **two** parameter groups:


| group    | members                                        | weight decay   | lr   | betas        |
| ---------- | ------------------------------------------------ | ---------------- | ------ | -------------- |
| decay    | weight matrices                                | `weight_decay` | `lr` | (0.9, 0.999) |
| no-decay | biases,`rms_norm`, `prelu`, `biasterm`, `_mbd` | 0.0            | `lr` | (0.9, 0.999) |

**Why `_mbd` left the decay group.** The embedding table is indexed, so a vocab entry absent from a batch receives no gradient, while decoupled decay applies every step regardless. The 2026-08-01 run had **52 of 211 rows with `exp_avg_sq == 0`** (never once seen in the data) shrinking monotonically toward zero. Decay is only meaningful for a parameter the loss pushes back on.

> **Resume compatibility.** The optimizer `state_dict` layout is keyed by the flattened group order, so a change in group count cannot be remapped. Checkpoints from before the current two-group split raise a `RuntimeError` naming the mismatch rather than silently attaching the wrong Adam moments.

### Learning Rate Schedule

`torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=lr_factor, patience=lr_patience, threshold=lr_threshold, threshold_mode="rel", min_lr=lr_min)`, stepped once per epoch (not per batch) on the **validation** loss. An epoch counts as progress only if `val_loss < best * (1 - lr_threshold)`; after `lr_patience` consecutive non-improving epochs the LR is multiplied by `lr_factor`, floored at `lr_min`. `lr_factor=1.0` disables decay.

Validation, not train loss (which is averaged over the epoch and so lags the current weights) and not test (which would leak the test split into the training procedure).

**Why plateau rather than exponential.** `ExponentialLR(gamma)` fires unconditionally, regardless of how training is going, and it assumes an epoch is a fixed amount of work. `dataset_frac` breaks that assumption: at a small `dataset_frac` an epoch is a small fraction of the optimiser steps, so a fixed per-epoch `gamma` decays much faster per step than the same number implies at `frac=1.0`, annealing the LR on a schedule unrelated to whether the model is still improving. Keying the cut off measured validation progress fixes that: the LR never decays while the model is still improving, and the schedule needs no known horizon, which matters because `cfg.epochs` is a per-invocation count rather than the run's total (a resume runs `cfg.epochs` MORE epochs from wherever it left off, so a horizon-based schedule like cosine has no endpoint to anneal toward).

It is **not** fully scale-free, though: `lr_patience` and `lr_threshold` are counted in *epochs*, so shrinking `dataset_frac` shortens each epoch, puts less progress in it, and makes a fixed relative-improvement threshold proportionally harder to clear, cutting the LR earlier in step-terms. Loosen `lr_patience`/`lr_threshold` when running at a small `dataset_frac`.

A reactive mid-epoch cut (windowed train-loss plateau detection) previously ran alongside this and was removed: it reacted to batch-to-batch noise rather than real stalls and throttled training far more aggressively than the per-epoch decay alone.

**Resume**: unlike `ExponentialLR`, this schedule is **path-dependent**. `best`, `num_bad_epochs`, and `cooldown_counter` cannot be reconstructed from the epoch number, so the checkpoint's `"scheduler"` state is load-bearing rather than an optimisation. The LR itself still rides in the optimizer: `ReduceLROnPlateau` mutates `optimizer.param_groups["lr"]` in place, so a checkpoint already has the true current lr baked into `optimizer.state_dict()`. On resume, `optimizer.load_state_dict(ckpt["optimizer"])` runs **before** `scheduler.load_state_dict(ckpt["scheduler"])`, and the latter only restores the scheduler's own bookkeeping, never touching `optimizer.param_groups`. So the lr used after a resume always comes from the checkpoint, never from `RunConfig.lr`; `cfg.lr` only matters for a fresh run. The checkpoint also saves the current lr as an explicit top-level `lr` field (redundant with `optimizer.state_dict()`, but avoids digging into optimizer state to inspect it).

### Per-Batch Step

```
1. Move batch to device
2. optimizer.zero_grad(set_to_none=True)
3. with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
       iq, coh, inc, fmags, sigmas = model(batch)
       loss = model.compute_loss(iq, coh, inc, fmags, batch, lambda_6, lambda_7)
4. loss.backward()
5. clip_grad_norm_(model.parameters(), grad_clip)
6. optimizer.step()
```

**BF16 always on.** `torch.autocast(device_type="cuda", dtype=torch.bfloat16)` wraps the forward and loss unconditionally, with no `enabled=` flag and no loss scaling. BF16 keeps fp32's exponent range (unlike fp16), so there is no overflow/underflow tradeoff to tune and no `GradScaler` is needed; `loss.backward()` and `optimizer.step()` run directly on the resulting gradients. There is no distributed all-reduce step: training runs on a single GPU.

### Epoch Metrics

After all training batches, `evaluate()` runs over `val_loader` and `test_loader` with `torch.no_grad()`, under the same always-on BF16 autocast as training.

R² is computed in the log1p domain: `1 - SS_res / SS_tot`, where `SS_tot = sum(y²) - (sum(y))²/n` (online, one pass), accumulated in fp32 regardless of autocast.

Evaluation is done once per epoch for both val and test. Val is used for checkpointing (best model selection); test is strictly a held-out report and does not influence any decision.

### Checkpoint and Resume


| File                                                                                                  | Contents                                                                                                                                                  | Saved when                                                   |
| ------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| `ckpt_best`                                                                                           | model weights only                                                                                                                                        | val_loss improves                                            |
| `ckpt_dir/checkpoint_<epoch>_<batch>.pt` (train) / `checkpoint_<epoch>_<phase>_<batch>.pt` (val/test) | weights + optimizer + scheduler + epoch + phase + batch_idx +`best_val` + this epoch's already-known train/val/test scalars + phase's partial accumulator | every`ckpt_interval_sec` seconds and at every phase boundary |

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

## 11. Hyperparameter Reference


| Name                | Default | What it controls                                                                                                                                                                                                                         |
| --------------------- | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `lambda_1`          | 128     | Atom embedding dimension. Width of the per-atom vector throughout MessagePass.                                                                                                                                                           |
| `lambda_2`          | 4       | Message-passing rounds. Also bounds the maximum cumulative sigma change.                                                                                                                                                                 |
| `lambda_3`          | 256     | OutputHead MLP starting width (bilinear output width, MLP input). Must satisfy`lambda_3 >= 2^lambda_4`.                                                                                                                                  |
| `lambda_4`          | 4       | OutputHead halving steps. The ladder always terminates at width 2 (the w and c channels), appending a step if the halvings do not land there. With defaults: 256->128->64->32->16->2.                                                    |
| `lambda_5`          | 128     | RFF count. More = tighter kernel approximation, higher memory cost.                                                                                                                                                                      |
| `lambda_6`          | 1.0     | Form-factor penalty weight.                                                                                                                                                                                                              |
| `lambda_7`          | 0.25    | Second-derivative smoothness penalty weight, applied to the pre-`SplineSmooth` coherent/incoherent curves.                                                                                                                               |
| `msg_seed`          | 42      | Seed for fixed RFF frequency matrix Ω.                                                                                                                                                                                                  |
| `atm_chunk`         | 512     | Atoms per M-chunk. Reduce to lower VRAM.                                                                                                                                                                                                 |
| `mol_chunk`         | 256     | Molecules per N-chunk. Reduce to lower VRAM on large molecules.                                                                                                                                                                          |
| `compile`           | True    | torch.compile Embed/MessagePass/OutputHead's checkpointed step functions (fullgraph=True, dynamic=True).                                                                                                                                 |
| `eps_embd`          | 1e-8    | Numerical floor in Embed (hypot, form factors).                                                                                                                                                                                          |
| `eps_msgp`          | 1e-3    | Numerical floor in MessagePass (sigma clamp, aggregate denominator).                                                                                                                                                                     |
| `lr`                | 1.3e-4  | Adam learning rate (starting value; see`lr_factor`).                                                                                                                                                                                     |
| `lr_factor`         | 0.5     | ReduceLROnPlateau decay factor, applied when val loss stops improving. 1.0 = no decay.                                                                                                                                                   |
| `lr_patience`       | 2       | Consecutive non-improving epochs tolerated before the LR is cut.                                                                                                                                                                         |
| `lr_threshold`      | 1e-3    | Relative improvement needed to count as progress:`val_loss < best * (1 - lr_threshold)`.                                                                                                                                                 |
| `lr_min`            | 1e-6    | Floor the LR is never reduced below.                                                                                                                                                                                                     |
| `weight_decay`      | 0.1     | AdamW decoupled decay. Applied to weight matrices only: not`_mbd`, biases, norm/gain params, or the RFF phases.                                                                                                                          |
| `grad_clip`         | 5.0     | Max gradient L2 norm before clipping.                                                                                                                                                                                                    |
| `epochs`            | 20      | Training epochs.                                                                                                                                                                                                                         |
| `batcher_seed`      | 0       | Seed for train/val/test split and per-epoch shuffle.                                                                                                                                                                                     |
| `dataset_frac`      | 1.0     | Fraction of each split's batches to use, (0.0, 1.0]. Applies to**train, val and test** - eval costs on the order of a train epoch, so thinning train alone just lets eval dominate. Deterministic off `batcher_seed`, fixed for the run. |
| `atom_size_ceil`    | -1      | Max total atoms per batch (-1 = 3x largest molecule).                                                                                                                                                                                    |
| `num_workers`       | 4       | DataLoader worker processes.                                                                                                                                                                                                             |
| `ckpt_interval_sec` | 600     | Seconds between mid-epoch resume checkpoints.                                                                                                                                                                                            |
| `profiler`          | False   | Diagnostic run: torch.profiler + per-section wall-clock timers, then stop. Traces written to`./profiler_trace/`.                                                                                                                         |
| `prof_warmup`       | 1       | Profiler warmup batches (profiled, discarded).                                                                                                                                                                                           |
| `prof_active`       | 3       | Profiler active batches (recorded). Loop runs`1 + prof_warmup + prof_active` batches; raise `prof_active` for more representative stats.                                                                                                 |

---

## 12. End-to-End Data Flow

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
       └─ returns (embeds, f_mags, sigmas): (N,M,1,λ₁), (N,M,Q,1), (N,M,Q,1)

  └─ MessagePass x λ₂ rounds
       └─ embeds.expand Q dim        -> (N, M, Q, λ₁)
       └─ [per round]
            └─ _pass_1: accumulate features (Nc,Q,λ₅), chem_env (Nc,Q,λ₅,λ₁)
            └─ mean-normalise features/chem_env by per-molecule atom count
            └─ _pass_2: locality, weights, agg, MishGLU gate -> new embeds
                        qdiag + ptanhshrink + arctan window   -> new sigmas
       └─ returns (embeds, f_mags, sigmas): embeds (N,M,Q,λ₁), sigmas updated, f_mags unchanged

  └─ OutputHead
       └─ [per M-chunk]
            └─ bilinear(embeds, f_mags) -> (N,mc,Q,λ₃)
            └─ Mish -> MLP -> (N,mc,Q,2) -> PBId -> w (signed)
                                         └─ SqrP     -> c (positive)
            └─ w * f_mags  * mask -> sum over atoms -> coh_accum (N,Q)
            └─ c * f_mags² * mask -> sum over atoms -> inc_accum (N,Q)
       └─ returns coh_accum, inc_accum, RAW and unsquared  (N,Q) each

  └─ SplineSmooth (see SplineSmooth section)
       └─ pool sigma, f_mag over atoms -> sigmas_pooled, f_mags_pooled (N,Q)
       └─ LambdaHead(coh), LambdaHead(inc) -> lam_coh, lam_inc (N,Q), >=0
       └─ PPSpline(coh_accum, lam_coh)  -> coh_smooth (N,Q)
       └─ PPSpline(inc_accum, lam_inc)  -> inc_smooth (N,Q)
       └─ returns coh_smooth**2 + inc_smooth -> I(q) (N, Q)

Loss
  └─ _kratky_MSLE:    (1+q²)*(log1p(Î)-log1p(I))²          (N,Q)
  └─ _ff_penalty:     λ₆*(log1p(f̂)-log1p(f_ref))²/n        (N,Q)
  └─ _smooth_penalty: λ₇*(d²coh_accum + d²inc_accum)², pre-spline curvature via IQ2ndDerivative (N,Q)
  └─ .mean() -> scalar

Optimizer: Adam(clip at grad_clip) -> parameter update
```

---

## 13. Profiling and Optimization

### Running the Profiler

Set `profiler: true` in your YAML config or pass `--profiler` on the CLI. This runs a short **diagnostic** instead of normal training: the loop runs `1 + prof_warmup + prof_active` batches (defaults `1 + 1 + 3 = 5`; tune with `--prof_warmup`/`--prof_active`), then stops - no eval or checkpointing. Adjust `prof_active` higher (e.g. 20-50) to average over many buckets.

### Which Buckets Get Profiled

The `1 + prof_warmup + prof_active` budget is split three ways across bucket **metadata** (atom counts only - no tensors loaded), not drawn randomly from the shuffled train loader:

1. **Heaviest by `N*M`** - the compute-time proxy.
2. **Heaviest by raw `M`** - the memory-risk proxy. A large-`M`/small-`N` bucket can rank low on `N*M` (small `N` keeps the product down) while still having the largest per-chunk RFF tensors (`atm_chunk`-sized chunks are actually full when `M` is large) - invisible to the first ranking alone.
3. **A band around the median `N*M`** - a "regular" batch baseline, so the two worst-case groups have something typical to compare against instead of only ever showing outliers.

Each group is deduplicated against the ones before it. The single heaviest bucket by `N*M` stays first (`bi=0`, the profiler's "wait" step) so it gets a clean `peak_alloc` reading with no torch-trace overhead. The startup log line reports one example bucket from each group (`mols x atoms`) so you can sanity-check what got selected.

**The section-timer report breaks these three groups out separately** (`---- per-group breakdown ----`, printed after the combined summary), each with its own batch count, mean `compute`/`forward`/`backward` (ms/batch), and peak `peak_alloc`. The combined numbers blend three structurally different populations, so a profiler run's *combined* average isn't comparable across runs with a different bucket mix (e.g. after a config change that shifts which buckets fall in each group). Compare **within the same group** across runs to isolate the effect you actually changed.

Two decoupled layers of profiling run each time:

1. **Section timers** - a CUDA-synced wall-clock breakdown printed at the end, over the **full** `prof_active` window (so averages are representative): time spent in `data_wait` / `h2d` / `forward` / `loss` / `backward` / `clip` / `step`, plus the heaviest batches by data-wait and by compute (with molecule count, max atoms, and real atoms). Costs ~no extra memory.
2. **torch.profiler** - a CPU+CUDA TensorBoard trace at `./profiler_trace/` for kernel-level drill-down. This buffers every op in host RAM and materializes them at export, so it is memory-heavy: it samples only `min(prof_active, 3)` steady-state steps regardless of `prof_active`, and runs with `with_stack`/`profile_memory` **off** (a long active window or `with_stack` triggers the host OOM-killer -> process `SIGKILL`). Raising `prof_active` lengthens the cheap section-timer window, not the heavy trace.

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

**Start with the section timers** (printed to stdout) before opening the trace - they tell you which bucket of time to chase. High `data_wait` means the DataLoader is starving the GPU (see below); high `forward`/`backward` means go into the trace.

**High `data_wait`**: CPU/IO is the bottleneck. Each `BatchSet.__getitem__` re-opens the HDF5 file and runs a Python encode loop per atom, so heavy buckets stall. Raise `num_workers`; if `data_wait` stays high and tracks the heavy buckets in the report, cache `Batch` objects as `.pt` files or hoist the HDF5 handle out of `__getitem__`.

**GPU idle gaps** between CUDA kernels in the trace: CPU is the bottleneck, usually the DataLoader (HDF5 reads) or Python overhead between chunks.

**Short CUDA bars, lots of idle**: chunks are too small and kernel launch overhead dominates. Increase `atm_chunk`.

**OOM**: chunks are too large. Reduce `atm_chunk` or `mol_chunk`.

**Memory fragmentation**: `torch.cuda.memory_reserved()` greatly exceeds `torch.cuda.memory_allocated()`. If OOM despite low `memory_allocated`, reduce `atm_chunk` to create less fragmentation.

### Quick Optimization Checklist (single L4)


| Setting       | Recommendation                                                                                                                                            |
| --------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `num_workers` | 2-4 (0 = serial, >4 = diminishing returns)                                                                                                                |
| `pin_memory`  | True (already set in train.py for GPU)                                                                                                                    |
| `atm_chunk`   | Start at 512; raising it buys no steady-state speed (compute is roughly chunk-invariant, Appendix A1) and only costs memory + bigger torch.compile stalls |
| `mol_chunk`   | Start at 256; drop to 128 if OOM on large molecules                                                                                                       |
| `verbosity`   | `"batch"` for loss + memory stats every 20 batches                                                                                                        |
| `max_batches` | Set to 20 for a quick smoke test before a full run                                                                                                        |

---

## Appendix

### A1. mol_chunk and atm_chunk optimizations

`atm_chunk`/`mol_chunk` still gate local memory tiering in `MessagePass` post-refactor (`ScatterNet/model/message_pass.py`, `_CHUNK_TIERS`) and remain live hyperparameters.

**2026-07-10 sweep** (2x T4, fp16/AMP): predates the BF16-always-on, single-GPU migration, so the numbers below need re-measuring on an L4. Kept for the qualitative, architecture-level relationship, which should still hold: **steady-state compute is roughly invariant to chunk size**; larger chunks buy no speed and cost more peak memory. Default `atm_chunk=512, mol_chunk=256`; don't raise them expecting a speedup, don't lower them without a memory-pressure reason.

Two bugs found and fixed in the course of that sweep, still applicable:

- **RMSNorm/autocast dtype mismatch**: `MessagePass._rms_norm` ran under autocast against its fp32 weight, forcing an unfused fallback kernel. Fixed by running that call outside autocast, matching the nearby RFF block.
- **Non-contiguous chunk-slice views causing extra torch.compile recompiles**: chunked slices in `MessagePass`/`OutputHead`/`Loss` are views into larger padded tensors, and their stride/storage_offset vary by chunk index; torch.compile guards on that independently of shape, multiplying recompiles past the intended `_CHUNK_TIERS` axis. Fixed with `.contiguous()` at each call site: recompiles dropped from ~14 distinct guard failures to 4 legitimate ones.

---

### Low-q coherent limit

**Symptom**: low-q percent error diverged over training while high q converged normally.

**Correction (2026-08-04)**: an earlier version of this section blamed the "Kratky" diagnostic plot for suppressing low-q via a `q²` weighting. That's wrong: `plot_kratky` (`Baselines/run/metrics.py:977`) applies no `q²` at all; only its name did.

The plot did hide the failure, but differently: it draws `mean_j ln(1+I_true(q))` against `mean_j ln(1+I_pred(q))`, averaged **separately** per curve (`metrics.py:354-355, 446-447`). That's the mean *signed* log residual, so a model over-predicting half the set by +3 log units and under-predicting the other half by −3 draws two coincident curves, with no band, percentile, or `n` shown. `ln(1+I) ≈ 2 ln M + const`, so the mean is also dominated by the largest molecules. The per-molecule residual histogram (`metrics.py:1102`) doesn't have this defect. Two related caps also participated: `metrics.py:906-907` clamps per-q R² at `-3.0`, and `metrics.py:963-964` clamps per-q percent error at `500%`, both silently.

**Root cause**: the old `OutputHead` couldn't represent the low-q limit at any parameter setting. Ground truth is the Debye sum `I(q) = sum_i sum_j f_i f_j sinc(q r_ij)`:

- At `q -> 0`, `sinc -> 1` for every pair, so `I(0) = (sum_j f_j)²`: diagonal plus all M² - M off-diagonal pairs contribute, scaling as **M²**.
- At high q, off-diagonal pairs oscillate and average away, leaving the M diagonal terms `sum_j f_j²`, scaling as **M**.

The old head computed `I(q) = sum_j contribs_j(q) · f_j(q)²`, a single linear sum over atoms with no operation that squares a sum. That's structurally the incoherent (high-q) limit only: `(sum f)²` isn't reachable by any weights, since a linear functional can't produce the M² - M cross terms. The only way to approach the right low-q magnitude was to inflate `contribs_j` toward ~M, a molecule-size-dependent quantity the MLP had to learn from scratch, which explains why high q converged immediately while low q climbed slowly and dragged on molecule size.

**Fix**: two-channel coherent + incoherent head, `I(q) = (sum_j w_j f_j)² + sum_j c_j f_j²` (see [§5](#5-outputhead)). Squaring a sum over atoms *is* the pairwise double sum, so the M² term is recovered in one O(M) pass, and the M-scaling comes from architecture rather than learned magnitude.

**Retracted**: an earlier version blamed the `(1 + q²)` Kratky loss weight and a secondary story about σ being crushed at low q. Both wrong: over `q ∈ [0, 0.5] Å⁻¹`, `1 + q²` spans 1.00 to 1.25, a 25% spread that cannot explain an orders-of-magnitude error.

---

### The σ⁻² pole (2026-08-03)

**Symptom**: gradients exploding and the clip firing every step, with the σ pathway simultaneously frozen. At `grad_clip = 1.0` against measured norms of 10-18, the clip was doing normalisation rather than acting as a safety valve, and since buckets are size-sorted, the clip factor varied systematically with molecule size.

**Root cause**: σ is the RBF bandwidth; the kernel forms `r/σ`, whose backward carries `-r/σ²`, so `∂z/∂σ` has a σ⁻² pole and the σ → z map's condition number is `‖r‖/σ²`. `Embed` bounds σ into `[0.5, 100]` via an arctan squash to cap that at ~4, but `MessagePass` discarded the window on round 1: its `squareplus(σ + shrink(...))` update is unbounded below, asymptoting to `b/(4|x|) → 0⁺`. Measured after one round at init: `min σ = 0.0107`, 41% of entries below Embed's floor, `|dL/dσ|` of 1.5e8 at σ = 0.002 vs 2.4e3 at the floor. The resulting 1e6-1e8 global-norm clip rescaled every other parameter's gradient to ~1e-8, which against Adam's default `eps = 1e-8` gave updates of `lr·g/eps` instead of `lr`: 59% of the model was effectively frozen.

**Fix**: the σ update is now additive in log space and re-squashed by the same `arctan_log_sigma_window` every round, so the bound holds on the σ that reaches the kernel and loss. Also: the head reads `RMSNorm(new_emb)` instead of the raw residual, `NoTrilinBilin` inits from the true `in1*in2` fan-in, `adam_eps` drops to 1e-12, `grad_clip` rises to 5.0. Measured after: σ ∈ [1.4, 29.2], 0% below the floor, no zero/non-finite gradients, global grad norm 1.1 (from 1062).

The sigma *penalty* in the loss is gone with it: it measured identically 0 at init, making it the only (and dead) gradient path to σ. The window now bounds σ structurally, so there's nothing left for a penalty to charge for.

### Form-factor init (2026-08-03)

Untrained `Embed` emits `f_mag ~ 0.5` where the physical value is ~7. Both head channels are homogeneous of degree 2 in `f`, so `d log I / d log f_mag = 2` exactly: a ~200x error in `I(q)`, measured as 99.5% of the entire gradient norm at init (`_f0f1`/`_f2` at |g| = 96/102 vs 0.10 for the largest MessagePass weight). `ScatterNet` now passes a physical `f_init` curve into `Embed`, biasing `_f0f1` onto it and shrinking both weight matrices by `_F_INIT_WEIGHT_GAIN`.

`f_init` averages the organic elements (H/C/N/O/P/S), not the whole vocabulary: the 211-ion xraydb vocab has mean/median `|f|(0)` ≈ 46.7, but training structures are proteins/capsids where `|f|(0)` is 6.0 (C), 7.0 (N), 8.0 (O). The vocab mean overshoots ~6.7x, and since `I ~ f²` that's a ~45x error. Measured `I_pred/I_true` at init: 30.3 with the vocab mean, 1.09 with the organic mean.
