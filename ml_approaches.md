# ML Approaches for Structure → B_lm(q) Prediction

## Problem Statement

**Input:** molecular structure — N atoms with 3D positions (x, y, z) and atom types (Z)  
**Output:** B_lm(q) ∈ ℂ, shape `(lMax+1)² × N_q`  
**Ground truth:** analytic computation via existing codebase (free, unlimited labels)  
**Key symmetry:** B_lm transforms as an irrep of SO(3) under rotation of the input structure

---

## 1. Output Parameterisation Strategies

Before choosing a model, decide how to handle the (l, m, q) structure of the output.

### 1a. Full tensor prediction
- Output the entire `(lMax+1)² × N_q` complex tensor at once
- Simple; works if N_q is fixed across all structures
- Output dimension for lMax=10, N_q=100: 121 × 100 × 2 (real/imag) = 24,200 scalars
- **Pros:** single forward pass, captures q-correlations
- **Cons:** large output head, doesn't generalise to new q grids

### 1b. Condition on q (recommended starting point)
- Input: structure + scalar q; output: B_lm(q) for that q (121 complex numbers)
- Train with all (structure, q) pairs — effectively N_q × more training samples
- q can be encoded as sin/cos Fourier features or log-scaled
- **Pros:** small output, flexible q grid, natural curriculum (train easy q first)
- **Cons:** no explicit q-correlation in output; correlations must be learned implicitly

### 1c. Neural operator / basis decomposition
- Learn a set of K radial basis functions φ_k(q), predict coefficients c_{lm,k}
- B_lm(q) ≈ Σ_k c_{lm,k} · φ_k(q)
- Related to: Fourier Neural Operator (FNO), DeepONet
- **Pros:** compact latent, smooth q-dependence built in, generalisable q grid
- **Cons:** more complex; basis choice matters

### 1d. Separate model per l
- One model per l value (l=0..lMax), predicting all m at that l for all q
- Simpler output per model, easy to parallelize
- **Pros:** each model is smaller, easier to debug
- **Cons:** 11 models to train, no cross-l information sharing

---

## 2. Neural Network Architectures

### 2a. Equivariant GNNs (strongly recommended)

These respect SO(3) symmetry by construction. Since B_lm transforms as spherical harmonic
irreps under rotation, equivariant networks can output them directly in the correct basis.

#### e3nn (Euclidean Neural Networks)
- General-purpose equivariant NN library
- Operates on spherical harmonic features of type (l, parity)
- Message passing with tensor products between irreps
- Output layer can directly produce l=0..lMax features → B_lm naturally
- **Physics fit:** the internal feature space IS the spherical harmonic basis
- **Complexity:** O(N · L⁴) per layer due to tensor products; expensive for large lMax
- **Best for:** small-medium molecules, lMax ≤ 6 practically
- Reference: Geiger & Smidt, 2022

#### MACE (Message passing Atomic Cluster Expansion)
- Builds on e3nn; uses higher-order equivariant messages
- State of the art for molecular property prediction (energy, forces)
- Very data-efficient; works well with hundreds of structures
- **Physics fit:** excellent — designed for exactly this kind of atomic property prediction
- **Complexity:** more efficient than naive e3nn via ACE body-order expansion
- **Best for:** this project if you want one model to bet on
- Reference: Batatia et al., NeurIPS 2022

#### NequIP
- Equivariant interatomic potential; similar to MACE but slightly simpler
- Shown to work with very few training examples (tens)
- **Best for:** small datasets

#### PaiNN (Polarizable Atom Interaction Neural Network)
- Equivariant but uses vector (l=1) features only, not full spherical harmonics
- Faster than full e3nn but less expressive for higher-l outputs
- **Best for:** if you truncate output to l ≤ 1 or care mostly about l=0

#### SE(3)-Transformer
- Attention mechanism over equivariant features
- Captures long-range interactions better than local message passing
- **Best for:** large molecules where global structure matters (your PhiTE baseplate, STRIPAK)

#### Equiformer / EquiformerV2
- Transformer architecture in the equivariant space
- State of the art on OC20/OC22 benchmarks
- More expressive than message-passing-only approaches
- **Best for:** if you have enough data (thousands of structures)

### 2b. Invariant GNNs

These produce rotationally invariant outputs. B_lm is NOT invariant (it transforms under
rotation), so these cannot directly output B_lm. However:
- They CAN predict I(q) (which is invariant) as an auxiliary task
- They CAN predict |B_lm|² (invariant) — loses phase but retains power spectrum
- Useful as baselines to verify the problem is learnable

#### SchNet
- Distance-based message passing; ignores angles
- Simple, fast, well-understood
- Good baseline for I(q) prediction
- Reference: Schütt et al., NeurIPS 2017

#### DimeNet / DimeNet++
- Adds angular information (bond angles) to SchNet
- Better for directional properties
- Still invariant so can't output full B_lm

#### SphereNet
- Adds torsion angles; most expressive invariant GNN
- Better than DimeNet for complex molecular properties

### 2c. Point Cloud Methods

Treat the molecule as an unordered set of 3D points (atoms).

#### PointNet
- MLP on each point, then global max-pool
- No equivariance; rotation must be handled by data augmentation (random rotations)
- Very simple to implement; good sanity-check baseline
- For B_lm prediction: augment with 10,000+ random rotations per structure

#### PointNet++
- Hierarchical version; captures local + global structure
- Still not equivariant; same augmentation caveat

#### Point Transformer (V1/V2/V3)
- Self-attention over point cloud
- State of the art among non-equivariant point cloud methods
- Can be made approximately equivariant with augmentation

### 2d. Transformer / Attention-based

#### Equiformer
- Full transformer with equivariant attention; mentioned above
- Best of both worlds (attention + equivariance)

#### TokenGT (Tokenized Graph Transformer)
- Treats atoms as tokens, edges as tokens
- Standard transformer backbone; fast but not equivariant

#### Perceiver IO
- Handles variable-length inputs and outputs
- Could handle variable N_q naturally
- Not equivariant by default

### 2e. Graph Neural Networks (non-equivariant)

#### GCN / GAT / MPNN
- Standard graph convolutions over molecular graph
- No geometric awareness beyond connectivity
- Weakest baseline but fastest to implement

### 2f. Convolutional on Voxelised Structure

Voxelise the molecule onto a 3D grid, then apply 3D CNNs.

#### 3D CNN
- Standard conv layers on electron density or atomic density grid
- Naturally captures spatial relationships
- **Pros:** simple, GPU-efficient, well-understood
- **Cons:** loses atomic resolution; not equivariant; grid resolution vs. memory tradeoff
- Can be made approximately equivariant with group-equivariant CNNs (Cohen & Welling)

#### Spherical CNN
- Project structure onto concentric spherical shells, apply convolutions on S²
- Naturally produces spherical harmonic features
- Very natural fit for B_lm prediction
- Reference: Cohen et al. 2018, Esteves et al. 2018
- **Best for:** if you want an alternative equivariant approach to e3nn

### 2g. Recurrent / Sequential Models

Not a natural fit for point clouds, but possible if you serialise atoms (e.g., by distance from centroid).

#### LSTM / GRU over sorted atom sequence
- Weak baseline; ordering is arbitrary
- Only mention to rule it out

#### Set Transformer
- Attention over sets; permutation invariant
- Better than LSTM for unordered atoms; still not equivariant

---

## 3. Classical / Non-Neural Algorithms

### 3a. Random Forests / Gradient Boosting

**Input features must be hand-crafted** (since these don't operate on graphs/point clouds natively):
- Radial distribution function g(r) histogram
- Pair distance histogram
- Moment of inertia tensor eigenvalues
- Zernike 3D moments
- SOAP (Smooth Overlap of Atomic Positions) descriptors — most powerful option
- Coulomb matrix eigenvalues

**Algorithms:**
- Random Forest (scikit-learn)
- XGBoost / LightGBM / CatBoost
- Extra Trees

**Pros:** fast to train, interpretable feature importances, good with small datasets  
**Cons:** B_lm is complex-valued and high-dimensional — need one model per (l, m, q) or flatten; loses correlations  
**Best for:** quick baseline with SOAP features; surprisingly competitive for simple molecules

### 3b. Kernel Methods / Gaussian Processes

#### Gaussian Process Regression with SOAP kernel
- SOAP (Smooth Overlap of Atomic Positions): computes similarity between atomic environments as overlap of Gaussians expanded in spherical harmonics
- Kernel: k(A, B) = Σ_i Σ_j soap(a_i, b_j)^ξ
- GP gives uncertainty estimates — useful for knowing when the model is out-of-distribution
- **Pros:** works with very few structures (10-50); principled uncertainty; no gradient needed
- **Cons:** O(N³) training; doesn't scale past ~10,000 structures

#### SOAP + Ridge Regression (linear ACE)
- Fit linear model on SOAP features
- Surprisingly strong baseline; essentially what early ML potentials did
- Fast, interpretable

#### REMatch kernel
- Optimal-transport-based kernel over atomic environments
- More expressive than naive SOAP sum

### 3c. Monte Carlo Methods

#### Monte Carlo sampling of conformational space
- If structures have flexibility, sample conformations via MC and average B_lm
- Builds in thermal averaging that rigid-body formula misses
- Not a prediction algorithm per se — more a data generation / physics correction

#### Markov Chain Monte Carlo for inverse problem
- Sample structures consistent with observed B_lm via MCMC
- Defines a posterior p(structure | B_lm) which can be sampled
- Very slow; more of a baseline for the inverse problem

#### Monte Carlo Tree Search
- Not applicable here

### 3d. Variational / Generative Approaches

#### VAE (Variational Autoencoder)
- Encode structure to latent z, decode to B_lm
- Latent space could enable interpolation between structures
- Could also condition on q in latent space

#### Normalising Flows
- Learn invertible map structure ↔ B_lm
- Enables both forward and inverse prediction
- Equivariant flows exist (e.g., EquivariantFlow)

#### Diffusion Models
- State of the art for molecular generation
- Could generate structures conditioned on B_lm (inverse direction)
- DiffSBDD, DiffDock as references

#### Score Matching / Flow Matching
- More stable training than diffusion
- Faster sampling
- FrameDiff, FoldFlow as references for equivariant versions

### 3e. Symbolic / Physics-Based Regression

#### Sparse regression (SINDy-style)
- Fit B_lm as a sparse combination of physically-motivated basis functions of structure
- e.g., B_lm(q) ≈ Σ_k α_k · Φ_k(structure, q) where Φ_k are physical observables
- Produces interpretable closed-form expressions

#### Genetic algorithms / evolutionary strategies
- Evolve functional forms for B_lm from structure features
- Very slow; only worth it if interpretability is the main goal

### 3f. Zernike Moments

- 3D Zernike polynomials form an orthonormal basis on the unit ball
- Naturally related to spherical harmonics (same angular part)
- Can compute Zernike moments of atomic density analytically, then regress to B_lm
- Simpler and more interpretable than a full GNN
- Reference: Novotni & Klein 2003

---

## 4. Physics-Informed Loss Terms

### 4a. I(q) Consistency (highest priority)
```
L_iq = || I(q)_from_pred_Blm - I(q)_true ||²
```
where I(q) = (4π)² · Σ_{l,m} |B_lm(q)|²

- Free to compute; directly connects B_lm to measurable observable
- Differentiable; easy to add
- Enforces that the l-channel power sums are correct even if individual phases drift

### 4b. Conjugate Symmetry
For real structures: B_{l,-m}(q) = (-1)^m · B*_{lm}(q)
```
L_sym = || B_{l,-m} - (-1)^m · conj(B_{lm}) ||²
```
- Hard constraint: can also be enforced architecturally (output only m ≥ 0, reconstruct m < 0)
- Architectural enforcement is cleaner than a loss term

### 4c. Guinier Regime
At low q: I(q) ≈ I(0) · exp(-q²Rg²/3)  
Rg² = (1/N) Σ_i |r_i - r_cm|² (computable from structure)
```
L_guinier = || log I(q→0)_from_pred - log I(0)_true ||² + || Rg_from_pred - Rg_true ||²
```
- Enforces correct low-q behaviour; important for size/shape determination

### 4d. Porod Law (high q)
At high q: I(q) ~ q⁻⁴ · (surface area)
```
L_porod = || q⁴ · I(q)_from_pred - const ||² for large q
```
- Enforces correct high-q power law decay
- Useful regularizer if training data is sparse in high-q regime

### 4e. Positivity of I(q)
I(q) ≥ 0 always (it's an intensity)
```
L_pos = || ReLU(-I(q)_from_pred) ||²
```
- Or enforce via output activation: output |B_lm|² directly for l=0 component

### 4f. Radius of Gyration
Rg is computable from structure analytically. I(q) at low q gives Rg via Guinier.
Can add explicit Rg term:
```
L_Rg = (Rg_from_pred_Blm - Rg_true)²
```

### 4g. Sum Rule / Total Intensity
∫ I(q) q² dq relates to the total electron density (Parseval-like relation)
Provides a global normalisation constraint

### 4h. Smoothness of B_lm(q) in q
B_lm(q) should be smooth (no discontinuities) — analytic formula guarantees this
```
L_smooth = || dB_lm/dq ||²  (finite difference penalty)
```
Useful regulariser when predicting all q at once

### 4i. Phase Consistency across l
B_lm(q) phases are not independent — they encode the same physical structure.
Hard to formulate exactly, but can penalise rapid phase flipping:
```
L_phase = || angle(B_lm(q+dq)) - angle(B_lm(q)) ||²
```

### 4j. Debye Formula Consistency (expensive)
The full Debye formula gives I(q) = Σ_{ij} f_i f_j · sinc(q·r_ij)
Predicted B_lm should reproduce I(q) consistent with Debye (same as 4a but derived differently)
- Expensive to compute for large molecules; useful as validation metric

---

## 5. Featurisation / Input Representations

### 5a. Raw atomic coordinates + types
- Positions: (x, y, z) per atom
- Types: one-hot or learned embedding over Z (atomic number)
- Radial cutoff graph: connect atoms within r_cut (typically 5-8 Å)
- **Best for:** equivariant GNNs

### 5b. SOAP descriptors
- Smooth Overlap of Atomic Positions
- Expands atomic density in Gaussian × spherical harmonic basis
- Per-atom or global (summed/averaged)
- **Best for:** kernel methods, random forests, linear models
- Library: DScribe

### 5c. Coulomb Matrix
- M_ij = Z_i Z_j / |r_i - r_j| for i≠j; M_ii = 0.5 Z_i^2.4
- Eigenvalues as features (invariant)
- Simple but ignores angular structure

### 5d. Radial Distribution Function g(r)
- Histogram of pairwise distances
- Rotationally invariant
- Loses 3D shape; good only for l=0 (isotropic) component

### 5e. Voxelised Electron Density
- Place Gaussian blobs at atom positions, discretise to 3D grid
- Resolution tradeoff: 1 Å → large memory for big molecules
- **Best for:** 3D CNNs

### 5f. Spherical Shell Projections
- For each shell radius r_k, project atomic density onto S² → N_shells × (lMax+1)² coefficients
- Natural intermediate representation for B_lm prediction
- Can feed directly into a simpler MLP after this projection

---

## 6. Training Strategies

### 6a. Data Sources
- PDB structures (your demo_structures): 8 structures currently — need more
- PDB database: 200,000+ structures available; filter by resolution, R-factor
- QM9 / MD17 / GEOM datasets: small organic molecules with conformations
- Synthetic: random packing of atoms, random polymers

### 6b. Data Augmentation
- Random rotations (SO(3)): critical for non-equivariant models; free for equivariant
- Random translations: should be centred already
- Subsampling atoms: test robustness
- Adding Gaussian noise to coordinates: simulates experimental uncertainty / flexibility

### 6c. Curriculum Learning
- Start with small molecules (small N), increase size
- Start with low q values (smoother B_lm), add high q later
- Start with l=0 (isotropic), add higher l progressively

### 6d. Multi-task Learning
- Jointly predict B_lm(q) + I(q) + Rg + other physical observables
- Shared encoder, separate heads
- Often improves generalisation

### 6e. Transfer Learning
- Pretrain on large dataset of simple structures
- Fine-tune on your specific molecule types
- Can use pretrained MACE/NequIP weights as encoder

### 6f. Loss Weighting Schedule
- Start with just MSE on B_lm
- Anneal in physics-informed terms (I(q) consistency, symmetry) over training
- Prevents physics terms from dominating early when predictions are random

---

## 7. Full Pipeline Options (Summary)

### Option A: MACE + physics loss (recommended for term project)
- Model: MACE encoder → linear head for B_lm(q) conditioned on q
- Loss: MSE(B_lm) + λ₁·L_iq + λ₂·L_sym
- Data: PDB structures, compute labels with existing pipeline
- Expected effort: 4-6 weeks

### Option B: e3nn custom network
- Build bespoke equivariant network with explicit Bessel radial functions
- More physics-informed architecture; higher ceiling, higher effort
- Loss: same as A
- Expected effort: 8-10 weeks

### Option C: SOAP + Gaussian Process
- Featurise with DScribe SOAP; fit GP per (l, m) channel
- No GPU needed; works with small dataset (< 100 structures)
- Loss: standard GP marginal likelihood
- Expected effort: 1-2 weeks — good sanity check

### Option D: Spherical CNN on voxelised structure
- Voxelise molecule; apply S²-equivariant convolutions
- Output: spherical harmonic coefficients per shell → decode to B_lm(q)
- Naturally produces the right output type
- Expected effort: 6-8 weeks

### Option E: Neural operator (DeepONet / FNO)
- Encode structure with any GNN; decode B_lm(q) as continuous function of q
- Handles variable q grids; smooth q-dependence
- Expected effort: 5-7 weeks

### Option F: Linear ACE / SOAP ridge regression
- Simplest possible model; pure baseline
- Good to run first to establish a lower bound on achievable error
- Expected effort: 3-5 days

---

## 8. Evaluation Metrics

- **MSE / MAE on B_lm(q)**: direct; but complex-valued, so report real/imag separately
- **I(q) error**: MSE on I(q) reconstructed from predicted B_lm; physically interpretable
- **Rg error**: scalar, easy to communicate
- **Angular power spectrum error**: per-l power Σ_m |B_lm|² — tests if angular channels correct
- **χ² vs. experimental SAXS**: ultimate test if real experimental data available
- **Phase error**: angle(B_lm_pred) vs angle(B_lm_true) — harder to interpret but useful

---

## 9. Recommended Starting Point

1. **Week 1-2:** Run Option F (SOAP + ridge regression) as baseline. Proves the problem is learnable.
2. **Week 3-4:** Implement Option A (MACE). Finetune from pretrained weights.
3. **Week 5-6:** Add physics-informed loss terms (L_iq, L_sym). Compare against baseline.
4. **Week 7-8:** Scale data (pull more PDB structures). Evaluate generalisation.
5. **Week 9-10:** Write up; ablation studies on loss terms and architecture choices.
