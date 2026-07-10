import torch
import torch.nn.functional as F
import torch.distributed   as dist
import numpy               as np

from torch                  import cos, nn
from torch.utils.checkpoint import checkpoint
from ..batching             import Batch
from jaxtyping              import jaxtyped, Float, Bool
from beartype               import beartype
from .layer_head            import LayerHead
from ..utils.no_trilin_bilin import NoTrilinBilin
from typing                 import NamedTuple, Callable

class MessagePass(nn.Module):

    """
    RFF-based message passing: λ₂ rounds of kernel-weighted neighbourhood aggregation.

    Mathematical Formulation
    ------------------------
    Each atom m has an embedding e_m ∈ R^λ₁ and a per-q bandwidth σ_m ∈ R^Q.
    At each q-point, coordinates are scaled by σ to make the kernel range q-dependent:
        ```
        r̃_m(q) = r_m / σ_m(q)          (Å → dimensionless; large σ → long range)
        ```

    RFF approximation of the RBF kernel exp(-‖r̃_i − r̃_j‖² / 2):
        ``` 
        φ_m(q) = √(2/λ₅) · cos(Ω · r̃_m(q) + b)    ∈ R^λ₅
        ```
    where Ω ∈ R^{λ₅×3} is a fixed random frequency matrix and b ∈ R^λ₅ random phases.
    Then φ_i · φ_j ≈ k(r̃_i, r̃_j).

    Per round, two passes over M-chunks:

    Pass 1; accumulate global context for the N-chunk:

    ```
    features[q, d]      = Σ_m  φ_m(q, d)                        shape (Q, λ₅)
    chem_env[q, d, l]   = Σ_m  φ_m(q, d) · e_m(q, l)            shape (Q, λ₅, λ₁)
    ``` 
    Pass 2; per-atom update using the accumulated context:
        
        ```
        locality_m[q, l]    = Σ_d  φ_m(q, d) · chem_env[q, d, l]   shape (Q, λ₁)
                            ≈ Σ_{m'} k(r̃_m, r̃_{m'}) · e_{m'}(q, l)
    
        weights_m[q]        = |Σ_d  φ_m(q, d) · features[q, d]|     shape (Q,)
                            ≈ Σ_{m'} k(r̃_m, r̃_{m'})   (denominator for normalised avg)
                            
        agg_m               = RMSNorm(locality_m / weights_m)         shape (Q, λ₁)

        [p1, p2]            = Linear(agg_m)                           shape (Q, 2*λ₁)
        gate_m              = p1 · Mish(p2)                           shape (Q, λ₁)

        e_m  ←  e_m + gate_m                                       (residual update)
        σ_m  ←  softplus(σ_m + tanhshrink(bilinear(e_m, f_m)))    (sigma update)
        ```

    Memory strategy
    ---------------
    Checkpointing at the N-chunk level with use_reentrant=False: chem_env
    (Nc, Q, λ₅, λ₁) only exists inside the per-N-chunk closure's own call
    frame, so it's created and freed per N-chunk rather than kept alive for
    all N molecules simultaneously.

    Within each pass, each M-chunk gets its own inner checkpoint() call too
    (in `_pass_1`/`_pass_2`), so only one M-chunk's RFF intermediates are ever
    alive at once - never the full (Nc, M, Q, λ₅) tensor. The two heavy
    contractions use bmm on reshaped 3-D tensors rather than einsum, because
    einsum('nmqd,nmql->nqdl') would create an (Nc,mc,Q,λ₅,λ₁) intermediate
    before contracting over m; the tensor that caused OOM.

    Compile shapes
    --------------
    `forward` pads N and M up to multiples of `n_chunk`/`m_chunk` before
    chunking, so every N-chunk is exactly `n_chunk` molecules and every
    M-chunk is exactly `m_chunk` atoms, every call - regardless of which
    dataset bucket the batch came from. `_step1_fn`/`_step2_fn` (the compiled
    step functions) therefore only ever see one shape for the life of the
    module: no recompiles from a ragged tail chunk or from bucket-to-bucket
    shape variety.
    """

    _lambda_1: int                          # atom embedding dimension (λ₁)
    _lambda_2: int                          # number of message-passing rounds (λ₂)
    _lambda_5: int                          # number of Random Fourier Features (λ₅)
    _nchunk:   int                          # molecules per N-chunk
    _mchunk:   int                          # atoms per M-chunk

    # Shared small-shape tiers for Cn_eff/Cm_eff in forward(): when a bucket's real N or M
    # is a small fraction of n_chunk/m_chunk, round up to the smallest tier instead of the
    # full chunk size, bounding the number of distinct compiled shapes to len(_CHUNK_TIERS)
    # per axis instead of one shape per distinct real size.
    _CHUNK_TIERS: tuple[int, ...] = (8, 16, 32, 64, 128)
    _proj_agg: nn.Linear                    # projects aggregated context λ₁ → 2λ₁ for MishGLU gating
    _omegafrq: Float[torch.Tensor, "λ₅ 3"]  # fixed RFF frequency matrix: λ₅ random 3-D directions
    _biasterm: nn.Parameter                 # RFF random phase offsets b ∈ R^λ₅
    _sigbilin: NoTrilinBilin                # updates σ from (updated embedding, form factor)
    _rms_norm: nn.RMSNorm                   # normalises aggregated context before gating
    _q_points: int                          # number of q-points (Q)
    _rffscale: float                        # rff_scale = (2/lambda_5) ** 0.5 to prevent triton bug
    _step1_fn: Callable
    _step2_fn: Callable
    
    class _AllReduce(torch.autograd.Function):

        """
        Differentiable all-reduce (SUM) across the distributed process group.

        Forward: each rank contributes a partial sum; all_reduce gives every rank
        the global sum. Backward: the incoming gradient is itself a partial sum
        (each rank only saw its own atoms), so the same all_reduce(SUM) is applied
        to give every rank the global gradient, by the chain rule since
        ∂(sum)/∂(each_input) = 1.
        """

        @staticmethod
        def forward(_ctx, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            """Sum-reduce `x` across the distributed process group.

            Parameters
            ----------
            _ctx : torch.autograd.function.FunctionCtx
                Autograd context (unused).
            x : torch.Tensor
                Local partial tensor (e.g. `features` or `chem_env`) to be
                summed across ranks.

            Returns
            -------
            torch.Tensor
                `x` all-reduced (summed) across every rank.
            """
            x = x.clone()
            dist.all_reduce(x, op=dist.ReduceOp.SUM)
            return x

        @staticmethod
        def backward(_ctx, grad: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            """Sum-reduce the incoming gradient across the process group.

            Parameters
            ----------
            _ctx : torch.autograd.function.FunctionCtx
                Autograd context (unused).
            grad : torch.Tensor
                Gradient of the loss with respect to this function's
                output, as seen by this rank (itself a partial sum since
                each rank only saw its own atoms).

            Returns
            -------
            torch.Tensor
                `grad` all-reduced (summed) across every rank, giving each
                rank the global gradient.
            """
            dist.all_reduce(grad, op=dist.ReduceOp.SUM)
            return grad

    class _PassContainer(NamedTuple):

        """Working state for one N-chunk during a message-passing round.

        Attributes
        ----------
        M : int
            Total (padded) number of atoms across the whole batch.
        Nchnk : int
            Number of molecules in this N-chunk.
        Mchnk : int
            Number of atoms per M-chunk used when iterating over M.
        emb_n : torch.Tensor
            Atom embeddings for this N-chunk, shape (Nchnk, M, Q, λ₁).
        msk_n : torch.Tensor
            Padding mask for this N-chunk, shape (Nchnk, M); True = real
            atom.
        ffs_n : torch.Tensor
            Form factor magnitudes for this N-chunk, shape
            (Nchnk, M, Q, 1).
        sig_n : torch.Tensor
            Per-atom per-q RBF bandwidths for this N-chunk, shape
            (Nchnk, M, Q, 1).
        crd_n : torch.Tensor
            Atom Cartesian coordinates (Å) for this N-chunk, shape
            (Nchnk, M, 3).
        features : torch.Tensor
            Accumulated sum of RFF features over atoms, shape
            (Nchnk, Q, λ₅).
        chem_env : torch.Tensor
            Accumulated kernel-weighted embedding sum, shape
            (Nchnk, Q, λ₅, λ₁).
        """

        M:        int
        Nchnk:    int
        Mchnk:    int
        emb_n:    Float[torch.Tensor, "Nchnk Mc Q λ₁"]  # atom embeddings for this N-chunk
        msk_n:    Bool[torch.Tensor,  "Nchnk Mc"]         # padding mask (True = real atom)
        ffs_n:    Float[torch.Tensor, "Nchnk Mc Q 1"]    # form factor magnitudes
        sig_n:    Float[torch.Tensor, "Nchnk Mc Q 1"]    # per-atom per-q RBF bandwidths
        crd_n:    Float[torch.Tensor, "Nchnk Mc 3"]      # atom Cartesian coordinates (Å)
        features: Float[torch.Tensor, "Nchnk Q λ₅"]      # Σ_m φ_m; kernel weight normaliser
        chem_env: Float[torch.Tensor, "Nchnk Q λ₅ λ₁"]   # Σ_m φ_m ⊗ e_m; chemical environment

    def __init__(
        self,
        lambda_1: int,
        lambda_2: int,
        lambda_5: int,
        seed:     int,
        q_points: int,
        n_chunk:  int,
        m_chunk:  int,
        compile:  bool = False
        ):

        """Validate hyperparameters and build MessagePass's learned layers.

        Parameters
        ----------
        lambda_1 : int
            Atom embedding dimension.
        lambda_2 : int
            Number of message-passing rounds; must be > 0.
        lambda_5 : int
            Number of Random Fourier Features.
        seed : int
            RNG seed for the fixed RFF frequency matrix and phase offsets.
        q_points : int
            Number of q-points (Q).
        n_chunk : int
            Molecules per N-chunk; must be > 0.
        m_chunk : int
            Atoms per M-chunk; must be > 0.
        compile : bool, optional
            If True, torch.compile `_step1`/`_step2`. Default is False.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If `lambda_2`, `n_chunk`, or `m_chunk` is not greater than 0.
        """

        super().__init__()

        if lambda_2 <= 0:
            raise ValueError(f"invalid lambda_2 (must be > 0): {lambda_2}")
        if n_chunk <= 0:
            raise ValueError(f"invalid n_chunk (must be > 0): {n_chunk}")
        if m_chunk <= 0:
            raise ValueError(f"invalid m_chunk (must be > 0): {m_chunk}")

        rng = np.random.default_rng(seed=seed)

        self._lambda_1 = lambda_1
        self._lambda_2 = lambda_2
        self._lambda_5 = lambda_5
        self._nchunk   = n_chunk
        self._mchunk   = m_chunk
        self._proj_agg = nn.Linear(lambda_1, 2 * lambda_1)
        self.register_buffer('_omegafrq', torch.from_numpy(rng.standard_normal((lambda_5, 3))).float())
        self._biasterm = nn.Parameter(torch.from_numpy(rng.uniform(0, 2*np.pi, size=(self._lambda_5))).float())
        # NoTrilinBilin, not nn.Bilinear: out_features=1 here, the exact degenerate
        # case NoTrilinBilin targets (see its docstring) - also avoids nn.Bilinear's
        # F.bilinear op forcing a torch.compile graph break (aten::_trilinear isn't
        # Inductor-supported), so this now fuses into the surrounding compiled graph.
        self._sigbilin = NoTrilinBilin(lambda_1, q_points, 1)
        self._rms_norm = nn.RMSNorm(lambda_1)
        self._q_points = q_points
        
        # precomputed here so torch.compile never sees Python
        # arithmetic on lambda_5: under dynamic=True it treats lambda_5 as a SymInt (it's
        # also used in reshapes below), and `(2/lambda_5) ** 0.5` would then trace as a
        # SymFloat that specializes to float64, which Triton's pow() can't mix with the
        # float32 RFF tensor (Triton has no (float32, float64) overload).
        self._rffscale = (2 / lambda_5) ** 0.5
        
        # dynamic=False: forward() now pads every N-/M-chunk fed to these
        # to exactly n_chunk/m_chunk atoms/molecules, so these functions only ever see one
        # shape for the module's whole lifetime - no need to trace shapes as dynamic.
        # Exception: forward() rounds down to a small tier (_CHUNK_TIERS) instead of the
        # full chunk size when a bucket's real N or M is a small fraction of n_chunk/m_chunk
        # (avoids an 8x+ memory blowup on those buckets - see forward()'s Cn_eff/Cm_eff).
        # Each axis independently sees len(_CHUNK_TIERS) tiered shapes PLUS the one
        # full-chunk shape (used whenever real N/M exceeds the chunk) - so the real ceiling
        # on distinct (Cn_eff, Cm_eff) combos is (len(_CHUNK_TIERS)+1)**2, not just
        # len(_CHUNK_TIERS). A limit of 4*len(_CHUNK_TIERS)=20 undercounted this (real
        # datasets routinely exercise >20 distinct combos across an epoch's bucket
        # diversity), causing mid-run cache evictions -> full Dynamo retraces costing
        # tens of seconds to minutes each, observed directly in profiler runs as isolated
        # batches taking 40-250s versus a few seconds for the same bucket group otherwise.
        # Set comfortably above the real ceiling instead of guessing.
        if compile:
            torch._dynamo.config.cache_size_limit = max(
                torch._dynamo.config.cache_size_limit, 2 * (len(self._CHUNK_TIERS) + 1) ** 2
            )
        self._step1_fn = torch.compile(self._step1, fullgraph=True) if compile else self._step1
        self._step2_fn = torch.compile(self._step2, fullgraph=True) if compile else self._step2

    def _step1(
        self,
        embslice: torch.Tensor,
        crdslice: torch.Tensor,
        sigslice: torch.Tensor,
        mskslice: torch.Tensor,
        epsilon_: float,
    ):

        """Compute per-M-chunk RFF features and the kernel-weighted embedding sum.

        Parameters
        ----------
        embslice : torch.Tensor
            Atom embeddings for this M-chunk, shape (Nc, mc, Q, λ₁).
        crdslice : torch.Tensor
            Atom coordinates for this M-chunk, shape (Nc, mc, 3).
        sigslice : torch.Tensor
            Per-atom per-q bandwidths for this M-chunk, shape
            (Nc, mc, Q, 1).
        mskslice : torch.Tensor
            Padding mask for this M-chunk, shape (Nc, mc).
        epsilon_ : float
            Numerical floor for clamping sigma before division.

        Returns
        -------
        torch.Tensor
            `step_features`, partial sum of RFF features over this
            M-chunk's atoms, shape (Nc, Q, λ₅).
        torch.Tensor
            `step_chem_env`, partial kernel-weighted embedding sum, shape
            (Nc, Q, λ₅, λ₁).
        """

        # not a staticmethod: biasterm/omegafrq are read off self rather than passed as
        # positional args. Passing an nn.Parameter as a positional arg into a compiled,
        # checkpointed function is unsound - checkpoint's recompute-on-backward sometimes
        # hands back a detached plain Tensor for the same saved argument, so its *type*
        # (Parameter vs Tensor) differs between the original forward call and the backward
        # recompute; torch.compile treats that as a reason to recompile, every single time,
        # eventually hitting the recompile limit. Reading self._biasterm/self._omegafrq
        # inside the method body avoids threading them through checkpoint's argument list.
        lambda_1, lambda_5 = self._lambda_1, self._lambda_5

        # r̃_m = r_m / σ_m: scale coords by bandwidth so kernel range is q-dependent.
        # clamp(min=eps) caps the max RFF frequency and avoids 1/0.
        # Forced to fp32 (autocast disabled): under AMP the fp16 path overflows here.
        # sigslice.clamp(min=eps) can be ~1e-3, so scaled_coords reaches ~1e5 and the
        # `@ omega.T` projection blows past fp16's 65504 -> cos(inf) = NaN. This is a tiny
        # inner-dim-3 -> λ₅ contraction, so fp32 costs almost nothing; the heavy bmm below
        # still runs in fp16 under the outer autocast (no-op when AMP is off).
        with torch.autocast(device_type=crdslice.device.type, enabled=False):
            scaled_coords = crdslice.float().unsqueeze(-2) / sigslice.float().clamp(min=epsilon_) # (Nc, mc, Q, 3)

            # φ_m = √(2/λ₅) · cos(Ω · r̃_m + b): RFF feature vector per atom per q-point
            proj = scaled_coords @ self._omegafrq.T + self._biasterm  # (Nc, mc, Q, λ₅)
            zrff = self._rffscale * cos(proj)                         # (Nc, mc, Q, λ₅)
        zrff = zrff * mskslice.unsqueeze(-1).unsqueeze(-1)           # zero padding atoms

        # Σ_m φ_m: partial sum of RFF features over atoms in this M-chunk
        step_features = zrff.sum(dim=1) # (Nc, Q, λ₅)

        # Σ_m φ_m ⊗ e_m: partial outer-product sum (kernel-weighted embedding accumulator).
        # bmm on (Nc*Q, λ₅, mc) @ (Nc*Q, mc, λ₁) avoids the (Nc, mc, Q, λ₅, λ₁) intermediate
        # that einsum('nmqd,nmql->nqdl') would create before contracting over m.
        Nc, mc, Q = zrff.shape[0], zrff.shape[1], zrff.shape[2]
        zb = zrff.permute(0, 2, 3, 1).reshape(Nc * Q, lambda_5, mc)      # (Nc*Q, λ₅, mc)
        eb = embslice.permute(0, 2, 1, 3).reshape(Nc * Q, mc, lambda_1) # (Nc*Q, mc, λ₁)
        step_chem_env = torch.bmm(zb, eb).reshape(Nc, Q, lambda_5, lambda_1)

        return step_features, step_chem_env

    def _pass_1(self, cont: _PassContainer, eps: float) -> _PassContainer:

        """
        Accumulate global context (features, chem_env) for this N-chunk across all M-chunks.

        After this pass:
            features  = Σ_m φ_m        (kernel weight normaliser)
            chem_env  = Σ_m φ_m ⊗ e_m (kernel-weighted embedding sum)

        Parameters
        ----------
        cont : MessagePass._PassContainer
            Working state for this N-chunk; `features` and `chem_env` are
            accumulated (functionally) from their current values.
        eps : float
            Numerical floor for sigma clamping inside `_step1`.

        Returns
        -------
        MessagePass._PassContainer
            `cont` with `features` and `chem_env` updated to their fully
            accumulated values across all M-chunks.
        """

        features: Float[torch.Tensor, "Nchnk Q λ₅"]    = cont.features
        chem_env: Float[torch.Tensor, "Nchnk Q λ₅ λ₁"] = cont.chem_env

        for m0 in range(0, cont.M, cont.Mchnk):
            m1 = min(m0 + cont.Mchnk, cont.M)
            
            # .contiguous(): these are views into cont.*_n, so their stride/storage_offset
            # depend on m0 (which chunk) and the batch's own padding. torch.compile guards
            # on stride()/storage_offset() as well as shape, so distinct views recompile
            # even at the same chunk-tier shape. Materializing a canonical contiguous
            # layout collapses that to one guard set per tier.
            emb_slice = cont.emb_n[:, m0:m1].contiguous()
            crd_slice = cont.crd_n[:, m0:m1].contiguous()
            sig_slice = cont.sig_n[:, m0:m1].contiguous()
            msk_slice = cont.msk_n[:, m0:m1].contiguous()

            args = (emb_slice, crd_slice, sig_slice, msk_slice, eps)

            step_feat, step_chem = checkpoint(self._step1_fn, *args, use_reentrant=False)  # type: ignore[misc]
            features = features + step_feat
            chem_env = chem_env + step_chem

        return cont._replace(features=features, chem_env=chem_env)

    def _step2(
        self,
        embslice: torch.Tensor,
        crdslice: torch.Tensor,
        sigslice: torch.Tensor,
        mskslice: torch.Tensor,
        ffsslice: torch.Tensor,
        features: torch.Tensor,
        chem_env: torch.Tensor,
        epsilon_: float,
        ):

        """Compute the per-atom neighbourhood aggregate and updated embedding/sigma.

        Parameters
        ----------
        embslice : torch.Tensor
            Atom embeddings for this M-chunk, shape (Nc, mc, Q, λ₁).
        crdslice : torch.Tensor
            Atom coordinates for this M-chunk, shape (Nc, mc, 3).
        sigslice : torch.Tensor
            Per-atom per-q bandwidths for this M-chunk, shape
            (Nc, mc, Q, 1).
        mskslice : torch.Tensor
            Padding mask for this M-chunk, shape (Nc, mc).
        ffsslice : torch.Tensor
            Form factor magnitudes for this M-chunk, shape (Nc, mc, Q, 1).
        features : torch.Tensor
            Fully accumulated RFF feature sum from `_pass_1`, shape
            (Nc, Q, λ₅).
        chem_env : torch.Tensor
            Fully accumulated kernel-weighted embedding sum from
            `_pass_1`, shape (Nc, Q, λ₅, λ₁).
        epsilon_ : float
            Numerical floor for clamping sigma and the aggregate
            denominator.

        Returns
        -------
        torch.Tensor
            `new_emb`, updated atom embeddings, shape (Nc, mc, Q, λ₁).
        torch.Tensor
            `new_sig`, updated per-atom per-q bandwidths, shape
            (Nc, mc, Q, 1).
        """

        q_points, lambda_1, lambda_5 = self._q_points, self._lambda_1, self._lambda_5

        # recompute φ_m for this M-chunk (same as _pass_1, but now chem_env is complete).
        # fp32 (autocast disabled) for the same fp16-overflow reason as _step1 - see there.
        with torch.autocast(device_type=crdslice.device.type, enabled=False):
            scaled_coords = crdslice.float().unsqueeze(-2) / sigslice.float().clamp(min=epsilon_) # (Nc, mc, Q, 3)
            proj = scaled_coords @ self._omegafrq.T + self._biasterm # (Nc, mc, Q, λ₅)
            zrff = self._rffscale * cos(proj) # (Nc, mc, Q, λ₅)
        mask = mskslice.unsqueeze(-1).unsqueeze(-1) # (Nc, mc, 1, 1)
        zrff = zrff * mask

        # locality_m = φ_m · chem_env ≈ Σ_{m'} k(r̃_m, r̃_{m'}) · e_{m'}
        # neighbourhood embedding for each atom: weighted sum of all other atoms' embeddings.
        # Same bmm trick as _pass_1: avoids (Nc, mc, Q, λ₅, λ₁) intermediate.
        Nc, mc, Q = zrff.shape[0], zrff.shape[1], zrff.shape[2]
        zb        = zrff.permute(0, 2, 1, 3).reshape(Nc * Q, mc, lambda_5)  # (Nc*Q, mc, λ₅); query features
        cb        = chem_env.reshape(Nc * Q, lambda_5, lambda_1) # (Nc*Q, λ₅, λ₁); accumulated context

        # locality: (Nc, mc, Q, λ₁); approximate kernel-weighted neighbour embedding sum
        locality = torch.bmm(zb, cb).reshape(Nc, Q, mc, lambda_1).permute(0, 2, 1, 3)

        # weights_m = |φ_m · features| ≈ Σ_{m'} k(r̃_m, r̃_{m'})
        # denominator for the normalised average: total kernel weight seen by atom m.
        # abs() because cosine RFF features can be negative, making the dot product negative.
        # clamp(min=eps): avoids 0/0 in both forward and backward (nan_to_num only fixes
        # the forward value but still produces grad/0 = NaN during backward).
        weights = torch.einsum('nmqd, nqd -> nmq', zrff, features).abs() # (Nc, mc, Q)

        # normalised aggregate: kernel-weighted average of neighbour embeddings
        # fp32 (autocast disabled) so RMSNorm's fp32 weight matches its input and
        # dispatches to the fused kernel - under fp16 autocast the input arrives
        # fp16 against the fp32 weight, which falls back to a slow unfused path.
        pre_norm = locality / weights.unsqueeze(-1).clamp(min=epsilon_) # (Nc, mc, Q, λ₁)
        with torch.autocast(device_type=pre_norm.device.type, enabled=False):
            agg = self._rms_norm(pre_norm.float())
        agg = agg.to(locality.dtype)

        # MishGLU gate: one linear projects to 2λ₁, split into value p1 and gate p2.
        # gate = p1 · Mish(p2) selectively passes neighbourhood signal into the residual stream.
        p1, p2 = self._proj_agg(agg).chunk(2, dim=-1) # each (Nc, mc, Q, λ₁)
        gate   = p1 * F.mish(p2) * mask               # (Nc, mc, Q, λ₁)

        # residual update
        new_emb = embslice + gate

        # sigma update: tanhshrink(x) = x - tanh(x) is near-zero 
        # for small x ("sticky" sigma barely moves when the bilinear output is small) 
        # and grows linearly for large x. softplus keeps σ strictly positive.
        f_in    = ffsslice.transpose(-1, -2).expand(-1, -1, q_points, -1)
        new_sig = F.softplus(sigslice + F.tanhshrink(self._sigbilin(new_emb, f_in)))

        return new_emb, new_sig

    def _pass_2(self, cont: _PassContainer, eps: float) -> _PassContainer:

        """
        Compute per-atom neighbourhood aggregate and update embeddings and sigmas.

        Uses the fully-accumulated chem_env from _pass_1 so every atom attends to
        all others despite M-chunking (all-pairs coverage is preserved).

        Parameters
        ----------
        cont : MessagePass._PassContainer
            Working state for this N-chunk, with `features` and
            `chem_env` already fully accumulated by `_pass_1`.
        eps : float
            Numerical floor for sigma clamping and the aggregate
            denominator inside `_step2`.

        Returns
        -------
        MessagePass._PassContainer
            `cont` with `emb_n` and `sig_n` replaced by their updated
            values.
        """

        new_emb_m = []
        new_sig_m = []

        for m0 in range(0, cont.M, cont.Mchnk):
            m1 = min(m0 + cont.Mchnk, cont.M)
            
            # .contiguous(): see the matching comment in _pass_1 - these views' stride/
            # storage_offset vary by m0 and trigger extra torch.compile recompiles beyond
            # the intended chunk-tier shape guard.
            emb_slice = cont.emb_n[:, m0:m1].contiguous()
            ffs_slice = cont.ffs_n[:, m0:m1].contiguous()
            sig_slice = cont.sig_n[:, m0:m1].contiguous()
            crd_slice = cont.crd_n[:, m0:m1].contiguous()
            msk_slice = cont.msk_n[:, m0:m1].contiguous()
            cont_feat = cont.features
            cont_chnv = cont.chem_env

            args = (
                emb_slice, 
                crd_slice, 
                sig_slice, 
                msk_slice, 
                ffs_slice, 
                cont_feat, 
                cont_chnv, 
                eps
            )
            emb_c, sig_c = checkpoint(self._step2_fn, *args, use_reentrant=False)  # type: ignore[misc]
            new_emb_m.append(emb_c)
            new_sig_m.append(sig_c)

        new_emb = torch.cat(new_emb_m, dim=1)  # (Nchnk, M, Q, λ₁)
        new_sig = torch.cat(new_sig_m, dim=1)  # (Nchnk, M, Q, 1)
        return cont._replace(emb_n=new_emb, sig_n=new_sig)

    @staticmethod
    def _pad_to_multiple(t: torch.Tensor, chunk: int, dim: int) -> torch.Tensor:

        """
        Zero-pad `t` along `dim` up to the next multiple of `chunk`.

        Lets every N-chunk/M-chunk fed to the compiled step functions be exactly
        `chunk` elements wide, every call, so torch.compile only ever sees one
        shape per compiled function - no ragged tail chunk, no recompiles from
        shape variety. Padded rows are all-zero; the padding_mask entries built
        from them are False (real atom/molecule = True), so _step1/_step2 zero
        their RFF contribution and the caller slices the padding back off before
        returning.

        Parameters
        ----------
        t : torch.Tensor
            Tensor to pad.
        chunk : int
            Chunk size; `t` is padded so `dim`'s size becomes a multiple
            of this.
        dim : int
            Dimension along which to pad.

        Returns
        -------
        torch.Tensor
            `t` zero-padded along `dim` up to the next multiple of
            `chunk` (returned unchanged if already a multiple).
        """

        size = t.shape[dim]
        pad_to = -(-size // chunk) * chunk  # ceil division
        if pad_to == size:
            return t
        pad_shape = list(t.shape)
        pad_shape[dim] = pad_to - size
        return torch.cat([t, t.new_zeros(pad_shape)], dim=dim)

    @classmethod
    def _tier_chunk(cls, real: int, chunk: int) -> int:

        """
        Effective chunk size for a bucket whose real N or M may be far smaller
        than the configured chunk.

        Padding a bucket's real size up to the full `chunk` wastes
        `chunk/real` memory and compute when `real` is a small fraction of
        `chunk` (see forward()'s Cn_eff/Cm_eff - this is what caused two real
        OOMs, one on each axis). Rounding up to the exact next multiple of a
        fine granularity fixes the waste but produces one distinct compiled
        shape per distinct `real` value, which can exceed torch.compile's
        recompile cache limit under `fullgraph=True` when many different
        bucket sizes are seen over an epoch. Rounding up to the smallest
        shared tier in `_CHUNK_TIERS` instead bounds the number of distinct
        shapes to `len(_CHUNK_TIERS)`, regardless of how many distinct real
        sizes occur.

        Parameters
        ----------
        real : int
            The real (unpadded) size along this axis.
        chunk : int
            The configured chunk size (`n_chunk` or `m_chunk`). Only called
            when `real <= chunk`; the tier is capped at `chunk`.

        Returns
        -------
        int
            The smallest tier in `_CHUNK_TIERS` that is `>= real`, capped at
            `chunk`.
        """

        for tier in cls._CHUNK_TIERS:
            if tier >= real:
                return min(tier, chunk)
        return chunk

    def _all_reduce(self, x: torch.Tensor, use_all_reduce: bool) -> torch.Tensor:

        """Apply _AllReduce if a process group is active AND the caller wants one;
        otherwise pass through. `use_all_reduce=False` is for ScatterNet's DP path:
        each rank there already holds a complete, disjoint set of molecules, so
        there is nothing to reconcile across ranks (unlike TP, which shards atoms
        of the SAME molecules and must all-reduce to see the full neighbourhood).
        Calling all_reduce here unconditionally in DP mode would be a correctness
        bug, not just a slowdown: DP's two ranks can have different local molecule
        counts (an off-by-one from `ceil(N/ws)`), so they can end up issuing a
        different NUMBER of N-chunk rounds, and thus a different number of
        all_reduce calls overall, one rank hangs forever waiting for a collective
        the other rank never issues (NCCL watchdog timeout).

        Parameters
        ----------
        x : torch.Tensor
            Tensor to all-reduce (`features` or `chem_env`).
        use_all_reduce : bool
            Whether to actually apply the all-reduce.

        Returns
        -------
        torch.Tensor
            `x` all-reduced across ranks, or unchanged if no process
            group is active or `use_all_reduce` is False.
        """

        if not use_all_reduce or not dist.is_available() or not dist.is_initialized():
            return x
        return self._AllReduce.apply(x)  # type: ignore[return-value]

    def _count_all_reduce(self, x: torch.Tensor, use_all_reduce: bool) -> torch.Tensor:

        """Plain (non-differentiable) SUM all-reduce for the per-molecule atom count.

        Mirrors `_all_reduce`'s gating but without the autograd Function: the count is a
        constant divisor derived from the padding mask (no gradient flows to it), and under
        TP each rank holds only its atom shard, so the per-molecule counts must be summed
        across ranks to match the globally-summed features/chem_env they normalise.
        `use_all_reduce=False` (the DP path) leaves the local count untouched, exactly as
        for features/chem_env - see `_all_reduce`'s docstring for why that gating is a
        correctness requirement, not an optimization.

        Parameters
        ----------
        x : torch.Tensor
            Per-molecule real-atom counts for this N-chunk, shape (Nchnk,).
        use_all_reduce : bool
            Whether to sum the count across ranks (True for TP, False for DP).

        Returns
        -------
        torch.Tensor
            `x` summed across ranks (TP), or unchanged (DP / no process group).
        """

        if not use_all_reduce or not dist.is_available() or not dist.is_initialized():
            return x
        x = x.clone()
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        return x

    @jaxtyped(typechecker=beartype)
    def forward(self, batch: Batch, embed_head: LayerHead, eps: float, use_all_reduce: bool = True) -> LayerHead:

        """Run λ₂ rounds of RFF message passing.

        Parameters
        ----------
        batch : Batch
            Molecule geometry; uses `coord` (N, M, 3) Cartesian positions
            (Å).
        embed_head : LayerHead
            Output of Embed; embeds (N, M, 1, λ₁), f_mags and sigmas
            (N, M, Q, 1).
        eps : float
            Numerical floor for sigma clamping and the aggregate
            denominator.
        use_all_reduce : bool, optional
            Whether to all-reduce features/chem_env across ranks between
            the two passes. True (default) for TP and single-GPU, where
            the shard being processed is a slice of the SAME molecules on
            every rank. Must be False for ScatterNet's DP path, where each
            rank processes a disjoint set of molecules and there is
            nothing to reconcile - see `_all_reduce`'s docstring for why
            this isn't optional (it's a correctness/deadlock issue, not
            just a no-op).

        Returns
        -------
        LayerHead
            `embed_head` with `embeds` updated to (N, M, Q, λ₁) and
            `sigmas` updated; `f_mags` unchanged.
        """

        embeds_raw   = embed_head.embeds  # (N, M, 1, λ₁); Q not yet expanded
        sigmas       = embed_head.sigmas
        coord        = batch.coord
        padding_mask = batch.padding_mask()
        f_mags       = embed_head.f_mags

        N_real, M_real, _ = coord.shape
        Q       = sigmas.shape[2]
        Cn      = self._nchunk
        Cm      = self._mchunk

        # For most buckets N_real >> Cn, so padding N up to a multiple of Cn wastes at
        # most Cn-1 molecules of compute - negligible. But "heavy-M" buckets (a handful
        # of huge molecules, e.g. 7 mols x 6046 atoms) have N_real << Cn: padding all the
        # way up to Cn there inflates every per-chunk tensor (chem_env, RFF intermediates)
        # by Cn/N_real - an 8x+ memory/compute blowup on exactly the buckets already
        # tightest on memory (this caused a real OOM). Round up to the smallest shared
        # tier instead (see `_tier_chunk`), so distinct small-N buckets collapse onto a
        # handful of shared compiled shapes rather than either the full Cn (wasteful) or
        # one shape per distinct N_real (recompile-limit blowout - hit in practice).
        Cn_eff = self._tier_chunk(N_real, Cn) if N_real <= Cn else Cn

        # Same reasoning on the M (atom) axis: DP-routed buckets with huge N but tiny M
        # (e.g. 5273 mols x 11 atoms) keep M unsharded, so padding M all the way up to
        # atm_chunk (e.g. 80) when M_real is a fraction of that wastes ~Cm/M_real memory
        # on the (N_padded, M_padded, Q, λ₁) tensors materialised across all N-chunks
        # before the final `torch.cat` - this is what actually caused the OOM (a single
        # 5+ GiB allocation on that cat), not the N-axis case above.
        Cm_eff = self._tier_chunk(M_real, Cm) if M_real <= Cm else Cm

        # Pad N (molecules) and M (atoms) up to multiples of Cn_eff/Cm_eff so every
        # N-chunk/M-chunk fed to the compiled step functions is exactly (Cn_eff, Cm_eff)
        # every call - one shape per distinct bucket shape, no recompiles from a ragged
        # tail chunk. Padding is done on embeds_raw (Q still singleton) so the Q-expand
        # right after stays a zero-copy stride-0 view instead of materialising a
        # (N, M, Q, λ₁) tensor early.
        for dim, chunk in ((0, Cn_eff), (1, Cm_eff)):
            embeds_raw   = self._pad_to_multiple(embeds_raw,   chunk, dim)
            sigmas       = self._pad_to_multiple(sigmas,       chunk, dim)
            coord        = self._pad_to_multiple(coord,        chunk, dim)
            f_mags       = self._pad_to_multiple(f_mags,       chunk, dim)
            padding_mask = self._pad_to_multiple(padding_mask, chunk, dim)

        # expand Q dim from 1 → Q upfront; zero-copy stride-0 view until a contiguous op fires
        embeds = embeds_raw.expand(-1, -1, self._q_points, -1)
        N, M   = coord.shape[0], coord.shape[1]  # padded sizes; N a multiple of Cn_eff, M a multiple of Cm_eff

        for _ in range(self._lambda_2):
            new_embeds_n = []
            new_sigmas_n = []

            for n0 in range(0, N, Cn_eff):
                n1 = n0 + Cn_eff  # N is padded to a multiple of Cn_eff, so every N-chunk is exactly Cn_eff

                # Nc derived from emb_s.shape[0] inside the closure rather than closed over as a
                # variable; Python closures capture by reference so a loop variable would give the
                # last iteration's value when the checkpoint replays during backward.
                def _n_chunk_round(emb_s, msk_s, ffs_s, sig_s, crd_s):
                    Nc_s = emb_s.shape[0]
                    cont = self._PassContainer(
                        M        = M,
                        Nchnk    = Nc_s,
                        Mchnk    = Cm_eff,
                        emb_n    = emb_s,
                        msk_n    = msk_s,
                        ffs_n    = ffs_s,
                        sig_n    = sig_s,
                        crd_n    = crd_s,
                        features = crd_s.new_zeros(Nc_s, Q, self._lambda_5),
                        chem_env = crd_s.new_zeros(Nc_s, Q, self._lambda_5, self._lambda_1),
                    )
                    cont = self._pass_1(cont, eps)
                    cont = cont._replace(
                        features = self._all_reduce(cont.features, use_all_reduce),
                        chem_env = self._all_reduce(cont.chem_env, use_all_reduce),
                    )
                    # Mean-normalise the atom-sums. features/chem_env are Σ over all atoms
                    # (O(M): ~1e3-1e4 for a large molecule), which overflows fp16's 65504 in
                    # the _step2 contractions and their backward - the AMP NaN. Dividing both
                    # by the per-molecule real-atom count turns them into means (O(1)). _step2
                    # forms locality/weights, a RATIO of these two, so scaling both by the same
                    # per-molecule constant leaves the aggregate algebraically identical (to
                    # fp32 rounding) while keeping every fp16 intermediate in range. `count` is
                    # the GLOBAL count per molecule: msk_n is only this rank's atom shard under
                    # TP, so it is summed across ranks exactly like features/chem_env (a no-op
                    # under DP, where msk_n already spans whole molecules). clamp(min=1) guards
                    # padded (all-masked) molecules, whose sums are 0 anyway.
                    # NOTE: _step2's `weights.clamp(min=eps)` floor now acts on an O(1) mean
                    # rather than an O(M) sum, so it engages at a different absolute threshold.
                    # This does NOT change the output: _step2 feeds locality/weights straight
                    # into RMSNorm, which is invariant to any positive per-atom scalar on its
                    # input, so the magnitude of `weights` (and hence where the clamp bites)
                    # washes out - it only ever served as a 0/0 guard. Verified: post-RMSNorm
                    # `agg` matches the un-normalised sum form to ~2e-6 (fp32 rounding).
                    count = cont.msk_n.sum(dim=1, dtype=cont.features.dtype)          # (Nc,)
                    count = self._count_all_reduce(count, use_all_reduce).clamp(min=1.0)
                    inv   = (1.0 / count).view(-1, 1, 1)                              # (Nc, 1, 1)
                    cont  = cont._replace(
                        features = cont.features * inv,
                        chem_env = cont.chem_env * inv.unsqueeze(-1),
                    )
                    cont = self._pass_2(cont, eps)
                    return cont.emb_n, cont.sig_n

                # use_reentrant=False: chem_env lives only inside this closure's call frame,
                # not behind a persistent reference held elsewhere, so it's still freed once
                # this N-chunk's forward/recompute finishes rather than kept alive for every
                # N-chunk simultaneously - unlike the older nested-_step-closure version this
                # replaced, where a persistent closure held chem_env alive across N-chunks.
                #
                # embeds[n0:n1].contiguous(): embeds is a stride-0 broadcast view over Q
                # (see the expand() above). Left as a view, checkpoint's recompute-on-backward
                # sometimes hands the compiled step function a materialised (real-stride)
                # copy instead of the original broadcast view, so the same argument's memory
                # layout differs between the original forward call and the recompute -
                # torch.compile treats that as a reason to recompile, same failure mode as
                # passing biasterm as a positional arg (see _step1's comment). Materialising
                # per N-chunk here (Cn*M_pad*Q*λ1) is far smaller than doing it for the whole
                # batch up front, and keeps the layout identical on every call.
                args = (
                    embeds[n0:n1].contiguous(), 
                    padding_mask[n0:n1], 
                    f_mags[n0:n1], 
                    sigmas[n0:n1], 
                    coord[n0:n1]
                )
                new_emb, new_sig = checkpoint(_n_chunk_round, *args, use_reentrant=False)  # type: ignore[misc]
                new_embeds_n.append(new_emb)
                new_sigmas_n.append(new_sig)

            embeds = torch.cat(new_embeds_n, dim=0)  # (N, M, Q, λ₁)
            sigmas = torch.cat(new_sigmas_n, dim=0)  # (N, M, Q, 1)

        # strip the N/M padding added above before returning
        embeds = embeds[:N_real, :M_real]
        sigmas = sigmas[:N_real, :M_real]

        return embed_head._replace(embeds=embeds, sigmas=sigmas)
