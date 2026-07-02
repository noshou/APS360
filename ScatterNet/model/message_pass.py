import torch
import torch.nn.functional as F
import torch.distributed   as dist
import numpy               as np

from torch                   import cos, nn
from torch.utils.checkpoint  import checkpoint
from ..batching              import Batch
from jaxtyping               import jaxtyped, Float, Bool
from beartype                import beartype
from .layer_head             import LayerHead
from typing                  import NamedTuple, Callable

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
    Checkpointing at the N-chunk level with use_reentrant=True: each N-chunk's full
    round (_pass_1 → all_reduce → _pass_2) runs under no_grad in the forward pass,
    so chem_env (Nc, Q, λ₅, λ₁) is created and freed per N-chunk rather than
    kept alive for all N molecules simultaneously.

    Within each pass, M-chunking avoids materialising the full (Nc, M, Q, λ₅) RFF
    tensor. The two heavy contractions use bmm on reshaped 3-D tensors rather than
    einsum, because einsum('nmqd,nmql->nqdl') would create an (Nc,mc,Q,λ₅,λ₁)
    intermediate before contracting over m; the tensor that caused OOM.
    """

    _lambda_1: int                          # atom embedding dimension (λ₁)
    _lambda_2: int                          # number of message-passing rounds (λ₂)
    _lambda_5: int                          # number of Random Fourier Features (λ₅)
    _nchunk:   int                          # molecules per N-chunk
    _mchunk:   int                          # atoms per M-chunk
    _proj_agg: nn.Linear                    # projects aggregated context λ₁ → 2λ₁ for MishGLU gating
    _omegafrq: Float[torch.Tensor, "λ₅ 3"]  # fixed RFF frequency matrix: λ₅ random 3-D directions
    _biasterm: nn.Parameter                 # RFF random phase offsets b ∈ R^λ₅
    _sigbilin: nn.Bilinear                  # updates σ from (updated embedding, form factor)
    _rms_norm: nn.RMSNorm                   # normalises aggregated context before gating
    _q_points: int                          # number of q-points (Q)
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
            x = x.clone()
            dist.all_reduce(x, op=dist.ReduceOp.SUM)
            return x

        @staticmethod
        def backward(_ctx, grad: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            dist.all_reduce(grad, op=dist.ReduceOp.SUM)
            return grad

    class _PassContainer(NamedTuple):

        """Working state for one N-chunk during a message-passing round."""

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
        self._sigbilin = nn.Bilinear(lambda_1, q_points, 1)
        self._rms_norm = nn.RMSNorm(lambda_1)
        self._q_points = q_points
        self._step1_fn = torch.compile(self._step1, fullgraph=True, dynamic=True) if compile else self._step1
        self._step2_fn = torch.compile(self._step2, fullgraph=True, dynamic=True) if compile else self._step2

    @staticmethod
    def _step1(
        biasterm: nn.Parameter,
        omegafrq: torch.Tensor,
        embslice: torch.Tensor, 
        crdslice: torch.Tensor, 
        sigslice: torch.Tensor, 
        mskslice: torch.Tensor,
        epsilon_: float,
        lambda_1: int,
        lambda_5: int
    ):

        # r̃_m = r_m / σ_m: scale coords by bandwidth so kernel range is q-dependent.
        # clamp(min=eps) caps the max RFF frequency and avoids 1/0.
        scaled_coords = crdslice.unsqueeze(-2) / sigslice.clamp(min=epsilon_) # (Nc, mc, Q, 3)

        # φ_m = √(2/λ₅) · cos(Ω · r̃_m + b): RFF feature vector per atom per q-point
        proj = scaled_coords @ omegafrq.T + biasterm       # (Nc, mc, Q, λ₅)
        zrff = (2/lambda_5) ** 0.5 * cos(proj)             # (Nc, mc, Q, λ₅)
        zrff = zrff * mskslice.unsqueeze(-1).unsqueeze(-1) # zero padding atoms

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
        """

        features: Float[torch.Tensor, "Nchnk Q λ₅"]    = cont.features
        chem_env: Float[torch.Tensor, "Nchnk Q λ₅ λ₁"] = cont.chem_env

        for m0 in range(0, cont.M, cont.Mchnk):
            m1 = min(m0 + cont.Mchnk, cont.M)
            
            emb_slice = cont.emb_n[:, m0:m1]
            crd_slice = cont.crd_n[:, m0:m1]
            sig_slice = cont.sig_n[:, m0:m1]
            msk_slice = cont.msk_n[:, m0:m1]
            
            args = (
                self._biasterm,
                self._omegafrq,
                emb_slice,
                crd_slice,
                sig_slice,
                msk_slice,
                eps,
                self._lambda_1,
                self._lambda_5
            )
            
            step_feat, step_chem = checkpoint(self._step1_fn, *args, use_reentrant=False)  # type: ignore[misc]
            features = features + step_feat
            chem_env = chem_env + step_chem

        return cont._replace(features=features, chem_env=chem_env)

    @staticmethod
    def _step2(
        biasterm: nn.Parameter,
        rms_norm: nn.RMSNorm,
        proj_agg: nn.Linear,
        sigbilin: nn.Bilinear,
        omegafrq: torch.Tensor,
        embslice: torch.Tensor, 
        crdslice: torch.Tensor, 
        sigslice: torch.Tensor, 
        mskslice: torch.Tensor,
        ffsslice: torch.Tensor,
        features: torch.Tensor,
        chem_env: torch.Tensor,
        epsilon_: float,
        q_points: int,
        lambda_1: int,
        lambda_5: int
        ):

        # recompute φ_m for this M-chunk (same as _pass_1, but now chem_env is complete)
        scaled_coords = crdslice.unsqueeze(-2) / sigslice.clamp(min=epsilon_) # (Nc, mc, Q, 3)
        proj = scaled_coords @ omegafrq.T + biasterm # (Nc, mc, Q, λ₅)
        zrff = (2/lambda_5) ** 0.5 * cos(proj) # (Nc, mc, Q, λ₅)
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
        agg = rms_norm(locality / weights.unsqueeze(-1).clamp(min=epsilon_)) # (Nc, mc, Q, λ₁)

        # MishGLU gate: one linear projects to 2λ₁, split into value p1 and gate p2.
        # gate = p1 · Mish(p2) selectively passes neighbourhood signal into the residual stream.
        p1, p2 = proj_agg(agg).chunk(2, dim=-1) # each (Nc, mc, Q, λ₁)
        gate   = p1 * F.mish(p2) * mask               # (Nc, mc, Q, λ₁)

        # residual update
        new_emb = embslice + gate       

        # sigma update: tanhshrink(x) = x - tanh(x) is near-zero for small x ("sticky" -
        # sigma barely moves when the bilinear output is small) and grows linearly for large x.
        # softplus keeps σ strictly positive.
        f_in    = ffsslice.transpose(-1, -2).expand(-1, -1, q_points, -1)
        new_sig = F.softplus(sigslice + F.tanhshrink(sigbilin(new_emb, f_in)))

        return new_emb, new_sig

    def _pass_2(self, cont: _PassContainer, eps: float) -> _PassContainer:

        """
        Compute per-atom neighbourhood aggregate and update embeddings and sigmas.

        Uses the fully-accumulated chem_env from _pass_1 so every atom attends to
        all others despite M-chunking (all-pairs coverage is preserved).
        """

        new_emb_m = []
        new_sig_m = []

        for m0 in range(0, cont.M, cont.Mchnk):
            m1 = min(m0 + cont.Mchnk, cont.M)
            
            emb_slice = cont.emb_n[:, m0:m1]
            ffs_slice = cont.ffs_n[:, m0:m1]
            sig_slice = cont.sig_n[:, m0:m1]
            crd_slice = cont.crd_n[:, m0:m1]
            msk_slice = cont.msk_n[:, m0:m1]
            cont_feat = cont.features
            cont_chnv = cont.chem_env 
            
            args = (
                self._biasterm,
                self._rms_norm,
                self._proj_agg,
                self._sigbilin,
                self._omegafrq,
                emb_slice,
                crd_slice,
                sig_slice,
                msk_slice,
                ffs_slice,
                cont_feat,
                cont_chnv,
                eps,
                self._q_points,
                self._lambda_1,
                self._lambda_5
            )

            emb_c, sig_c = checkpoint(self._step2_fn, *args, use_reentrant=False)  # type: ignore[misc]
            new_emb_m.append(emb_c)
            new_sig_m.append(sig_c)

        new_emb = torch.cat(new_emb_m, dim=1)  # (Nchnk, M, Q, λ₁)
        new_sig = torch.cat(new_sig_m, dim=1)  # (Nchnk, M, Q, 1)
        return cont._replace(emb_n=new_emb, sig_n=new_sig)

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
        the other rank never issues (NCCL watchdog timeout)."""

        if not use_all_reduce or not dist.is_available() or not dist.is_initialized():
            return x
        return self._AllReduce.apply(x)  # type: ignore[return-value]

    @jaxtyped(typechecker=beartype)
    def forward(self, batch: Batch, embed_head: LayerHead, eps: float, use_all_reduce: bool = True) -> LayerHead:

        """
        Run λ₂ rounds of RFF message passing.

        Args:
            batch:          molecule geometry; uses coord (N, M, 3) Cartesian positions (Å)
            embed_head:     output of Embed; embeds (N, M, 1, λ₁), f_mags and sigmas (N, M, Q, 1)
            eps:            numerical floor for sigma clamping and aggregate denominator
            use_all_reduce: whether to all-reduce features/chem_env across ranks between
                            the two passes. True (default) for TP and single-GPU, where
                            the shard being processed is a slice of the SAME molecules on
                            every rank. Must be False for ScatterNet's DP path, where each
                            rank processes a disjoint set of molecules and there is nothing
                            to reconcile — see `_all_reduce`'s docstring for why this isn't
                            optional (it's a correctness/deadlock issue, not just a no-op).
        Returns:
            LayerHead with embeds updated to (N, M, Q, λ₁) and sigmas updated; f_mags unchanged
        """

        # expand Q dim from 1 → Q upfront; zero-copy stride-0 view until a contiguous op fires
        embeds       = embed_head.embeds.expand(-1, -1, self._q_points, -1)
        sigmas       = embed_head.sigmas
        coord        = batch.coord
        padding_mask = batch.padding_mask()
        f_mags       = embed_head.f_mags

        N, M, _ = coord.shape
        Q       = sigmas.shape[2]
        Cn      = self._nchunk
        Cm      = self._mchunk

        for _ in range(self._lambda_2):
            new_embeds_n = []
            new_sigmas_n = []

            for n0 in range(0, N, Cn):
                n1 = min(n0 + Cn, N)

                # Nc derived from emb_s.shape[0] inside the closure rather than closed over as a
                # variable; Python closures capture by reference so a loop variable would give the
                # last iteration's value when the checkpoint replays during backward.
                def _n_chunk_round(emb_s, msk_s, ffs_s, sig_s, crd_s):
                    Nc_s = emb_s.shape[0]
                    cont = self._PassContainer(
                        M        = M,
                        Nchnk    = Nc_s,
                        Mchnk    = Cm,
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
                    cont = self._pass_2(cont, eps)
                    return cont.emb_n, cont.sig_n

                # use_reentrant=True: forward runs under no_grad so chem_env is created and freed
                # per N-chunk. use_reentrant=False would keep all N-chunks' chem_env alive
                # simultaneously (the _pass_2._step closures hold a reference to it).
                args = (embeds[n0:n1], padding_mask[n0:n1], f_mags[n0:n1], sigmas[n0:n1], coord[n0:n1])
                new_emb, new_sig = checkpoint(_n_chunk_round, *args, use_reentrant=True)  # type: ignore[misc]
                new_embeds_n.append(new_emb)
                new_sigmas_n.append(new_sig)

            embeds = torch.cat(new_embeds_n, dim=0)  # (N, M, Q, λ₁)
            sigmas = torch.cat(new_sigmas_n, dim=0)  # (N, M, Q, 1)

        return embed_head._replace(embeds=embeds, sigmas=sigmas)
