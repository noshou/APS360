import numpy as np
import torch
import torch.nn.functional as F

from torch                   import cos, nn
from torch.utils.checkpoint  import checkpoint
from ..batching              import Batch
from jaxtyping               import jaxtyped, Float, Bool
from beartype                import beartype
from .layer_head             import LayerHead
from typing                  import NamedTuple
import torch.distributed as dist

class MessagePass(nn.Module):

    """
    RFF-based message passing module.

    Runs lambda_2 rounds of kernel-weighted neighbourhood aggregation using Random
    Fourier Features (RFF) to approximate an RBF kernel.

    Memory strategy: checkpoint at the N-chunk level. Each N-chunk's full
    computation (_pass_1 -> all_reduce -> _pass_2) is wrapped in a single
    checkpoint call. This means:
        - chem_env (Nc, Q, lambda_5, lambda_1) is created and freed within each
        N-chunk's recomputation — only one exists at a time regardless of batch size
        - saved state per N-chunk is just the small input slices
        - Python overhead is N/n_chunk * lambda_2 checkpoint calls (not multiplied
        by M-chunk count as with per-M-chunk checkpointing)

    Each round chunks over both N (molecules) and M (atoms):
        N-chunking: molecules are independent; chem_env is computed per N-chunk.
        M-chunking: atoms within each N-chunk are processed in M-chunks to avoid
            materialising the full (Nc, M, Q, lambda_5) RFF tensor at once.

    Per round:
        1. Scale atom coordinates by per-atom, per-q sigma bandwidths.
        2. Project scaled coords onto lambda_5 random frequency vectors (RFF).
        3. Compute kernel-weighted neighbourhood embedding (chem_env).
        4. Gate aggregated embedding with MishGLU and add residual.
        5. Update sigma bandwidths via a bilinear layer on the new embeddings.
    """
    _lambda_1: int                          # atom embedding dimension (λ₁)
    _lambda_2: int                          # number of message-passing rounds (λ₂)
    _lambda_5: int                          # number of Random Fourier Features (λ₅)
    _nchunk:   int                          # molecules per N-chunk
    _mchunk:   int                          # atoms per M-chunk
    _proj_agg: nn.Linear                    # MishGLU projection λ₁ → 2λ₁; gates the aggregated context
    _omegafrq: Float[torch.Tensor, "λ₅ 3"]  # fixed RFF frequency matrix: λ₅ random 3-D directions
    _biasterm: nn.Parameter                 # RFF random phase offsets, one per Fourier feature
    _sigbilin: nn.Bilinear                  # per-round σ update from (embedding, form factor)
    _rms_norm: nn.RMSNorm                   # per-round embedding norm; caps activation magnitude (prevents fp16 overflow)
    _q_points: int                          # number of q-points (Q)
        
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
        def forward(ctx, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            x = x.clone()
            dist.all_reduce(x, op=dist.ReduceOp.SUM)
            return x
    
        @staticmethod
        def backward(ctx, grad: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            dist.all_reduce(grad, op=dist.ReduceOp.SUM)
            return grad
    
    class _PassContainer(NamedTuple):

        """
        Per-N-chunk working state for one message-passing round.

        Bundles sliced input tensors and zero-initialised accumulators so that
        _pass_1 and _pass_2 can be called as self-contained functions without
        checkpointing overhead at the pass level. Checkpointing lives at the
        N-chunk level in forward(), so chem_env is created and freed within
        each N-chunk's recomputation rather than being held for the full batch.
        """

        M:        int
        Nchnk:    int
        Mchnk:    int
        emb_n:    Float[torch.Tensor, "Nchnk Mc Q λ₁"]
        msk_n:    Bool[torch.Tensor,  "Nchnk Mc"]
        ffs_n:    Float[torch.Tensor, "Nchnk Mc Q 1"]
        sig_n:    Float[torch.Tensor, "Nchnk Mc Q 1"]
        crd_n:    Float[torch.Tensor, "Nchnk Mc 3"]
        features: Float[torch.Tensor, "Nchnk Q λ₅"]
        chem_env: Float[torch.Tensor, "Nchnk Q λ₅ λ₁"]

    def __init__(
        self,
        lambda_1:  int,
        lambda_2:  int,
        lambda_5:  int,
        seed:      int,
        q_points:  int,
        n_chunk:   int,
        m_chunk:   int,
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

    def _pass_1(self, cont: _PassContainer, eps: float) -> _PassContainer:

        """
            Accumulates global context for this N-chunk across all M-chunks.
            ```
                features[nc, q, λ₅]     = sum_m  zrff[nc,mc,q,λ₅]
                chem_env[nc, q, λ₅, λ₁] = sum_m  zrff[nc,mc,q,λ₅] * embeds[nc,mc,q,λ₁]
            ```
            Each M-chunk step is checkpointed so only the small input slices
            are saved for recomputation rather than the full RFF tensors.
        """

        def _step(emb_slice, crd_slice, sig_slice, msk_slice):
            
            # crds_c: (Nc, mc, 1, 3) / (Nc, mc, Q, 1) -> (Nc, mc, Q, 3)
            # eps_msgp (sigma floor) must keep proj_c = crds_c @ omega within fp16 range (~65504).
            # With max coord ~60 Å and omega ~ N(0,1) over 3 dims, proj_c ≈ (coord/sigma) × 3×3.5.
            # eps=0.1 → crds_c ≤ 600 → proj_c ≤ 2100, safely within fp16.
            # eps=1e-3 → crds_c ≤ 60000 → proj_c ≤ 180000, overflows fp16 → inf → cos=NaN.
            crds_c = crd_slice.unsqueeze(-2) / sig_slice.clamp(min=eps)
            proj_c = crds_c @ self._omegafrq.T + self._biasterm
            zrff_c = (2/self._lambda_5) ** 0.5 * cos(proj_c)
            zrff_c = zrff_c * msk_slice.unsqueeze(-1).unsqueeze(-1)
            step_features = zrff_c.sum(dim=1)
            
            # chem_env[nc, q, d, l] = Σ_m  zrff[nc, mc, q, d] * emb[nc, mc, q, l]
            # reshape to (Nc*Q, λ₅, mc) @ (Nc*Q, mc, λ₁) to avoid 5D intermediate
            Nc, mc, Q = zrff_c.shape[0], zrff_c.shape[1], zrff_c.shape[2]
            zb = zrff_c.permute(0, 2, 3, 1).reshape(Nc * Q, self._lambda_5, mc)
            eb = emb_slice.permute(0, 2, 1, 3).reshape(Nc * Q, mc, self._lambda_1)
            step_chem_env = torch.bmm(zb, eb).reshape(Nc, Q, self._lambda_5, self._lambda_1)
            return step_features, step_chem_env

        features: Float[torch.Tensor, "Nchnk Q λ₅"]    = cont.features
        chem_env: Float[torch.Tensor, "Nchnk Q λ₅ λ₁"] = cont.chem_env

        for m0 in range(0, cont.M, cont.Mchnk):
            m1 = min(m0 + cont.Mchnk, cont.M)
            step_feat, step_chem = checkpoint(  # type: ignore[misc]
                _step,
                cont.emb_n[:, m0:m1],
                cont.crd_n[:, m0:m1],
                cont.sig_n[:, m0:m1],
                cont.msk_n[:, m0:m1],
                use_reentrant=False,
            )
            features = features + step_feat
            chem_env = chem_env + step_chem

        return cont._replace(features=features, chem_env=chem_env)

    def _pass_2(self, cont: _PassContainer, eps: float) -> _PassContainer:

        """
            Computes per-M-chunk aggregate using the fully-accumulated chem_env.
            Every atom still attends to all others via chem_env (all-pairs preserved).
            Each M-chunk step is checkpointed; chem_env and features are captured
            via closure (they exist for the entire N-chunk's backward recomputation
            so they are available when each M-chunk step is replayed).
        """

        def _step(emb_slice, ffs_slice, sig_slice, crd_slice, msk_slice):
            crds_c = crd_slice.unsqueeze(-2) / sig_slice.clamp(min=eps)
            proj_c = crds_c @ self._omegafrq.T + self._biasterm
            zrff_c = (2/self._lambda_5) ** 0.5 * cos(proj_c)
            mask_c = msk_slice.unsqueeze(-1).unsqueeze(-1)
            zrff_c = zrff_c * mask_c

            # locality[nc, m, q, λ₁] = sum_λ₅  zrff[nc,mc,q,λ₅] * chem_env[nc,q,λ₅,λ₁]
            #                       ≈ sum_{m'} k(m, m') * embeds[m']
            # reshape to (Nc*Q, mc, λ₅) @ (Nc*Q, λ₅, λ₁) to avoid 5D intermediate
            Nc, mc, Q = zrff_c.shape[0], zrff_c.shape[1], zrff_c.shape[2]
            zb = zrff_c.permute(0, 2, 1, 3).reshape(Nc * Q, mc, self._lambda_5)
            cb = cont.chem_env.reshape(Nc * Q, self._lambda_5, self._lambda_1)
            locality_c = torch.bmm(zb, cb).reshape(Nc, Q, mc, self._lambda_1).permute(0, 2, 1, 3)

            # denominator: kernel weight sum per atom
            # weights can be negative due to cosine features; take abs so denominator is non-negative.
            # clamp(min=eps) avoids 0/0 in both forward and backward — nan_to_num only fixes the
            # forward value but still computes grad_output/0 = NaN during backward.
            weighted_c = torch.einsum('nmqd, nqd -> nmq', zrff_c, cont.features).abs()

            # weighted average: aggregate[nc,mc,q,λ₁] = locality / weighted
            agg_c = self._rms_norm(locality_c / weighted_c.unsqueeze(-1).clamp(min=eps))

            # gate aggregate using MishGLU; mask after gating (MishGLU ≠ 0 at x=0)
            p1_c, p2_c = self._proj_agg(agg_c).chunk(2, dim=-1)
            gate_c = p1_c * F.mish(p2_c) * mask_c

            emb_c  = emb_slice + gate_c
            f_in_c = ffs_slice.transpose(-1, -2).expand(-1, -1, self._q_points, -1)
            sig_c  = sig_slice + F.softplus(self._sigbilin(emb_c, f_in_c)) + eps
            return emb_c, sig_c

        new_emb_m = []
        new_sig_m = []

        for m0 in range(0, cont.M, cont.Mchnk):
            m1 = min(m0 + cont.Mchnk, cont.M)
            emb_c, sig_c = checkpoint(  # type: ignore[misc]
                _step,
                cont.emb_n[:, m0:m1],
                cont.ffs_n[:, m0:m1],
                cont.sig_n[:, m0:m1],
                cont.crd_n[:, m0:m1],
                cont.msk_n[:, m0:m1],
                use_reentrant=False,
            )
            new_emb_m.append(emb_c)
            new_sig_m.append(sig_c)

        new_emb = torch.cat(new_emb_m, dim=1)  # (Nchnk, M, Q, λ₁)
        new_sig = torch.cat(new_sig_m, dim=1)  # (Nchnk, M, Q, 1)
        return cont._replace(emb_n=new_emb, sig_n=new_sig)

    def _all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        """Apply _AllReduce if a process group is active; otherwise pass through."""
        if not dist.is_available() or not dist.is_initialized():
            return x
        return self._AllReduce.apply(x)  # type: ignore[return-value]

    @jaxtyped(typechecker=beartype)
    def forward(self, batch: Batch, embed_head: LayerHead, eps: float) -> LayerHead:
        
        """
        Run lambda_2 rounds of RFF-based message passing on the given batch.

        Checkpointing is at the N-chunk level: each N-chunk's full round
        (_pass_1 -> all_reduce -> _pass_2) is one checkpoint. This ensures
        chem_env is created and freed per N-chunk during recomputation, so
        peak memory is O(n_chunk * Q * lambda_5 * lambda_1) regardless of
        total batch size N.

        Args:
            batch:      molecule geometry; uses coord (N,M,3) Cartesian positions
            embed_head: output of Embed; embeds (N,M,1,lambda_1), f_mags, sigmas (N,M,Q,1)
            eps:        numerical floor for sigma clamping and aggregate denominator

        Returns:
            LayerHead with embeds updated to (N,M,Q,lambda_1), sigmas updated; f_mags unchanged
        """
        
        # expand Q dim from 1 to q_points upfront so both passes see uniform shape;
        # expand is zero-copy (stride-0 view) until a contiguous op materialises it
        embeds       = embed_head.embeds.expand(-1, -1, self._q_points, -1)
        sigmas       = embed_head.sigmas
        coord        = batch.coord
        padding_mask = batch.padding_mask()
        f_mags       = embed_head.f_mags

        N, M, _ = coord.shape
        Q       = sigmas.shape[2]
        Cn      = self._nchunk
        Cm      = self._mchunk

        # RFF kernel approximates RBF kernel exp(-||r_i - r_j||^2 / 2*sigma^2).
        # Scaling coordinates by sigma is equivalent to scaling the kernel bandwidth:
        #   exp(-||r_i - r_j||^2 / 2*sigma^2) = exp(-||r_i/sigma - r_j/sigma||^2 / 2)
        # large sigma -> small scaled coords -> long-range aggregation (low q)
        # small sigma -> large scaled coords -> short-range aggregation (high q)

        for _ in range(self._lambda_2):
            new_embeds_n = []
            new_sigmas_n = []

            for n0 in range(0, N, Cn):
                n1 = min(n0 + Cn, N)

                # Nc derived from emb_s.shape[0] inside the closure rather than
                # closed over as a variable; Python closures capture by reference
                # so a loop variable would give the last iteration's value on replay.
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
                        features = self._all_reduce(cont.features),
                        chem_env = self._all_reduce(cont.chem_env),
                    )
                    cont = self._pass_2(cont, eps)
                    return cont.emb_n, cont.sig_n

                # use_reentrant=True runs _n_chunk_round under no_grad() during
                # forward, so no internal autograd graph is built. Without this,
                # use_reentrant=False keeps the full internal graph alive (including
                # _pass_2._step closures that hold cont.chem_env), meaning all
                # N-chunks' chem_env tensors are live simultaneously. With
                # use_reentrant=True, chem_env is created and freed per N-chunk;
                # during backward each N-chunk is rerun once under enable_grad().
                new_emb, new_sig = checkpoint(  # type: ignore[misc]
                    _n_chunk_round,
                    embeds[n0:n1],
                    padding_mask[n0:n1],
                    f_mags[n0:n1],
                    sigmas[n0:n1],
                    coord[n0:n1],
                    use_reentrant=True,
                )
                new_embeds_n.append(new_emb)
                new_sigmas_n.append(new_sig)

            embeds = torch.cat(new_embeds_n, dim=0)  # (N, M, Q, λ₁)
            sigmas = torch.cat(new_sigmas_n, dim=0)  # (N, M, Q, 1)

        return embed_head._replace(embeds=embeds, sigmas=sigmas)
