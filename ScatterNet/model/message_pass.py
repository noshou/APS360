import numpy as np
import torch
import torch.nn.functional as F

from torch                        import cos, nn
from torch.utils.checkpoint       import checkpoint
from ..batching                   import Batch
from jaxtyping                    import jaxtyped, Float
from beartype                     import beartype
from .layer_head                  import LayerHead


class MessagePass(nn.Module):

    """
    RFF-based message passing module.

    Runs lambda_2 rounds of kernel-weighted neighbourhood aggregation using Random
    Fourier Features (RFF) to approximate an RBF kernel. Each round is gradient
    checkpointed: intermediates are not stored during the forward pass and are
    recomputed during backward. This trades extra compute for significantly reduced
    peak activation memory, which is critical for large molecules where the
    (N, M, Q, lambda_5) RFF tensor would otherwise be stored lambda_2 times.

    Each round runs in two passes over M-sized chunks of atoms (controlled by
    msg_chunk) to avoid materialising the full (N, M, Q, lambda_5) tensor at once:
        Pass 1: accumulate features (N, Q, lambda_5) and chem_env (N, Q, lambda_5, lambda_1)
                by summing over all atom chunks. The full chem_env captures global
                context: every atom contributes to every other atom's neighbourhood.
        Pass 2: for each chunk, compute locality and weighted aggregate using the
                globally-accumulated chem_env and features, then apply MishGLU and
                update the chunk's embeddings and sigma bandwidths.

    Per round:
        1. Scale atom coordinates by per-atom, per-q sigma bandwidths.
        2. Project scaled coords onto lambda_5 random frequency vectors (RFF).
        3. Compute kernel-weighted neighbourhood embedding (chem_env).
        4. Gate aggregated embedding with MishGLU and add residual.
        5. Update sigma bandwidths via a bilinear layer on the new embeddings.
    """

    _lambda_2:  int                          # number of message passing rounds
    _lambda_5:  int                          # number of RFF features
    _msg_chunk: int                          # atoms per M-chunk to cap peak tensor size
    _proj_agg:  nn.Linear                   # projects lambda_1 to 2*lambda_1 for MishGLU gate
    _omegafrq:  Float[torch.Tensor, "λ₅ 3"] # fixed random frequency matrix for RFF
    _biasterm:  nn.Parameter                # learned phase shift; preserves rotational invariance
    _sigbilin:  nn.Bilinear                 # updates sigma bandwidths from embeddings and form factors

    def __init__(self, lambda_1: int, lambda_2: int, lambda_5: int, seed: int, q_points: int, msg_chunk: int):
        super().__init__()

        if lambda_2 <= 0:
            raise ValueError(f"invalid lambda_2 (must be > 0): {lambda_2}")
        if msg_chunk <= 0:
            raise ValueError(f"invalid msg_chunk (must be > 0): {msg_chunk}")

        rng = np.random.default_rng(seed=seed)

        self._lambda_2  = lambda_2
        self._lambda_5  = lambda_5
        self._msg_chunk = msg_chunk
        self._proj_agg  = nn.Linear(lambda_1, 2 * lambda_1)
        self.register_buffer('_omegafrq', torch.from_numpy(rng.standard_normal((lambda_5, 3))).float())
        self._biasterm  = nn.Parameter(torch.from_numpy(rng.uniform(0, 2*np.pi, size=(self._lambda_5))).float())
        self._sigbilin  = nn.Bilinear(lambda_1, q_points, 1)

    def _one_round(
        self,
        embeds:       torch.Tensor,  # (N, M, Q, lambda_1)
        sigmas:       torch.Tensor,  # (N, M, Q, 1)
        coord:        torch.Tensor,  # (N, M, 3)
        padding_mask: torch.Tensor,  # (N, M)
        f_mags:       torch.Tensor,  # (N, M, Q, 1)
        eps:          float,
    ):
        """
        One round of RFF message passing, chunked along the M (atom) dimension.
        Invoked via gradient checkpoint in forward.

        Two passes over M-chunks avoid materialising the full (N, M, Q, lambda_5)
        RFF tensor. The globally-accumulated chem_env ensures every atom still
        attends to all others, preserving the all-pairs interaction.

        Args:
            embeds:       atom embeddings, shape (N, M, Q, lambda_1)
            sigmas:       RBF kernel bandwidths per atom per q, shape (N, M, Q, 1)
            coord:        centroid-subtracted Cartesian coordinates, shape (N, M, 3)
            padding_mask: True for real atoms, False for padding, shape (N, M)
            f_mags:       form factor magnitudes per atom per q, shape (N, M, Q, 1)
            eps:          numerical floor for sigma clamping and aggregate denominator

        Returns:
            updated embeds (N, M, Q, lambda_1) and sigmas (N, M, Q, 1)
        """

        N, M, _ = coord.shape
        Q       = sigmas.shape[2]
        L       = embeds.shape[-1]   # lambda_1
        C       = self._msg_chunk

        # RFF kernel approximates RBF kernel exp(-||r_i - r_j||^2 / 2*sigma^2).
        # Scaling coordinates by sigma is equivalent to scaling the kernel bandwidth:
        #   exp(-||r_i - r_j||^2 / 2*sigma^2) = exp(-||r_i/sigma - r_j/sigma||^2 / 2)
        # large sigma -> small scaled coords -> long-range aggregation (low q)
        # small sigma -> large scaled coords -> short-range aggregation (high q)

        # Pass 1: accumulate global context across all atom chunks.
        # features[n, q, d]    = sum_m  zrff[n,m,q,d]
        # chem_env[n, q, d, l] = sum_m  zrff[n,m,q,d] * embeds[n,m,q,l]
        # Neither output has the M dimension, so they fit in memory regardless of M.
        features = coord.new_zeros(N, Q, self._lambda_5)
        chem_env = coord.new_zeros(N, Q, self._lambda_5, L)

        for m0 in range(0, M, C):
            m1 = min(m0 + C, M)

            # crds_c: (N, chunk, 1, 3) / (N, chunk, Q, 1) -> (N, chunk, Q, 3)
            crds_c = coord[:, m0:m1].unsqueeze(-2) / sigmas[:, m0:m1].clamp(min=eps)

            # proj_c[n, m, q, d] = sum_k  omega[d, k] * crds_c[n, m, q, k]  +  bias[d]
            # proj_c: (N, chunk, Q, lambda_5)
            proj_c = crds_c @ self._omegafrq.T + self._biasterm

            # RFF features: z[n,m,q,d] = sqrt(2/lambda_5) * cos(proj[n,m,q,d])
            # dot product z[i] . z[j] ≈ exp(-||r_i/sigma - r_j/sigma||^2 / 2)
            # zrff_c: (N, chunk, Q, lambda_5)
            zrff_c   = (2/self._lambda_5) ** 0.5 * cos(proj_c)
            mask_c   = padding_mask[:, m0:m1].unsqueeze(-1).unsqueeze(-1)
            zrff_c   = zrff_c * mask_c

            features = features + zrff_c.sum(dim=1)
            chem_env = chem_env + torch.einsum('nmqd, nmql -> nqdl', zrff_c, embeds[:, m0:m1])

        # Pass 2: compute per-chunk aggregate and update embeddings and sigmas.
        # Each atom chunk uses the fully-accumulated chem_env and features, so every
        # atom's output still depends on all other atoms (all-pairs interaction preserved).
        new_embeds_chunks = []
        new_sigmas_chunks = []

        for m0 in range(0, M, C):
            m1 = min(m0 + C, M)

            crds_c = coord[:, m0:m1].unsqueeze(-2) / sigmas[:, m0:m1].clamp(min=eps)
            proj_c = crds_c @ self._omegafrq.T + self._biasterm
            zrff_c = (2/self._lambda_5) ** 0.5 * cos(proj_c)
            mask_c = padding_mask[:, m0:m1].unsqueeze(-1).unsqueeze(-1)
            zrff_c = zrff_c * mask_c

            # locality[n, m, q, l] = sum_d  zrff[n,m,q,d] * chem_env[n,q,d,l]
            #                      ≈ sum_{m'} k(m, m') * embeds[m']
            # locality_c: (N, chunk, Q, lambda_1)
            locality_c = torch.einsum('nmqd, nqdl -> nmql', zrff_c, chem_env)

            # denominator: kernel weight sum per atom
            # weighted[n, m, q] = sum_d  zrff[n,m,q,d] * features[n,q,d] ≈ sum_{m'} k(m, m')
            # weights can be negative due to cosine features, clamp to avoid division issues
            # weighted_c: (N, chunk, Q)
            weighted_c = torch.einsum('nmqd, nqd -> nmq', zrff_c, features).abs().clamp(min=0)

            # weighted average: aggregate[n,m,q,l] = locality / weighted
            # (N, chunk, Q, lambda_1) / (N, chunk, Q, 1)
            agg_c = torch.nan_to_num(locality_c / weighted_c.unsqueeze(-1), nan=0.0, posinf=0.0, neginf=0.0)

            # gate aggregate using MishGLU
            # mask gated output instead of agg since MishGLU allows small outputs at x=0
            p1_c, p2_c = self._proj_agg(agg_c).chunk(2, dim=-1)
            gate_c  = p1_c * F.mish(p2_c)
            gate_c  = gate_c * mask_c

            # update embeddings and sigmas via residual connections
            emb_c  = embeds[:, m0:m1] + gate_c
            f_in_c = f_mags[:, m0:m1].transpose(-1, -2).expand(-1, -1, emb_c.shape[2], -1)
            sig_c  = sigmas[:, m0:m1] + F.softplus(self._sigbilin(emb_c, f_in_c)) + eps

            new_embeds_chunks.append(emb_c)
            new_sigmas_chunks.append(sig_c)

        new_embeds = torch.cat(new_embeds_chunks, dim=1)
        new_sigmas = torch.cat(new_sigmas_chunks, dim=1)

        return new_embeds, new_sigmas

    @jaxtyped(typechecker=beartype)
    def forward(self, batch: Batch, embed_head: LayerHead, eps: float) -> LayerHead:
        """
        Run lambda_2 rounds of RFF-based message passing on the given batch.

        Args:
            batch:      molecule geometry; uses coord (N,M,3) Cartesian positions
            embed_head: output of Embed; embeds (N,M,1,lambda_1), f_mags (N,M,Q,1), sigmas (N,M,Q,1)
            eps:        numerical floor for sigma clamping and aggregate denominator

        Returns:
            LayerHead with embeds updated to (N,M,Q,lambda_1), sigmas updated; f_mags unchanged
        """
        embeds       = embed_head.embeds
        sigmas       = embed_head.sigmas
        coord        = batch.coord
        padding_mask = batch.padding_mask()
        f_mags       = embed_head.f_mags

        for _ in range(self._lambda_2):
            embeds, sigmas = checkpoint(  # type: ignore[assignment]
                self._one_round, embeds, sigmas, coord, padding_mask, f_mags, eps,
                use_reentrant=False,
            )

        return embed_head._replace(embeds=embeds, sigmas=sigmas)
