import numpy as np 
import torch
import torch.nn.functional as F

from torch        import cos, nn
from ..batching   import Batch
from jaxtyping    import jaxtyped, Float
from beartype     import beartype
from .layer_head  import LayerHead
class MessagePass(nn.Module):

    _lambda_2: int                          # number of message passing rounds
    _lambda_5: int                          # resolution for RFF
    _proj_agg: nn.Linear                    # projects λ₁ dimension to 2 * λ₁ for aggregate GLU
    _omegafrq: Float[torch.Tensor, "λ₅ 3"] # rotational frequencies; isotropic when sampled randomly enough times
    _biasterm: nn.Parameter                 # phase shift bias term; learned, does not break rotational invariance
    _sigbilin: nn.Bilinear                  # bilinear layer for updating sigmas
    
    def __init__(self, lambda_1: int, lambda_2: int, lambda_5: int, seed: int, q_points: int):
        super().__init__()

        if lambda_2 <= 0:
            raise ValueError(f"invalid lambda_2 (must be > 0): {lambda_2}")

        rng = np.random.default_rng(seed=seed)
        
        self._lambda_2 = lambda_2
        self._lambda_5 = lambda_5
        self._proj_agg = nn.Linear(lambda_1, 2 * lambda_1)
        self.register_buffer('_omegafrq', torch.from_numpy(rng.standard_normal((lambda_5, 3))).float())
        self._biasterm = nn.Parameter(torch.from_numpy(rng.uniform(0, 2*np.pi, size=(self._lambda_5))).float())
        self._sigbilin = nn.Bilinear(lambda_1, q_points, 1)
        
    @jaxtyped(typechecker=beartype)
    def forward(self, batch: Batch, embed_head: LayerHead, eps: float) -> LayerHead:
        
        """
        Run λ₂ rounds of RFF-based message passing on the given batch.

        Args:
            batch:      molecule geometry; uses coord (N,M,3) Cartesian positions
            embed_head: output of Embed; embeds (N,M,1,λ₁), f_mags (N,M,Q,1), sigmas (N,M,Q,1)
            eps:        numerical floor for sigma clamping and aggregate denominator

        Returns:
            LayerHead with embeds updated to (N,M,Q,λ₁), sigmas updated; f_mags unchanged
        """
        
        # create mutable embed and sigmas, message pass λ₂ times
        embeds = embed_head.embeds
        sigmas = embed_head.sigmas
        for _ in range(self._lambda_2):
            
            
            # RFF kernel approximates RBF kernel exp(-||r_i - r_j||² / 2σ²)
            # scaling coordinates by sigma is equivalent to scaling the kernel bandwidth:
            # exp(-||r_i - r_j||² / 2σ²) = exp(-||r_i/σ - r_j/σ||² / 2)
            # large sigma → small scaled coords → long-range aggregation (low q)
            # small sigma → large scaled coords → short-range aggregation (high q)
            # crds: (N, M, 1, 3) / (N, M, Q, 1) → (N, M, Q, 3)
            crds = batch.coord.unsqueeze(-2) / sigmas.clamp(min=eps)

            # project each atom's scaled position onto λ₅ frequency vectors, add phase shift
            # proj[n, m, q, d] = Σ_k  omega[d, k] * crds[n, m, q, k]  +  bias[d]
            # proj: (N, M, Q, λ₅)
            proj = crds @ self._omegafrq.T + self._biasterm

            # RFF features: z[n,m,q,d] = sqrt(2/λ₅) * cos(proj[n,m,q,d])
            # dot product z[i]·z[j] ≈ exp(-||r_i/σ - r_j/σ||² / 2)
            # zrff: (N, M, Q, λ₅)
            zrff = (2/self._lambda_5) ** 0.5 * cos(proj)
            zrff = zrff * batch.padding_mask().unsqueeze(-1).unsqueeze(-1)

            # precompute sum of RFF features over atoms (loop-invariant given fixed zrff)
            # features[n, q, d] = Σ_m  zrff[n, m, q, d]
            # features: (N, Q, λ₅)
            features = zrff.sum(dim=1)
            
            # numerator: kernel-weighted sum of neighbour embeddings
            # chem_env[n, q, d, l] = Σ_m  zrff[n,m,q,d] * embeds[n,m,q,l]
            # chem_env: (N, Q, λ₅, λ₁)
            chem_env = torch.einsum('nmqd, nmql -> nqdl', zrff, embeds)

            # locality[n, m, q, l] = Σ_d  zrff[n,m,q,d] * chem_env[n,q,d,l]
            #                      ≈ Σ_{m'} k(m, m') * embeds[m']
            # locality: (N, M, Q, λ₁)
            locality = torch.einsum('nmqd, nqdl -> nmql', zrff, chem_env)

            # denominator: kernel weight sum per atom
            # weighted[n, m, q] = Σ_d  zrff[n,m,q,d] * features[n,q,d] ≈ Σ_{m'} k(m, m')
            # weights can be negative, but this is just noise, so we clamp it to 0.            
            # weighted: (N, M, Q)
            weighted = (torch.einsum('nmqd, nqd -> nmq', zrff, features)).abs().clamp(min=0)

            # weighted average: aggregate[n,m,q,l] = locality / weighted
            # (N, M, Q, λ₁) / (N, M, Q, 1)
            aggregate = torch.nan_to_num(locality / weighted.unsqueeze(-1), nan=0.0, posinf=0.0, neginf=0.0)

            # gate aggregate using MishGLU
            # mask gated_aggregate instead of aggregate since
            # MishGLU allows small outputs at x=0
            embed_proj = self._proj_agg(aggregate)
            embed_proj1, embed_proj2 = embed_proj.chunk(2, dim=-1)
            gated_aggregate = embed_proj1 * F.mish(embed_proj2)
            gated_aggregate = gated_aggregate * batch.padding_mask().unsqueeze(-1).unsqueeze(-1)

            # update embeddings and sigmas via residual connections
            embeds = embeds + gated_aggregate
            sigmas = sigmas + F.softplus(self._sigbilin(embeds, embed_head.f_mags.transpose(-1, -2))) + eps

        return embed_head._replace(embeds=embeds, sigmas=sigmas)