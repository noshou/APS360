import numpy as np 
import torch
import torch.nn.functional as F

from torch        import cos, nn
from ..batching   import Batch
from jaxtyping    import jaxtyped
from beartype     import beartype
from .layer_head  import LayerHead
from numpy.random import Generator
class MessagePass(nn.Module):

    _lambda_2: int          # number of message passing rounds
    _lambda_5: int          # resolution for RFF
    _project2: nn.Linear    # projects λ₁ dimension to 2 * λ₁ for GLU
    _rng_gen_: Generator    # random number generator
    _omegafrq: nn.Parameter
    _biasterm: nn.Parameter    
    def __init__(self, lambda_1: int, lambda_2: int, lambda_5: int, seed):
        super().__init__()

        if lambda_2 <= 0:
            raise ValueError(f"invalid lambda_2 (must be > 0): {lambda_2}")
        
        self._lambda_2 = lambda_2
        self._lambda_5 = lambda_5
        self._project2 = nn.Linear(lambda_1, 2 * lambda_1)
        self._rng_gen_ = np.random.default_rng(seed=seed)
        self._omegafrq = nn.Parameter(torch.from_numpy(self._rng_gen_.standard_normal((self._lambda_5, 3))).float())
        self._biasterm = nn.Parameter(torch.from_numpy(self._rng_gen_.uniform(0, 2*np.pi, size=(self._lambda_5))).float())
        
    @jaxtyped(typechecker=beartype)
    def forward(self, batch: Batch, embed_head: LayerHead, eps: float=1e-8) -> LayerHead:
        
        """
        Run λ₂ rounds of RFF-based message passing on the given batch.

        Args:
            batch:      molecule geometry; uses coord (N,M,3) Cartesian positions
            embed_head: output of Embed; embeds (N,M,1,λ₁), f_mags (N,M,Q,1)

        Returns:
            new LayerHead with embeds updated; f_mags unchanged
        """

        # calc dot product btwn atom's position and d-th frequency vector, add phase shift (bias)
        # proj[N, M, λ₅] = Σ_k  omega[d, k] * coord[n, m, k]  +  bias[d]
        proj = batch.coord @ self._omegafrq.T + self._biasterm
        
        # calculate matrix of rff features per atom
        zrff = (2/self._lambda_5) ** 0.5 * cos(proj)
        zrff = zrff * batch.padding_mask().unsqueeze(-1)
                    
        # aggregate total rff features per atom per frequency per molecule
        # features[N, λ₅] = Σ_m  zrff[n, m, λ₅]
        features = zrff.sum(dim=1)
        
        # normalize against locality to get weighted average vs weighted sum
        # weighted[N, M] = Σ_d  zrff[n, m, λ₅] * denom_sum[n, λ₅]
        weighted = torch.einsum('nmd, nd -> nm', zrff, features)

        # create mutable embed, message pass λ₂ times
        embeds = embed_head.embeds
        for _ in range(self._lambda_2):
            
            # aggregate over all M neighbours to get chemical environment 
            # chem_env[N, Q, λ₅, λ₁] = Σ_m  zrff[N, M, λ₅] * embeds[N, M, Q, λ₁]
            chem_env = torch.einsum('nmd, nmql -> nqdl', zrff, embeds)              
            
            # aggregate neighbourhood embeddings by takeing dot product 
            # of each atom against the chemical environment
            # locality[N, M, Q, λ₁] = Σ_d  zrff[n, m, λ₅] * chem_env[n, q, λ₅, λ₁]
            locality = torch.einsum('nmd, nqdl -> nmql', zrff, chem_env)
            
            # aggregate (N,M,Q,λ₁)
            aggregate = locality / weighted.unsqueeze(-1).unsqueeze(-1).clamp(min=eps) 
            
            # gate aggregate using MishGLU
            # mask gated_aggregate instead of aggregate since
            # SwiGLU allows small outputs at x=0
            embed_proj = self._project2(aggregate)
            embed_proj1, embed_proj2 = embed_proj.chunk(2, dim=-1)
            gated_aggregate = embed_proj1 * F.mish(embed_proj2)
            gated_aggregate = gated_aggregate * batch.padding_mask().unsqueeze(-1).unsqueeze(-1)
            
            # update embedding vector using residual connection
            embeds = embeds + gated_aggregate

        # output layer head
        return embed_head._replace(embeds=embeds)