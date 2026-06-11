import torch.nn.functional as F
from torch import sin, cos, exp, nn
from ..batching import Batch
from jaxtyping import jaxtyped
from beartype import beartype
from .layer_head import LayerHead



class MessagePass(nn.Module):
    
    """
    MessagePass: 
        λ₂ rounds of Gaussian-weighted neighbour aggregation.
        Each round, atom i collects a weighted mean of neighbour embeddings,
        weighted by rbf = exp(-d²/2σ²), where σ is atom i's learned bandwidth.
        Geometry is fixed per batch; only embed.embeds evolves across rounds.
    
    Training: O(M²) full distance matrix.
    
    Inference: replace with KD-tree + σ cutoff for O(M·K).
    """

    _lambda_2: int          # number of message passing rounds
    _project2: nn.Linear    # projects λ₁ dimension to 2 * λ₁ for GLU
    
    def __init__(self, lambda_1: int, lambda_2: int):
        super().__init__()

        if lambda_2 <= 0:
            raise ValueError(f"invalid lambda_2 (must be > 0): {lambda_2}")
        
        self._lambda_2 = lambda_2
        self._project2 = nn.Linear(lambda_1, 2 * lambda_1)
            
    @jaxtyped(typechecker=beartype)
    def forward(self, batch: Batch, embed_head: LayerHead, eps: float=1e-8) -> LayerHead:
        
        """
        Run λ₂ rounds of message passing on the given batch.

        Args:
            batch:      molecule geometry; provides radii (N,M) and angle (N,M,2)
            embed_head: output of Embed; embeds (N,M,1,λ₁) and sigmas (N,M,Q,1)

        Returns:
            new LayerHead with embeds updated to (N,M,Q,λ₁); f_mags and sigmas unchanged
        """

        # broadcast theta, phi angles
        t_i = batch.angle[:,:,0].unsqueeze(-1)
        t_j = batch.angle[:,:,0].unsqueeze(-2)
        p_i = batch.angle[:,:,1].unsqueeze(-1)
        p_j = batch.angle[:,:,1].unsqueeze(-2)

        # broadcast radial distances
        r_i = batch.radii.unsqueeze(-1)
        r_j = batch.radii.unsqueeze(-2)

        # d_ij² = r_i² + r_j² - 2·r_i·r_j·(sin(θ_i)sin(θ_j)cos(φ_i - φ_j) + cos(θ_i)cos(θ_j))
        # d_ij_sq is N x M_i x M_j, so need to transform to N x M_i x M_j x 1 x 1
        d_ij_sq = r_i ** 2 + r_j ** 2 - 2 * r_i * r_j * (sin(t_i) * sin(t_j) * cos(p_i-p_j) + cos(t_i) * cos(t_j))
        d_ij_sq = d_ij_sq.unsqueeze(-1).unsqueeze(-1)

        # now we calculate radial basis kernel, mask padded atoms 
        # where rbf has dimension N X M_i x M_j x Q x 1
        # since sigma has dimensions N x M x Q x 1, we must unsqueeze it to N x M_i x 1 x Q x 1
        # we clamp sigma since masked elements result in NaN which could cause issues
        rbf = exp(-(d_ij_sq/(2*embed_head.sigmas.clamp(min=eps).unsqueeze(-3)**2)))
        rbf = rbf * batch.padding_mask().unsqueeze(-2).unsqueeze(-1).unsqueeze(-1)
        
        # create mutable embed, message pass λ₂ times
        embeds = embed_head.embeds
        for _ in range(self._lambda_2):
            
            # aggregate over M_j neighbours to get chemical environment
            # we use weighted average since sum would skew towards larger aggregates
            # aggregate has dimension N x M x Q x λ₁, since M_j collapses with sum dim=-3
            # need to unsqueeze embeds from N x M x Q x λ₁ -> N x 1 x M x Q x λ₁
            aggregate = ((rbf*embeds.unsqueeze(-4)).sum(dim=-3) / (rbf.sum(dim=-3) + eps)) 

            # gate aggregate using SwiGLU
            # mask gated_aggregate instead of aggregate since
            # SwiGLU allows small outputs at x=0
            embed_proj = self._project2(aggregate)
            embed_proj1, embed_proj2 = embed_proj.chunk(2, dim=-1)
            gated_aggregate = embed_proj1 * F.silu(embed_proj2)
            gated_aggregate = gated_aggregate * batch.padding_mask().unsqueeze(-1).unsqueeze(-1)
            
            # update embedding vector using residual connection
            embeds = embeds + gated_aggregate
            
        # output layer head
        return embed_head._replace(embeds=embeds)