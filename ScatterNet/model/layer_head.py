import torch
from jaxtyping import Float
from typing import NamedTuple

class LayerHead(NamedTuple):
    
    """
    Immutable container passed between Embed, MessagePass, and OutputHead.

    Fields:
        embeds: per-atom learned embeddings; (N, M, 1, λ₁) from Embed, (N, M, Q, λ₁) after MessagePass
        f_mags: form factor magnitudes |f0+f1 + i*f2| per atom per q-point, shape (N, M, Q, 1)
        sigmas: per-atom RFF kernel bandwidth per q-point, shape (N, M, Q, 1)
    """
    
    embeds: Float[torch.Tensor, "N M * λ₁"] # atom embeddings
    f_mags: Float[torch.Tensor, "N M Q 1"]  # form factor magnitudes
    sigmas: Float[torch.Tensor, "N M Q 1"]  # per-atom positional scaling factor