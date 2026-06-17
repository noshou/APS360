import torch
from jaxtyping import Float
from typing import NamedTuple

class LayerHead(NamedTuple):
    
    """
    Immutable output container for a hidden layer output head.

    Fields:
        embeds: per-atom learned embeddings; (N, M, 1, λ₁) from Embed, (N, M, Q, λ₁) after MessagePass
        f_mags: complex form factor magnitudes |f_re + i·f_im| (N, M, Q, 1)
    """
    
    embeds: Float[torch.Tensor, "N M * λ₁"] # atom embeddings
    f_mags: Float[torch.Tensor, "N M Q 1"]  # form factor magnitudes
