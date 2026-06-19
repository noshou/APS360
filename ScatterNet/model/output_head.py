import torch
import torch.nn.functional as F

from torch           import nn
from jaxtyping       import Float, jaxtyped
from beartype        import beartype
from beartype.typing import Tuple
from ..batching      import Batch
from .layer_head     import LayerHead
from collections     import OrderedDict
from numpy           import log2, floor

class OutputHead(nn.Module):
    
    """
    Collapses per-atom contributions into a predicted I(q) curve.

    For each atom, combines its embedding with its form factor magnitude via a bilinear
    layer, passes through an MLP, weights by f_mags^2 (Debye diagonal prior), and sums
    over atoms to produce I(q) per molecule.
    """
    
    _bilinear: nn.Bilinear   
    _mlp:      nn.Sequential 

    def __init__(
        self,
        lambda_1: int,
        lambda_3: int,
        lambda_4: int,
    ) -> None:
        super().__init__()
        
        if lambda_4 <= 0:
            raise ValueError("lambda_4 must be > 0")
        if lambda_4 > floor(log2(lambda_3)):
            raise ValueError(f"lambda_4 must be <= {int(floor(log2(lambda_3)))} for lambda_3={lambda_3}")

        self._bilinear = nn.Bilinear(lambda_1, 1, lambda_3)

        dims = [lambda_3 // 2**i for i in range(lambda_4 + 1)]
        if dims[-1] != 1:
            dims.append(1)

        ldicts = OrderedDict()
        for i in range(len(dims) - 1):
            ldicts[f"layer_{i}"] = nn.Linear(dims[i], dims[i+1])
            if i < len(dims) - 2:
                ldicts[f"activation_{i}"] = nn.Mish()
        self._mlp = nn.Sequential(ldicts)

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        batch: Batch,
        msg_head: LayerHead,
    ) -> Tuple[Float[torch.Tensor, "N Q"], Float[torch.Tensor, "N M Q"]]:
        
        # 1. Pass through bilinear layer: (N,M,Q, λ₁) x (N,M,Q,1) -> (N,M,Q,λ₃)
        atomic_contrib = F.mish(self._bilinear(msg_head.embeds, msg_head.f_mags))

        # 2. MLP compression -> (N,M,Q,1) -> Squeeze to (N,M,Q)
        contribs = F.softplus(self._mlp(atomic_contrib)).squeeze(-1)

        # 3. Squeeze true form factors down to (N,M,Q)
        f_mags = msg_head.f_mags.squeeze(-1)                          
        
        # 4. Get mask and cast to float to prevent multiplication type errors
        mask = batch.padding_mask().unsqueeze(-1).to(contribs.dtype) # (N,M,1)
        
        # 5. Apply weights and mask zeroing
        weighted = contribs * f_mags**2 * mask  # (N,M,Q)

        # 6. Sum along atom dimension M -> (N,Q)
        return weighted.sum(dim=1), f_mags
