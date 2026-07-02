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
from typing          import Callable

class OutputHead(nn.Module):
    
    """
    Collapses per-atom contributions into a predicted I(q) curve.

    For each atom, combines its embedding with its form factor magnitude via a bilinear
    layer, passes the result through an MLP, weights by f_mags^2 (Debye diagonal prior),
    and sums over atoms to produce I(q) per molecule.

    The forward pass is chunked to avoid storing the
    (N, M, Q, lambda_3) bilinear output tensor.
    """
    
    _bilinear: nn.Bilinear   
    _mlp:      nn.Sequential 
    _fwd_fn:   Callable
    
    def __init__(
        self,
        lambda_1:  int,
        lambda_3:  int,
        lambda_4:  int,
        out_chunk: int,
        compile:   bool = False
    ) -> None:
        
        super().__init__()
        self._out_chunk = out_chunk
        
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
        self._fwd_fn = torch.compile(self._forward_fn, dynamic=True, fullgraph=True) if compile else self._forward_fn
        
    @staticmethod
    def _forward_fn(bilinear, mlp, emb_c, fmag_c, mask_c):

        # (N,M,Q, λ₁) x (N,M,Q,1) -> (N,M,Q,λ₃)
        atomic   = F.mish(bilinear(emb_c, fmag_c))       
        
        # MLP compression -> (N,M,Q,1) -> Squeeze to (N,M,Q)
        contribs = F.softplus(mlp(atomic)).squeeze(-1)
        
        # Squeeze form factors and mask down to (N,M,Q) and (N,M,1)
        fmc = fmag_c.squeeze(-1)
        
        # return intensity value (N,M,Q)
        return (contribs * fmc**2 * mask_c).sum(dim=1)
        
    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        batch:    Batch,
        msg_head: LayerHead,
    ) -> Tuple[
            Float[torch.Tensor, "N Q"], 
            Float[torch.Tensor, "N M Q"],
            Float[torch.Tensor, "N M Q"]
        ]:
            
        
        # 1. initialize values
        N, M, Q, _ = msg_head.embeds.shape
        mask       = batch.padding_mask().unsqueeze(-1).to(msg_head.embeds.dtype)  # (N, M, 1)
        
        # accumulate the Debye sum over atom-chunks; chunking avoids materialising
        # the full (N, M, Q, λ₃) bilinear tensor for large molecules
        iq_accum   = torch.zeros(N, Q, device=msg_head.embeds.device, dtype=msg_head.embeds.dtype)
        
        # 2. Accumulate over chunks
        for mol1 in range(0, M, self._out_chunk):
            mol2 = min(mol1 + self._out_chunk, M)
            iq_accum += self._fwd_fn(
                self._bilinear,
                self._mlp,
                msg_head.embeds[:, mol1:mol2],
                msg_head.f_mags[:, mol1:mol2],
                mask[:, mol1:mol2]
            )

        # 3. return output head
        f_mags = msg_head.f_mags.squeeze(-1)
        sigmas = msg_head.sigmas.squeeze(-1)
        return iq_accum, f_mags, sigmas
