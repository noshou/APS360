import torch
from torch import log
from jaxtyping import Float, jaxtyped
from beartype import beartype
from ScatterNet import Batch

@jaxtyped(typechecker=beartype)
def _MLSE(batch: Batch, calc: Float[torch.Tensor, "N Q"]) -> Float[torch.Tensor, ""]:
    
    ln_truth = torch.log(batch.iqval) # (N x Q)
    ln_calcs = torch.log(calc)        # (N x Q)
 
    N, Q = ln_truth.shape[0], ln_truth.shape[1]
    
    # mlse_NQ = (1/Q) Σ_n Σ_i (log Î(q)_n,i − log I(q)_n,i)²
    # has shape [N]
    mlse_Q = ((ln_calcs - ln_truth)**2).sum(dim=1) / Q
    
    # mlse_N = (1/N) Σ (mlse_Q)
    # scalar
    mlse_N = mlse_Q.sum(dim=0) / N
    
    return mlse_N

def _sigma_L1()