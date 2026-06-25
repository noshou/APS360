import torch

from torch                  import nn
from torch.utils.checkpoint import checkpoint
from jaxtyping              import Float, jaxtyped
from beartype               import beartype
from beartype.typing        import Tuple
from .batching              import Batch
from .model                 import Embed, MessagePass, OutputHead

class ScatterNet(nn.Module):

    """
    Full ScatterNet pipeline: Embed -> MessagePass -> OutputHead -> I(q).

    Both MessagePass and OutputHead use gradient checkpointing to reduce peak
    activation memory. This allows training on large molecules without OOM at
    the cost of recomputing intermediates during backward.

    Hyperparameters:
        lambda_1:  atom embedding dimension
        lambda_2:  number of message passing rounds
        lambda_3:  OutputHead hidden width
        lambda_4:  number of halving steps in OutputHead MLP
        lambda_5:  number of RFF features in MessagePass
        atm_chunk: atoms per chunk in MessagePass and OutputHead; peak tensor is (N, atm_chunk, Q, lambda_5)
        q_points:  number of q-grid points (Q)
        eps_embd:  numerical floor for Embed (avoids division by zero)
        eps_msgp:  numerical floor for MessagePass (avoids division by zero)
    """

    _emd:      Embed
    _msg:      MessagePass
    _out:      OutputHead
    _eps_embd: Float 
    _eps_msgp: Float
    
    def __init__(
        self,
        lambda_1: int,
        lambda_2: int,
        lambda_3: int,
        lambda_4: int,
        lambda_5: int,
        msg_seed: int,
        atm_chunk: int,
        q_points: int,
        eps_embd: float,
        eps_msgp: float
    ) -> None:

        super().__init__()
        self._emb      = Embed(lambda_1, q_points)
        self._msg      = MessagePass(lambda_1, lambda_2, lambda_5, msg_seed, q_points, atm_chunk)
        self._out      = OutputHead(lambda_1, lambda_3, lambda_4, atm_chunk)
        self._eps_embd = eps_embd
        self._eps_msgp = eps_msgp
        
    @jaxtyped(typechecker=beartype)
    def forward(self, batch: Batch) -> Tuple[
            Float[torch.Tensor, "N Q"], 
            Float[torch.Tensor, "N M Q"],
            Float[torch.Tensor, "N M Q"]
        ]:
        
        """
        Args:
            batch: input batch of molecules
        Returns:
            iq:     predicted I(q) per molecule,       shape (N, Q)
            f_mags: per-atom form factor magnitudes,   shape (N, M, Q)
            sigmas: per-atom gaussian kernel bandwith, shape (N, M, Q)
        """
        
        embed_head = self._emb(batch, self._eps_embd)
        msg_head   = self._msg(batch, embed_head, self._eps_msgp)
        return       self._out(batch, msg_head)
