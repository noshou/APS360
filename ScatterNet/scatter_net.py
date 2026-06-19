import torch

from torch           import nn
from jaxtyping       import Float, jaxtyped
from beartype        import beartype
from beartype.typing import Tuple
from .batching       import Batch
from .model          import Embed, MessagePass, OutputHead

class ScatterNet(nn.Module):

    """
    Full ScatterNet pipeline: Embed -> MessagePass -> OutputHead -> I(q).

    Hyperparameters
    ---------------
    lambda_1: atom embedding dimension
    lambda_2: number of message-passing rounds
    lambda_3: OutputHead hidden width
    lambda_4: number of halving steps in OutputHead MLP
    lambda_5: number of RFF features in MessagePass
    q_points: number of q-grid points (Q)
    eps_embd: numerical floor for Embed (avoids division by zero)
    eps_msgp: numerical floor for MessagePass (avoids division by zero)
    """

    _embed:    Embed
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
        q_points: int,
        eps_embd: float,
        eps_msgp: float
    ) -> None:
        
        super().__init__()
        self._embed    = Embed(lambda_1, q_points)
        self._msg      = MessagePass(lambda_1, lambda_2, lambda_5, msg_seed, q_points)
        self._out      = OutputHead(lambda_1, lambda_3, lambda_4)
        self._eps_embd = eps_embd
        self._eps_msgp = eps_msgp
        
    @jaxtyped(typechecker=beartype)
    def forward(self, batch: Batch) -> Tuple[Float[torch.Tensor, "N Q"], Float[torch.Tensor, "N M Q"]]:
        """
        Args:
            batch: input batch of molecules
        Returns:
            iq:     predicted I(q) per molecule, shape (N, Q)
            f_mags: per-atom form factor magnitudes, shape (N, M, Q)
        """
        embed_head = self._embed(batch, self._eps_embd)
        msg_head   = self._msg(batch, embed_head, self._eps_msgp)
        return self._out(batch, msg_head)
