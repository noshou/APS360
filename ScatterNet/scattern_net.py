import torch
from torch import nn
from jaxtyping import Float, jaxtyped
from beartype import beartype
from beartype.typing import Tuple
from .batching import Batch
from .model import Embed, MessagePass, OutputHead


class ScatterNet(nn.Module):

    """
    Full ScatterNet pipeline: Embed → MessagePass → OutputHead → I(q).

    Hyperparameters
    ---------------
    lambda_1 : atom embedding dimension
    lambda_2 : number of message-passing rounds
    lambda_3 : IntensityMLP hidden width
    lambda_4 : number of halving steps in IntensityMLP
    qPoints  : number of q-grid points (Q)
    """

    _embed:   Embed
    _msg:     MessagePass
    _out:     OutputHead

    def __init__(
        self,
        lambda_1: int,
        lambda_2: int,
        lambda_3: int,
        lambda_4: int,
        qPoints:  int,
    ) -> None:
        super().__init__()
        self._embed = Embed(lambda_1, qPoints)
        self._msg   = MessagePass(lambda_1, lambda_2, qPoints)
        self._out   = OutputHead(lambda_1, lambda_3, lambda_4)

    @jaxtyped(typechecker=beartype)
    def forward(self, batch: Batch) -> Tuple[Float[torch.Tensor, "N Q"], Float[torch.Tensor, "N M Q"]]:
        """
        Args:
            batch: input batch of molecules
        Returns:
            predicted I(q) per molecule (N x Q), and the sigma values (N x M x Q).
        """
        embed_head = self._embed(batch)
        msg_head   = self._msg(batch, embed_head)
        return self._out(batch, msg_head), msg_head.sigmas.squeeze(-1)
