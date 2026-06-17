import torch
from torch               import nn
from jaxtyping           import Float, jaxtyped
from beartype            import beartype
from ScatterNet.batching import Batch
from ScatterNet.model    import Embed, MessagePass, OutputHead


class ScatterNet(nn.Module):

    """
    Full ScatterNet pipeline: Embed → MessagePass → OutputHead → I(q).

    Hyperparameters
    ---------------
    lambda_1 : atom embedding dimension
    lambda_2 : number of message-passing rounds
    lambda_3 : IntensityMLP hidden width
    lambda_4 : number of halving steps in IntensityMLP
    lambda_5 : number of RFF features in MessagePass
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
        lambda_5: int,
        q_points: int,
        msg_seed: int
    ) -> None:
        super().__init__()
        self._embed = Embed(lambda_1, q_points)
        self._msg   = MessagePass(lambda_1, lambda_2, lambda_5, msg_seed)
        self._out   = OutputHead(lambda_1, lambda_3, lambda_4)

    @jaxtyped(typechecker=beartype)
    def forward(self, batch: Batch) -> Float[torch.Tensor, "N Q"]:
        """
        Args:
            batch: input batch of molecules
        Returns:
            predicted I(q) per molecule (N x Q).
        """
        embed_head = self._embed(batch)
        msg_head   = self._msg(batch, embed_head)
        return self._out(batch, msg_head)
