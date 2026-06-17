import torch
from torch import nn
from jaxtyping import Float, jaxtyped
from beartype import beartype
from beartype.typing import Tuple
from ..batching import Batch
from .layer_head import LayerHead
from .intensity_mlp import IntensityMLP

class OutputHead(nn.Module):

    """
    OutputHead: collapses per-atom intensity contributions to a predicted I(q) curve.

    For each atom, IntensityMLP produces a scalar contribution per q-point.
    Those contributions are weighted by the atom's form factor magnitude and
    summed over all atoms to give the predicted I(q) for each molecule.
    """

    _mlp: IntensityMLP

    def __init__(
        self,
        lambda_1: int,
        lambda_3: int,
        lambda_4: int,
    ) -> None:
        """
        Args:
            lambda_1: atom embedding dimension
            lambda_3: hidden dimension of IntensityMLP
            lambda_4: number of halving steps in IntensityMLP compression
        """
        super().__init__()
        self._mlp = IntensityMLP(lambda_1, lambda_3, lambda_4)

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        batch: Batch,
        msg_head: LayerHead,
    ) -> Tuple[Float[torch.Tensor, "N Q"], Float[torch.Tensor, "N,M,Q"]]:
        """
        Args:
            batch:    input batch; uses padding_mask (N,M) to zero padded atoms
            msg_head: output of MessagePass; embeds (N,M,Q,λ₁), f_mags (N,M,Q,1)

        Returns:
            predicted I(q) per molecule: (N, Q)
        """
        # (N,M,Q,1) -> (N,M,Q)
        contribs = self._mlp(msg_head).squeeze(-1)

        # weight by form factor magnitude and zero out padded atoms
        f_mags = msg_head.f_mags.squeeze(-1)                          # (N,M,Q)
        mask   = batch.padding_mask().unsqueeze(-1)                   # (N,M,1)
        weighted = contribs * f_mags * mask                           # (N,M,Q)

        # sum atom contributions -> (N,Q), form factors -> (N,M,Q)
        return weighted.sum(dim=1), f_mags
