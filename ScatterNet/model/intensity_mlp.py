import torch
from torch import nn
from jaxtyping import Float, jaxtyped
from beartype import beartype
from .layer_head import LayerHead
from numpy import log2, floor
from collections import OrderedDict
import torch.nn.functional as F

class IntensityMLP(nn.Module):

    """
    Per-atom intensity contribution network.

    Maps each atom's learned embedding and form factor magnitude to a scalar
    contribution to I(q), using a bilinear interaction followed by a halving MLP.

    Architecture:
        Bilinear(λ₁, 1 → λ₃) → Mish → Linear(λ₃ → λ₃//2) → ... → Linear(· → 1) → Softplus

    The bilinear layer captures interactions between the atom's identity (λ₁)
    and its scattering strength (f_mag scalar) before compression to a scalar vote.
    """

    _bilinear: nn.Bilinear  # maps (λ₁, 1) -> λ₃
    _mlp:      nn.Sequential # halving compression: λ₃ -> 1

    def __init__(
        self,
        lambda_1: int,
        lambda_3: int,
        lambda_4: int,
    ) -> None:
        """
        Args:
            lambda_1: atom embedding dimension (input to bilinear)
            lambda_3: hidden dimension after bilinear interaction
            lambda_4: number of halving steps from λ₃ to 1; must satisfy
                      0 < lambda_4 <= floor(log2(lambda_3))
        """
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
    def forward(self, msg_head: LayerHead) -> Float[torch.Tensor, "N M Q 1"]:
        """
        Args:
            msg_head: output of MessagePass; uses embeds (N,M,Q,λ₁) and f_mags (N,M,Q,1)

        Returns:
            per-atom scalar contribution to I(q): (N, M, Q, 1)
        """
        # (N,M,Q,λ₁) x (N,M,Q,1) -> (N,M,Q,λ₃)
        atomic_contribs = F.mish(self._bilinear(msg_head.embeds, msg_head.f_mags))

        # (N,M,Q,λ₃) -> (N,M,Q,1)
        return F.softplus(self._mlp(atomic_contribs))
