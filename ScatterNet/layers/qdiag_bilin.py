import math

import torch
from torch import nn


class QDiagBilin(nn.Module):
    """Bilinear form with one weight matrix per q-point.

    ::

        y[..., q] = Σ_a Σ_b W[q, a, b] · x1[..., q, a] · x2[..., b] + bias[q]

    """

    weight: nn.Parameter
    bias: "nn.Parameter | None"

    def __init__(
        self,
        in1_features: int,
        in2_features: int,
        q_points: int,
        bias: bool = True,
    ) -> None:
        """Build the per-q weight stack and optional per-q bias.

        Parameters
        ----------
        in1_features : int
            Size of `x1`'s trailing axis.
        in2_features : int
            Size of `x2`'s trailing axis.
        q_points : int
            Number of q-points; both the output width and the length of
            `x1`'s second-to-last axis.
        bias : bool, optional
            If True (default), adds a learnable bias of shape
            `(q_points,)`.

        Returns
        -------
        None
        """

        super().__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.q_points = q_points
        self.out_features = q_points
        self.weight = nn.Parameter(
            torch.empty(q_points, in1_features, in2_features)
        )
        self.bias = nn.Parameter(torch.empty(q_points)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset `weight` and `bias` in place, from the true fan-in.

        Every output sums `in1_features * in2_features` products, so the
        bound is `1 / sqrt(in1_features * in2_features)` rather than
        `nn.Bilinear`'s `1 / sqrt(in1_features)`, which would leave the
        output variance growing linearly in `in2_features`.

        Returns
        -------
        None
        """

        bound = 1 / math.sqrt(self.in1_features * self.in2_features)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x1 : torch.Tensor
            First input, shape `(..., q_points, in1_features)`.
        x2 : torch.Tensor
            Second input, shape `(..., in2_features)`, sharing `x1`'s
            leading (batch) dimensions but WITHOUT the q axis.

        Returns
        -------
        torch.Tensor
            Output, shape `(..., q_points)`.
        """

        # Contract in1 first: (..., Q, in1) x (Q, in1, in2) -> (..., Q, in2).
        # This is the whole point of the class; see the class docstring.
        temp = torch.einsum("qab,...qa->...qb", self.weight, x1)

        # Elementwise multiply by x2 (broadcast over Q) and reduce over in2.
        y = (temp * x2.unsqueeze(-2)).sum(-1)
        return y + self.bias if self.bias is not None else y
