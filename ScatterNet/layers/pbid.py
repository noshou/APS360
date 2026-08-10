import math

import torch
from torch import nn

# Hard cap on |w|. Monotonicity needs |w| < 2 (see the class docstring), and
# `OutputHead`'s init inverse divides by `2*(1 - w²/4)`, which is 0 at
# |w| = 2 and underflows to 0 in fp16 once |w - 2| < 1e-4. Capping strictly
# at 1.9 the smallest denominator is 0.195 and the smallest slope is 0.05.
_W_MAX = 1.9


class PBId(nn.Module):
    """Bent identity with a learned, bounded per-feature bend.

    ::

        f(x) = w * (sqrt(x² + 1) - 1) / 2 + x + b
        w    = _W_MAX * tanh(raw_weight)

    `w = 0` is the exact identity; `w = 1` is the standard bent identity.
    `w` controls only how much bend is applied so the layer interpolates
    continuously between "no nonlinearity" and "bent identity" rather than
    rescaling the whole function.

    Monotonicity
    -------------
    ``f'(x) = w * x / (2 * sqrt(x² + 1)) + 1``, and `x / sqrt(x² + 1)`
    lies in (-1, 1), so ``f'`` lies in the open interval
    ``(1 - |w|/2, 1 + |w|/2)``; the endpoints are infima approached only
    as ``x -> -+inf``, never attained at finite `x`. `f` is therefore
    strictly increasing for ``|w| <= 2``.

    Beyond ``|w| = 2`` the function folds: at ``|w| = 2.5`` there is a
    turning point at ``x = -(2/|w|)/sqrt(1 - (2/|w|)²) = -4/3``
    where ``f' = 0``, and ``f'`` bottoms out at an infimum of -0.25 far
    out on the negative branch. `w` is therefore bounded to
    ``|w| <= _W_MAX = 1.9``.

    Layout
    -------
    Features are taken on the **last** axis.
    """

    raw_weight: nn.Parameter
    bias: "nn.Parameter | None"
    _in_feats: int

    def __init__(self, in_feats: int, bias: bool = True) -> None:
        """Build the bend weight and optional bias.

        Parameters
        ----------
        in_feats : int
            Size of the trailing (feature) axis. Use 1 for a single bend
            shared across the whole channel.
        bias : bool, optional
            If True (default), adds a learnable additive bias of shape
            `(in_feats,)`.

        Returns
        -------
        None
        """

        super().__init__()
        self._in_feats = in_feats
        self.raw_weight = nn.Parameter(torch.empty(in_feats))
        self.bias = nn.Parameter(torch.empty(in_feats)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset `weight` to one and `bias` to zero, in place.

        `raw_weight` is set to `atanh(1 / _W_MAX)` so that the bounded
        `weight` starts at exactly 1, the standard bent identity. `b = 0`
        leaves `f(0) = 0`, so a zero pre-activation maps to a zero
        coherent weight and the channel starts able to cancel.

        Returns
        -------
        None
        """

        nn.init.constant_(self.raw_weight, math.atanh(1.0 / _W_MAX))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    @property
    def weight(self) -> torch.Tensor:
        """The bend strength `w`, bounded to `(-_W_MAX, _W_MAX)`."""

        return _W_MAX * torch.tanh(self.raw_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the bent identity elementwise.

        Parameters
        ----------
        x : torch.Tensor
            Input, shape `(..., in_feats)`.

        Returns
        -------
        torch.Tensor
            Output, same shape as `x`.
        """

        # A (C,) weight broadcasts against a (..., C) input with no
        # reshape at any rank, so there is no channels-first branch.
        bent = (torch.sqrt(x * x + 1.0) - 1.0) / 2.0
        out = self.weight * bent + x
        return out if self.bias is None else out + self.bias
