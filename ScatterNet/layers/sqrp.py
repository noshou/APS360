import math

import torch
from torch import nn

# Floor on `exp(_b)`. `_b` is unconstrained, and once `exp(_b)` underflows to
# exactly 0 the forward becomes `(x + sqrt(x**2))/2`, whose derivative at
# x = 0 is `0 * inf = NaN`. fp32 underflows at `_b ~ -104`, fp16 at
# `_b ~ -17.3`, both reachable by an unregularized parameter.
_B_FLOOR = 1e-12


class SqrP(nn.Module):
    """Square-plus activation with a learned width.

    ::

        f(x) = (x + sqrt(x**2 + b)) / 2,     b = exp(_b) > 0

    A smooth, strictly positive approximation to `max(x, 0)`:
    `f'(x) = (1 + x / sqrt(x**2 + b)) / 2` lies in `(0, 1)` and
    `f''(x) = b / (2 (x**2 + b)**1.5) > 0`, so `f` is strictly increasing
    and strictly convex with no dead unit anywhere.

    Tail behaviour vs softplus
    --------------------------
    `f(x) -> x + b/(4x)` as `x -> +inf` and `f(x) -> b/(4|x|)` as `x -> -inf`.
    The negative tail decays as `1/|x|`: reaching an output of 1e-6 at `b = 4`
    needs `x = -1e6`, where softplus needs only `x = -13.8`. Use this where a
    strictly positive output of order 1 is wanted and exact zero is not; do not
    use it where the network must be able to switch a channel off.

    Numerical form
    --------------
    `forward` evaluates the `x < 0` branch as `b / (2 (sqrt(x**2 + b) - x))`
    rather than `(x + sqrt(x**2 + b)) / 2`. The two are algebraically
    identical (multiply by the conjugate), but the second subtracts two
    nearly equal positives and cancels catastrophically: in fp32 at
    `b = 4` it returns 2.441e-4 for the true 2.000e-4 at `x = -5e3`, and
    exactly 0.0 for `x <~ -1e4`, which then makes any `1 / f(x)` infinite.
    """

    _b: "nn.Parameter | None"

    def __init__(self, bias: bool = True) -> None:
        """Build the optional learnable width.

        Parameters
        ----------
        bias : bool, optional
            If True (default), `b` is learnable via `_b = log(b)`. If
            False, `b` is fixed at 4.0.

        Returns
        -------
        None
        """

        super().__init__()
        if bias:
            self._b = nn.Parameter(torch.empty(1))
        else:
            self.register_parameter("_b", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset `_b` to log(4.0), in place.

        `b = 4` puts `f(0) = sqrt(b)/2 = 1`, so a zero
        pre-activation maps to a unit output.

        Returns
        -------
        None
        """

        if self._b is not None:
            nn.init.constant_(self._b, math.log(4.0))

    @property
    def b(self) -> "torch.Tensor | None":
        """The width `b = exp(_b)`, floored, or None if not learnable."""

        if self._b is None:
            return None
        return torch.exp(self._b).clamp(min=_B_FLOOR)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply square-plus elementwise.

        Parameters
        ----------
        x : torch.Tensor
            Input, any shape.

        Returns
        -------
        torch.Tensor
            Output, same shape as `x`, strictly positive.
        """

        b = self.b
        if b is None:
            b = torch.full((), 4.0, dtype=x.dtype, device=x.device)

        # Conjugate form on the cancelling branch only; see class docstring.
        # `torch.where` evaluates both branches, and its backward hands each
        # one `grad * mask`, so a branch that produces inf on the side it is
        # not selected on still poisons the gradient via `0 * inf = NaN`.
        # Clamping x at 0 for the conjugate branch keeps `root_n - x_n >=
        # sqrt(b)` everywhere, so neither branch can blow up on the inputs it
        # is evaluated at.
        x_neg = x.clamp(max=0.0)
        root_n = torch.sqrt(x_neg * x_neg + b)
        return torch.where(
            x < 0.0,
            b / (2.0 * (root_n - x_neg)),
            (x + torch.sqrt(x * x + b)) / 2.0,
        )
