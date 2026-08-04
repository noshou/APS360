import math

import torch
from torch import nn

# Bound on how far the learned width may drift from `init_width`, in log
# units: `c = init_width * exp(_W_SPAN * tanh(raw_log_c))`, so
# `c / init_width` stays in `[exp(-1), exp(1)] = [0.368, 2.718]`.
#
# `c` is NOT left free. dg/dc = -tanh(u) + u*sech²(u) with u = y/c is odd
# with sign(dg/dc) = -sign(u), so raising `c` shrinks |g| for EVERY y with
# no opposing term: any loss that prefers a smaller update has a strictly
# monotone incentive to grow `c` forever. Measured under plain SGD on
# mean(g²), `c` went 1 -> 30 in 600 steps and was still climbing. Two
# things break as it does: dg/dlog_c -> -c in the linear regime, which is
# unbounded and then summed over Nc*mc*Q elements, and `y - c*tanh(y/c)`
# starts cancelling catastrophically (in fp16 it is identically 0 by
# c ~ 100). Bounding the ratio caps both.
_W_SPAN = 1.0


class PTanhShrink(nn.Module):
    """Bilinear layer followed by a width-parametric tanh shrink.

    ::

        y = bilinear(x1, x2)
        f(y) = y - c * tanh(y / c)

    `f` is a soft shrink toward 0: `f(0) = 0`, `f'(y) = tanh²(y / c)`, and
    `f'' = 2 tanh(u) sech²(u) / c`. Near the origin the response is
    **cubic**, `f(y) ~ y³ / (3c²)`, so a small bilinear output barely
    moves the result. That is the "sticky" behaviour the sigma update
    wants: the layer only acts once the bilinear has something decisive
    to say.

    Dead zone
    ---------
    `f'` is bounded in `[0, 1)` with a double zero at `y = 0`, so the
    shrink also gates its own gradient. Averaged uniformly over
    `|y| < c/2`, only **7.6%** of the gradient survives
    (`(2/c)∫₀^{c/2} tanh²(y/c) dy = 2(0.5 - tanh 0.5) = 0.0758`); at
    `|y| = c` it is 58%, and it reaches 99% only at `|y| = 3c`. `c`
    therefore sets a real threshold, not just a scale, which is why it is
    bounded rather than free (see `_W_SPAN`).

    For `|y| >> c` the layer is the identity minus a bounded offset,
    `f(y) -> y - c*sign(y)`, so it does NOT bound its own output. Anything
    downstream that needs a bounded result has to impose that itself.
    """

    bilinear: nn.Module
    raw_log_c: nn.Parameter

    def __init__(
        self,
        bilinear: nn.Module,
        out_features: int,
        init_width: float = 1.0,
    ) -> None:
        """Wrap a bilinear layer with a bounded per-channel shrink width.

        Parameters
        ----------
        bilinear : torch.nn.Module
            Any module mapping `(x1, x2)` to `(..., out_features)`. Taken
            as an already-constructed instance rather than built here so
            the caller picks the right bilinear form; `MessagePass` passes
            a `QDiagBilin`, which makes `out_features` the q-axis.
        out_features : int
            Width of `bilinear`'s trailing output axis. One shrink width
            is learned per channel.
        init_width : float, optional
            Starting value of `c`, and the centre of the range it is
            bounded to. Default 1.0.

        Returns
        -------
        None
        """

        super().__init__()
        self.bilinear = bilinear
        self._log_init_width = math.log(init_width)
        self.raw_log_c = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset `raw_log_c` to zero, in place, so `c == init_width`.

        Returns
        -------
        None
        """

        nn.init.zeros_(self.raw_log_c)

    @property
    def c(self) -> torch.Tensor:
        """The shrink width, bounded to `init_width * [e^-1, e^+1]`."""

        return torch.exp(
            self._log_init_width + _W_SPAN * torch.tanh(self.raw_log_c)
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Apply the bilinear form, then the shrink.

        Parameters
        ----------
        x1 : torch.Tensor
            First bilinear input.
        x2 : torch.Tensor
            Second bilinear input.

        Returns
        -------
        torch.Tensor
            Output, shape `(..., out_features)`.
        """

        y = self.bilinear(x1, x2)
        c = self.c
        return y - c * torch.tanh(y / c)
