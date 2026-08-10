import math

import torch
from torch import nn


class NoTrilinBilin(nn.Module):

    weight: nn.Parameter
    bias: "nn.Parameter | None"

    def __init__(
        self,
        in1_features: int,
        in2_features: int,
        out_features: int,
        bias: bool = True,
    ) -> None:
        """Mathematically identical replacement
        for ``nn.Bilinear`` that avoids the `_trilinear` kernel.

        ``nn.Bilinear`` dispatches to a generic `_trilinear` autograd op that
        can be very inneficient with small input features due to CUDA
        kernel launch overheads. This class computes the exact same bilinear
        form via one matmul (`x1 @ W`) plus an elementwise multiply-and-reduce
        over `in2_features`, routing through the well-optimized GEMM kernel.

        When to use it
        ---------------
        Use this in place of ``nn.Bilinear`` whenever `min(out_features,
        in2_features) == 1`, or is otherwise small - i.e. one side of the
        bilinear form is a near-scalar.

        When NOT to use it
        -------------------
        Avoid it when both `out_features` and `in2_features` are large. The
        intermediate `temp` tensor this class materializes has shape
        ``(..., out_features, in2_features)`` which needs to be allocated.
        `nn.Bilinear`'s fused kernel never allocates that full tensor, so on a
        genuine (large, large) bilinear form, this decomposition costs
        strictly more memory for no compute benefit. Profile before reusing
        this class outside its current call sites;
        don't assume the win here generalizes.

        Parameters
        ----------
        in1_features : int
            Size of the first input, `x1`.
        in2_features : int
            Size of the second input, `x2`.
        out_features : int
            Size of the output.
        bias : bool, optional
            If True (default), adds a learnable additive bias of shape
            `(out_features,)`.

        Returns
        -------
        None
        """

        super().__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(out_features, in1_features, in2_features)
        )
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset `weight` and `bias` in place, from the true bilinear fan-in.

        Both tensors are drawn from `Uniform(-bound, bound)` with
        `bound = 1 / sqrt(in1_features * in2_features)`.

        `nn.Bilinear` (and this class, previously) uses
        `1 / sqrt(in1_features)`, which ignores `in2_features` even though
        the forward sums `in1_features * in2_features` products into every
        output. That leaves the output variance growing linearly in
        `in2_features`: at `in2_features = 1` the two bounds agree exactly,
        but at the `in2_features = Q = 51` call sites it inflated the
        pre-activation std from 4.0 to 28.4 (max 147), which is a large
        constant sitting in front of every sigma gradient.

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
            First input, shape `(..., in1_features)`.
        x2 : torch.Tensor
            Second input, shape `(..., in2_features)`; must share `x1`'s
            leading (batch) dimensions.

        Returns
        -------
        torch.Tensor
            Output, shape `(..., out_features)`.
        """

        # Permute (out, in1, in2) -> (out, in2, in1) before flattening to
        # (out*in2, in1). A plain reshape without the permute would chop the
        # flat buffer without regrouping by in2, silently computing the wrong
        # thing whenever in2_features > 1.
        W = self.weight.permute(0, 2, 1).reshape(
            self.out_features * self.in2_features, self.in1_features
        )

        # (..., in1) @ (in1, out*in2) -> (..., out*in2) -> (..., out, in2)
        temp = (x1 @ W.T).reshape(
            *x1.shape[:-1], self.out_features, self.in2_features
        )

        # Elementwise multiply by x2 (broadcast over out) and reduce over in2
        y = (temp * x2.unsqueeze(-2)).sum(-1)
        return y + self.bias if self.bias is not None else y
