import torch
import math

from torch import nn

class NoTrilinBilin(nn.Module):

    """Drop-in replacement for ``nn.Bilinear`` that avoids the `_trilinear` kernel.

    ``nn.Bilinear`` dispatches to a generic `_trilinear` autograd op that
    profiled as one of the two dominant CUDA kernels in this model (tied with
    `bmm`), and it doesn't specialize for the degenerate shapes this model
    actually uses (`in2_features=1` in `OutputHead`, `out_features=1` in
    `MessagePass`). This class computes the exact same bilinear form via one
    matmul (`x1 @ W`) plus an elementwise multiply-and-reduce over
    `in2_features`, routing through the same well-optimized GEMM kernel
    already dominant elsewhere in the model instead of a separate, less
    launch-efficient op. Mathematically identical to ``nn.Bilinear`` — not an
    approximation — and uses the same parameter init scheme, so it is a
    direct substitute at any call site.

    ``nn.Bilinear``'s `F.bilinear` also isn't supported by torch.compile's
    Inductor backend, so it forces a graph break whenever a compiled function
    calls it - splitting one fused compiled region into pieces around an
    eager fallback. This class uses only matmul/reshape/elementwise ops,
    all of which Inductor fuses natively, so swapping it in also lets
    torch.compile absorb the whole computation into its surrounding
    compiled graph instead of breaking around it.

    When to use it
    ---------------
    Use this in place of ``nn.Bilinear`` whenever `min(out_features,
    in2_features) == 1`, or is otherwise small - i.e. one side of the
    bilinear form is a near-scalar. Both current call sites are this shape:
    `OutputHead` (`in2_features=1`) and `MessagePass._sigbilin`
    (`out_features=1`). In that regime this class strictly wins: same math,
    same init, fewer/cheaper kernel launches (matmul + elementwise instead of
    `_trilinear`), and compile-friendly.

    When NOT to use it
    -------------------
    Avoid it when both `out_features` and `in2_features` are large. The
    intermediate `temp` tensor this class materializes has shape
    ``(..., out_features, in2_features)`` - `nn.Bilinear`'s fused kernel
    never allocates that full tensor, so on a genuine (large, large) bilinear
    form, this decomposition costs strictly more memory for no compute
    benefit. Profile before reusing this class outside its current call
    sites; don't assume the win here generalizes.
    """

    weight: nn.Parameter
    bias:   "nn.Parameter | None"

    def __init__(
        self,
        in1_features: int,
        in2_features: int,
        out_features: int,
        bias:         bool = True,
    ) -> None:

        """Construct the layer and initialize its parameters.

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
        self.weight = nn.Parameter(torch.empty(
            out_features, 
            in1_features, 
            in2_features
            ))
        self.bias   = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:

        """Reset `weight` and `bias` in place, matching `nn.Bilinear`'s scheme.

        Both tensors are drawn from `Uniform(-bound, bound)` with
        `bound = 1 / sqrt(in1_features)`, identical to `nn.Bilinear`'s own
        `reset_parameters`, so a freshly constructed model trains from the
        same initialization distribution regardless of which class is used.

        Returns
        -------
        None
        """

        bound = 1 / math.sqrt(self.in1_features)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:

        """Compute the bilinear form `y = x1 @ W @ x2 + bias` without `_trilinear`.

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

        # Permute (out, in1, in2) -> (out, in2, in1) BEFORE flattening to
        # (out*in2, in1) - a plain reshape without the permute would chop the
        # flat buffer without regrouping by in2, silently computing the wrong
        # thing whenever in2_features > 1 (masked when in2_features == 1,
        # where permuting a size-1 axis is a no-op - caught via parity test
        # against nn.Bilinear on the out_features=1 case, not the in2=1 case).
        W = self.weight.permute(0, 2, 1).reshape(
            self.out_features * self.in2_features, 
            self.in1_features
            )

        # (..., in1) @ (in1, out*in2) -> (..., out*in2) -> (..., out, in2)
        temp = (x1 @ W.T).reshape(*x1.shape[:-1], self.out_features, self.in2_features)

        # Elementwise multiply by x2 (broadcast over out) and reduce over in2 -
        # the remaining contraction nn.Bilinear's kernel does internally.
        y = (temp * x2.unsqueeze(-2)).sum(-1)
        return y + self.bias if self.bias is not None else y
