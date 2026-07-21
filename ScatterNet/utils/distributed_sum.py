import torch
import torch.distributed as dist
from torch.autograd.function import FunctionCtx as FuncCtx


class DistributedSum(torch.autograd.Function):
    """
    All-reduce (SUM) in the forward pass; identity in the backward pass.

    Used for partial I(q) sums from each rank's atom shard. Since both ranks
    compute the same loss after the all-reduce, ∂L/∂partial_iq is already the
    global gradient on every rank - no backward communication is needed.
    """

    @staticmethod
    def forward(ctx: FuncCtx, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Sum-reduce `x` across all ranks in the process group.

        Parameters
        ----------
        ctx : torch.autograd.function.FunctionCtx
            Autograd context (unused; nothing needs to be saved for backward).
        x : torch.Tensor
            Local partial tensor (e.g. a partial I(q) sum) to be summed
            across ranks.

        Returns
        -------
        torch.Tensor
            `x` all-reduced (summed) across every rank in the process group.
        """
        x = x.clone()
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        return x

    @staticmethod
    def backward(ctx: FuncCtx, grad: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Pass the incoming gradient through unchanged.

        Parameters
        ----------
        ctx : torch.autograd.function.FunctionCtx
            Autograd context (unused).
        grad : torch.Tensor
            Gradient of the loss with respect to this function's output.

        Returns
        -------
        torch.Tensor
            `grad` unchanged; the gradient is already global on every rank
            since all ranks compute the same loss after the forward all-reduce.
        """
        return grad
