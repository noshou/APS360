import torch
import torch.distributed as dist
from torch.autograd.function import FunctionCtx as FuncCtx


class AllGatherDim1(torch.autograd.Function):
    """
    Gather per-atom tensors from all ranks along the atom dimension (dim 1).

    Forward: each rank contributes its shard; all ranks receive the
    full M tensor. Padding is used when the last rank's shard
    is smaller than ceil(M / world_size).

    Backward: each rank receives only the gradient slice for its own atoms.
    No cross-rank communication is needed in the backward pass because the
    gradient is already global (loss was computed identically on all ranks).
    """

    @staticmethod
    def forward(
        ctx: FuncCtx,
        x: torch.Tensor,
        m0: int,
        M_full: int,
    ) -> torch.Tensor:
        """Gather per-atom shards from every rank along the atom dimension.

        Parameters
        ----------
        ctx : torch.autograd.function.FunctionCtx
            Autograd context; used to stash `m0` and the local shard width
            (`M_local`) for `backward`.
        x : torch.Tensor
            This rank's local shard, shape (N, M_local, ...).
        m0 : int
            Starting atom index of this rank's shard within the full
            `M_full` atoms.
        M_full : int
            Total (unsharded) number of atoms across all ranks.

        Returns
        -------
        torch.Tensor
            Tensor of shape (N, M_full, ...) with every rank's shard
            concatenated along dim 1, with any padding trimmed off.
        """
        ws = dist.get_world_size()
        N = x.shape[0]
        M_local = x.shape[1]
        rest = x.shape[2:]
        max_shard = (M_full + ws - 1) // ws

        if M_local < max_shard:
            pad = x.new_zeros(N, max_shard - M_local, *rest)
            x_pad = torch.cat([x, pad], dim=1)
        else:
            x_pad = x

        gathered = [torch.zeros_like(x_pad) for _ in range(ws)]
        dist.all_gather(gathered, x_pad.contiguous())

        ctx.m0 = m0  # pyright: ignore[reportAttributeAccessIssue]
        ctx.M_local = M_local  # pyright: ignore[reportAttributeAccessIssue]
        return torch.cat(gathered, dim=1)[:, :M_full]

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: FuncCtx, grad: torch.Tensor
    ) -> tuple[torch.Tensor, None, None]:
        """Slice out the gradient belonging to this rank's atom shard.

        Parameters
        ----------
        ctx : torch.autograd.function.FunctionCtx
            Autograd context holding `m0` and `M_local` saved in `forward`.
        grad : torch.Tensor
            Gradient of the loss with respect to the gathered
            (N, M_full, ...) output.

        Returns
        -------
        torch.Tensor
            Gradient slice for this rank's own atoms, shape
            (N, M_local, ...).
        None
            Placeholder gradient for the `m0` argument (not a tensor).
        None
            Placeholder gradient for the `M_full` argument (not a tensor).
        """
        m0 = ctx.m0  # pyright: ignore[reportAttributeAccessIssue]
        m_local = ctx.M_local  # pyright: ignore[reportAttributeAccessIssue]
        return grad[:, m0 : m0 + m_local].contiguous(), None, None
