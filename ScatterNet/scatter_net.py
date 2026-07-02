import torch
import torch.distributed as dist

from dataclasses        import replace as _dc_replace
from torch              import nn
from jaxtyping          import Float, jaxtyped
from beartype           import beartype
from beartype.typing    import Tuple
from .batching          import Batch
from .model             import Embed, MessagePass, OutputHead


class _DistributedSum(torch.autograd.Function):
    """
    All-reduce (SUM) in the forward pass; identity in the backward pass.

    Used for partial I(q) sums from each rank's atom shard. Since both ranks
    compute the same loss after the all-reduce, ∂L/∂partial_iq is already the
    global gradient on every rank - no backward communication is needed.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = x.clone()
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        return x

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return grad


class _AllGatherDim1(torch.autograd.Function):
    """
    Gather per-atom tensors from all ranks along the atom dimension (dim 1).

    Forward: each rank contributes its shard; all ranks receive the full M tensor.
    Padding is used when the last rank's shard is smaller than ceil(M / world_size).

    Backward: each rank receives only the gradient slice for its own atoms.
    No cross-rank communication is needed in the backward pass because the
    gradient is already global (loss was computed identically on all ranks).
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        x:      torch.Tensor,
        m0:     int,
        M_full: int,
    ) -> torch.Tensor:
        ws       = dist.get_world_size()
        N        = x.shape[0]
        M_local  = x.shape[1]
        rest     = x.shape[2:]
        max_shard = (M_full + ws - 1) // ws

        if M_local < max_shard:
            pad   = x.new_zeros(N, max_shard - M_local, *rest)
            x_pad = torch.cat([x, pad], dim=1)
        else:
            x_pad = x

        gathered = [torch.zeros_like(x_pad) for _ in range(ws)]
        dist.all_gather(gathered, x_pad.contiguous())

        ctx.m0      = m0
        ctx.M_local = M_local
        return torch.cat(gathered, dim=1)[:, :M_full]

    @staticmethod
    def backward(ctx, grad: torch.Tensor):  # type: ignore[override]
        return grad[:, ctx.m0 : ctx.m0 + ctx.M_local].contiguous(), None, None


class ScatterNet(nn.Module):

    """
    Full ScatterNet pipeline: Embed → MessagePass → OutputHead → I(q).

    When a distributed process group is active (torch.distributed initialised)
    and the model is training, each batch is routed to one of two parallelism
    strategies based on M (padded atoms per molecule), N (molecule count), and
    `dp_atom_threshold`:

    Tensor-parallel (TP) - the default; used whenever DP's conditions (below)
    aren't met, or dp_atom_threshold <= 0, or not self.training (eval always
    uses this path):
        1. Embed runs on the full batch on every rank (cheap, per-atom lookup).
        2. Each rank slices its atom shard from embed_head.
        3. MessagePass runs on the shard; an all-reduce inside MessagePass
           reconstructs the global RFF context (features, chem_env) between
           the two passes so every atom still attends to all others.
        4. OutputHead produces partial I(q) and per-atom f_mags/sigmas for
           the shard.
        5. I(q) is summed across ranks (all-reduce); f_mags and sigmas are
           gathered back to full M.

    Data-parallel (DP) - training, M < dp_atom_threshold, AND N >= 2*mol_chunk:
        Molecules are divided across ranks (no atom sharding, no in-model
        communication at all - small M means TP's all-reduce cost would
        dwarf the tiny amount of per-rank compute it parallelises). Each
        rank runs the ordinary single-GPU forward on its own molecule slice
        and returns a `loss_scale = local_N / global_N` so that, once the
        training loop's existing grad all-reduce (SUM) runs, summing the
        two ranks' scaled-local-mean gradients reconstructs the exact
        global-mean gradient - the same mechanism already used to combine
        TP's partial gradients, just fed a differently-scaled loss.

        The `N >= 2*mol_chunk` guard matters: DP halves the outer N-chunk
        loop but does NOT halve M before MessagePass's own atm_chunk-loop
        runs (TP does, by sharding M first). So a DP-routed bucket runs
        ~2x the inner M-chunk-loop launches TP would've had on the same
        bucket - only worth it if halving N actually shrinks the outer
        loop. If N already fits in one N-chunk, DP is pure overhead.

    Without a process group the forward is identical to the single-GPU path.

    Hyperparameters:
        lambda_1:  atom embedding dimension
        lambda_2:  number of message passing rounds
        lambda_3:  OutputHead hidden width
        lambda_4:  number of halving steps in OutputHead MLP
        lambda_5:  number of RFF features in MessagePass
        atm_chunk: atoms per M-chunk in MessagePass and OutputHead
        mol_chunk: molecules per N-chunk in MessagePass (controls chem_env peak size)
        dp_atom_threshold: see class docstring; 0 = always TP
        q_points:  number of q-grid points (Q)
        eps_embd:  numerical floor for Embed
        eps_msgp:  numerical floor for MessagePass
        compile:   torch.compile Embed/MessagePass/OutputHead's checkpointed step functions
    """

    _emb:      Embed
    _msg:      MessagePass
    _out:      OutputHead
    _eps_embd: Float
    _eps_msgp: Float
    _dp_atom_threshold: int
    _mol_chunk:         int

    def __init__(
        self,
        lambda_1:  int,
        lambda_2:  int,
        lambda_3:  int,
        lambda_4:  int,
        lambda_5:  int,
        msg_seed:  int,
        atm_chunk: int,
        mol_chunk: int,
        q_points:  int,
        eps_embd:  float,
        eps_msgp:  float,
        dp_atom_threshold: int = 0,
        compile:   bool = False,
    ) -> None:

        super().__init__()
        self._emb      = Embed(lambda_1, q_points, compile=compile)
        self._msg      = MessagePass(lambda_1, lambda_2, lambda_5, msg_seed, q_points, n_chunk=mol_chunk, m_chunk=atm_chunk, compile=compile)
        self._out      = OutputHead(lambda_1, lambda_3, lambda_4, atm_chunk, compile=compile)
        self._eps_embd = eps_embd
        self._eps_msgp = eps_msgp
        self._dp_atom_threshold = dp_atom_threshold
        self._mol_chunk = mol_chunk

    @jaxtyped(typechecker=beartype)
    def forward(self, batch: Batch) -> Tuple[
            Float[torch.Tensor, "N Q"],
            Float[torch.Tensor, "N M Q"],
            Float[torch.Tensor, "N M Q"],
            Batch,
            float,
        ]:

        """
        Args:
            batch: input batch of molecules
        Returns:
            iq:          predicted I(q),                     shape (N, Q)
            f_mags:      per-atom form factor magnitudes,    shape (N, M, Q)
            sigmas:      per-atom gaussian kernel bandwidth,  shape (N, M, Q)
            local_batch: the slice of `batch` these outputs correspond to.
                         Equal to `batch` unchanged unless DP-routed this
                         batch (in which case N above is the local molecule
                         count, not the global one) - pass this to the loss,
                         not the original `batch`.
            loss_scale:  local_N / global_N when DP-routed; 1.0 otherwise.
                         Multiply the loss by this before backward().
        """

        embed_head = self._emb(batch, self._eps_embd)
        dist_on    = dist.is_available() and dist.is_initialized()

        if not dist_on:
            msg_head = self._msg(batch, embed_head, self._eps_msgp)
            iq, f_mags, sigmas = self._out(batch, msg_head)
            return iq, f_mags, sigmas, batch, 1.0

        M = embed_head.embeds.shape[1]
        N = batch.vocab.shape[0]
        route_dp = (
            self.training
            and self._dp_atom_threshold > 0
            and M < self._dp_atom_threshold
            # DP halves the N-chunk loop but does NOT halve M (unlike TP, which
            # shards M before MessagePass's own atm_chunk-loop runs on it), so a
            # DP-routed bucket runs the M-chunk loop over the *full* M - roughly
            # 2x the inner-loop launches TP would've had on the same bucket. That
            # only pays for itself if halving N actually shrinks the outer loop;
            # if N already fits in one N-chunk (N < 2*mol_chunk), splitting it
            # buys nothing while still eating the un-halved-M cost, so stay on TP.
            and N >= 2 * self._mol_chunk
        )

        if route_dp:
            rank  = dist.get_rank()
            ws    = dist.get_world_size()
            N     = batch.vocab.shape[0]
            shard = (N + ws - 1) // ws
            n0    = rank * shard
            n1    = min(n0 + shard, N)

            local_batch = _dc_replace(
                batch,
                vocab = batch.vocab[n0:n1],
                iqval = batch.iqval[n0:n1],
                coord = batch.coord[n0:n1],
            )
            local_head = embed_head._replace(
                embeds = embed_head.embeds[n0:n1],
                f_mags = embed_head.f_mags[n0:n1],
                sigmas = embed_head.sigmas[n0:n1],
            )
            # use_all_reduce=False: each rank holds a disjoint set of molecules here,
            # not a shard of the SAME molecules like TP, so there is nothing to
            # reconcile across ranks. This is required for correctness, not just an
            # optimization - see MessagePass._all_reduce's docstring for the deadlock
            # this avoids (DP's two ranks can have different local N, hence a
            # different number of N-chunk rounds and all_reduce calls, if this were
            # left on).
            msg_head           = self._msg(local_batch, local_head, self._eps_msgp, use_all_reduce=False)
            iq, f_mags, sigmas = self._out(local_batch, msg_head)
            return iq, f_mags, sigmas, local_batch, (n1 - n0) / N

        rank  = dist.get_rank()
        ws    = dist.get_world_size()
        shard = (M + ws - 1) // ws
        m0    = rank * shard
        m1    = min(m0 + shard, M)

        shard_batch = _dc_replace(
            batch,
            vocab = batch.vocab[:, m0:m1],
            coord = batch.coord[:, m0:m1],
        )
        shard_head = embed_head._replace(
            embeds = embed_head.embeds[:, m0:m1],
            f_mags = embed_head.f_mags[:, m0:m1],
            sigmas = embed_head.sigmas[:, m0:m1],
        )

        msg_head           = self._msg(shard_batch, shard_head, self._eps_msgp)
        iq, f_mags, sigmas = self._out(shard_batch, msg_head)

        iq     = _DistributedSum.apply(iq)                       # type: ignore[assignment]
        f_mags = _AllGatherDim1.apply(f_mags, m0, M)             # type: ignore[assignment]
        sigmas = _AllGatherDim1.apply(sigmas, m0, M)             # type: ignore[assignment]
        return iq, f_mags, sigmas, batch, 1.0
