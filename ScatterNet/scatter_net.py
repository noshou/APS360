import re
from dataclasses import replace as _dc_replace
from typing import Callable

import numpy as np
import torch
import torch.distributed as dist
import xraydb
from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import Float, jaxtyped
from torch import nn

from Preprocess import VOCAB

from .batching import Batch
from .model import Embed, MessagePass, OutputHead
from .utils.all_gather import AllGatherDim1
from .utils.distributed_sum import DistributedSum


class ScatterNet(nn.Module):
    """
    Full ScatterNet pipeline: Embed → MessagePass → OutputHead → I(q).

    Also computes the training loss: Kratky MSLE plus form-factor and
    sigma penalties.

    When a distributed process group is active (torch.distributed initialised)
    and the model is training, each batch is routed to one of two parallelism
    strategies based on M (padded atoms per molecule), N (molecule count), and
    `dp_atom_threshold`:

    Tensor-parallel (TP) - the default; used whenever DP's conditions (below)
    aren't met, or dp_atom_threshold <= 0. Applies in both train and eval -
    routing is keyed on M/N/dp_atom_threshold only, not self.training, so a
    given bucket takes the same path (and therefore the same compiled shapes)
    in both modes. Forcing eval onto TP unconditionally used to introduce
    fresh dynamic shapes into the compiled Embed/MessagePass/OutputHead
    functions the first time evaluate() ran each session - a compile storm at
    the first epoch boundary. eval() correctly aggregates DP's per-rank
    molecule shards (see evaluate() in train.py), so there's no correctness
    reason to force TP there anymore:
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
        mol_chunk: molecules per N-chunk in MessagePass
        dp_atom_threshold: see class docstring; 0 = always TP
        qgrid:     q-grid tensor, shape (Q,)
        energy:    X-ray energy (eV) for xraydb form factors
        eps_embd:  numerical floor for Embed
        eps_msgp:  numerical floor for MessagePass
        compile:   torch.compile checkpointed step functions

    Attributes
    ----------
    _fmag_table : torch.Tensor
        Reference form factor magnitudes per vocabulary entry per
        q-point, shape (V, Q) where V = len(VOCAB) + 1.
    _q_weights_ : torch.Tensor
        Kratky weighting (1 + q^2), shape (1, Q).
    _s_weights_ : torch.Tensor
        q-dependent weighting for the sigma penalty, shape (1, Q).
        Proportional to q^2, normalised to mean 1 over the grid (see
        __init__).
    """

    _emb: Embed
    _msg: MessagePass
    _out: OutputHead
    _eps_embd: Float
    _eps_msgp: Float
    _dp_atom_threshold: int
    _mol_chunk: int
    _fmag_table: Float[torch.Tensor, "V Q"]  # noqa: F722
    _q_weights_: Float[torch.Tensor, "1 Q"]  # noqa: F722
    _s_weights_: Float[torch.Tensor, "1 Q"]  # noqa: F722
    _fwd_fn: Callable

    def __init__(
        self,
        lambda_1: int,
        lambda_2: int,
        lambda_3: int,
        lambda_4: int,
        lambda_5: int,
        msg_seed: int,
        atm_chunk: int,
        mol_chunk: int,
        qgrid: Float[torch.Tensor, "Q"],  # noqa: F722,F821
        energy: float,
        eps_embd: float,
        eps_msgp: float,
        sigma_max: float = 100.0,
        sigma_floor: float = 0.5,
        sigma_init_gain: float = 0.1,
        dp_atom_threshold: int = 0,
        compile: bool = False,
    ) -> None:
        """Construct the Embed, MessagePass, and OutputHead submodules.

        Parameters
        ----------
        lambda_1 : int
            Atom embedding dimension.
        lambda_2 : int
            Number of message passing rounds.
        lambda_3 : int
            OutputHead hidden width.
        lambda_4 : int
            Number of halving steps in the OutputHead MLP.
        lambda_5 : int
            Number of Random Fourier Features used in MessagePass.
        msg_seed : int
            RNG seed for MessagePass's fixed RFF frequency matrix.
        atm_chunk : int
            Atoms per M-chunk in MessagePass and OutputHead.
        mol_chunk : int
            Molecules per N-chunk in MessagePass (controls chem_env peak
            size).
        qgrid : torch.Tensor
            Q-grid points, shape (Q,).
        energy : float
            X-ray energy (eV) used to evaluate anomalous scattering
            factors f1/f2 via xraydb.
        eps_embd : float
            Numerical floor used inside Embed.
        eps_msgp : float
            Numerical floor for the kernel-weight sum at
            message_pass.py:614. The sigma floor it also used to serve now
            lives in Embed as `sigma_floor`.
        sigma_max : float, optional
            Cap on Embed's 1/q sigma envelope, in Angstroms. Default 100.0.
        sigma_floor : float, optional
            Lower clamp on sigma, in Angstroms. Default 0.5.
        sigma_init_gain : float, optional
            Multiplier on Embed's `_sigma` init. Default 0.1.
        dp_atom_threshold : int, optional
            Atom-count threshold below which training batches may be
            routed to the data-parallel path instead of tensor-parallel.
            0 (default) always uses tensor-parallel.
        compile : bool, optional
            If True, torch.compile Embed/MessagePass/OutputHead's
            checkpointed step functions. Default is False.

        Returns
        -------
        None
        """

        super().__init__()
        q_points = qgrid.shape[0]
        self._emb = Embed(
            lambda_1,
            qgrid,
            sigma_max=sigma_max,
            sigma_floor=sigma_floor,
            sigma_init_gain=sigma_init_gain,
            compile=compile,
        )
        self._msg = MessagePass(
            lambda_1,
            lambda_2,
            lambda_5,
            msg_seed,
            q_points,
            n_chunk=mol_chunk,
            m_chunk=atm_chunk,
            compile=compile,
        )
        self._out = OutputHead(
            lambda_1, lambda_3, lambda_4, atm_chunk, compile=compile
        )
        self._eps_embd = eps_embd
        self._eps_msgp = eps_msgp
        self._dp_atom_threshold = dp_atom_threshold
        self._mol_chunk = mol_chunk

        fmag_table = torch.zeros(len(VOCAB) + 1, len(qgrid))

        # convert q vector to s vector
        sgrid = (qgrid / (4 * torch.pi)).numpy()

        # special cases:
        # 1. ion is transuranic,  skip f1/f2
        # 2. ion is special case, map to appropriate base case
        # 3. ion is not in vocab, use ground state
        for idx, ion in enumerate(VOCAB.ions):
            key = ion.lower()
            if key in VOCAB.TRANSURANICS:
                f_mag = torch.tensor(xraydb.f0(ion, sgrid)).float()
            elif key in VOCAB.SPECIAL_CASES:
                resolved = VOCAB.SPECIAL_CASES[key]
                f_mag = torch.tensor(
                    np.hypot(
                        xraydb.f0(resolved, sgrid)
                        + xraydb.f1_chantler(resolved, energy),
                        xraydb.f2_chantler(resolved, energy),
                    )
                ).float()
            else:
                elem = re.sub(r"[0-9+\-]+$", "", key)
                f_mag = torch.tensor(
                    np.hypot(
                        xraydb.f0(ion, sgrid)
                        + xraydb.f1_chantler(elem, energy),
                        xraydb.f2_chantler(elem, energy),
                    )
                ).float()
            fmag_table[idx + 1] = f_mag

        self.register_buffer("_fmag_table", fmag_table, persistent=False)
        self.register_buffer(
            "_q_weights_", (1 + qgrid**2).unsqueeze(0), persistent=False
        )

        # Sigma-penalty weighting, ~q^2. sigma is the message-passing kernel's
        # range (MessagePass scales coordinates as r/sigma,
        # so large sigma = long range), and the real-space distances
        # a scattering vector q is sensitive to go like 1/q.
        # A flat L2 penalty on sigma fights the model everywhere in the same
        # units, and since low q is exactly long-range structure info lives,
        # it crushes the kernel range where I(q) needs it most.
        #
        # At q = 0 the weight is exactly 0 (the grid starts there),
        # i.e. sigma is left unconstrained at that q-point. This works out
        # since I(0) is the forward-scattering limit, where sin(qr)/(qr) -> 1
        # and the intensity reduces to (sum of form factors)^2.
        # This is independent of the coordinates, and thus of the kernel range.
        s_weights = qgrid**2
        s_weights = s_weights / s_weights.mean().clamp(min=1e-12)
        self.register_buffer(
            "_s_weights_", s_weights.unsqueeze(0), persistent=False
        )
        self._fwd_fn = (
            torch.compile(self._loss_fn, dynamic=True, fullgraph=True)
            if compile
            else self._loss_fn
        )

    @staticmethod
    def _loss_fn(  # can't jaxtype here bc of torch.compile graph breaking
        fmag_table: torch.Tensor,
        q_weights: torch.Tensor,
        s_weights: torch.Tensor,
        output_head: torch.Tensor,
        f_mag_pred: torch.Tensor,
        sigma_pred: torch.Tensor,
        iqval: torch.Tensor,
        vocab: torch.Tensor,
        mask: torch.Tensor,
        lambda_6: float,
        lambda_7: float,
    ) -> torch.Tensor:
        """Compute the total training loss.

        Sums the Kratky MSLE, form-factor penalty, and sigma penalty,
        then averages over molecules and q-points.

        Parameters
        ----------
        fmag_table : torch.Tensor
            Reference form factor magnitudes per vocabulary entry per
            q-point, shape (V, Q).
        q_weights : torch.Tensor
            Kratky weighting (1 + q^2), shape (1, Q).
        s_weights : torch.Tensor
            Sigma-penalty weighting (~q^2, mean 1), shape (1, Q).
        output_head : torch.Tensor
            Predicted I(q), shape (N, Q).
        f_mag_pred : torch.Tensor
            Predicted form factor magnitudes, shape (N, M, Q).
        sigma_pred : torch.Tensor
            Predicted sigma bandwidths, shape (N, M, Q).
        iqval : torch.Tensor
            Reference I(q), shape (N, Q).
        vocab : torch.Tensor
            Atom vocabulary indices, shape (N, M).
        mask : torch.Tensor
            Padding mask, shape (N, M); True marks a real (non-padding)
            atom.
        lambda_6 : float
            Weight on the form-factor penalty term.
        lambda_7 : float
            Weight on the sigma L2 penalty term (q-weighted; see __init__).

        Returns
        -------
        torch.Tensor
            Scalar total loss.
        """

        # Kratky-weighted MSLE: (1+q²) * (log1p(Î(q)) - log1p(I(q)))²
        residual = torch.log1p(output_head) - torch.log1p(iqval)
        msle_loss = q_weights * residual**2  # (N, Q)

        mask_2d = mask.unsqueeze(-1)  # (N, M, 1)
        n_atoms = mask.sum(dim=1, keepdim=True).float().clamp(min=1)  # (N, 1)

        # form-factor penalty.
        # log1p-normalized L2 vs xraydb reference, atom-count-normalized
        f_mag_real = torch.log1p(fmag_table[vocab])
        f_mag_pred = torch.log1p(f_mag_pred)
        ff_penalty = (
            (lambda_6 * ((f_mag_pred - f_mag_real) ** 2)) * mask_2d
        ).sum(dim=1) / n_atoms  # (N, Q)

        # sigma L2 penalty.
        sg_sum = (torch.pow(sigma_pred, 2) * mask_2d).sum(
            dim=1
        ) / n_atoms  # (N, Q)
        sg_penalty = lambda_7 * s_weights * sg_sum  # (N, Q)

        return (msle_loss + ff_penalty + sg_penalty).mean()

    @jaxtyped(typechecker=beartype)
    def compute_loss(
        self,
        output_head: Float[torch.Tensor, "N Q"],  # noqa: F722
        f_mag_pred: Float[torch.Tensor, "N M Q"],  # noqa: F722
        sigma_pred: Float[torch.Tensor, "N M Q"],  # noqa: F722
        batch: Batch,
        lambda_6: float,
        lambda_7: float,
    ) -> Float[torch.Tensor, ""]:  # noqa: F722
        """Compute the total training loss.

        Sums the Kratky MSLE, form-factor penalty, and sigma penalty,
        then averages over molecules and q-points.

        Parameters
        ----------
        output_head : torch.Tensor
            Predicted I(q), shape (N, Q).
        f_mag_pred : torch.Tensor
            Predicted form factor magnitudes, shape (N, M, Q).
        sigma_pred : torch.Tensor
            Predicted sigma bandwidths, shape (N, M, Q).
        batch : Batch
            Input batch with reference I(q), vocab, and padding mask.
        lambda_6 : float
            Weight on the form-factor penalty term.
        lambda_7 : float
            Weight on the sigma L2 penalty term (q-weighted; see __init__).

        Returns
        -------
        torch.Tensor
            Scalar total loss.
        """
        # .contiguous(): output_head/f_mag_pred/sigma_pred/mask come from
        # upstream concatenation/squeeze ops whose stride can vary run to
        # run, and torch.compile guards on stride() in addition to shape -
        # causing avoidable recompiles.
        return self._fwd_fn(
            self._fmag_table,
            self._q_weights_,
            self._s_weights_,
            output_head.contiguous(),
            f_mag_pred.contiguous(),
            sigma_pred.contiguous(),
            batch.iqval,
            batch.vocab,
            batch.padding_mask(),
            lambda_6,
            lambda_7,
        )

    @jaxtyped(typechecker=beartype)
    def forward(
        self, batch: Batch
    ) -> Tuple[
        Float[torch.Tensor, "N Q"],  # noqa: F722
        Float[torch.Tensor, "N M Q"],  # noqa: F722
        Float[torch.Tensor, "N M Q"],  # noqa: F722
        Batch,
        float,
    ]:
        """Run the ScatterNet forward pass, routing to TP, DP, or single-GPU.

        Parameters
        ----------
        batch : Batch
            Input batch of molecules (atom vocabulary, coordinates, and
            target I(q) values).

        Returns
        -------
        torch.Tensor
            `iq`, predicted I(q), shape (N, Q).
        torch.Tensor
            `f_mags`, per-atom form factor magnitudes, shape (N, M, Q).
        torch.Tensor
            `sigmas`, per-atom Gaussian kernel bandwidth, shape (N, M, Q).
        Batch
            `local_batch`, the slice of `batch` these outputs correspond
            to. Equal to `batch` unchanged unless this batch was routed to
            the data-parallel path (in which case N above is the local
            molecule count, not the global one) - pass this to the loss,
            not the original `batch`.
        float
            `loss_scale`, local_N / global_N when data-parallel routed;
            1.0 otherwise. Multiply the loss by this before calling
            backward().
        """

        embed_head = self._emb(batch, self._eps_embd)
        dist_on = dist.is_available() and dist.is_initialized()

        if not dist_on:
            msg_head = self._msg(batch, embed_head, self._eps_msgp)
            iq, f_mags, sigmas = self._out(batch, msg_head)
            return iq, f_mags, sigmas, batch, 1.0

        M = embed_head.embeds.shape[1]
        N = batch.vocab.shape[0]
        route_dp = (
            self._dp_atom_threshold > 0
            and M < self._dp_atom_threshold
            # DP halves the N-chunk loop but does NOT halve M (unlike TP, which
            # shards M before MessagePass's own atm_chunk-loop runs on it), so
            # DP-routed bucket runs the M-chunk loop over the *full* M. That
            # only pays for itself if halving N shrinks the outer loop,
            # if N already fits in one N-chunk (N < 2*mol_chunk), splitting it
            # buys nothing while but eats the un-halved-M cost, so stay on TP.
            and N >= 2 * self._mol_chunk
        )

        if route_dp:
            rank = dist.get_rank()
            ws = dist.get_world_size()
            N = batch.vocab.shape[0]
            shard = (N + ws - 1) // ws
            n0 = rank * shard
            n1 = min(n0 + shard, N)

            local_batch = _dc_replace(
                batch,
                vocab=batch.vocab[n0:n1],
                iqval=batch.iqval[n0:n1],
                coord=batch.coord[n0:n1],
            )
            local_head = embed_head._replace(
                embeds=embed_head.embeds[n0:n1],
                f_mags=embed_head.f_mags[n0:n1],
                sigmas=embed_head.sigmas[n0:n1],
            )
            # use_all_reduce=False: each rank holds a disjoint set of mols,
            # not a shard of the SAME molecules like TP, so there is nothing to
            # reconcile across ranks.
            msg_head = self._msg(
                local_batch, local_head, self._eps_msgp, use_all_reduce=False
            )
            iq, f_mags, sigmas = self._out(local_batch, msg_head)
            return iq, f_mags, sigmas, local_batch, (n1 - n0) / N

        rank = dist.get_rank()
        ws = dist.get_world_size()
        shard = (M + ws - 1) // ws
        m0 = rank * shard
        m1 = min(m0 + shard, M)

        shard_batch = _dc_replace(
            batch,
            vocab=batch.vocab[:, m0:m1],
            coord=batch.coord[:, m0:m1],
        )
        shard_head = embed_head._replace(
            embeds=embed_head.embeds[:, m0:m1],
            f_mags=embed_head.f_mags[:, m0:m1],
            sigmas=embed_head.sigmas[:, m0:m1],
        )

        msg_head = self._msg(shard_batch, shard_head, self._eps_msgp)
        iq, f_mags, sigmas = self._out(shard_batch, msg_head)

        iq = DistributedSum.apply(iq)  # type: ignore[assignment]
        f_mags = AllGatherDim1.apply(f_mags, m0, M)  # type: ignore[assignment]
        sigmas = AllGatherDim1.apply(sigmas, m0, M)  # type: ignore[assignment]
        return iq, f_mags, sigmas, batch, 1.0
