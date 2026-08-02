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
    _resid_var_ : torch.Tensor
        Running mean of the squared log-residual per q-point, EMA over
        `per_q_momentum`. Drives `_per_q_scale`. Not checkpointed; it
        re-warms in ~1/(1-momentum) batches.
    _log_env_ : torch.Tensor
        log of Embed's sigma envelope, log(min(1/q, sigma_max)), shape
        (1, Q). The sigma penalty measures deviation from this (see
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
    _log_env_: Float[torch.Tensor, "1 Q"]  # noqa: F722
    _resid_var_: Float[torch.Tensor, "1 Q"]  # noqa: F722
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
        per_q_norm: float = 1.0,
        per_q_momentum: float = 0.99,
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
        per_q_norm : float, optional
            Exponent on the per-q inverse-spread rescaling of the MSLE
            term, in [0, 1]. 0 disables it (every q-point weighted
            equally, the original behaviour); 1 divides each q's residual
            by that q's running RMS. Default 1.0. See `_per_q_scale`.
        per_q_momentum : float, optional
            EMA momentum for the per-q residual variance. Default 0.99,
            i.e. a ~100-batch window.
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

        # Reference sigma envelope, matching Embed's:
        # sigma(m,q) = exp(clamp(z + log env)), env(q) = min(1/q, sigma_max).
        # The penalty measures z = log(sigma) - log(env), the LOG-space
        # deviation of the learned bandwidth from that 1/q prior.
        #
        # This replaces a q^2-weighted L2 on sigma itself, which became
        # inert once Embed moved the 1/q envelope inside sigma: with
        # sigma ~ exp(z)/q, the penalty q^2 * sigma^2 = exp(2z) cancels its
        # own weighting exactly, so the q-dependence the weight existed to
        # provide no longer did anything. Penalising z^2 instead is
        # - symmetric: sigma^2 only punishes LARGE sigma, dragging z toward
        #   the clamp floor, which is the boundary it should stay away from
        # - scale-free: sigma spans 0.25 to 1e4 A^2 across the clamp range,
        #   so one lambda_7 cannot mean the same thing at both ends of the
        #   grid; z is a dimensionless log-ratio, O(1) everywhere
        #
        # Purpose is gradient flow, not bounding: Embed's clamp already
        # bounds sigma hard, but clamp has ZERO gradient outside its range,
        # so a sigma driven past a boundary pins there and cannot recover.
        # Keeping z near 0 keeps it off those boundaries.
        log_env = torch.log(
            torch.clamp(1.0 / qgrid.clamp(min=1e-12), max=sigma_max)
        )
        self.register_buffer(
            "_log_env_", log_env.unsqueeze(0), persistent=False
        )

        # Running mean of the squared log-residual per q-point, for the
        # optional per-q rescaling in `_per_q_scale`. persistent=False to
        # match the other derived buffers here: it re-warms from the data
        # in ~1/(1-momentum) batches, so carrying it in the checkpoint
        # would buy nothing and would break loading older ones.
        self.register_buffer(
            "_resid_var_", torch.ones(1, qgrid.shape[0]), persistent=False
        )
        self._per_q_norm = per_q_norm
        self._per_q_momentum = per_q_momentum
        self._fwd_fn = (
            torch.compile(self._loss_fn, dynamic=True, fullgraph=True)
            if compile
            else self._loss_fn
        )

    @staticmethod
    def _loss_fn(  # can't jaxtype here bc of torch.compile graph breaking
        fmag_table: torch.Tensor,
        q_weights: torch.Tensor,
        q_scale: torch.Tensor,
        log_env: torch.Tensor,
        output_head: torch.Tensor,
        f_mag_pred: torch.Tensor,
        sigma_pred: torch.Tensor,
        z_sigma_pred: torch.Tensor,
        iqval: torch.Tensor,
        vocab: torch.Tensor,
        mask: torch.Tensor,
        lambda_6: float,
        lambda_7: float,
        lambda_8: float,
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
        q_scale : torch.Tensor
            Per-q inverse-spread scaling, shape (1, Q), mean 1. All-ones
            when `per_q_norm` is 0.
        log_env : torch.Tensor
            log of the sigma envelope, log(min(1/q, sigma_max)), shape
            (1, Q). Subtracted from log(sigma) to give z.
        output_head : torch.Tensor
            Predicted I(q), shape (N, Q).
        f_mag_pred : torch.Tensor
            Predicted form factor magnitudes, shape (N, M, Q).
        sigma_pred : torch.Tensor
            Predicted sigma bandwidths, shape (N, M, Q).
        z_sigma_pred : torch.Tensor
            Embed's PRE-saturation sigma exponent offset, shape (N, M, Q).
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
            Weight on the sigma penalty, an L2 on z = log(sigma) -
            log(env) (see __init__).
        lambda_8 : float
            Weight on the q-curvature penalty, an L2 on the second
            difference of z and of log1p(f_mag) along the q axis.

        Returns
        -------
        torch.Tensor
            Scalar total loss.
        """

        # Kratky-weighted MSLE: (1+q²) * (log1p(Î(q)) - log1p(I(q)))²,
        # then scaled per q by `q_scale` (see `_per_q_scale`). q_scale is
        # all-ones unless per_q_norm > 0, and is normalised to mean 1 either
        # way, so lr / lambda_6 / lambda_7 keep their meaning.
        residual = torch.log1p(output_head) - torch.log1p(iqval)
        msle_loss = q_scale * q_weights * residual**2  # (N, Q)

        mask_2d = mask.unsqueeze(-1)  # (N, M, 1)
        n_atoms = mask.sum(dim=1, keepdim=True).float().clamp(min=1)  # (N, 1)

        # form-factor penalty.
        # log1p-normalized L2 vs xraydb reference, atom-count-normalized
        f_mag_real = torch.log1p(fmag_table[vocab])
        f_mag_pred = torch.log1p(f_mag_pred)
        ff_penalty = (
            (lambda_6 * ((f_mag_pred - f_mag_real) ** 2)) * mask_2d
        ).sum(dim=1) / n_atoms  # (N, Q)

        # sigma penalty, two terms, both L2 on a log-space deviation from
        # Embed's 1/q envelope. z = 0 means sigma sits exactly on the
        # physical prior; |z| = 1 means e-fold off it. Both are
        # atom-count-normalized like the form-factor term.
        #
        # (1) POST-saturation, on the final sigma. This is the only term
        #     that sees MessagePass's per-round
        #     softplus(sig + tanhshrink(...)) update, which is unbounded
        #     above and has no saturation of its own, so it stays.
        z = torch.log(sigma_pred.clamp(min=1e-12)) - log_env.unsqueeze(1)
        sg_sum = (torch.pow(z, 2) * mask_2d).sum(dim=1) / n_atoms  # (N, Q)

        # (2) PRE-saturation, on Embed's raw exponent. Term (1) alone
        #     cannot do this job: it reaches _emb._sigma only through the
        #     saturation, whose derivative decays to ~0 exactly for the
        #     entries that have already run away, so it is preventative
        #     with no restoring force and saturation becomes an absorbing
        #     state (measured: 4.1% of (atom, q) pairs pinned at step 207,
        #     38.5% at step 10661, 95.8% by the end of the previous run).
        #     d/dz of this term is 2*lambda_7*z no matter how saturated the
        #     entry is, so a runaway entry is always pulled back.
        zr_sum = (
            torch.pow(z_sigma_pred, 2) * mask_2d
        ).sum(dim=1) / n_atoms  # (N, Q)

        sg_penalty = lambda_7 * (sg_sum + zr_sum)  # (N, Q)

        # q-curvature penalty: second difference along the q axis.
        #
        # `_f0f1`, `_f2` and `_sigma` each emit Q INDEPENDENT output
        # channels (nn.Linear(lambda_1, qPoints) etc.), so nothing couples
        # q_j to q_{j+1}. The true I(q) is an orientational average of a
        # bounded object, i.e. a finite sum of sinc terms and therefore
        # analytic in q, but the model is free to make adjacent q-points
        # disagree arbitrarily. It does: measured on the 2026-08-01 run,
        # mean log1p(I(q)) over 8165 molecules oscillates +-1.5 nats
        # between ADJACENT grid points (a ~20x swing) against a perfectly
        # smooth target, and per-q R2 crosses zero at q ~ 0.22 and reaches
        # -2.26 beyond it.
        #
        # Applied to z, not to log(sigma): log_env carries the intended 1/q
        # shape and has real curvature of its own, so penalising
        # d2(log sigma) would fight the envelope. Penalising d2(z) asks the
        # learned DEVIATION from the envelope to be smooth and leaves the
        # physical shape alone. Form factors are penalised in log1p space,
        # matching the lambda_6 term, and their xraydb reference is smooth
        # so this does not fight it either.
        def _d2(x: torch.Tensor) -> torch.Tensor:
            return x[..., 2:] - 2.0 * x[..., 1:-1] + x[..., :-2]

        curv = (
            torch.pow(_d2(z_sigma_pred), 2) + torch.pow(_d2(f_mag_pred), 2)
        ) * mask_2d  # (N, M, Q-2)
        curv_penalty = lambda_8 * (curv.sum(dim=1) / n_atoms).mean()

        return (msle_loss + ff_penalty + sg_penalty).mean() + curv_penalty

    def _per_q_scale(
        self,
        output_head: torch.Tensor,
        iqval: torch.Tensor,
        sync_stats: bool,
    ) -> torch.Tensor:
        """Per-q inverse-spread weights for the MSLE term.

        Every q-point currently enters the loss with equal weight, but not
        with equal difficulty: measured per-q R2 (log1p) on the 2026-08-01
        run is ~0.85 below q=0.17, crosses zero at q~0.22, and bottoms at
        -2.26. Because the loss is quadratic, the q-range with the largest
        residuals also contributes the largest gradient, so the hardest
        third of the grid dominates the update while remaining the part
        the model fits worst. Dividing each q's residual by that q's
        running RMS turns the objective into a balanced multi-task one:
        every q-point contributes comparably, and no single band can
        capture the update.

        NOTE this reduces the weight on high q rather than raising it. The
        justification is that more raw gradient there has demonstrably not
        helped; the aim is to stop an ill-conditioned band from crowding
        out the rest, not to push harder on it. If high-q R2 gets worse
        rather than better, set `per_q_norm` to 0.0 to recover the
        original equal weighting; that knob is the whole experiment.

        Normalised to mean 1 so total loss magnitude is unchanged and lr,
        lambda_6 and lambda_7 keep their calibration.

        Runs eager (not inside the compiled `_loss_fn`) because it mutates
        a buffer and may issue a collective, both of which would break a
        fullgraph capture.

        Parameters
        ----------
        output_head : torch.Tensor
            Predicted I(q), shape (N, Q).
        iqval : torch.Tensor
            Reference I(q), shape (N, Q).
        sync_stats : bool
            All-reduce the batch statistic across ranks before folding it
            into the EMA. Required on the DP path, where each rank holds a
            DIFFERENT set of molecules and would otherwise drift into a
            different weighting. Must be False on the TP path, where every
            rank already holds the same molecules and the same all-reduced
            `output_head`, so reducing again would just double-count.

        Returns
        -------
        torch.Tensor
            Per-q weights, shape (1, Q), mean 1.
        """

        if self._per_q_norm <= 0.0:
            return torch.ones_like(self._q_weights_)

        if self.training:
            with torch.no_grad():
                pred = torch.log1p(output_head.detach().float())
                resid = pred - torch.log1p(iqval.float())
                batch_var = (resid**2).mean(dim=0, keepdim=True)  # (1, Q)
                if (
                    sync_stats
                    and dist.is_available()
                    and dist.is_initialized()
                ):
                    dist.all_reduce(batch_var, op=dist.ReduceOp.SUM)
                    batch_var /= dist.get_world_size()
                m = self._per_q_momentum
                self._resid_var_.mul_(m).add_(batch_var, alpha=1.0 - m)

        # eps floors the RMS so an already-solved q-point cannot acquire an
        # unbounded weight and take over the update.
        scale = (self._resid_var_ + 1e-6) ** (-0.5 * self._per_q_norm)
        return scale / scale.mean().clamp(min=1e-12)

    @jaxtyped(typechecker=beartype)
    def compute_loss(
        self,
        output_head: Float[torch.Tensor, "N Q"],  # noqa: F722
        f_mag_pred: Float[torch.Tensor, "N M Q"],  # noqa: F722
        sigma_pred: Float[torch.Tensor, "N M Q"],  # noqa: F722
        z_sigma_pred: Float[torch.Tensor, "N M Q"],  # noqa: F722
        batch: Batch,
        lambda_6: float,
        lambda_7: float,
        lambda_8: float = 0.0,
        sync_stats: bool = False,
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
        z_sigma_pred : torch.Tensor
            Embed's PRE-saturation sigma exponent offset, shape (N, M, Q).
        batch : Batch
            Input batch with reference I(q), vocab, and padding mask.
        lambda_6 : float
            Weight on the form-factor penalty term.
        lambda_7 : float
            Weight on the sigma penalty, an L2 on z = log(sigma) -
            log(env) (see __init__).
        lambda_8 : float, optional
            Weight on the q-curvature penalty. Default 0.0 (off).
        sync_stats : bool, optional
            Forwarded to `_per_q_scale`; True only on the DP path. Default
            False.

        Returns
        -------
        torch.Tensor
            Scalar total loss.
        """
        q_scale = self._per_q_scale(output_head, batch.iqval, sync_stats)
        # .contiguous(): output_head/f_mag_pred/sigma_pred/mask come from
        # upstream concatenation/squeeze ops whose stride can vary run to
        # run, and torch.compile guards on stride() in addition to shape -
        # causing avoidable recompiles.
        return self._fwd_fn(
            self._fmag_table,
            self._q_weights_,
            q_scale,
            self._log_env_,
            output_head.contiguous(),
            f_mag_pred.contiguous(),
            sigma_pred.contiguous(),
            z_sigma_pred.contiguous(),
            batch.iqval,
            batch.vocab,
            batch.padding_mask(),
            lambda_6,
            lambda_7,
            lambda_8,
        )

    @jaxtyped(typechecker=beartype)
    def forward(
        self, batch: Batch
    ) -> Tuple[
        Float[torch.Tensor, "N Q"],  # noqa: F722
        Float[torch.Tensor, "N M Q"],  # noqa: F722
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
        torch.Tensor
            `z_sigmas`, Embed's pre-saturation sigma exponent offset, shape
            (N, M, Q). Feed to `compute_loss`; see `_loss_fn` for why the
            penalty needs the pre-saturation value.
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
            # whole molecule on one device, so coh already covers every atom
            coh, inc, f_mags, sigmas, z_sigmas = self._out(batch, msg_head)
            iq = OutputHead.combine(coh, inc)
            return iq, f_mags, sigmas, z_sigmas, batch, 1.0

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
                z_sigma=embed_head.z_sigma[n0:n1],  # type: ignore[index]
            )
            # use_all_reduce=False: each rank holds a disjoint set of mols,
            # not a shard of the SAME molecules like TP, so there is nothing to
            # reconcile across ranks.
            msg_head = self._msg(
                local_batch, local_head, self._eps_msgp, use_all_reduce=False
            )
            # DP splits molecules, not atoms, so each rank holds every atom
            # of the molecules it owns and coh is already complete.
            coh, inc, f_mags, sigmas, z_sigmas = self._out(
                local_batch, msg_head
            )
            iq = OutputHead.combine(coh, inc)
            return iq, f_mags, sigmas, z_sigmas, local_batch, (n1 - n0) / N

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
            z_sigma=embed_head.z_sigma[:, m0:m1],  # type: ignore[index]
        )

        msg_head = self._msg(shard_batch, shard_head, self._eps_msgp)
        coh, inc, f_mags, sigmas, z_sigmas = self._out(shard_batch, msg_head)

        # TP shards the ATOM dim, so coh/inc here cover only this rank's
        # atoms. Both partials must be reduced across ranks BEFORE the
        # coherent term is squared: squaring first would give
        # sum_ranks (sum_j w_j f_j)^2 instead of (sum_ranks sum_j w_j f_j)^2,
        # dropping every atom pair split across the shard boundary. Stacked
        # into one tensor so this stays a single collective.
        parts = DistributedSum.apply(  # type: ignore[assignment]
            torch.stack((coh, inc), dim=-1)
        )
        iq = OutputHead.combine(parts[..., 0], parts[..., 1])
        f_mags = AllGatherDim1.apply(f_mags, m0, M)  # type: ignore[assignment]
        sigmas = AllGatherDim1.apply(sigmas, m0, M)  # type: ignore[assignment]
        # gathered for the same reason as sigmas: the penalty reduces over
        # the FULL atom dim against the full-batch padding mask, so a
        # per-rank shard would not line up with mask_2d.
        z_sigmas = AllGatherDim1.apply(  # type: ignore[assignment]
            z_sigmas, m0, M
        )
        return iq, f_mags, sigmas, z_sigmas, batch, 1.0
