import math
from typing import Callable, NamedTuple, Tuple, cast

import numpy as np
import torch
from pyteals import PBId, QDiagBilin
from torch import cos, nn
from torch.nn.functional import mish
from torch.utils.checkpoint import checkpoint

from ..batching import Batch
from ..utils.sigma_window import arctan_log_sigma_window


class MessagePass(nn.Module):
    """
    RFF-based message passing: λ₂ rounds of
    kernel-weighted neighbourhood aggregation.

    Mathematical Formulation
    ------------------------
    Each atom m has an embedding e_m ∈ R^λ₁ and a per-q bandwidth σ_m ∈ R^Q.
    At each q, coords are scaled by σ to make the kernel range q-dependent:
        ```
        r̃_m(q) = r_m / σ_m(q)   [Å → dimensionless; large σ → long range]
        ```

    RFF approximation of the RBF kernel exp(-‖r̃_i − r̃_j‖² / 2):
        ```
        φ_m(q) = √(2/λ₅) · cos(Ω · r̃_m(q) + b)    ∈ R^λ₅
        ```
    where Ω ∈ R^{λ₅×3} is a fixed random freq matrix and b ∈ R^λ₅ rand phases.
    Then φ_i · φ_j ≈ k(r̃_i, r̃_j).

    Per round, two passes over M-chunks:

    Pass 1; accumulate global context for the N-chunk:

    ```
    features[q, d]      = Σ_m  φ_m(q, d)             [shape:: (Q, λ₅)]
    glob_ctx[q, d, l]   = Σ_m  φ_m(q, d) · e_m(q, l) [shape:: (Q, λ₅, λ₁)]
    ```
    Pass 2; per-atom update using the accumulated context:

        ```
        atmenv_m[q,l]   = Σ_d  φ_m(q, d) · glob_ctx[q, d, l] [shape:: (Q, λ₁)]
                        ≈ Σ_{m'} k(r̃_m, r̃_{m'}) · e_{m'}(q, l)

        weights_m[q]    = |Σ_d  φ_m(q, d) · features[q, d]| [shape:: (Q,)]
                        ≈ Σ_{m'} k(r̃_m, r̃_{m'}) [denom for normalised avg]

        agg_m           = RMSNorm(atmenv_m / weights_m)   [shape:: (Q, λ₁)]

        [p1, p2]        = Linear(agg_m)                     [shape:: (Q, 2*λ₁)]
        gate_m          = p1 · Mish(p2)                     [shape:: (Q, λ₁)]
        e_m  ←  e_m + gate_m
        σ_m  ←  exp(window(log σ_m + pshrink(qdiag(RMSNorm(e_m), f_m))))
        ```

    Memory strategy
    ---------------
    Checkpointing at the N-chunk level with use_reentrant=False: glob_ctx
    (Nc, Q, λ₅, λ₁) only exists inside the per-N-chunk closure's own call
    frame, so it's created and freed per N-chunk rather than kept alive for
    all N molecules simultaneously.

    Within each pass, each M-chunk gets its own inner checkpoint() call too
    (in `_pass_1`/`_pass_2`), so only one M-chunk's RFF intermediates are ever
    alive at once - never the full (Nc, M, Q, λ₅) tensor. The two heavy
    contractions use bmm on reshaped 3-D tensors rather than einsum, because
    einsum('nmqd,nmql->nqdl') would create an (Nc,mc,Q,λ₅,λ₁) intermediate
    before contracting over m; the tensor that caused OOM.

    Compile shapes
    --------------
    `_step1`/`_step2` are run eager, never `torch.compile`d (see
    `__init__`). Several compiled configurations were tried and all hit
    real crashes:

    - Static compilation (`dynamic=False`), padding every chunk to the
      full `n_chunk`/`m_chunk`: guaranteed one shape per axis, but forced
      small buckets (e.g. 7 molecules against a 128-molecule chunk) to pad
      18x+ past their real size, which OOM'd on a live run.
    - A small-bucket tiering scheme (round up to the nearest of a handful
      of shared tiers instead of the full chunk): bounded that padding
      waste, but under static compilation each tier was its own compiled
      graph, and a live run hit `FailOnRecompileLimitHit`.
    - `dynamic=True` with alignment-only padding (round up to a small
      granularity, not the full chunk): fixed the OOM, but a live run
      still hit the recompile cap. `cache_size_limit`, raised in
      `__init__`, measurably held at both construction and the start of
      `forward` (confirmed via debug prints), yet the crash's own error
      message reported the pre-raise default at the moment it was hit.
    - `fullgraph=False` (fall back to eager past the recompile cap instead
      of hard-failing) plus disabling dynamo's eval_frame LRU cache
      (`torch._C._dynamo.eval_frame._set_lru_cache(False)`, the workaround
      suggested by pytorch/pytorch#166926): still hit a `CheckpointError`,
      backward's recompute pass produced tensors with different shapes
      (in several cases transposed or a different rank entirely) than
      what forward had saved.

    All of these point to the same underlying interaction: this module
    nests checkpointing (`forward`'s outer per-N-chunk `checkpoint`,
    `__pass_1`/`__pass_2`'s inner per-M-chunk `checkpoint` around
    `_step1_fn`/`_step2_fn`), and checkpoint-inside-checkpoint under
    `torch.compile`'s backward recompute is a far less battle-tested path
    than either alone. Running `_step1`/`_step2` eager sidesteps the whole
    class of bug at some speed cost; `Embed`/`OutputHead`/`LambdaHead`/
    `SplineSmooth` do not nest checkpointing and are unaffected, so they
    stay compiled.
    """

    __lambda_1: int  # atom embedding dimension (λ₁)
    __lambda_2: int  # number of message-passing rounds (λ₂)
    __lambda_5: int  # number of Random Fourier Features (λ₅)
    __nchunk: int  # molecules per N-chunk
    __mchunk: int  # atoms per M-chunk

    # projects aggregated context λ₁ → 2λ₁ for MishGLU gating.
    # one nn.Linear per message-passing round (length λ₂); rounds do NOT
    # share weights.
    __proj_agg: nn.ModuleList

    # fixed RFF frequency matrix: λ₅ random 3-D directions
    __omegafrq: torch.Tensor  # shape (λ₅, 3)

    # RFF random phase offsets b ∈ R^λ₅
    __biasterm: nn.Parameter

    # updates σ from (updated embedding, form factor): QDiagBilin -> PBId.
    # one instance of each per message-passing round (length λ₂); rounds do
    # NOT share weights.
    __sigbilin: nn.ModuleList
    __sigact: nn.ModuleList

    # normalises aggregated context before gating.
    # one nn.RMSNorm per message-passing round (length λ₂); rounds do NOT
    # share weights.
    __rms_norm: nn.ModuleList

    # number of q-points (Q)
    __q_points: int

    # precomputed once here rather than recomputed inside _step1/_step2.
    __rffscale: float

    __step1_fn: Callable
    __step2_fn: Callable

    class _PassContainer(NamedTuple):
        """Working state for one N-chunk during a message-passing round.

        Attributes
        ----------
        M : int
            Total (padded) number of atoms across the whole batch.
        Nchnk : int
            Number of molecules in this N-chunk.
        Mchnk : int
            Number of atoms per M-chunk used when iterating over M.
        emb_n : torch.Tensor
            Atom embeddings for this N-chunk, shape (Nchnk, M, Q, λ₁).
        msk_n : torch.Tensor
            Padding mask for this N-chunk, shape (Nchnk, M); True = real
            atom.
        ffs_n : torch.Tensor
            Form factor magnitudes for this N-chunk, shape
            (Nchnk, M, Q, 1).
        sig_n : torch.Tensor
            Per-atom per-q RBF bandwidths for this N-chunk, shape
            (Nchnk, M, Q, 1).
        crd_n : torch.Tensor
            Atom Cartesian coordinates (Å) for this N-chunk, shape
            (Nchnk, M, 3).
        features : torch.Tensor
            Accumulated sum of RFF features over atoms, shape
            (Nchnk, Q, λ₅).
        glob_ctx : torch.Tensor
            Accumulated kernel-weighted embedding sum, shape
            (Nchnk, Q, λ₅, λ₁).
        """

        M: int
        Nchnk: int
        Mchnk: int

        # atom embeddings for this N-chunk
        emb_n: torch.Tensor  # shape (Nchnk, Mc, Q, λ₁)

        # padding mask (True = real atom)
        msk_n: torch.Tensor  # shape (Nchnk, Mc), bool

        # form factor magnitudes
        ffs_n: torch.Tensor  # shape (Nchnk, Mc, Q, 1)

        # per-atom per-q RBF bandwidths
        sig_n: torch.Tensor  # shape (Nchnk, Mc, Q, 1)

        # atom Cartesian coordinates (Å)
        crd_n: torch.Tensor  # shape (Nchnk, Mc, 3)

        # Σ_m φ_m; kernel weight normaliser
        features: torch.Tensor  # shape (Nchnk, Q, λ₅)

        # Σ_m φ_m ⊗ e_m; chemical environment
        glob_ctx: torch.Tensor  # shape (Nchnk, Q, λ₅, λ₁)

    def __init__(
        self,
        lambda_1: int,
        lambda_2: int,
        lambda_5: int,
        seed: int,
        q_points: int,
        n_chunk: int,
        m_chunk: int,
        sigma_max: float = 100.0,
        sigma_floor: float = 0.5,
        sigma_init_gain: float = 0.1,
        compile: bool = False,
    ):
        """Validate hyperparameters and build MessagePass's learned layers.

        Parameters
        ----------
        lambda_1 : int
            Atom embedding dimension.
        lambda_2 : int
            Number of message-passing rounds; must be > 0.
        lambda_5 : int
            Number of Random Fourier Features.
        seed : int
            RNG seed for the fixed RFF frequency matrix and phase offsets.
        q_points : int
            Number of q-points (Q).
        n_chunk : int
            Molecules per N-chunk; must be > 0.
        m_chunk : int
            Atoms per M-chunk; must be > 0.
        sigma_max : float, optional
            Upper bound of the per-round sigma window, in Angstroms.
            Must match `Embed`'s. Default 100.0.
        sigma_floor : float, optional
            Lower bound of the per-round sigma window, in Angstroms.
            Must match `Embed`'s. Default 0.5.
        sigma_init_gain : float, optional
            Multiplier on `_sigbilin`'s weight init, so the round-0 sigma
            update is a small offset in log space rather than a saturating
            one. Default 0.1.
        compile : bool, optional
            Accepted for interface consistency with Embed/OutputHead/
            LambdaHead/SplineSmooth (`ScatterNet.__init__` passes the same
            `compile` value to all five uniformly) but otherwise unused:
            `_step1`/`_step2` are never torch.compiled here, see the class
            docstring's "Compile shapes" section for why. Default is False.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If `lambda_2`, `n_chunk`, or `m_chunk` is not greater than 0.
        """

        super().__init__()

        if lambda_2 <= 0:
            raise ValueError(f"invalid lambda_2 (must be > 0): {lambda_2}")
        if n_chunk <= 0:
            raise ValueError(f"invalid n_chunk (must be > 0): {n_chunk}")
        if m_chunk <= 0:
            raise ValueError(f"invalid m_chunk (must be > 0): {m_chunk}")

        rng = np.random.default_rng(seed=seed)

        self.__lambda_1 = lambda_1
        self.__lambda_2 = lambda_2
        self.__lambda_5 = lambda_5
        self.__nchunk = n_chunk
        self.__mchunk = m_chunk
        # Same log-sigma window Embed uses, re-applied every round; see the
        # "Sigma window" section of the class docstring.
        self.__log_sigma_floor = math.log(sigma_floor)
        self.__log_sigma_mid = 0.5 * (
            math.log(sigma_floor) + math.log(sigma_max)
        )
        self.__log_sigma_half = 0.5 * (
            math.log(sigma_max) - math.log(sigma_floor)
        )

        # one instance per message-passing round (length lambda_2): each
        # round gets its own learned transform rather than reusing the same
        # weights lambda_2 times. Constructed in a loop so each element's
        # torch-RNG-driven init (nn.Linear/nn.RMSNorm/NoTrilinBilin all use
        # torch's default/global RNG, not the numpy `rng` above) draws a
        # distinct set of initial values per round without needing explicit
        # per-round seeding.
        self.__proj_agg = nn.ModuleList(
            [nn.Linear(lambda_1, 2 * lambda_1) for _ in range(lambda_2)]
        )
        self.register_buffer(
            "_MessagePass__omegafrq",
            torch.from_numpy(rng.standard_normal((lambda_5, 3))).float(),
        )
        # Buffer, NOT a parameter. phi_i . phi_j is a Rahimi-Recht estimator
        # of the RBF kernel only because
        #   2 cos(A+b) cos(A'+b) = cos(A-A') + cos(A+A'+2b)
        # and E_{b ~ U(0, 2pi)} kills the second term. Learning b breaks
        # that expectation: measured at lambda_5 = 8192 and r = 1 A, the
        # estimate is 0.596 against a true kernel of 0.607 with b uniform,
        # but 1.202 (a 2x bias) once b collapses toward 0.
        self.register_buffer(
            "_MessagePass__biasterm",
            torch.from_numpy(
                rng.uniform(0, 2 * np.pi, size=(self.__lambda_5))
            ).float(),
        )
        # QDiagBilin, not nn.Bilinear and not NoTrilinBilin: the sigma head
        # emits one value per q-point from that q-point's embedding and the
        # atom's whole form-factor spectrum, so the output index and the
        # input's q axis are the SAME index. A generic bilinear treats them
        # as independent and materializes (Nc, mc, Q, Q, Q) - 1.012 GiB at
        # (4, 512, 51) - to then use only the diagonal. QDiagBilin contracts
        # lambda_1 first and never builds the third axis: 20.3 MiB, exact
        # same result. Both also avoid nn.Bilinear's F.bilinear op forcing a
        # torch.compile graph break (aten::_trilinear isn't Inductor-
        # supported), so this fuses into the surrounding compiled graph.
        #
        # PBId, not PTanhShrink: the hard bound on the sigma update lives in
        # arctan_log_sigma_window downstream (see __pass_1's "Sigma update"
        # comment), so PTanhShrink's dead zone near 0 wasn't adding safety -
        # it was just killing gradient in the "small controlled nudge"
        # regime this update is supposed to live in (measured: sigma's
        # per-element grad RMS was 9.4e-13 against ~1e0 elsewhere, and
        # PTanhShrink's learned width `c` never moved off its init across
        # multiple independent runs - it wasn't using the escape hatch
        # either). PBId's f'(0) = 1 regardless of its bend `w`, so small
        # inputs pass through near-identity; the bend only engages at large
        # |x|, exactly where arctan_log_sigma_window already bounds things.
        #
        # The last round is skipped: nothing consumes its sigma. Sigma feeds
        # the RFF kernel of the NEXT round, and OutputHead ignores sigmas
        # entirely, so round lambda_2-1's sigma parameters measured
        # grad = None / |g| = 0.0 and sat permanently at init.
        self.__sigbilin = nn.ModuleList(
            [
                QDiagBilin(
                    in1_features=lambda_1,
                    in2_features=q_points,
                    q_points=q_points,
                    bias=True,
                )
                for _ in range(lambda_2 - 1)
            ]
        )
        self.__sigact = nn.ModuleList(
            [PBId(q_points) for _ in range(lambda_2 - 1)]
        )
        with torch.no_grad():
            for bilin in self.__sigbilin:
                bilin = cast(QDiagBilin, bilin)
                bilin.weight.mul_(sigma_init_gain)
                if bilin.bias is not None:
                    bilin.bias.mul_(sigma_init_gain)

        # The sigma head reads the raw residual stream, which is the only
        # unnormalized linear map in the block (`agg` is RMSNormed). Without
        # this the pre-activation std is sqrt(Q/3)*std(e)*f_rms ~ 25-90,
        # which in log-sigma units saturates the window on round 0.
        self.__sig_norm = nn.ModuleList(
            [nn.RMSNorm(lambda_1) for _ in range(lambda_2 - 1)]
        )

        self.__rms_norm = nn.ModuleList(
            [nn.RMSNorm(lambda_1) for _ in range(lambda_2)]
        )
        self.__q_points = q_points

        self.__rffscale = (2 / lambda_5) ** 0.5

        # _step1/_step2 are never torch.compiled (see the class docstring's
        # "Compile shapes" section): every combination tried (static
        # padding, dynamic shapes with alignment padding, fullgraph=False,
        # disabling dynamo's eval_frame LRU cache) still hit real crashes
        # from the interaction between this module's nested checkpointing
        # (forward()'s outer per-N-chunk checkpoint, __pass_1/__pass_2's
        # inner per-M-chunk checkpoint around _step1_fn/_step2_fn) and
        # torch.compile's backward recompute path. Running eager here costs
        # some speed but is unconditionally correct; Embed/OutputHead/
        # LambdaHead/SplineSmooth are unaffected and stay compiled.
        self.__step1_fn = self.__step1
        self.__step2_fn = self.__step2

    def __step1(
        self,
        embslice: torch.Tensor,
        crdslice: torch.Tensor,
        sigslice: torch.Tensor,
        mskslice: torch.Tensor,
        epsilon_: float,
    ):
        """Compute per-M-chunk RFF features
        and the kernel-weighted embedding sum.

        Parameters
        ----------
        embslice : torch.Tensor
            Atom embeddings for this M-chunk, shape (Nc, mc, Q, λ₁).
        crdslice : torch.Tensor
            Atom coordinates for this M-chunk, shape (Nc, mc, 3).
        sigslice : torch.Tensor
            Per-atom per-q bandwidths for this M-chunk, shape
            (Nc, mc, Q, 1).
        mskslice : torch.Tensor
            Padding mask for this M-chunk, shape (Nc, mc).
        epsilon_ : float
            Numerical floor for clamping sigma before division.

        Returns
        -------
        torch.Tensor
            `step_features`, partial sum of RFF features over this
            M-chunk's atoms, shape (Nc, Q, λ₅).
        torch.Tensor
            `step_glob_ctx`, partial kernel-weighted embedding sum, shape
            (Nc, Q, λ₅, λ₁).
        """

        lambda_1, lambda_5 = self.__lambda_1, self.__lambda_5

        # r̃_m = r_m / σ_m: scale coords by bandwidth so
        # kernel range is q-dependent. clamp(min=eps) caps the max RFF
        # frequency and avoids 1/0. Forced to fp32 (autocast disabled): under
        # BF16 the magnitude can't overflow (BF16 shares fp32's exponent
        # range), but sigslice.clamp(min=eps) can be ~1e-3, so scaled_coords
        # reaches ~1e5 and BF16's ~7 mantissa bits leave the `@ omega.T`
        # projection too coarse for cos()'s phase to stay accurate at that
        # range. This is a tiny inner-dim-3 -> λ₅ contraction, so fp32 costs
        # almost nothing; the heavy bmm below still runs in BF16 under the
        # outer autocast.
        with torch.autocast(device_type=crdslice.device.type, enabled=False):
            scaled_coords = crdslice.float().unsqueeze(
                -2
            ) / sigslice.float().clamp(min=epsilon_)  # (Nc, mc, Q, 3)

            # φ_m = √(2/λ₅) · cos(Ω · r̃_m + b):
            # RFF feature vector per atom per q-point
            proj = (
                scaled_coords @ self.__omegafrq.T + self.__biasterm
            )  # (Nc, mc, Q, λ₅)
            zrff = self.__rffscale * cos(proj)  # (Nc, mc, Q, λ₅)
        zrff = zrff * mskslice.unsqueeze(-1).unsqueeze(
            -1
        )  # zero padding atoms

        # Σ_m φ_m: partial sum of RFF features over atoms in this M-chunk
        step_features = zrff.sum(dim=1)  # (Nc, Q, λ₅)

        # Σ_m φ_m ⊗ e_m: partial outer-product sum
        # (kernel-weighted embedding accumulator).
        # bmm on (Nc*Q, λ₅, mc) @ (Nc*Q, mc, λ₁) avoids the
        # (Nc, mc, Q, λ₅, λ₁) intermediate that einsum('nmqd,nmql->nqdl')
        # would create before contracting over m.
        Nc, mc, Q = zrff.shape[0], zrff.shape[1], zrff.shape[2]
        zb = zrff.permute(0, 2, 3, 1).reshape(
            Nc * Q, lambda_5, mc
        )  # (Nc*Q, λ₅, mc)
        eb = embslice.permute(0, 2, 1, 3).reshape(
            Nc * Q, mc, lambda_1
        )  # (Nc*Q, mc, λ₁)
        step_glob_ctx = torch.bmm(zb, eb).reshape(Nc, Q, lambda_5, lambda_1)

        return step_features, step_glob_ctx

    def __pass_1(self, cont: _PassContainer, eps: float) -> _PassContainer:
        """
        Accumulate global context (features, glob_ctx)
        for this N-chunk across all M-chunks.

        After this pass:
            features  = Σ_m φ_m        (kernel weight normaliser)
            glob_ctx  = Σ_m φ_m ⊗ e_m (kernel-weighted embedding sum)

        Parameters
        ----------
        cont : MessagePass._PassContainer
            Working state for this N-chunk; `features` and `glob_ctx` are
            accumulated (functionally) from their current values.
        eps : float
            Numerical floor for sigma clamping inside `_step1`.

        Returns
        -------
        MessagePass._PassContainer
            `cont` with `features` and `glob_ctx` updated to their fully
            accumulated values across all M-chunks.
        """

        features: torch.Tensor  # shape (Nchnk, Q, λ₅)
        glob_ctx: torch.Tensor  # shape (Nchnk, Q, λ₅, λ₁)

        features = cont.features
        glob_ctx = cont.glob_ctx

        for m0 in range(0, cont.M, cont.Mchnk):
            m1 = min(m0 + cont.Mchnk, cont.M)

            # .contiguous(): these are views into cont.*_n, so their
            # stride/storage_offset depend on m0 (which chunk) and the batch's
            # own padding. torch.compile guards on stride()/storage_offset()
            # as well as shape, so distinct views recompile even at the same
            # chunk-tier shape. Materializing a canonical contiguous
            # layout collapses that to one guard set per tier.
            emb_slice = cont.emb_n[:, m0:m1].contiguous()
            crd_slice = cont.crd_n[:, m0:m1].contiguous()
            sig_slice = cont.sig_n[:, m0:m1].contiguous()
            msk_slice = cont.msk_n[:, m0:m1].contiguous()

            args = (emb_slice, crd_slice, sig_slice, msk_slice, eps)

            step_feat, step_chem = checkpoint(
                self.__step1_fn, *args, use_reentrant=False
            )  # type: ignore[misc]
            features = features + step_feat
            glob_ctx = glob_ctx + step_chem

        return cont._replace(features=features, glob_ctx=glob_ctx)

    def __step2(
        self,
        embslice: torch.Tensor,
        crdslice: torch.Tensor,
        sigslice: torch.Tensor,
        mskslice: torch.Tensor,
        ffsslice: torch.Tensor,
        features: torch.Tensor,
        glob_ctx: torch.Tensor,
        epsilon_: float,
        rnd_idx: int,
    ):
        """Compute the per-atom neighbourhood
        aggregate and updated embedding/sigma.

        Parameters
        ----------
        embslice : torch.Tensor
            Atom embeddings for this M-chunk, shape (Nc, mc, Q, λ₁).
        crdslice : torch.Tensor
            Atom coordinates for this M-chunk, shape (Nc, mc, 3).
        sigslice : torch.Tensor
            Per-atom per-q bandwidths for this M-chunk, shape
            (Nc, mc, Q, 1).
        mskslice : torch.Tensor
            Padding mask for this M-chunk, shape (Nc, mc).
        ffsslice : torch.Tensor
            Form factor magnitudes for this M-chunk, shape (Nc, mc, Q, 1).
        features : torch.Tensor
            Fully accumulated RFF feature sum from `_pass_1`, shape
            (Nc, Q, λ₅).
        glob_ctx : torch.Tensor
            Fully accumulated kernel-weighted embedding sum from
            `_pass_1`, shape (Nc, Q, λ₅, λ₁).
        epsilon_ : float
            Numerical floor for clamping sigma and the aggregate
            denominator.
        rnd_idx : int
            Index (0-based) of the current message-passing round; selects
            which round's entry of `_proj_agg`/`_sigbilin`/`_rms_norm` to
            use, since rounds no longer share weights.

        Returns
        -------
        torch.Tensor
            `new_emb`, updated atom embeddings, shape (Nc, mc, Q, λ₁).
        torch.Tensor
            `new_sig`, updated per-atom per-q bandwidths, shape
            (Nc, mc, Q, 1).
        """

        lambda_1, lambda_5 = self.__lambda_1, self.__lambda_5

        # recompute φ_m for this M-chunk
        # (same as _pass_1, but now glob_ctx is complete).
        # fp32 (autocast disabled) for the same BF16-precision
        # reason as _step1 - see there.
        with torch.autocast(device_type=crdslice.device.type, enabled=False):
            scaled_coords = crdslice.float().unsqueeze(
                -2
            ) / sigslice.float().clamp(min=epsilon_)  # (Nc, mc, Q, 3)
            proj = (
                scaled_coords @ self.__omegafrq.T + self.__biasterm
            )  # (Nc, mc, Q, λ₅)
            zrff = self.__rffscale * cos(proj)  # (Nc, mc, Q, λ₅)
        mask = mskslice.unsqueeze(-1).unsqueeze(-1)  # (Nc, mc, 1, 1)
        zrff = zrff * mask

        # atmenv_m = φ_m · glob_ctx ≈ Σ_{m'} k(r̃_m, r̃_{m'}) · e_{m'}
        # neighbourhood embedding for each atom:
        # weighted sum of all other atoms' embeddings.
        # Same bmm trick as _pass_1: avoids (Nc, mc, Q, λ₅, λ₁) intermediate.
        Nc, mc, Q = zrff.shape[0], zrff.shape[1], zrff.shape[2]
        zb = zrff.permute(0, 2, 1, 3).reshape(
            Nc * Q, mc, lambda_5
        )  # (Nc*Q, mc, λ₅); query features
        cb = glob_ctx.reshape(
            Nc * Q, lambda_5, lambda_1
        )  # (Nc*Q, λ₅, λ₁); accumulated context

        # atmenv: (Nc, mc, Q, λ₁);
        # approximate kernel-weighted neighbour embedding sum
        atmenv = (
            torch.bmm(zb, cb).reshape(Nc, Q, mc, lambda_1).permute(0, 2, 1, 3)
        )

        # weights_m = |φ_m · features| ≈ Σ_{m'} k(r̃_m, r̃_{m'})
        # denominator for the normalised average w/ shape (Nc, mc, Q).
        # Represents total kernel weight seen by atom m.
        # abs() because cosine RFF features can be negative, making
        # the dot product negative. clamp(min=eps) avoids 0/0 in
        # both forward and backward (nan_to_num only fixes
        # the forward value but still produces grad/0 = NaN during backward).
        weights = torch.einsum("nmqd, nqd -> nmq", zrff, features).abs()

        # normalised aggregate w/ shape # (Nc, mc, Q, λ₁).
        # Kernel-weighted average of neighbour embeddings
        # fp32 (autocast disabled) so RMSNorm's fp32 weight matches
        # its input and dispatches to the fused kernel.
        pre_norm = atmenv / weights.unsqueeze(-1).clamp(min=epsilon_)
        with torch.autocast(device_type=pre_norm.device.type, enabled=False):
            agg = self.__rms_norm[rnd_idx](pre_norm.float())
        agg = agg.to(atmenv.dtype)

        # MishGLU gate: one linear projects to 2λ₁,
        # split into value p1 and gate p2.
        # gate = p1 · Mish(p2) selectively passes neighbourhood signal
        # into the residual stream.
        p1, p2 = self.__proj_agg[rnd_idx](agg).chunk(
            2, dim=-1
        )  # each (Nc, mc, Q, λ₁)
        gate = p1 * mish(p2) * mask  # (Nc, mc, Q, λ₁)

        # residual update
        new_emb = embslice + gate

        # Sigma update, in LOG space and inside the same window Embed uses.
        #
        # The update is an additive offset on log sigma: QDiagBilin's raw
        # output through `PBId` (near-identity at small |x|, bounded bend
        # at large |x| - see the `_sigbilin`/`_sigact` construction
        # comment for why this replaced PTanhShrink), then squashed back
        # into (sigma_floor, sigma_max) by `arctan_log_sigma_window`.
        #
        # The bound is load-bearing, not cosmetic. sigma is the RBF
        # bandwidth: _step1/_step2 form r/sigma, whose backward carries
        # -r/sigma**2, so d(kernel)/d(sigma) has a sigma^-2 pole. The
        # previous squareplus update was unbounded below (it asymptotes to
        # b/(4|x|) -> 0+), which discarded Embed's window on round 1:
        # measured min sigma 0.0107 with 41% of entries under Embed's 0.5
        # floor, and |dL/dsigma| up to 1.5e8 at sigma = 0.002 against 2.4e3
        # at the floor - an exact sigma^-2 scaling. Under grad_clip that
        # rescaled every other parameter's gradient toward zero.
        #
        # Masked at the end so padded atoms leave at sigma = 0, matching the
        # contract Embed establishes (embed.py: "masked last so padding
        # atoms come out at 0"). Without it they exit at sigma ~ 1.
        if rnd_idx >= len(self.__sigbilin):
            # Last round: nothing consumes this sigma, so don't build the
            # update at all. See the `_sigbilin` construction comment.
            return new_emb, sigslice

        # (Nc, mc, Q, 1) -> (Nc, mc, Q); f is the same spectrum at every q
        # slot, so QDiagBilin takes it without the expand a generic bilinear
        # would need.
        f_in = ffsslice.squeeze(-1)
        delta = self.__sigact[rnd_idx](
            self.__sigbilin[rnd_idx](self.__sig_norm[rnd_idx](new_emb), f_in)
        ).unsqueeze(-1)  # (Nc, mc, Q, 1)

        # clamp before log: Embed masks padded sigma to exactly 0.
        log_sig = torch.log(sigslice.clamp(min=math.exp(
            self.__log_sigma_floor
        )))
        new_sig = torch.exp(
            arctan_log_sigma_window(
                log_sig + delta, self.__log_sigma_mid, self.__log_sigma_half
            )
        ) * mask

        return new_emb, new_sig

    def __pass_2(
        self, cont: _PassContainer, eps: float, rnd_idx: int
    ) -> _PassContainer:
        """
        Compute per-atom neighbourhood aggregate
        and update embeddings and sigmas.

        Uses the fully-accumulated glob_ctx from _pass_1
        so every atom attends to all others despite M-chunking
        (all-pairs coverage is preserved).

        Parameters
        ----------
        cont : MessagePass._PassContainer
            Working state for this N-chunk, with `features` and
            `glob_ctx` already fully accumulated by `_pass_1`.
        eps : float
            Numerical floor for sigma clamping and the aggregate
            denominator inside `_step2`.
        rnd_idx : int
            Index (0-based) of the current message-passing round; passed
            through to `_step2` to select that round's weights.

        Returns
        -------
        MessagePass._PassContainer
            `cont` with `emb_n` and `sig_n` replaced by their updated
            values.
        """

        new_emb_m = []
        new_sig_m = []

        for m0 in range(0, cont.M, cont.Mchnk):
            m1 = min(m0 + cont.Mchnk, cont.M)

            # .contiguous(): see the matching comment in _pass_1
            emb_slice = cont.emb_n[:, m0:m1].contiguous()
            ffs_slice = cont.ffs_n[:, m0:m1].contiguous()
            sig_slice = cont.sig_n[:, m0:m1].contiguous()
            crd_slice = cont.crd_n[:, m0:m1].contiguous()
            msk_slice = cont.msk_n[:, m0:m1].contiguous()
            cont_feat = cont.features
            cont_chnv = cont.glob_ctx

            args = (
                emb_slice,
                crd_slice,
                sig_slice,
                msk_slice,
                ffs_slice,
                cont_feat,
                cont_chnv,
                eps,
                rnd_idx,
            )
            emb_c, sig_c = checkpoint(
                self.__step2_fn, *args, use_reentrant=False
            )  # type: ignore[misc]
            new_emb_m.append(emb_c)
            new_sig_m.append(sig_c)

        new_emb = torch.cat(new_emb_m, dim=1)  # (Nchnk, M, Q, λ₁)
        new_sig = torch.cat(new_sig_m, dim=1)  # (Nchnk, M, Q, 1)
        return cont._replace(emb_n=new_emb, sig_n=new_sig)

    @staticmethod
    def __align_chunk(real: int, chunk: int, align: int = 8) -> int:
        """
        Effective chunk size for a bucket whose real N or M may be far
        smaller than the configured chunk.

        Padding a bucket's real size all the way up to the full `chunk`
        wastes `chunk/real` memory and compute when `real` is a small
        fraction of `chunk` (a live run OOM'd on exactly this: a 7-molecule
        bucket forced to pad to a 128-molecule chunk, an 18x blowup).
        Rounding up to the nearest multiple of `align` instead bounds the
        padding waste to under `align` extra rows, independent of `chunk`.
        This produces a different concrete shape per distinct `real` value;
        that used to matter for `torch.compile`'s recompile budget, but
        `_step1_fn`/`_step2_fn` run eager now (see the class docstring's
        "Compile shapes" section), so shape variety here is free.

        Parameters
        ----------
        real : int
            The real (unpadded) size along this axis.
        chunk : int
            The configured chunk size (`n_chunk` or `m_chunk`). Only called
            when `real <= chunk`; the result is capped at `chunk`.
        align : int, optional
            Rounding granularity. Default 8.

        Returns
        -------
        int
            `real` rounded up to the next multiple of `align`, capped at
            `chunk`.
        """

        aligned = -(-real // align) * align  # ceil division
        return min(aligned, chunk)

    @staticmethod
    def __pad_to_multiple(
        t: torch.Tensor,
        chunk: int,
        dim: int
    ) -> torch.Tensor:
        """
        Zero-pad `t` along `dim` up to the next multiple of `chunk`.

        Lets every N-chunk/M-chunk fed to `_step1_fn`/`_step2_fn` be
        `chunk` elements wide, every call. Padded rows are all-zero; the
        padding_mask entries built from them are False (real atom/molecule
        = True), so _step1/_step2 zero their RFF contribution and the
        caller slices the padding back off before returning.

        Parameters
        ----------
        t : torch.Tensor
            Tensor to pad.
        chunk : int
            Chunk size; `t` is padded so `dim`'s size becomes a multiple
            of this.
        dim : int
            Dimension along which to pad.

        Returns
        -------
        torch.Tensor
            `t` zero-padded along `dim` up to the next multiple of
            `chunk` (returned unchanged if already a multiple).
        """

        size = t.shape[dim]
        pad_to = -(-size // chunk) * chunk  # ceil division
        if pad_to == size:
            return t
        pad_shape = list(t.shape)
        pad_shape[dim] = pad_to - size
        return torch.cat([t, t.new_zeros(pad_shape)], dim=dim)

    def forward(
        self,
        batch: Batch,
        embed_head: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        eps: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run λ₂ rounds of RFF message passing.

        Parameters
        ----------
        batch : Batch
            Molecule geometry; uses `coord` (N, M, 3) Cartesian positions
            (Å).
        embed_head : Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Output of Embed; `(embeds, f_mags, sigmas)` with embeds
            (N, M, 1, λ₁), f_mags and sigmas (N, M, Q, 1).
        eps : float
            Numerical floor for sigma clamping and the aggregate
            denominator.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            `(embeds, f_mags, sigmas)`, embeds updated to (N, M, Q, λ₁)
            and sigmas updated; f_mags unchanged.
        """

        embeds_raw, f_mags, sigmas = embed_head  # (N,M,1,λ₁); Q not expanded
        coord = batch.coord
        padding_mask = batch.padding_mask()
        # f_mags is padded below for the chunked passes but is returned
        # unchanged (per this method's contract), so keep the pristine,
        # un-padded reference to return at the end.
        f_mags_out = f_mags

        N_real, M_real, _ = coord.shape
        Q = sigmas.shape[2]
        Cn = self.__nchunk
        Cm = self.__mchunk

        # Pad only up to a small alignment (see __align_chunk), not all the
        # way to the full configured chunk, when the bucket's real N/M is
        # smaller than the chunk. _step1/_step2 run eager (see __init__),
        # so the resulting shape variety costs nothing extra.
        Cn_eff = self.__align_chunk(N_real, Cn) if N_real <= Cn else Cn
        Cm_eff = self.__align_chunk(M_real, Cm) if M_real <= Cm else Cm

        # Pad N (molecules) and M (atoms) up to multiples of Cn_eff/Cm_eff
        # so every N-chunk/M-chunk fed to _step1_fn/_step2_fn is exactly
        # (Cn_eff, Cm_eff) every call.
        for dim, chunk in ((0, Cn_eff), (1, Cm_eff)):
            embeds_raw = self.__pad_to_multiple(embeds_raw, chunk, dim)
            sigmas = self.__pad_to_multiple(sigmas, chunk, dim)
            coord = self.__pad_to_multiple(coord, chunk, dim)
            f_mags = self.__pad_to_multiple(f_mags, chunk, dim)
            padding_mask = self.__pad_to_multiple(padding_mask, chunk, dim)

        embeds = embeds_raw.expand(-1, -1, self.__q_points, -1)
        N, M = (coord.shape[0], coord.shape[1])

        for rnd_idx in range(self.__lambda_2):
            new_embeds_n = []
            new_sigmas_n = []

            for n0 in range(0, N, Cn_eff):
                n1 = (n0 + Cn_eff)

                # Nc derived from emb_s.shape[0] inside the closure rather
                # than closed over as a variable. Python closures capture
                # by reference so a loop variable would give the
                # last iteration's value when the checkpoint replays during
                # backward. rnd_idx is captured the same way it would be
                # via reference, so it's bound as a default arg (evaluated
                # once, at definition time, per outer-loop iteration)
                # instead, for the same reason.
                def _n_chunk_round(emb_s, msk_s, ffs_s, sig_s, crd_s, rnd_idx=rnd_idx):  # noqa: ANN001, E501
                    Nc_s = emb_s.shape[0]
                    cont = self._PassContainer(
                        M=M,
                        Nchnk=Nc_s,
                        Mchnk=Cm_eff,
                        emb_n=emb_s,
                        msk_n=msk_s,
                        ffs_n=ffs_s,
                        sig_n=sig_s,
                        crd_n=crd_s,
                        features=crd_s.new_zeros(Nc_s, Q, self.__lambda_5),
                        glob_ctx=crd_s.new_zeros(
                            Nc_s, Q, self.__lambda_5, self.__lambda_1
                        ),
                    )
                    cont = self.__pass_1(cont, eps)
                    # Mean-normalise the atom-sums. features/glob_ctx are
                    # Σ over all atoms (O(M): ~1e3-1e4 for a large molecule);
                    # BF16's ~7 mantissa bits lose relative precision at that
                    # magnitude in the _step2 contractions and their
                    # backward. Dividing both by the per-molecule real-atom
                    # count turns them into means (O(1)), keeping values in
                    # BF16's well-represented range.
                    count = cont.msk_n.sum(
                        dim=1, dtype=cont.features.dtype
                    ).clamp(min=1.0)  # (Nc,)
                    inv = (1.0 / count).view(-1, 1, 1)  # (Nc, 1, 1)
                    cont = cont._replace(
                        features=cont.features * inv,
                        glob_ctx=cont.glob_ctx * inv.unsqueeze(-1),
                    )
                    cont = self.__pass_2(cont, eps, rnd_idx)
                    return cont.emb_n, cont.sig_n

                # use_reentrant=False: glob_ctx lives only inside this
                # closure's call frame, so it's still freed once this N-chunk's
                # forward/recompute finishes rather than kept alive for every
                # N-chunk simultaneously.
                args = (
                    embeds[n0:n1].contiguous(),
                    padding_mask[n0:n1],
                    f_mags[n0:n1],
                    sigmas[n0:n1],
                    coord[n0:n1],
                )
                new_emb, new_sig = checkpoint(
                    _n_chunk_round, *args, use_reentrant=False
                )  # type: ignore[misc]
                new_embeds_n.append(new_emb)
                new_sigmas_n.append(new_sig)

            embeds = torch.cat(new_embeds_n, dim=0)  # (N, M, Q, λ₁)
            sigmas = torch.cat(new_sigmas_n, dim=0)  # (N, M, Q, 1)

        # strip the N/M padding added above before returning
        embeds = embeds[:N_real, :M_real]
        sigmas = sigmas[:N_real, :M_real]

        return embeds, f_mags_out, sigmas
