import math
from typing import Callable

import torch
from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import jaxtyped
from torch import nn

from Preprocess import VOCAB

from ..batching import Batch
from ..utils.no_trilin_bilin import NoTrilinBilin
from ..utils.sigma_window import arctan_log_sigma_window
from .layer_head import LayerHead

# Multiplier on `_f0f1`/`_f2`'s default weight init when a physical
# `f_init` bias is supplied. See `Embed`'s "Form-factor init" docstring
# section for how this number was chosen; it is a constant rather than a
# constructor keyword because `Embed.__init__`'s signature is a fixed
# contract with `ScatterNet.__init__`.
_F_INIT_WEIGHT_GAIN = 0.05


class Embed(nn.Module):
    """
    Embed each atom into a learned representation
    and estimate its scattering strength.

    Produces three outputs per atom:
        - embeds: learned identity embedding, shape (N, M, 1, λ₁)
        - f_mags: form factor magnitude |f(q)| per q-point, shape (N, M, Q, 1)
        - sigmas: RFF kernel bandwidth per q-point, shape (N, M, Q, 1)

    Sigma parameterisation
    ----------------------
        u          = z + log env
        sigma(m,q) = exp(mid + half * (2/pi) * arctan(pi/2 * (u-mid)/half))
        env(q)     = min(1/q, sigma_max)
        mid, half  = midpoint and half-width of [log floor, log max]

    z is `_sigma`'s bilinear output. The arctan squashes log sigma into
    the open interval (log floor, log max), and is scaled so the map is
    the identity to first order at the midpoint (d/du = 1 at u = mid).

    Why arctan and not tanh: arctan's derivative decays polynomially
    (1/x^2) where tanh's decays exponentially (e^-2x), so arctan keeps a
    usable gradient far outside the window. At |u-mid| = 57 nats, the
    previous run's mean, tanh gives 0.0 in fp32 and arctan gives 8.7e-4.
    Plain torch.atan also has a single-op backward, 1/(1+x^2), that
    cannot overflow or emit NaN for any finite input. The cost is that
    arctan approaches its asymptote more slowly, so a given |z| reaches a
    less extreme sigma than tanh would (61% of the window vs 71% at the
    run's mean |z| = 2.36); `_sigma`'s weights absorb that.

    env ~ 1/q: sigma is the RBF bandwidth (MessagePass forms r/sigma). In
    I(q) = sum_mm' f f' sinc(q r), sinc first vanishes at q r = pi, so q is
    sensitive to separations O(1/q). Low q sees global geometry, high q
    local. Same reasoning as the sigma-penalty weighting in scatter_net.py,
    applied in the forward rather than the loss.

    exp not softplus: what matters is relative sensitivity
    dln(sigma)/dz = g'/g. For exp it is 1 (constant); for softplus it decays
    as ~1/sigma. Since env spans 2 A (q=0.5) to 100 A (q=0.01), softplus
    makes low q ~50x less responsive per step than high q, the wrong way
    round. (ELU+1 shares softplus's slope-1 large-sigma limit, so it fixes
    nothing here.)

    Envelope applied in log space because exp(z)*env == exp(z + log env),
    which also lets the clamp act on one summed exponent. Clamping z and env
    separately would compose their bounds into [floor*env_min, max*env_max].

    Clamp on the exponent, not the output: exp overflows to inf long before
    the result could be clamped, so clamping first bounds the largest value
    at sigma_max.

    Bounds come from the grid: q=0.01 probes 1/q = 100 A (longest resolvable
    scale), q=0.5 probes 2 A. floor=0.5 leaves margin and caps the -r/sigma^2
    term in r/sigma's backward at ~4, versus ~1e6 at the old 1e-3.

    MessagePass re-applies this same window (via the shared
    `arctan_log_sigma_window`) on every round, so the bound holds on the
    sigma that actually reaches the kernel and the loss, not just on
    Embed's output.

    Form-factor init
    ----------------
    With ordinary random init the untrained `_f0f1`/`_f2` emit
    f_mag ~ 0.5 while the physical value is ~7 (vocab mean over the
    xraydb table). Both OutputHead channels are homogeneous of degree 2
    in f, so d log I / d log f_mag = 2 exactly, and a 14x error in f is a
    ~200x error in I(q). At init that single mismatch was 99.5% of the
    whole gradient norm (`_f0f1` and `_f2` measured at |g| = 96 and 102,
    against 0.10 for the largest MessagePass weight), which is what the
    global-norm clip was almost entirely spending itself on.

    Passing `f_init` puts `_f0f1.bias` at the physical curve and
    `_f2.bias` at zero, so f_mag starts at the right magnitude and the
    layers only have to learn the per-element correction. The weights are
    additionally scaled by `_F_INIT_WEIGHT_GAIN = 0.05` so the random part
    (std ~0.58 at lambda_1 = 128) sits well under the ~7 bias rather than
    swamping it. The gain does not slow learning: for a linear layer
    dL/dW = dL/dy * x, which is independent of W's magnitude.
    """

    _mbd: nn.Embedding      # atom identity in learned space
    _f0f1: nn.Linear        # approximate real form factor at each q
    _f2: nn.Linear          # approximate imaginary form factor at each q
    _prelu: nn.PReLU        # prelu activation function
    _sigma: NoTrilinBilin   # computes positional scaling from embed and f_mag
    _fwd_fn: (Callable)     # torch.compiled or plain _forward_fn flag

    def __init__(
        self,
        lambda_1: int,
        qgrid: torch.Tensor,
        sigma_max: float = 100.0,
        sigma_floor: float = 0.5,
        sigma_init_gain: float = 1.0,
        f_init: "torch.Tensor | None" = None,
        compile: bool = False
    ) -> None:
        """Construct the embedding, form-factor, and sigma-estimation layers.

        Parameters
        ----------
        lambda_1 : int
            Embedding dimension (λ₁).
        qgrid : torch.Tensor
            The q-grid, shape (Q,). Values (not just the length) are used to
            build the 1/q sigma envelope.
        sigma_max : float, optional
            Cap on the 1/q envelope, in Angstroms. Default 100.
        sigma_floor : float, optional
            Lower clamp on sigma, in Angstroms. Default 0.5. Replaces the
            1e-3 floor `eps_msgp` applied at `message_pass.py:426`. Moved
            here because `eps_msgp` also floors the dimensionless
            kernel-weight sum at `message_pass.py:614`, so raising the shared
            constant would have changed the aggregation normalisation too.
        sigma_init_gain : float, optional
            Multiplier on `_sigma`'s default init. Default 1.0. See the note
            at the call site.
        f_init : torch.Tensor, optional
            Physical form-factor magnitudes to bias `_f0f1` toward, shape
            (Q,). If given, `_f0f1.bias` is set to it and `_f2.bias` to
            zero so an untrained `Embed` emits `f_mag ~ f_init` instead of
            a random O(0.5) value. See the "Form-factor init" section of
            the class docstring. Default None (ordinary random init).
        compile : bool, optional
            If True, torch.compile `_forward_fn`. Default is False.

        Returns
        -------
        None
        """

        super().__init__()
        qPoints = qgrid.shape[0]
        self._log_sigma_floor = math.log(sigma_floor)
        self._log_sigma_max = math.log(sigma_max)
        # midpoint / half-width of the log-sigma window, for the arctan
        # saturation that replaced the hard clamp (see class docstring).
        self._log_sigma_mid = 0.5 * (
            self._log_sigma_floor + self._log_sigma_max
        )
        self._log_sigma_half = 0.5 * (
            self._log_sigma_max - self._log_sigma_floor
        )

        # 1/q sigma envelope, stored in log space (see class docstring).
        # Before this, the forward never saw q's values at all, only Q, so
        # the bilinear had to discover the 1/q scaling across Q independent
        # output channels. Now it only learns an O(1) correction on top.
        q = qgrid.detach().clone().float().flatten()
        positive = q[q > 0]
        # q = 0 has no finite 1/q and is the longest-range point, so it caps.
        q_min = positive.min() if positive.numel() else torch.tensor(1.0)
        env = torch.where(
            q > 0, 1.0 / q.clamp(min=q_min), torch.full_like(q, sigma_max)
        ).clamp(max=sigma_max)
        # persistent=False matches _q_weights_: derived from the q-grid,
        # not learned, so it stays out of the checkpoint.
        self.register_buffer(
            "_log_sigma_env", env.log().view(1, 1, -1), persistent=False
        )
        self._mbd = nn.Embedding(
            len(VOCAB) + 1, lambda_1, padding_idx=0 # +1 since 0 is padding idx
        )
        self._f0f1 = nn.Linear(lambda_1, qPoints)
        self._f2 = nn.Linear(lambda_1, qPoints)
        self._prelu = nn.PReLU(lambda_1)

        # Form-factor init: see the class docstring section of the same name.
        if f_init is not None:
            with torch.no_grad():
                self._f0f1.weight.mul_(_F_INIT_WEIGHT_GAIN)
                self._f2.weight.mul_(_F_INIT_WEIGHT_GAIN)
                self._f0f1.bias.copy_(f_init.to(self._f0f1.bias.dtype))
                self._f2.bias.zero_()

        # The F.bilinear op dispatches to aten::_trilinear,
        # which profiled as a top CUDA op here (741ms / 12 calls) and forces
        # a torch.compile graph break. NoTrilinBilin is mathematically
        # identical (verified to fp32 rounding) with the same init, routing
        # through GEMM + elementwise instead.
        self._sigma = NoTrilinBilin(lambda_1, qPoints, qPoints)

        # Shrink _sigma's init. Under exp, z is in log-sigma units, so its
        # init spread is a spread in orders of magnitude: a pre-activation
        # std of ~3 is a ~400x spread of kernel range at step 0, parking
        # many entries against the window where the gradient is smallest.
        # The gain puts z near 0, so sigma starts near the envelope.
        # Softplus did not need this, being linear for large z.
        #
        # Default is 1.0, not the old 0.1, because NoTrilinBilin's init now
        # uses the true fan-in `in1*in2` instead of `in1`. That divides the
        # bound by sqrt(in2) = sqrt(Q) = 7.14 on its own, so gain 1.0 here
        # reproduces the old 0.1-with-wrong-fan-in scale to within 40%
        # (0.1/sqrt(128) = 8.8e-3 vs 1.0/sqrt(128*51) = 1.2e-2).
        with torch.no_grad():
            self._sigma.weight.mul_(sigma_init_gain)
            if self._sigma.bias is not None:
                self._sigma.bias.mul_(sigma_init_gain)

        self._fwd_fn = (
            torch.compile(self._forward_fn, dynamic=True, fullgraph=True)
            if compile
            else self._forward_fn
        )

    @staticmethod
    def _forward_fn( # can't use jaxtyping/beartype due to torch.compile
        prelu: nn.PReLU,
        mbd: nn.Embedding,
        f0f1: nn.Linear,
        f2: nn.Linear,
        sigma: NoTrilinBilin,
        log_sigma_env: torch.Tensor,
        log_sigma_mid: float,
        log_sigma_half: float,
        vocabs: torch.Tensor,
        mask: torch.Tensor,
        eps: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute atom embeddings, form factor magnitudes, and sigmas.

        Parameters
        ----------
        prelu : torch.nn.PReLU
            Activation applied to the raw atom embedding.
        mbd : torch.nn.Embedding
            Atom identity embedding table.
        f0f1 : torch.nn.Linear
            Linear layer approximating the real part of the form factor.
        f2 : torch.nn.Linear
            Linear layer approximating the imaginary part of the form
            factor.
        sigma : NoTrilinBilin
            Bilinear layer (nn.Bilinear-equivalent) computing the positional
            scaling factor from the embedding and form factor magnitude.
        log_sigma_env : torch.Tensor
            Fixed (non-learned) log of the 1/q kernel-range envelope, shape
            (1, 1, Q).
        log_sigma_mid : float
            Midpoint of the log-sigma window, (log floor + log max) / 2.
        log_sigma_half : float
            Half-width of the log-sigma window, (log max - log floor) / 2.
        vocabs : torch.Tensor
            Atom vocabulary indices, shape (N, M).
        mask : torch.Tensor
            Padding mask, shape (N, M); True marks a real (non-padding)
            atom.
        eps : float
            Numerical floor added to form factor magnitudes and sigmas.

        Returns
        -------
        torch.Tensor
            `embeds`, atom identity representations, shape (N, M, 1, λ₁).
        torch.Tensor
            `f_mags`, form factor magnitudes, shape (N, M, Q, 1).
        torch.Tensor
            `sigmas`, kernel variance bandwidths, shape (N, M, Q, 1).
        """

        # get embedding: "what is this atom?" → (N, M, λ₁)
        # since linear layers applies only to last dimension, we must
        # keep this order stable since we want to activate using prelu.
        # However, since nn.PReLU() acts on dim=1 for |dim| >= 2,
        # first we must transpose embed to (N,λ₁,M),
        # do prelu, then re-transpose to (N,M,λ₁)
        embed = (prelu(mbd(vocabs).transpose(-1, -2))).transpose(-1, -2)

        # estimate form factors → (N, M, Q)
        # as above, must keep this stable at 3 dimensions
        f_rel = f0f1(embed) + eps
        f_img = f2(embed)
        f_mag = torch.hypot(f_rel, f_img) + eps

        # get padding mask (N,M) -> (N,M,1)
        masks = mask.unsqueeze(-1)

        # z is the raw exponent OFFSET from the 1/q envelope.
        z_sig = sigma(embed, f_mag)  # (N, M, Q)

        # sigmas (N, M, Q, 1): squash log sigma into the open window
        # (log floor, log max) with a scaled arctan, so a saturated entry
        # keeps a nonzero gradient and can be pulled back. The 2/pi and
        # pi/2 make d(log_sigmas)/du = 1 at the midpoint, i.e. the map is
        # the identity where it does not need to bend. arctan not tanh:
        # 1/x^2 tails instead of e^-2x, and torch.atan's backward cannot
        # overflow or NaN. See the class docstring.
        #
        # Saturation is on the SUM; saturating z and log_env separately
        # would compose their bounds. No `+ eps` needed, unlike softplus:
        # exp is positive and the lower asymptote floors it.
        u = z_sig + log_sigma_env
        log_sigmas = arctan_log_sigma_window(
            u, log_sigma_mid, log_sigma_half
        )
        # masked last so padding atoms come out at 0, not at sigma_floor.
        sigmas = torch.exp(log_sigmas).unsqueeze(-1) * masks.unsqueeze(-1)

        # squeeze padding mask (N,M,1) -> (N,M,1,1)
        masks = masks.unsqueeze(-1)

        # pad the layer heads to have correct dimensions, drop masked atoms:
        # embeds → (N, M, 1, λ₁); f_mags → (N, M, Q, 1)
        embeds = embed.unsqueeze(-2)
        f_mags = f_mag.unsqueeze(-1) * masks

        return embeds, f_mags, sigmas

    @jaxtyped(typechecker=beartype)
    def forward(self, batch: Batch, eps: float) -> LayerHead:
        """Embed a batch's atoms and estimate their scattering properties.

        Parameters
        ----------
        batch : Batch
            Input batch; uses `batch.vocab`, shape (N, M), atom indices.
        eps : float
            Numerical floor to avoid zero form factors and sigmas.

        Returns
        -------
        LayerHead
            Container with `embeds` (N, M, 1, λ₁), `f_mags` (N, M, Q, 1),
            and `sigmas` (N, M, Q, 1).
        """

        embeds, f_mags, sigmas = self._fwd_fn(
            self._prelu,
            self._mbd,
            self._f0f1,
            self._f2,
            self._sigma,
            self._log_sigma_env,
            self._log_sigma_mid,
            self._log_sigma_half,
            batch.vocab,
            batch.padding_mask(),
            eps,
        )

        return LayerHead(embeds=embeds, f_mags=f_mags, sigmas=sigmas)
