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
from .layer_head import LayerHead


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
        sigma(m,q) = exp(mid + half * tanh((z + log env - mid) / half))
        env(q)     = min(1/q, sigma_max)
        mid, half  = midpoint and half-width of [log floor, log max]

    z is `_sigma`'s bilinear output, returned alongside sigma as
    `LayerHead.z_sigma` so the loss can penalise it directly.

    tanh, not clamp: a hard clamp has EXACTLY zero gradient outside its
    range, so an entry driven past a boundary pins there permanently. The
    sigma penalty in scatter_net.py was meant to keep z away from those
    boundaries, but it is computed on the post-clamp sigma, so its
    gradient is zeroed for precisely the entries that already escaped: it
    is preventative with no restoring force, and pinning becomes an
    absorbing state. Measured on the 2026-08-01 run, the pinned fraction
    ratcheted 4.1% (step 207) -> 38.5% (step 10661) and never recovered;
    the previous run ended at 95.8% pinned with |z| mean 57 against a
    5.30-nat window, i.e. sigma reduced to a binary floor-or-max switch.

    tanh maps R onto the OPEN interval, so it degrades gracefully instead
    of switching off. Measured d(sigma)/du against the run's own |z|
    distribution (mean 2.36, p99 7.78), taken at the high-q end where the
    envelope pushes hardest:

        z =  0.00 -> 4.2e-01      z = 15.00 -> 6.6e-06
        z =  2.36 -> 8.7e-02      z = 23.00 -> 0.0
        z =  7.78 -> 1.5e-03      z = 30.00 -> 0.0

    So this is NOT a guarantee of nonzero gradient: tanh hits exactly 1.0
    in fp32 once |u - mid|/half exceeds ~8.7, and the previous run's |z|
    mean of 57 is far past that. It buys a usable gradient across the
    range this run actually occupies, and nothing beyond it.

    The unconditional part is the paired penalty on `z_sigma` (see
    ScatterNet._loss_fn), which acts on the raw exponent and so has
    derivative 2*lambda_7*z no matter how saturated the entry is. That is
    the term that makes saturation recoverable; this one just widens the
    window in which it does not have to do all the work.

    Scaled so the map is the identity to first order at the midpoint
    (d/dz = 1 at z + log env = mid), matching the old clamp's behaviour in
    the interior where it was already well-behaved.

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
        sigma_init_gain: float = 0.1,
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
            Multiplier on `_sigma`'s default init. Default 0.1. See the note
            at the call site.
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
        # midpoint / half-width of the log-sigma window, for the tanh
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
        # persistent=False matches _q_weights_/_log_env_: derived from the
        # q-grid, not learned, so it stays out of the checkpoint.
        self.register_buffer(
            "_log_sigma_env", env.log().view(1, 1, -1), persistent=False
        )
        self._mbd = nn.Embedding(
            len(VOCAB) + 1, lambda_1, padding_idx=0 # +1 since 0 is padding idx
        )
        self._f0f1 = nn.Linear(lambda_1, qPoints)
        self._f2 = nn.Linear(lambda_1, qPoints)
        self._prelu = nn.PReLU(lambda_1)

        # The F.bilinear op dispatches to aten::_trilinear,
        # which profiled as a top CUDA op here (741ms / 12 calls) and forces
        # a torch.compile graph break. NoTrilinBilin is mathematically
        # identical (verified to fp32 rounding) with the same init, routing
        # through GEMM + elementwise instead.
        self._sigma = NoTrilinBilin(lambda_1, qPoints, qPoints)

        # Shrink _sigma's init. Under exp, z is in log-sigma units, so its
        # init spread is a spread in orders of magnitude: the default gives a
        # pre-activation std of ~3, i.e. a ~400x spread of kernel range at
        # step 0, parking many entries against the clamp where the gradient
        # is zero. The gain puts z near 0, so sigma starts near the envelope.
        # Softplus did not need this, being linear for large z.
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
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
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
        torch.Tensor
            `z_sigs`, the pre-saturation exponent offset (`sigma`'s raw
            bilinear output), shape (N, M, Q, 1).
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

        # z is the raw exponent OFFSET from the 1/q envelope; it is exactly
        # what the sigma penalty targets, so it is returned rather than
        # reconstructed downstream (reconstruction would have to invert the
        # saturation, which is ill-conditioned near the boundaries).
        z_sig = sigma(embed, f_mag)  # (N, M, Q)

        # sigmas (N, M, Q, 1): exp(mid + half*tanh((z + log_env - mid)/half)).
        # Saturation is on the SUM; saturating z and log_env separately would
        # compose their bounds. No `+ eps` needed, unlike softplus: exp is
        # positive and the lower asymptote floors it. See the class docstring
        # for why this is tanh and not clamp.
        u = z_sig + log_sigma_env
        log_sigmas = log_sigma_mid + log_sigma_half * torch.tanh(
            (u - log_sigma_mid) / log_sigma_half
        )
        # masked last so padding atoms come out at 0, not at sigma_floor.
        sigmas = torch.exp(log_sigmas).unsqueeze(-1) * masks.unsqueeze(-1)
        # masked for the same reason: padding atoms must not contribute to
        # the penalty's atom-count-normalised sum.
        z_sigs = z_sig.unsqueeze(-1) * masks.unsqueeze(-1)

        # squeeze padding mask (N,M,1) -> (N,M,1,1)
        masks = masks.unsqueeze(-1)

        # pad the layer heads to have correct dimensions, drop masked atoms:
        # embeds → (N, M, 1, λ₁); f_mags → (N, M, Q, 1)
        embeds = embed.unsqueeze(-2)
        f_mags = f_mag.unsqueeze(-1) * masks

        return embeds, f_mags, sigmas, z_sigs

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
            `sigmas` (N, M, Q, 1), and `z_sigma` (N, M, Q, 1).
        """

        embeds, f_mags, sigmas, z_sigma = self._fwd_fn(
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

        return LayerHead(
            embeds=embeds, f_mags=f_mags, sigmas=sigmas, z_sigma=z_sigma
        )
