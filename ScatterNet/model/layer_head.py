from typing import NamedTuple, Optional

import torch
from jaxtyping import Float


class LayerHead(NamedTuple):
    """Immutable container passed between Embed, MessagePass, and OutputHead.

    Attributes
    ----------
    embeds : torch.Tensor
        Per-atom learned embeddings; shape (N, M, 1, λ₁) from Embed,
        (N, M, Q, λ₁) after MessagePass.
    f_mags : torch.Tensor
        Form factor magnitudes |f0+f1 + i*f2| per atom per q-point, shape
        (N, M, Q, 1).
    sigmas : torch.Tensor
        Per-atom RFF kernel bandwidth per q-point, shape (N, M, Q, 1).
    z_sigma : torch.Tensor or None
        Embed's PRE-saturation sigma exponent offset, shape (N, M, Q, 1);
        literally `_sigma`'s raw bilinear output, since sigma is built as
        exp(saturate(z + log env)) and the penalty's target is
        z = log(sigma) - log(env). Carried through to the loss so the
        sigma penalty keeps a gradient even where the saturation has
        flattened d(sigma)/dz to ~0. None outside Embed's own output.

        MessagePass passes this through untouched: it describes Embed's
        parameterisation of the round-0 bandwidth, not the per-round
        `softplus(sig + tanhshrink(...))` updates, which are unbounded but
        smooth and are covered by the penalty on the final sigma instead.
    """

    embeds: Float[torch.Tensor, "N M * λ₁"] # atom embeddings        #noqa:F722
    f_mags: Float[torch.Tensor, "N M Q 1"]  # form factor magnitudes #noqa:F722
    sigmas: Float[
        torch.Tensor, "N M Q 1"                                      #noqa:F722
    ]  # per-atom positional scaling factor
    z_sigma: Optional[
        Float[torch.Tensor, "N M Q 1"]                               #noqa:F722
    ] = None  # pre-saturation sigma exponent offset
