from typing import Callable

import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from pyteals import PPSpline
from torch import nn

from ..batching import Batch
from .lambda_head import LambdaHead


class SplineSmooth(nn.Module):
    """Smooths and combines the coherent/incoherent Debye limits.

    Pools per-atom sigma/f_mag to per-molecule, per-q features, predicts a
    non-negative smoothing penalty Λ per channel via `LambdaHead`, and smooths
    each channel's raw curve with its own `PPSpline` instance before
    squaring the coherent channel and adding the incoherent one.
    """

    __pps_coh: PPSpline
    __pps_inc: PPSpline
    __lmb_coh: LambdaHead
    __lmb_inc: LambdaHead
    __fwd_fn: Callable

    def __init__(
        self,
        q_points: int,
        delta_q: float = 0.01,
        compile: bool = False,
    ) -> None:
        """Build the per-channel spline smoothers and lambda heads.

        Parameters
        ----------
        q_points : int
            Number of points on the (uniform) q-grid, `Q`.
        delta_q : float, optional
            Grid spacing, passed through to `PPSpline`. Default 0.01.
        compile : bool, optional
            If True, `torch.compile` the pooling/combine math in
            `forward`, and pass `compile=True` (with `dynamic=True,
            fullgraph=True`) to both `PPSpline` instances. Default False.

        Returns
        -------
        None
        """

        super().__init__()
        self.__pps_coh = PPSpline(q_points, delta_q, compile, True, True)
        self.__pps_inc = PPSpline(q_points, delta_q, compile, True, True)
        self.__lmb_coh = LambdaHead(compile=compile)
        self.__lmb_inc = LambdaHead(compile=compile)
        self.__fwd_fn = (
            torch.compile(self.__forward_fn, dynamic=True, fullgraph=True)
            if compile
            else self.__forward_fn
        )

    @staticmethod
    def __forward_fn(  # no jaxtype/beartype here, torch.compile will break
        pps_coh: PPSpline,
        pps_inc: PPSpline,
        lmb_coh: LambdaHead,
        lmb_inc: LambdaHead,
        sigmas: torch.Tensor,
        f_mags: torch.Tensor,
        coh: torch.Tensor,
        inc: torch.Tensor,
        batch: Batch,
    ) -> torch.Tensor:
        """Pool, predict Λ, smooth, and combine.

        Parameters
        ----------
        pps_coh, pps_inc : PPSpline
            Per-channel spline smoothers.
        lmb_coh, lmb_inc : LambdaHead
            Per-channel amortized Λ heads, fed the same pooled features.
        sigmas, f_mags : torch.Tensor
            Per-atom sigma/form-factor magnitude, shape (N, M, Q).
        coh, inc : torch.Tensor
            Raw accumulated coherent/incoherent curves, shape (N, Q).
        batch : Batch
            Input batch; uses `batch.padding_mask()`.

        Returns
        -------
        torch.Tensor
            Combined I(q), shape (N, Q).
        """

        mask = batch.padding_mask()  # (N,M)
        n_atms = mask.sum(dim=1, keepdim=True).float().clamp(min=1)  # (N,1)
        mask2d = mask.unsqueeze(-1)  # (N,M,1)
        sigmas_pooled = (sigmas * mask2d).sum(dim=1) / n_atms  # (N,Q)
        f_mags_pooled = (f_mags * mask2d).sum(dim=1) / n_atms  # (N,Q)

        lam_coh = lmb_coh(sigmas_pooled, f_mags_pooled)
        lam_inc = lmb_inc(sigmas_pooled, f_mags_pooled)

        coh_smooth = pps_coh(coh, lam_coh)
        inc_smooth = pps_inc(inc, lam_inc)

        return coh_smooth**2 + inc_smooth

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        sigmas: Float[torch.Tensor, "N M Q"],  # noqa: F722
        f_mags: Float[torch.Tensor, "N M Q"],  # noqa: F722
        coh: Float[torch.Tensor, "N Q"],  # noqa: F722
        inc: Float[torch.Tensor, "N Q"],  # noqa: F722
        batch: Batch,
    ) -> Float[torch.Tensor, "N Q"]:  # noqa: F722
        """Smooth and combine the coherent/incoherent Debye limits.

        Parameters
        ----------
        sigmas : torch.Tensor
            Per-atom RFF kernel bandwidth, shape (N, M, Q).
        f_mags : torch.Tensor
            Per-atom form factor magnitude, shape (N, M, Q).
        coh : torch.Tensor
            Raw accumulated coherent sum, unsquared, shape (N, Q).
        inc : torch.Tensor
            Raw accumulated incoherent sum, shape (N, Q).
        batch : Batch
            Input batch; uses `batch.padding_mask()`.

        Returns
        -------
        torch.Tensor
            Predicted I(q), shape (N, Q).
        """

        return self.__fwd_fn(
            self.__pps_coh,
            self.__pps_inc,
            self.__lmb_coh,
            self.__lmb_inc,
            sigmas,
            f_mags,
            coh,
            inc,
            batch,
        )
