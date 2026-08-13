from typing import Callable

import torch
from torch import nn
from torch.nn.functional import softplus


class LambdaHead(nn.Module):
    """Amortized hyperparameter head: pooled (sigma, f_mag) -> per-q Λ.

    Maps a molecule's pooled per-q sigma and form-factor magnitude to a
    non-negative smoothing penalty Λ(q) for `PPSpline`, so Λ is learned
    per molecule and per q-point rather than fixed or free-floating.

    `Λ` must never go negative, PPSpline's `(I + DᵀΛD)` solve needs
    `Λ >= 0` to stay positive-definite. `softplus` gives that everywhere
    with a nonzero gradient at every input, unlike `relu` (dead/zero
    gradient once the pre-activation goes negative) and unlike `SqrP`
    (technically live everywhere too, but its tail decays so slowly,
    `~1/|x|`, that reaching a small Λ needs an impractically large
    negative pre-activation, `~1e6`; `softplus`'s tail decays
    exponentially, so a modest bias already gets Λ small at init).
    """

    __net: nn.Sequential
    __fwd_fn: Callable

    def __init__(self, hidden: int = 16, compile: bool = False) -> None:
        """Build the pooled-feature MLP.

        Parameters
        ----------
        hidden : int, optional
            Hidden width of the single Mish layer. Default 16.
        compile : bool, optional
            If True, `torch.compile` the elementwise/matmul math in
            `forward`. Default False.

        Returns
        -------
        None
        """

        super().__init__()
        self.__net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.Mish(),
            nn.Linear(hidden, 1),
        )
        # Bias-init the terminal layer toward a small Λ at step 0
        # (softplus(-10) ~= 4.5e-5), matching this codebase's "start
        # near the identity" convention. Weight is left at its default
        # random init, not zeroed: zeroing it would block gradient from
        # ever reaching the hidden layer (d(output)/d(hidden) scales
        # with the weight regardless of the activation's own gradient).
        terminal: nn.Linear = self.__net[-1]  # type: ignore[assignment]
        nn.init.constant_(terminal.bias, -10.0)
        self.__fwd_fn = (
            torch.compile(self.__forward_fn, dynamic=True, fullgraph=True)
            if compile
            else self.__forward_fn
        )

    @staticmethod
    def __forward_fn(  # no runtime checks here, torch.compile would break
        net: nn.Sequential,
        pooled_sigma: torch.Tensor,
        pooled_fmag: torch.Tensor,
    ) -> torch.Tensor:
        """Stack the two pooled features, apply the MLP, then softplus.

        Parameters
        ----------
        net : nn.Sequential
            The Linear -> Mish -> Linear stack.
        pooled_sigma : torch.Tensor
            Masked-mean sigma over atoms, shape (N, Q).
        pooled_fmag : torch.Tensor
            Masked-mean form factor magnitude over atoms, shape (N, Q).

        Returns
        -------
        torch.Tensor
            Λ, shape (N, Q), positive.
        """

        x = torch.stack((pooled_sigma, pooled_fmag), dim=-1)  # (N,Q,2)
        return softplus(net(x)).squeeze(-1)  # (N,Q)

    def forward(
        self,
        pooled_sigma: torch.Tensor,  # shape (N, Q)
        pooled_fmag: torch.Tensor,  # shape (N, Q)
    ) -> torch.Tensor:  # shape (N, Q)
        """Predict the smoothing penalty Λ for one channel.

        Parameters
        ----------
        pooled_sigma : torch.Tensor
            Masked-mean sigma over atoms, shape (N, Q).
        pooled_fmag : torch.Tensor
            Masked-mean form factor magnitude over atoms, shape (N, Q).

        Returns
        -------
        torch.Tensor
            Λ, shape (N, Q), positive.
        """

        return self.__fwd_fn(self.__net, pooled_sigma, pooled_fmag)
