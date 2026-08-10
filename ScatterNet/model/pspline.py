import math

import torch
from torch import nn
from typing import Any, Callable, NamedTuple, cast

import numpy as np
import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped


class PSpline(nn.Module):
    """
    Takes a raw input, Y_r, and outputs a smoothed spline, Y_s, via an
    adaptive P-spline with learned penalties.

    Given a B-spline basis matrix B with dim (U, V), and a diagonal matrix of
    local penalty terms Λ with dimensions (V-k, V-k), where
    Λ = diag(λ₁,...,λ_{V-k}), the adaptive penalized spline is a least-squares
    regression with a difference penalty matrix D of order k and
    dimensions (V-k, V):

        Y_s = B(BᵀB + DᵀΛD)⁻¹BᵀY_r

    Since we pick exactly one coefficient per grid point, B simplifies
    to the identity matrix with dimensions (V,V):

        Y_s = B(BᵀB + DᵀΛD)⁻¹BᵀY_r
            = I(IᵀI + DᵀΛD)⁻¹IᵀY_r
            = (I + DᵀΛD)⁻¹Y_r

    The scattering limit is:

        F_r   = (Y_coh_r)² + Y_inc_r

    Which when applied to our spline gives:

        F_s = ((I + DᵀΛ_cD)⁻¹Y_coh_r)² + ((I + DᵀΛ_iD)⁻¹Y_inc_r)²
    """

    _D:         Float[torch.Tensor, "G Q"] #noqa G = Q-2
    _DT:        Float[torch.Tensor, "G Q"] #noqa G = Q-2
    _I:         Float[torch.Tensor, "Q Q"] #noqa
    _delta_q:   float

    def __init__(self, q_points: int, delta_q: float=0.01) -> None:
        super().__init__()
        self._delta_q = delta_q
        self.register_buffer("_I", torch.eye(q_points))
        self.register_buffer (
            "_D",
            torch.diff(self._I, 2, dim=0) / (delta_q ** 2)
        )
        self.register_buffer(
            "_DT",
            torch.transpose_copy(self._D, 0, 1)
        )
