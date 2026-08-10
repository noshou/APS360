from typing import Callable, Tuple

import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped


class IQ2ndDerivative:
    """
    Analytical and Numerical Derivation of 2nd derivative
    of the Scattering Limits.

    Notes
    -----
    The total scattering limit is the sum of the limits of the coherent
    scattering contribution and the incoherent scattering contribution:

        f(q)    = coh(q) + inc(q)
                = [Σ_j c_j(q) · f_j(q)]² + Σ_j i_j(q) · [f_j(q)]²

    where:
        * q ∈ [0, 0.5]  : Scattering vector magnitude
        * Δq = 0.01     : Step size
        * j ∈ [1, M]    : Atom index

    Analytical Derivatives
    ----------------------
    Let u = Σ_j c_j(q) · f_j(q), then coh(q) = u².

    First derivative:

        coh'(q) = 2u · (du/dq)
                = 2·[Σ_j c_j(q)·f_j(q)]·Σ_k [c_k(q)f_k'(q) + f_k(q)c_k'(q)]

        inc'(q) = Σ_j [i_j'(q)·f_j(q)² + 2·i_j(q)·f_j(q)·f_j'(q)]

        f'(q)   = coh'(q) + inc'(q)
                = 2·[Σ_j c_j(q)·f_j(q)]·Σ_k [c_k(q)f_k'(q) + f_k(q)c_k'(q)]
                + Σ_j [i_j'(q)·f_j(q)² + 2·i_j(q)·f_j(q)·f_j'(q)]
    Second derivative:

        coh"(q) = 2 · [Σ_j (c_j(q)f_j'(q) + f_j(q)·c_j'·(q))]²
                + 2 · [Σ_j c_j(q)f_j(q)] · Σ_k [c_k"(q)f_k(q)
                + 2 · c_k'(q)·f_k'(q) + c_k(q)·f_k"(q)]

        inc"(q) = Σ_j [i_j"(q) f_j(q)² + 4 i_j'(q) f_j'(q) f_j(q)
                + 2·i_j(q)(f_j'(q))² + 2·i_j(q) f_j(q) f_j"(q)]

        f"(q)   = coh"(q) + inc"(q)
                = 2 · [Σ_j (c_j(q)f_j'(q) + f_j(q)·c_j'·(q))]²
                + 2 · [Σ_j c_j(q)f_j(q)] · Σ_k [c_k"(q)f_k(q)
                + 2 · c_k'(q)·f_k'(q) + c_k(q)·f_k"(q)]
                + Σ_j [i_j"(q) f_j(q)² + 4 i_j'(q) f_j'(q) f_j(q)
                + 2·i_j(q)(f_j'(q))² + 2·i_j(q) f_j(q) f_j"(q)]
    Numerical Derivatives
    ---------------------
    Central difference approximations:

        f(q+Δq) = [Σ_j c_j(q+Δq) · f_j(q+Δq)]² + Σ_j i_j(q+Δq)·[f_j(q+Δq)]²

        f(q-Δq) = [Σ_j c_j(q-Δq) · f_j(q-Δq)]² + Σ_j i_j(q-Δq)·[f_j(q-Δq)]²

        f'(q)   ≈ [f(q + Δq) - f(q - Δq)] / (2 · Δq)

        f"(q)   ≈ [f(q + Δq) - 2f(q) + f(q - Δq)] / (Δq)²

    Attributes
    ----------
    delta_q : float, default: 0.01
        Fixed step size between q‑points. If None, the spacing is inferred
        from the q grid passed to `forward`.
    q_grid  : the step size of the grid
    """

    _delta_q:   float
    _fwd_fn:    (Callable)

    def __init__(
        self,
        q_grid: Float[torch.Tensor, "Q"], #noqa
        delta_q: float=0.01,
        compile: bool = False
    ) -> None:
        """
        Parameters
        ----------
        delta_q : float, optional
            Spacing between consecutive q‑points.
        q_grid  : float, optional
        """
        super().__init__()

        assert torch.allclose(
            torch.diff(q_grid),
            torch.tensor(delta_q, dtype=q_grid.dtype, device=q_grid.device),
            atol=1e-5
        ), f"Grid steps must be uniformly equal to delta_q ({delta_q})."

        self._delta_q = delta_q

        self._fwd_fn = (
            torch.compile(self._forward_fn, dynamic=True, fullgraph=True)
            if compile
            else self._forward_fn
        )

    @staticmethod
    def _forward_fn( # can't use jaxtyping/beartype due to torch.compile
        intensity: torch.Tensor,
        delta: float
    ) -> torch.Tensor:

        # Compute second derivative via central differences
        d2I = torch.zeros_like(intensity)
        d2I[:, 1:-1] = (
            intensity[:, 2:]
            - 2.0 * intensity[:, 1:-1]
            + intensity[:, :-2]
        ) / (delta * delta)

        # Boundaries: second-order one-sided (using forward/backward)
        # Forward at left:   f"(x0) ≈ (2f0 - 5f1 + 4f2 - f3) / h²
        # Backward at right: f"(xN) ≈ (2fN - 5f_{N-1} + 4f_{N-2} - f_{N-3})/h²
        if intensity.size(1) >= 4:
            d2I[:, 0] = (
                2.0 * intensity[:, 0]
                - 5.0 * intensity[:, 1]
                + 4.0 * intensity[:, 2]
                - intensity[:, 3]
            ) / (delta * delta)
            d2I[:, -1] = (
                2.0 * intensity[:, -1]
                - 5.0 * intensity[:, -2]
                + 4.0 * intensity[:, -3]
                - intensity[:, -4]
            ) / (delta * delta)
        else:
            # Fallback to simpler one-sided if too few points
            d2I[:, 0] = (
                intensity[:, 2]
                - 2.0 * intensity[:, 1]
                + intensity[:, 0]
            ) / (delta * delta)
            d2I[:, -1] = (
                intensity[:, -1]
                - 2.0 * intensity[:, -2]
                + intensity[:, -3]
            ) / (delta * delta)

        return d2I

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        intensity: Float[torch.Tensor, "N Q"],  # noqa: F722
    ) -> Float[torch.Tensor, "N Q"]:  #noqa: F722
        """
        Compute second derivative of the intensity with respect
        to q using central finite differences.

        Parameters
        ----------
        intensity : (N, Q) torch.Tensor
            Intensity values evaluated at each q‑point.

        Returns
        -------
        d2I : (N, Q) torch.Tensor
            Second derivative I"(q).

        Raises
        ------
        ValueError
            If neither `delta_q` nor `q` provides a valid spacing, or if the
            grid is not uniform.
        """
        return self._fwd_fn(intensity, self._delta_q)
