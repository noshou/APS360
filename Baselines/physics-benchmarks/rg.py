import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped

from Baselines.baseline import Baseline, build_fmag_table
from ScatterNet.batching import Batch


@jaxtyped(typechecker=beartype)
def _guinier_stats(
    batch: Batch,
    ftable: Float[torch.Tensor, "V Q"],  # noqa: F722
) -> tuple[
    Float[torch.Tensor, "N 1"], Float[torch.Tensor, "N 1"]  # noqa: F722
]:
    """Per-molecule Rg^2 and I(0), shared by both Rg baselines.

    Parameters
    ----------
    batch : Batch
        Batch of molecules.
    ftable : torch.Tensor
        Form factor magnitude table of shape ``(V, Q)``, already moved to
        ``batch``'s device.

    Returns
    -------
    tuple of (torch.Tensor, torch.Tensor)
        ``(rg2, i0)``, each of shape ``(N, 1)``. ``rg2`` is the
        centroid-subtracted mean squared radius; ``i0 = (sum_i f_i(0))^2``.
    """
    mask = batch.padding_mask().float()  # (N, M)
    n_atoms = mask.sum(dim=1, keepdim=True).clamp(min=1)  # (N, 1)

    r2 = (batch.coord**2).sum(dim=-1)  # (N, M)
    rg2 = (r2 * mask).sum(dim=1, keepdim=True) / n_atoms  # (N, 1)

    f0 = ftable[batch.vocab, 0]  # (N, M)
    i0 = ((f0 * mask).sum(dim=1, keepdim=True)) ** 2  # (N, 1)

    return rg2, i0


class RgBaseline(Baseline):
    """
    Global-geometry baseline: Guinier approximation
    I(q) = I(0)*exp(-q^2*Rg^2/3).

    Rg is computed from centroid-subtracted coordinates; I(0) = (sum
    f_i(0))^2. Only physically valid for q*Rg < 1.3 (Guinier region).
    For large molecules with Rg >> 1/q_max this covers only the first
    few q-points. Beating this baseline proves the model captures
    structure beyond global size.
    """

    _qgrid: Float[torch.Tensor, "Q"]  # noqa: F722,F821
    _fmag_table: Float[torch.Tensor, "V Q"]  # noqa: F722

    @jaxtyped(typechecker=beartype)
    def __init__(
        self,
        qgrid: Float[torch.Tensor, "Q"],  # noqa: F722,F821
        energy: float,
    ) -> None:
        """
        Args:
            qgrid:  q-point grid in Å⁻¹ (Q,)
            energy: X-ray energy in eV for anomalous f1/f2 corrections
        """
        self._qgrid = qgrid
        self._fmag_table = build_fmag_table(qgrid, energy)

    @jaxtyped(typechecker=beartype)
    def __call__(self, batch: Batch) -> Float[torch.Tensor, "N Q"]:  # noqa: F722
        """Return Guinier-approximated I(q) for each molecule."""
        device = batch.vocab.device
        rg2, i0 = _guinier_stats(batch, self._fmag_table.to(device))
        q2 = self._qgrid.to(device) ** 2  # (Q,)

        return i0 * torch.exp(-q2.unsqueeze(0) * rg2 / 3)  # (N, Q)


class GuinierPorodBaseline(Baseline):
    """
    Global-geometry baseline: unified Guinier-Porod model (Hammouda, 2010).

    ``RgBaseline``'s pure Guinier law I(q) = I0*exp(-q^2*Rg^2/3) is only
    physically valid for q*Rg < 1.3; beyond that it decays far too fast
    relative to real scattering curves, which instead settle into a
    power-law tail I(q) ~ q^-d. This model keeps the Guinier form below a
    crossover q1 and switches to a matched power-law tail above it, with q1
    and the tail prefactor chosen so I(q) and its slope are continuous at
    the crossover:

        q1 = sqrt(3*d/2) / Rg
        I(q) = I0*exp(-q^2*Rg^2/3)                    for q <= q1
        I(q) = I0*exp(-q1^2*Rg^2/3) * (q1/q)^d         for q >  q1

    Only Rg and I(0) are used (same as ``RgBaseline``); ``d`` is a fixed
    Porod exponent, default 4.0 (smooth-surface/compact-globular limit).
    Beating this baseline proves the model captures structure beyond a
    single global size scale over the *entire* q range, not just the
    low-q Guinier region.
    """

    _qgrid: Float[torch.Tensor, "Q"]  # noqa: F722,F821
    _fmag_table: Float[torch.Tensor, "V Q"]  # noqa: F722
    _porod_exponent: float

    @jaxtyped(typechecker=beartype)
    def __init__(
        self,
        qgrid: Float[torch.Tensor, "Q"],  # noqa: F722,F821
        energy: float,
        porod_exponent: float = 4.0,
    ) -> None:
        """
        Parameters
        ----------
        qgrid : torch.Tensor
            q-point grid in Å⁻¹, shape ``(Q,)``.
        energy : float
            X-ray energy in eV for anomalous f1/f2 corrections.
        porod_exponent : float, default 4.0
            High-q power-law exponent ``d`` (Porod's law: 4.0 for a smooth
            compact interface; lower values suit mass-fractal-like
            molecules).
        """
        self._qgrid = qgrid
        self._fmag_table = build_fmag_table(qgrid, energy)
        self._porod_exponent = porod_exponent

    @jaxtyped(typechecker=beartype)
    def __call__(self, batch: Batch) -> Float[torch.Tensor, "N Q"]:  # noqa: F722
        """Return the Guinier-Porod unified-model I(q) for each molecule.

        Parameters
        ----------
        batch : Batch
            Batch of molecules to predict scattering curves for.

        Returns
        -------
        torch.Tensor
            Predicted I(q) curves of shape ``(N, Q)``.
        """
        device = batch.vocab.device
        rg2, i0 = _guinier_stats(batch, self._fmag_table.to(device))
        rg = rg2.clamp(min=1e-8).sqrt()  # (N, 1)
        d = self._porod_exponent

        q1 = (1.5 * d) ** 0.5 / rg  # (N, 1)
        i_at_q1 = i0 * torch.exp(-(q1**2) * rg2 / 3)  # (N, 1)

        q = self._qgrid.to(device).unsqueeze(0)  # (1, Q)
        q_safe = q.clamp(min=1e-8)

        i_guinier = i0 * torch.exp(-(q**2) * rg2 / 3)  # (N, Q)
        i_porod = i_at_q1 * (q1 / q_safe) ** d  # (N, Q)

        return torch.where(q > q1, i_porod, i_guinier)
