import numpy as np
import torch

from collections.abc import Iterable
from jaxtyping       import Float, jaxtyped
from beartype        import beartype
from ScatterNet.batching import Batch
from Baselines.baseline  import Baseline


class PowerLawBaseline(Baseline):
    """
    Global-geometry baseline: power law I(q) = n_atoms · A · q^(-α).

    A and α are fit in log-log space from training data (Welford online average).
    q=0 is handled separately as n_atoms · I(0)/n_atoms from training.
    Most relevant for the high-q / Porod regime; less meaningful when q_max is small.
    Beating this baseline proves the model captures structure beyond a global power law.
    """

    _qgrid:          Float[torch.Tensor, "Q"]
    _A:              float | None
    _alpha:          float | None
    _i0_per_atom:    float | None

    @jaxtyped(typechecker=beartype)
    def __init__(self, qgrid: Float[torch.Tensor, "Q"]) -> None:
        """
        Args:
            qgrid: q-point grid in Å⁻¹ (Q,); q=0 is handled separately from the fit
        """
        self._qgrid       = qgrid
        self._A           = None
        self._alpha       = None
        self._i0_per_atom = None

    @jaxtyped(typechecker=beartype)
    def fit(self, loader: Iterable[Batch]) -> "PowerLawBaseline":
        """Fit A and α via least squares in log-log space over the training loader."""
        q_pos_mask = self._qgrid > 0
        log_q      = torch.log(self._qgrid[q_pos_mask]).numpy()        # (Q',)

        mean_log_iq = np.zeros(int(q_pos_mask.sum().item()))
        mean_i0     = 0.0
        count       = 0

        for batch in loader:
            n = batch.padding_mask().sum(dim=1).float()                 # (N,)
            N = batch.iqval.shape[0]
            for i in range(N):
                ni = n[i].item()
                if ni > 0:
                    I_pos   = batch.iqval[i, q_pos_mask].clamp(min=1e-10)
                    log_iq  = torch.log(I_pos / ni).numpy()
                    count  += 1
                    mean_log_iq += (log_iq - mean_log_iq) / count
                    mean_i0     += (batch.iqval[i, 0].item() / ni - mean_i0) / count

        X      = np.stack([np.ones_like(log_q), -log_q], axis=1)      # (Q', 2)
        coeffs, _, _, _ = np.linalg.lstsq(X, mean_log_iq, rcond=None)

        self._A           = float(np.exp(coeffs[0]))
        self._alpha       = float(coeffs[1])
        self._i0_per_atom = float(mean_i0)

        return self

    @jaxtyped(typechecker=beartype)
    def __call__(self, batch: Batch) -> Float[torch.Tensor, "N Q"]:
        """Return n_atoms · A · q^(-α) per molecule; q=0 uses the fitted I(0)/n estimate."""
        if self._A is None or self._alpha is None or self._i0_per_atom is None:
            raise RuntimeError("PowerLawBaseline must be fit before calling")

        device = batch.vocab.device
        n      = batch.padding_mask().sum(dim=1, keepdim=True).float() # (N, 1)
        q      = self._qgrid.to(device)                                # (Q,)

        pred = n * self._A * q.unsqueeze(0).clamp(min=1e-10) ** (-self._alpha)  # (N, Q)
        pred[:, 0] = n.squeeze(-1) * self._i0_per_atom

        return pred
