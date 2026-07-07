import torch

from collections.abc     import Iterable
from jaxtyping           import Float, jaxtyped
from beartype            import beartype
from ScatterNet.batching import Batch
from Baselines.baseline  import Baseline
from Preprocess          import VOCAB


@jaxtyped(typechecker=beartype)
def _composition_features(batch: Batch) -> Float[torch.Tensor, "N V"]:
    """Per-molecule element histogram, one column per VOCAB ion (padding excluded).

    Parameters
    ----------
    batch : Batch
        Batch of molecules.

    Returns
    -------
    torch.Tensor
        Counts of shape ``(N, V)`` where ``V = len(VOCAB) + 1``. Column 0
        (the padding sentinel) is always zero since ``vocab == 0`` entries
        are padding, not real atoms.
    """
    n_ions = len(VOCAB) + 1
    feats  = torch.zeros(batch.vocab.shape[0], n_ions, device=batch.vocab.device)
    for n in range(batch.vocab.shape[0]):
        v = batch.vocab[n][batch.vocab[n] != 0]
        feats[n] = torch.bincount(v, minlength=n_ions).float()
    return feats


class CompositionRegressionBaseline(Baseline):
    """Zero-order baseline: linear regression of I(q) on per-element atom counts.

    Unlike a plain atom-count-bucket mean, this conditions on *which*
    elements are present and how many of each, so two molecules with equal
    total atom count but different stoichiometry get different predictions.
    Fit via ridge-regularized normal equations accumulated online across
    the training loader, so the full dataset never needs to be held in
    memory at once.

    Beating this baseline proves the model uses more than a linear
    combination of element counts.
    """

    _weights: Float[torch.Tensor, "F Q"] | None
    _ridge:   float

    def __init__(self, ridge: float = 1e-3) -> None:
        """
        Parameters
        ----------
        ridge : float, default 1e-3
            L2 regularization strength added to the normal equations'
            diagonal, to keep the solve well-posed when some elements are
            rare or perfectly collinear with others across the training set.
        """
        self._weights = None
        self._ridge   = ridge

    def fit(self, loader: Iterable[Batch]) -> "CompositionRegressionBaseline":
        """Accumulate normal equations (XtX, XtY) over the loader and solve once.

        Parameters
        ----------
        loader : Iterable[Batch]
            Iterable of training batches.

        Returns
        -------
        CompositionRegressionBaseline
            This instance, fit and ready to call.
        """
        xtx: torch.Tensor | None = None
        xty: torch.Tensor | None = None

        for batch in loader:
            feats = _composition_features(batch)                       # (N, V)
            ones  = torch.ones(feats.shape[0], 1, device=feats.device)
            x     = torch.cat([feats, ones], dim=1)                    # (N, V+1)
            y     = batch.iqval.to(feats.device)                       # (N, Q)

            if xtx is None:
                f_dim = x.shape[1]
                q_dim = y.shape[1]
                xtx   = torch.zeros(f_dim, f_dim, device=feats.device)
                xty   = torch.zeros(f_dim, q_dim, device=feats.device)

            assert xtx is not None and xty is not None
            xtx += x.T @ x
            xty += x.T @ y

        assert xtx is not None and xty is not None, "fit() called with an empty loader"

        f_dim = xtx.shape[0]
        xtx  += self._ridge * torch.eye(f_dim, device=xtx.device)
        self._weights = torch.linalg.solve(xtx, xty)
        return self

    @jaxtyped(typechecker=beartype)
    def __call__(self, batch: Batch) -> Float[torch.Tensor, "N Q"]:
        """Return the fitted linear regression prediction for each molecule.

        Parameters
        ----------
        batch : Batch
            Batch of molecules to predict scattering curves for.

        Returns
        -------
        torch.Tensor
            Predicted I(q) curves of shape ``(N, Q)``.
        """
        if self._weights is None:
            raise RuntimeError("CompositionRegressionBaseline must be fit before calling")

        device = batch.vocab.device
        feats  = _composition_features(batch)
        ones   = torch.ones(feats.shape[0], 1, device=device)
        x      = torch.cat([feats, ones], dim=1)
        return x @ self._weights.to(device)
