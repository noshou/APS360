import torch
import numpy as np

from collections.abc          import Iterable
from jaxtyping                import Float, jaxtyped
from beartype                 import beartype
from sklearn.preprocessing    import StandardScaler
from sklearn.neural_network   import MLPRegressor
from ScatterNet.batching      import Batch
from Preprocess               import VOCAB
from Baselines.baseline       import Baseline


class Mlp(Baseline):
    """
    Learned baseline: a single scikit-learn MLPRegressor over pooled molecule features.

    Pools each molecule down to the same fixed feature vector as ``Linsvm``
    (per-element composition fractions, atom count, radius of gyration),
    standardizes it, and feeds it into one ``MLPRegressor`` with all Q
    q-points as simultaneous outputs (unlike ``Linsvm``, which needs one
    LinearSVR head per q-point, MLPRegressor natively supports multi-output
    regression). Runs entirely on CPU. This is the smallest fully-learned,
    non-kernel baseline in the suite: no message passing, no per-atom
    embeddings, just a plain feedforward network on hand-pooled features.
    """

    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (64, 64),
        random_state:       int             = 4209,
        max_iter:           int             = 2000,
    ) -> None:
        """Store hyperparameters; the pipeline itself is built in ``fit``.

        Parameters
        ----------
        hidden_layer_sizes : tuple of int, optional
            Sizes of the MLP's hidden layers, by default ``(64, 64)``.
        random_state : int, optional
            Seed for weight initialization, by default 4209.
        max_iter : int, optional
            Maximum training iterations, by default 2000.

        Returns
        -------
        None
        """
        self._hidden_layer_sizes = hidden_layer_sizes
        self._random_state       = random_state
        self._max_iter           = max_iter

        self._scaler: StandardScaler | None = None
        self._mlp:    MLPRegressor  | None = None

    def _molecule_features(self, batch: Batch) -> np.ndarray:
        """Compute a fixed-size feature vector per molecule in a batch.

        Parameters
        ----------
        batch : Batch
            Batch of molecules.

        Returns
        -------
        numpy.ndarray
            Feature matrix of shape ``(N, len(VOCAB) + 3)``: per-element
            composition fractions, atom count, and radius of gyration.
        """
        N, M = batch.vocab.shape
        V    = len(VOCAB) + 1

        mask     = batch.padding_mask().float()
        counts   = torch.zeros(N, V).scatter_add_(
            1, batch.vocab.long(), torch.ones(N, M)
        )
        counts[:, 0] = 0.0
        n_atoms      = counts.sum(dim=1, keepdim=True).clamp(min=1)
        fractions    = counts / n_atoms

        r2  = (batch.coord ** 2).sum(dim=-1)
        rg2 = (r2 * mask).sum(dim=1, keepdim=True) / n_atoms
        rg  = rg2.clamp(min=0).sqrt()

        feats = torch.cat([fractions, n_atoms, rg], dim=1)
        return feats.cpu().numpy()

    def fit(self, loader: Iterable[Batch]) -> "Mlp":
        """Fit the scaler and the multi-output MLPRegressor.

        Parameters
        ----------
        loader : Iterable[Batch]
            Iterable of training batches.

        Returns
        -------
        Mlp
            This instance, with a fitted pipeline.

        Raises
        ------
        RuntimeError
            If the loader yields no batches (empty training set).
        """
        X_parts: list[np.ndarray] = []
        Y_parts: list[np.ndarray] = []

        for batch in loader:
            X_parts.append(self._molecule_features(batch))
            Y_parts.append(torch.log1p(batch.iqval).cpu().numpy())

        if not X_parts:
            raise RuntimeError("fit() received an empty loader")

        X = np.concatenate(X_parts, axis=0)
        Y = np.concatenate(Y_parts, axis=0)  # (n_samples, Q)

        self._scaler = StandardScaler().fit(X)
        X_scaled     = self._scaler.transform(X)

        self._mlp = MLPRegressor(
            hidden_layer_sizes = self._hidden_layer_sizes,
            random_state       = self._random_state,
            max_iter           = self._max_iter,
        )
        self._mlp.fit(X_scaled, Y)

        return self

    @jaxtyped(typechecker=beartype)
    def __call__(self, batch: Batch) -> Float[torch.Tensor, "N Q"]:
        """Predict I(q) for a batch of molecules.

        Parameters
        ----------
        batch : Batch
            Batch of molecules to predict scattering curves for.

        Returns
        -------
        torch.Tensor
            Predicted I(q) curves of shape ``(N, Q)``, on the same device
            as the input batch.

        Raises
        ------
        RuntimeError
            If called before ``fit`` has been run.
        """
        if self._scaler is None or self._mlp is None:
            raise RuntimeError("Mlp must be fit before calling")

        device = batch.vocab.device
        X        = self._molecule_features(batch)
        X_scaled = self._scaler.transform(X)

        log_pred = self._mlp.predict(X_scaled)  # (N, Q), in log1p space
        pred = torch.from_numpy(np.expm1(log_pred)).float().to(device)
        return pred
