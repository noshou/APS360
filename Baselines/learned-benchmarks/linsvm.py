from collections.abc import Iterable

import numpy as np
import torch
from sklearn.kernel_approximation import Nystroem
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR

from Baselines.run.baseline import Baseline
from Preprocess import VOCAB
from ScatterNet.batching import Batch


def _median_heuristic_gamma(
    X: np.ndarray,
    sample_size: int = 1000,
    random_state: int = 0,
) -> float:
    """Estimate an RBF gamma via the median pairwise-distance heuristic.

    The RBF kernel is ``exp(-gamma * ||x - x'||**2)``: gamma controls how
    quickly similarity decays with distance. A large gamma makes each point
    influence only a tiny neighborhood (the kernel degenerates toward a
    nearest-neighbor lookup and overfits); a small gamma makes distant
    points still look similar (the kernel degenerates toward a constant
    function and underfits). A fixed gamma picked in advance (e.g. 0.1)
    only happens to be reasonable for one particular feature scale. This
    instead derives gamma from the training data itself: it samples pairs
    of feature rows, takes the median squared Euclidean distance between
    them, and inverts it, so the kernel's "reach" self-adjusts to whatever
    scale the features actually have after standardization.

    Parameters
    ----------
    X : numpy.ndarray
        Feature matrix, shape ``(n_samples, n_features)``, already scaled.
    sample_size : int, optional
        Number of rows to subsample for the pairwise-distance estimate, by
        default 1000.
    random_state : int, optional
        Seed for the row subsample, by default 0.

    Returns
    -------
    float
        Estimated gamma. Falls back to 1.0 if every sampled pairwise
        distance is zero (degenerate/constant features).
    """
    rng = np.random.default_rng(random_state)
    n = X.shape[0]
    idx = rng.choice(n, size=min(sample_size, n), replace=False)
    sample = X[idx]

    sq_dists = np.sum((sample[:, None, :] - sample[None, :, :]) ** 2, axis=-1)
    triu = sq_dists[np.triu_indices_from(sq_dists, k=1)]
    median_sq_dist = np.median(triu)

    return float(1.0 / median_sq_dist) if median_sq_dist > 0 else 1.0


class Linsvm(Baseline):
    """
    Learned baseline: per-q LinearSVR over Nystroem-approximated RBF features.

    Pools each molecule down to a fixed feature vector (per-element
    composition fractions, plus log1p atom count and log1p radius of
    gyration), standardizes the two size columns -- the bounded [0, 1]
    fractions are deliberately left alone, see ``_scale`` -- then maps it
    through a Nystroem approximation of the RBF kernel so a
    fast linear model can still capture nonlinear structure without ever
    building the O(n^2) kernel matrix a plain kernel SVR would need on a
    900,000-molecule training set. A separate LinearSVR head is trained per
    q-point on top of the shared Nystroem features. Runs entirely on CPU.
    The RBF gamma used by Nystroem is estimated from the training data
    (median pairwise-distance heuristic) rather than fixed in advance, so a
    poor guess at gamma doesn't silently cripple the whole baseline - see
    ``_median_heuristic_gamma`` for what gamma actually controls.
    """

    def __init__(
        self,
        n_components: int = 500,
        gamma: float | None = None,
        random_state: int = 4209,
    ) -> None:
        """Store hyperparameters; the pipeline itself is built in ``fit``.

        Parameters
        ----------
        n_components : int, optional
            Number of Nystroem landmark features, by default 500.
        gamma : float or None, optional
            RBF kernel gamma. If None (default), it is estimated from the
            training features via the median pairwise-distance heuristic
            instead of being fixed in advance.
        random_state : int, optional
            Seed for Nystroem's landmark sampling, the gamma heuristic's
            subsample, and the LinearSVR heads, by default 4209.

        Returns
        -------
        None
        """
        self._n_components = n_components
        self._gamma = gamma
        self._random_state = random_state

        self._scaler: StandardScaler | None = None
        self._nystroem: Nystroem | None = None
        self._heads: list[LinearSVR] = []

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
        V = len(VOCAB) + 1

        dev = batch.vocab.device
        mask = batch.padding_mask().float()
        counts = torch.zeros(N, V, device=dev).scatter_add_(
            1, batch.vocab.long(), torch.ones(N, M, device=dev)
        )
        counts[:, 0] = 0.0  # drop the padding token
        n_atoms = counts.sum(dim=1, keepdim=True).clamp(min=1)
        fractions = counts / n_atoms  # per-element composition

        r2 = (batch.coord**2).sum(dim=-1)  # squared distance per atom
        rg2 = (r2 * mask).sum(dim=1, keepdim=True) / n_atoms
        rg = rg2.clamp(min=0).sqrt()  # radius of gyration

        # log1p the size pair: atom counts span 1..6046 and I(0) grows
        # ~n^2, so log1p(I) is ~linear in log(n). On the raw scale a
        # handful of huge molecules sit orders of magnitude from the rest
        # and dominate the pairwise distances the RBF kernel is built
        # from.
        size = torch.log1p(torch.cat([n_atoms, rg], dim=1))
        feats = torch.cat([fractions, size], dim=1)
        return feats.cpu().numpy()

    def _scale(self, X: np.ndarray) -> np.ndarray:
        """Standardize the two trailing size columns; leave fractions alone.

        The leading block is per-element composition *fractions*: already
        bounded to [0, 1] and comparable column-to-column, so they need no
        rescaling. Standardizing them is actively harmful here. A rare ion
        appears in only a few molecules, so its column's std is ~1e-3, and
        dividing by that turns a fraction of 0.25 into a value of ~80.
        Those inflated columns then dominate the pairwise distances
        ``_median_heuristic_gamma`` measures, so the gamma it returns is
        fitted to an artifact of the scaling rather than to real
        composition structure (measured: gamma 0.11 with the fractions
        scaled vs 0.47 without, i.e. a kernel roughly 4x too wide, which
        underfits). Unlike the MLP the RBF kernel is bounded, so this
        shows up as a quietly mediocre fit rather than an explosion: MSLE
        0.359 -> 0.104 and R^2(log1p) 0.951 -> 0.986 on the same data once
        the fractions are left alone.

        Parameters
        ----------
        X : numpy.ndarray
            Raw feature matrix from ``_molecule_features``, shape
            ``(N, V + 2)``.

        Returns
        -------
        numpy.ndarray
            Same shape, with only the trailing ``n_atoms``/``rg`` columns
            scaled.
        """
        Z = X.copy()
        Z[:, -2:] = self._scaler.transform(X[:, -2:])  # type: ignore
        return Z

    def fit(self, loader: Iterable[Batch]) -> "Linsvm":
        """Fit the scaler, Nystroem transform, and one LinearSVR per q-point.

        Parameters
        ----------
        loader : Iterable[Batch]
            Iterable of training batches.

        Returns
        -------
        Linsvm
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

        # fit the scaler on the size pair only -- see _scale for why the
        # fraction block is deliberately left unscaled
        self._scaler = StandardScaler().fit(X[:, -2:])
        X_scaled = self._scale(X)

        gamma = (
            self._gamma
            if self._gamma is not None
            else _median_heuristic_gamma(
                X_scaled, random_state=self._random_state
            )
        )

        self._nystroem = Nystroem(
            kernel="rbf",
            gamma=gamma,
            n_components=self._n_components,
            random_state=self._random_state,
        ).fit(X_scaled)
        X_transformed = self._nystroem.transform(X_scaled)

        self._heads = []
        for q in range(Y.shape[1]):
            head = LinearSVR(
                dual=False,  # pyright: ignore[reportArgumentType]
                loss="squared_epsilon_insensitive",
                random_state=self._random_state,
            )
            head.fit(X_transformed, Y[:, q])
            self._heads.append(head)

        return self

    def __call__(self, batch: Batch) -> torch.Tensor:  # shape (N, Q)
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
        if self._scaler is None or self._nystroem is None or not self._heads:
            raise RuntimeError("Linsvm must be fit before calling")

        device = batch.vocab.device
        X = self._molecule_features(batch)
        X_scaled = self._scale(X)
        X_transformed = self._nystroem.transform(X_scaled)

        log_preds = np.stack(
            [head.predict(X_transformed) for head in self._heads], axis=1
        )  # (N, Q), in log1p space
        pred = torch.from_numpy(np.expm1(log_preds)).float().to(device)
        return pred
