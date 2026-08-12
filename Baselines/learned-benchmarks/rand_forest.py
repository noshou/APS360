from collections.abc import Iterable

import numpy as np
import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from sklearn.ensemble import RandomForestRegressor

from Baselines.run.baseline import Baseline
from Preprocess import VOCAB
from ScatterNet.batching import Batch


class RandForestBaseline(Baseline):
    """
    Random Forest over composition + size features.

    Uses the exact same pooled feature vector as Mlp2Baseline/Linsvm (per-
    element atom fractions plus log1p atom count and log1p radius of
    gyration) but a tree ensemble instead of a neural net or kernel model,
    to isolate model-class effects from feature effects: if this scores
    close to Mlp2Baseline/Linsvm, the current error floor is set by the
    features (composition + size alone), not by which model reads them.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int | None = 20,
        min_samples_leaf: int = 5,
        n_jobs: int = -1,
        random_state: int = 0,
    ) -> None:
        """
        Parameters
        ----------
        n_estimators : int, optional
            Number of trees in the forest, by default 200.
        max_depth : int or None, optional
            Max tree depth, by default 20. sklearn stores a ``(n_nodes, 1,
            Q)`` value array per tree, one row per q-point; with
            ``max_depth=None`` on a full training pass (tens of thousands
            of molecules) ``n_nodes`` grows to ~2N per tree and the forest
            no longer fits in RAM. Bounding depth keeps node count, and
            therefore memory, roughly constant regardless of N.
        min_samples_leaf : int, optional
            Minimum samples per leaf, by default 5. Complements
            ``max_depth`` in keeping leaf count bounded.
        n_jobs : int, optional
            Parallel jobs for both fit and predict, by default -1 (all
            cores).
        random_state : int, optional
            Seed for the forest's bootstrap sampling and feature subsets,
            by default 0.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_jobs = n_jobs
        self.random_state = random_state
        self._forest: RandomForestRegressor | None = None

    def _features(self, batch: Batch) -> np.ndarray:
        """Pool each molecule to composition fractions + log1p size pair.

        Parameters
        ----------
        batch : Batch
            Batch of molecules.

        Returns
        -------
        numpy.ndarray
            Feature matrix, shape ``(N, len(VOCAB) + 2)``.
        """
        N, M = batch.vocab.shape
        V = len(VOCAB) + 1
        dev = batch.vocab.device

        counts = torch.zeros(N, V, device=dev).scatter_add_(
            1, batch.vocab.long(), torch.ones(N, M, device=dev)
        )
        counts[:, 0] = 0.0  # drop the padding token
        n_atoms = counts.sum(dim=1, keepdim=True).clamp(min=1)

        mask = batch.padding_mask().float()
        r2 = (batch.coord**2).sum(dim=-1)  # squared distance per atom
        rg = (
            ((r2 * mask).sum(dim=1, keepdim=True) / n_atoms)
            .clamp(min=0)
            .sqrt()
        )  # radius of gyration

        # log1p the size pair -- see Mlp2Baseline._features for why raw
        # atom counts (spanning 1..6046) badly distort a model that has
        # to relate size to a target that grows ~n^2.
        size = torch.log1p(torch.cat([n_atoms, rg], dim=1))
        feats = torch.cat([counts / n_atoms, size], dim=1)
        return feats.cpu().numpy()

    def fit(self, loader: Iterable[Batch]) -> "RandForestBaseline":
        """Fit one multi-output RandomForestRegressor over all q-points.

        Parameters
        ----------
        loader : Iterable[Batch]
            Iterable of training batches.

        Returns
        -------
        RandForestBaseline
            This instance, with a fitted forest.

        Raises
        ------
        RuntimeError
            If the loader yields no batches (empty training set).
        """
        X_parts: list[np.ndarray] = []
        Y_parts: list[np.ndarray] = []
        for batch in loader:
            X_parts.append(self._features(batch))
            Y_parts.append(torch.log1p(batch.iqval).cpu().numpy())

        if not X_parts:
            raise RuntimeError("fit() received an empty loader")

        X = np.concatenate(X_parts, axis=0)
        Y = np.concatenate(Y_parts, axis=0)  # (n_samples, Q)

        self._forest = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        ).fit(X, Y)
        return self

    @jaxtyped(typechecker=beartype)
    def __call__(
        self, batch: Batch
    ) -> Float[torch.Tensor, "N Q"]:  # noqa: F722
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
        if self._forest is None:
            raise RuntimeError("RandForestBaseline must be fit before calling")

        device = batch.vocab.device
        X = self._features(batch)
        log_pred = self._forest.predict(X)  # (N, Q), in log1p space
        pred = torch.from_numpy(np.expm1(log_pred)).float().to(device)
        return pred
