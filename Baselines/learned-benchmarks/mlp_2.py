from collections.abc import Iterable

import torch
import torch.nn as nn

from Baselines.run.baseline import Baseline
from Preprocess import VOCAB
from ScatterNet.batching import Batch


class Mlp2Baseline(Baseline):
    """
    Learned baseline: 2-hidden-layer MLP over composition + size features.

    Predicts log1p(I(q)) from per-element atom fractions plus log1p atom count
    and log1p Rg. Only the two size features are standardized; the bounded
    [0, 1] fractions are deliberately left as-is (see ``_normalise``).
    Default architecture is two hidden layers (``hidden=(64, 64)``).
    """

    def __init__(
        self,
        hidden: tuple[int, ...] = (64, 64),
        lr: float = 1e-3,
        epochs: int = 200,
        mini_batch: int = 256,
        grad_clip: float = 1.0,
        device: torch.device | None = None,
    ):
        self.hidden = hidden
        self.lr = lr
        self.epochs = epochs
        self.mini_batch = mini_batch
        self.grad_clip = grad_clip
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._net = None
        self._x_mean = None
        self._x_std = None
        self._log_clamp = (
            None  # set in fit(): training target's own max + margin
        )

    def _features(self, batch: Batch) -> torch.Tensor:
        # pool each molecule to a fixed vector: composition + size
        N, M = batch.vocab.shape
        V = len(VOCAB) + 1
        dev = (
            batch.vocab.device
        )  # keep the scatter on the batch's device (CUDA-safe)
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
        # log1p the size pair: atom counts span 1..6046, and I(0) ~
        # (sum f_i)^2 grows ~n^2, so the target log1p(I) is ~linear in
        # log(n) but strongly curved in raw n. Feeding raw n forces a
        # small ReLU net to approximate a logarithm with piecewise-linear
        # segments across four orders of magnitude, which it fits badly;
        # log1p makes that relationship straight and the fit near-exact.
        size = torch.log1p(torch.cat([n_atoms, rg], dim=1))
        return torch.cat(
            [counts / n_atoms, size], dim=1
        ).cpu()  # element fractions ++ size

    def _normalise(self, X: torch.Tensor) -> torch.Tensor:
        """Standardize the two trailing size columns; leave fractions alone.

        The leading block is per-element composition *fractions*: already
        bounded to [0, 1] and directly comparable column-to-column, so
        they need no rescaling. Standardizing them is actively harmful --
        a rare ion occurs in a handful of molecules, so its column's std
        is ~1e-3, and dividing by that amplifies a fraction of 0.25 into a
        z-score of ~80. That is far outside the range the net saw while
        training, so at predict time it extrapolates linearly and emits
        log-space values around 60 for a target whose true max is ~17 --
        capped by ``_log_clamp`` into a huge over-prediction that makes
        R^2(raw) meaningless (measured: R^2(raw) ~ -8.5e4, MSLE 4.08;
        leaving the fractions alone gives R^2(raw) 0.996, MSLE 0.078 on
        the same data).

        Only ``n_atoms``/``rg`` genuinely need standardizing: they are
        unbounded and on their own scales.
        """
        Z = X.clone()
        Z[:, -2:] = (X[:, -2:] - self._x_mean) / self._x_std  # type: ignore
        return Z

    def fit(self, loader: Iterable[Batch]) -> "Mlp2Baseline":
        X_parts, Y_parts = [], []
        for batch in loader:
            X_parts.append(self._features(batch))
            Y_parts.append(torch.log1p(batch.iqval).cpu())
        X = torch.cat(X_parts)
        Y = torch.cat(Y_parts)
        # +5 margin: comfortable headroom above every log1p(I) value seen
        # in training, so a genuinely well-fit prediction is never
        # clamped, but a NaN/exploded-weight prediction (see grad
        # clipping below) gets capped near the real data's own scale
        # instead of at an arbitrary constant. expm1(30) ~ 1e13 -- a
        # single such point dwarfs every other term in a raw-space sum of
        # squares and makes R²(raw) meaningless (observed: R²(raw) in the
        # -1e8 range on a real run).
        self._log_clamp = float(Y.max().item()) + 5.0
        # stats for the size pair only -- see _normalise for why the
        # fraction block is deliberately left unscaled. A zero-variance
        # size column (a single-size training set) falls back to 1.0
        # rather than a tiny floor, so the column collapses to a constant
        # instead of being amplified by 1/eps.
        size_std = X[:, -2:].std(0)
        self._x_mean = X[:, -2:].mean(0)
        self._x_std = torch.where(
            size_std > 1e-6, size_std, torch.ones_like(size_std)
        )
        X = self._normalise(X)
        in_dim, out_dim = X.shape[1], Y.shape[1]
        layers, prev = [], in_dim
        for h in self.hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self._net = nn.Sequential(*layers).to(self.device)
        X, Y = X.to(self.device), Y.to(self.device)
        opt = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        n = len(X)
        self._net.train()
        for _ in range(self.epochs):
            perm = torch.randperm(n, device=self.device)
            for i in range(0, n, self.mini_batch):
                b = perm[i : i + self.mini_batch]
                loss = nn.functional.mse_loss(self._net(X[b]), Y[b])
                opt.zero_grad()
                loss.backward()
                # heavy-tailed features (atom counts / Rg span ~1 to 6046
                # atoms) can produce occasional large gradients that blow
                # Adam up to inf/nan over many epochs; clip so one bad
                # minibatch can't diverge the run.
                nn.utils.clip_grad_norm_(
                    self._net.parameters(), self.grad_clip
                )
                opt.step()
        self._net.eval()
        return self

    def __call__(self, batch: Batch) -> torch.Tensor:
        X = self._normalise(self._features(batch))
        with torch.no_grad():
            log_pred = self._net(X.to(self.device))  # type: ignore
        # safety net: clamp before expm1 so a NaN/exploded weight (despite grad
        # clipping) surfaces as a large finite MSE instead of inf, which would
        # otherwise break evaluate()'s accumulated sums and the summary plot.
        log_pred = torch.nan_to_num(
            log_pred, nan=0.0, posinf=self._log_clamp, neginf=0.0
        ).clamp(max=self._log_clamp)  # type: ignore
        return torch.expm1(log_pred).cpu()

    # timed_call is inherited from Baseline: it now measures end-to-end,
    # CUDA-synchronized wall time (feature build + forward + transfer),
    # the same definition used for every other baseline. The old override
    # timed only the net's forward pass with cuda.Event, which is not
    # comparable to the physics baselines' end-to-end cost, so it was
    # removed for a single coherent metric.
