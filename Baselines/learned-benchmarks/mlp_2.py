import torch
import torch.nn as nn

from Preprocess          import VOCAB
from Baselines.baseline  import Baseline

class Mlp2Baseline(Baseline):
    
    """
    Learned baseline: 2-hidden-layer MLP over composition + size features.

    Predicts log1p(I(q)) from per-element atom fractions, atom count, and Rg.
    Default architecture is two hidden layers (``hidden=(64, 64)``).
    """

    def __init__(
        self, 
        hidden=(64, 64), 
        lr=1e-3, 
        epochs=200, 
        mini_batch=256, 
        grad_clip=1.0,
        device: torch.device | None = None
    ):
        self.hidden     = hidden
        self.lr         = lr
        self.epochs     = epochs
        self.mini_batch = mini_batch
        self.grad_clip  = grad_clip
        self.device     = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._net    = None
        self._x_mean = None
        self._x_std  = None
        self._log_clamp = None  # set in fit(): training target's own max + margin

    def _features(self, batch):
        # pool each molecule to a fixed vector: composition + size (atom count, Rg)
        N, M = batch.vocab.shape
        V = len(VOCAB) + 1
        counts = torch.zeros(N, V).scatter_add_(1, batch.vocab.long(), torch.ones(N, M))
        counts[:, 0] = 0.0                                                           # drop the padding token
        n_atoms = counts.sum(dim=1, keepdim=True).clamp(min=1)
        mask = batch.padding_mask().float()
        r2   = (batch.coord ** 2).sum(dim=-1)                                        # squared distance per atom
        rg   = ((r2 * mask).sum(dim=1, keepdim=True) / n_atoms).clamp(min=0).sqrt()  # radius of gyration
        return torch.cat([counts / n_atoms, n_atoms, rg], dim=1).cpu()               # element fractions ++ size

    def fit(self, loader):
        X_parts, Y_parts = [], []
        for batch in loader:
            X_parts.append(self._features(batch))
            Y_parts.append(torch.log1p(batch.iqval).cpu())
        X = torch.cat(X_parts)
        Y = torch.cat(Y_parts)
        # +5 margin: comfortable headroom above every log1p(I) value actually seen in
        # training, so a genuinely well-fit prediction is never clamped, but a NaN/
        # exploded-weight prediction (see grad clipping below) gets capped near the
        # real data's own scale instead of at an arbitrary constant. expm1(30) ~ 1e13 --
        # a single such point dwarfs every other term in a raw-space sum of squares and
        # makes R²(raw) meaningless (observed: R²(raw) in the -1e8 range on a real run).
        self._log_clamp = float(Y.max().item()) + 5.0
        self._x_mean = X.mean(0)
        self._x_std  = X.std(0).clamp(min=1e-8)
        X = (X - self._x_mean) / self._x_std
        in_dim, out_dim = X.shape[1], Y.shape[1]
        layers, prev = [], in_dim
        for h in self.hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self._net = nn.Sequential(*layers).to(self.device)
        X, Y = X.to(self.device), Y.to(self.device)
        opt  = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        n    = len(X)
        self._net.train()
        for _ in range(self.epochs):
            perm = torch.randperm(n, device=self.device)
            for i in range(0, n, self.mini_batch):
                b    = perm[i:i + self.mini_batch]
                loss = nn.functional.mse_loss(self._net(X[b]), Y[b])
                opt.zero_grad()
                loss.backward()
                # heavy-tailed features (atom counts / Rg span ~1 to 6046 atoms) can
                # produce occasional large gradients that blow Adam up to inf/nan
                # over many epochs; clip so one bad minibatch can't diverge the run.
                nn.utils.clip_grad_norm_(self._net.parameters(), self.grad_clip)
                opt.step()
        self._net.eval()
        return self

    def __call__(self, batch):
        X = (self._features(batch) - self._x_mean) / self._x_std #type: ignore
        with torch.no_grad():
            log_pred = self._net(X.to(self.device))              #type: ignore
        # safety net: clamp before expm1 so a NaN/exploded weight (despite grad
        # clipping) surfaces as a large finite MSE instead of inf, which would
        # otherwise break evaluate()'s accumulated sums and the summary plot.
        log_pred = torch.nan_to_num(
            log_pred, 
            nan=0.0, 
            posinf=self._log_clamp, 
            neginf=0.0
        ).clamp(max=self._log_clamp) #type: ignore
        return torch.expm1(log_pred).cpu()

    # timed_call is inherited from Baseline: it now measures end-to-end,
    # CUDA-synchronized wall time (feature build + forward + transfer), the same
    # definition used for every other baseline. The old override timed only the
    # net's forward pass with cuda.Event, which is not comparable to the physics
    # baselines' end-to-end cost, so it was removed for a single coherent metric.
