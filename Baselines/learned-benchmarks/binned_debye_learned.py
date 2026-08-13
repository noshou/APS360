from collections.abc import Iterable

import torch
import torch.nn as nn
from beartype import beartype
from jaxtyping import Float, jaxtyped

from Baselines.run.baseline import Baseline, build_fmag_table
from ScatterNet.batching import Batch


class BinnedDebyeLearnedBaseline(Baseline):
    """
    Learned baseline: MLP over a binned pairwise-distance histogram (P(r)).

    Builds the exact same form-factor-weighted P(r) histogram as
    ``Baselines/physics-benchmarks/binned_debye.py`` but instead of applying
    ``binned_debye``'s closed-form ``sinc(q*r)`` Debye transform to turn
    that histogram into I(q), a network learns the P(r) -> I(q) mapping
    directly (predicting all q-points jointly, same as Mlp2Baseline).
    """

    def __init__(
        self,
        qgrid: Float[torch.Tensor, "Q"],  # noqa: F821
        energy: float,
        n_bins: int = 200,
        hidden: tuple[int, ...] = (256, 128),
        lr: float = 1e-3,
        epochs: int = 200,
        mini_batch: int = 256,
        grad_clip: float = 1.0,
        device: torch.device | None = None,
    ) -> None:
        """
        Parameters
        ----------
        qgrid : torch.Tensor
            q-point grid in inverse angstroms, shape ``(Q,)``. Only used
            to build the form-factor table.
        energy : float
            X-ray energy in eV, used for anomalous f1/f2 corrections.
        n_bins : int, optional
            Number of P(r) histogram bins per molecule, by default 200
            (matches ``binned_debye``'s default).
        hidden : tuple of int, optional
            Hidden layer widths, by default ``(256, 128)``
        lr, epochs, mini_batch, grad_clip : see Mlp2Baseline.
        device : torch.device, optional
            Defaults to CUDA if available.
        """
        self.n_bins = n_bins
        self.hidden = hidden
        self.lr = lr
        self.epochs = epochs
        self.mini_batch = mini_batch
        self.grad_clip = grad_clip
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._fmag_table = build_fmag_table(qgrid, energy)
        self._net = None
        self._log_clamp = None  # set in fit()

    def _histogram_features(self, batch: Batch) -> torch.Tensor:
        """Per-molecule form-factor-weighted P(r) histogram + log1p(r_max).

        Parameters
        ----------
        batch : Batch
            Batch of molecules.

        Returns
        -------
        torch.Tensor
            Feature matrix, shape ``(N, n_bins + 1)``, on CPU.
        """
        device = batch.coord.device
        mask = batch.padding_mask()  # (N, M)
        ftable = self._fmag_table.to(device)
        n_bins = self.n_bins

        feats = []
        for n in range(batch.coord.shape[0]):
            coords_n = batch.coord[n][mask[n]]  # (m, 3)
            vocab_n = batch.vocab[n][mask[n]]
            m = coords_n.shape[0]

            f0 = ftable[vocab_n, 0]  # (m,) |f_i(0)| per atom

            if m < 2:
                feats.append(torch.zeros(n_bins + 1, device=device))
                continue

            diff = coords_n.unsqueeze(0) - coords_n.unsqueeze(1)  # (m, m, 3)
            dists = diff.norm(dim=-1)  # (m, m)
            idx = torch.triu_indices(m, m, offset=1, device=device)
            r_ij = dists[idx[0], idx[1]]  # (P,)
            w_ij = f0[idx[0]] * f0[idx[1]]  # (P,)

            r_max = r_ij.max()
            if r_max <= 0:
                feats.append(torch.zeros(n_bins + 1, device=device))
                continue

            bin_idx = ((r_ij / r_max) * n_bins).long().clamp(max=n_bins - 1)
            weight_sum = torch.zeros(n_bins, device=device)
            weight_sum.scatter_add_(0, bin_idx, w_ij)

            feats.append(
                torch.cat([weight_sum, torch.log1p(r_max).unsqueeze(0)])
            )

        return torch.stack(feats).cpu()

    def fit(self, loader: Iterable[Batch]) -> "BinnedDebyeLearnedBaseline":
        X_parts, Y_parts = [], []
        for batch in loader:
            X_parts.append(self._histogram_features(batch))
            Y_parts.append(torch.log1p(batch.iqval).cpu())
        X = torch.cat(X_parts)
        Y = torch.cat(Y_parts)

        # see Mlp2Baseline.fit for why this margin exists: caps a NaN/
        # exploded-weight prediction near the real data's own scale
        # instead of at an arbitrary constant, so R^2(raw) doesn't become
        # meaningless.
        self._log_clamp = float(Y.max().item()) + 5.0

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
                # heavy-tailed histogram weights (rare heavy elements, huge
                # molecules) can produce occasional large gradients the
                # same way Mlp2Baseline's raw atom counts do -- clip so
                # one bad minibatch can't diverge the run.
                nn.utils.clip_grad_norm_(
                    self._net.parameters(), self.grad_clip
                )
                opt.step()
        self._net.eval()
        return self

    @jaxtyped(typechecker=beartype)
    def __call__(
        self, batch: Batch
    ) -> Float[torch.Tensor, "N Q"]:  # noqa: F722
        if self._net is None:
            raise RuntimeError(
                "BinnedDebyeLearnedBaseline must be fit before calling"
            )
        device = batch.vocab.device
        X = self._histogram_features(batch)
        with torch.no_grad():
            log_pred = self._net(X.to(self.device))
        # safety net: same rationale as Mlp2Baseline.__call__ -- clamp
        # before expm1 so a NaN/exploded weight surfaces as a large finite
        # error instead of inf, which would otherwise break evaluate()'s
        # accumulated sums and the summary plot.
        log_pred = torch.nan_to_num(
            log_pred, nan=0.0, posinf=self._log_clamp, neginf=0.0
        ).clamp(max=self._log_clamp)
        return torch.expm1(log_pred).to(device)
