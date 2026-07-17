import torch
import numpy as np

from jaxtyping           import Float, jaxtyped
from beartype            import beartype
from ScatterNet.batching import Batch
from Baselines.baseline  import Baseline, build_fmag_table, q_chunks
from sklearn.cluster     import HDBSCAN # type: ignore

class Hdbscan(Baseline):
    """
    Local-geometry baseline: two-level Debye sum over HDBSCAN atom clusters.

    Atoms are spatially clustered with HDBSCAN; within a cluster the exact
    pairwise Debye sum is used (cheap, since clusters are small), and across
    clusters each cluster is approximated as a single point scatterer at its
    centroid with amplitude Σf_i(q), combined via the Debye formula on
    cluster-centroid distances (cheap, since there are far fewer clusters
    than atoms). Captures coherent structure at both the bonded scale and
    the coarse blob-spacing scale without the O(M²) cost of a full Debye sum.
    Noise points (label -1) are treated as singleton clusters.
    """

    _qgrid:            Float[torch.Tensor, "Q"]
    _fmag_table:       Float[torch.Tensor, "V Q"]
    _min_cluster_size: int
    _min_samples:      int

    def __init__(
        self,
        qgrid:            Float[torch.Tensor, "Q"],
        energy:           float,
        min_cluster_size: int = 5,
        min_samples:      int = 3,
    ) -> None:
        """Build the form factor table and store HDBSCAN hyperparameters.

        Parameters
        ----------
        qgrid : torch.Tensor
            q-point grid in inverse angstroms, shape ``(Q,)``.
        energy : float
            X-ray energy in eV, used for anomalous f1/f2 corrections.
        min_cluster_size : int, optional
            HDBSCAN's minimum cluster size, by default 5.
        min_samples : int, optional
            HDBSCAN's ``min_samples`` (core-point density threshold), by
            default 3.

        Returns
        -------
        None
        """
        self._qgrid            = qgrid
        self._fmag_table       = build_fmag_table(qgrid, energy)
        self._min_cluster_size = min_cluster_size
        self._min_samples      = min_samples
    
    @jaxtyped(typechecker=beartype)
    def __call__(self, batch: Batch) -> Float[torch.Tensor, "N Q"]:
        
        """Return the two-level cluster Debye sum I(q) per molecule.

        Parameters
        ----------
        batch : Batch
            Batch of molecules to predict scattering curves for.

        Returns
        -------
        torch.Tensor
            Predicted I(q) curves of shape ``(N, Q)``.
        """
        
        device = batch.coord.device
        mask   = batch.padding_mask()           # (N, M)
        qgrid  = self._qgrid.to(device)
        ftable = self._fmag_table.to(device)

        preds: list[Float[torch.Tensor, "Q"]] = []

        for n in range(batch.coord.shape[0]):
            coords_n = batch.coord[n][mask[n]]  # (m, 3)
            vocab_n  = batch.vocab[n][mask[n]]  # (m,)
            m = coords_n.shape[0]

            if m < 2:
                preds.append(torch.zeros(qgrid.shape[0], device=device))
                continue

            f_atoms = ftable[vocab_n]  # (m, Q) — per-atom q-dependent form factor

            if m <= self._min_cluster_size:
                labels = np.zeros(m, dtype=np.int64)  # too few atoms to cluster: one blob
            else:
                labels = HDBSCAN(
                    min_cluster_size=self._min_cluster_size,
                    min_samples=self._min_samples,
                    copy=False,
                ).fit_predict(coords_n.detach().cpu().numpy())

            labels_t = torch.as_tensor(labels, device=device)
            noise    = labels_t == -1
            # give every noise point its own singleton cluster id, after the real labels
            n_real_clusters = int(labels_t[~noise].max().item()) + 1 if (~noise).any() else 0
            singleton_ids = n_real_clusters + torch.arange(noise.sum().item(), device=device)
            cluster_id = labels_t.clone()
            cluster_id[noise] = singleton_ids
            n_clusters = int(cluster_id.max().item()) + 1

            i_q = torch.zeros(qgrid.shape[0], device=device)
            centroids = []
            cluster_amps = []

            for c in range(n_clusters):
                sel = cluster_id == c
                coords_c = coords_n[sel]  # (mc, 3)
                f_c      = f_atoms[sel]   # (mc, Q)

                diff  = coords_c.unsqueeze(0) - coords_c.unsqueeze(1)  # (mc, mc, 3)
                dists = diff.norm(dim=-1)                               # (mc, mc)
                mc    = coords_c.shape[0]

                # sum_ij f_i(q) f_j(q) sinc(q d_ij) is a quadratic form in f at each
                # q, so evaluate it as f^T sinc(q) f in memory-bounded q-chunks: the
                # (Qc, mc, mc) sinc block stays ~<= budget instead of scaling with Q
                for a, b in q_chunks(qgrid.shape[0], mc):

                    fc   = f_c[:, a:b].transpose(0, 1)                     # (Qc, mc)
                    qr   = qgrid[a:b].view(-1, 1, 1) * dists.unsqueeze(0)  # (Qc, mc, mc)
                    sinc = torch.where(qr.abs() < 1e-8, torch.ones_like(qr), torch.sin(qr) / qr)

                    tmp  = torch.bmm(sinc, fc.unsqueeze(2)).squeeze(2)     # (Qc, mc) = sinc @ f
                    i_q[a:b] += (fc * tmp).sum(dim=1)                      # full double sum, i == j included

                centroids.append(coords_c.mean(dim=0))
                cluster_amps.append(f_c.sum(dim=0))  # (Q,) — Σf_i(q) for this cluster

            centroids    = torch.stack(centroids)     # (C, 3)
            cluster_amps = torch.stack(cluster_amps)  # (C, Q)

            if n_clusters > 1:
                cdiff  = centroids.unsqueeze(0) - centroids.unsqueeze(1)  # (C, C, 3)
                cdists = cdiff.norm(dim=-1)                                # (C, C)

                # same quadratic form over cluster centroids, but excluding c == c.
                # zero the sinc diagonal rather than subtracting the diagonal terms
                # afterwards: the cross terms oscillate and can cancel to much less
                # than the diagonal, so the subtraction would lose precision to
                # catastrophic cancellation. masking keeps it exact, as before
                for a, b in q_chunks(qgrid.shape[0], n_clusters):

                    ac    = cluster_amps[:, a:b].transpose(0, 1)             # (Qc, C)
                    cqr   = qgrid[a:b].view(-1, 1, 1) * cdists.unsqueeze(0)  # (Qc, C, C)
                    csinc = torch.where(cqr.abs() < 1e-8, torch.ones_like(cqr), torch.sin(cqr) / cqr)
                    csinc.diagonal(dim1=1, dim2=2).zero_()                   # drop the c == c terms

                    tmp   = torch.bmm(csinc, ac.unsqueeze(2)).squeeze(2)     # (Qc, C) = csinc @ amp
                    i_q[a:b] += (ac * tmp).sum(dim=1)                        # off-diagonal only

            preds.append(i_q)

        return torch.stack(preds)
