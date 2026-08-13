import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from sklearn.cluster import HDBSCAN  # type: ignore

from Baselines.run.baseline import (
    Baseline,
    autocast_dtype,
    build_fmag_table,
    q_chunks,
)
from ScatterNet.batching import Batch


class Hdbscan(Baseline):
    """
    Coarse-graining baseline: Debye sum over HDBSCAN-clustered "beads".

    Atoms are spatially clustered with HDBSCAN (``min_samples=1``, so it
    never errors on small molecules -- see the module's design notes).
    Each cluster is group-averaged into a single bead: position is the
    cluster's centroid, amplitude is Σf_i(q) over its member atoms. The
    exact Debye formula is then run once over the beads (not the atoms),
    including each bead's own self-scattering term -- there is no
    separate exact intra-cluster level, so nothing else counts that term.

    Noise points (label -1) are each given their own singleton cluster id,
    so the same per-cluster code path handles them: a size-1 "cluster" is
    already exactly represented by a bead at that atom's own position with
    that atom's own amplitude, no approximation lost.

    This is a lossy coarse-graining, not the exact all-atom Debye sum: two
    clusters' internal sizes/shapes collapse into point scatterers, so
    within-cluster structure at scales finer than a cluster is invisible
    to this baseline. Beating it proves the model resolves structure below
    the HDBSCAN cluster scale.
    """

    _qgrid: Float[torch.Tensor, "Q"]  # noqa: F821
    _fmag_table: Float[torch.Tensor, "V Q"]  # noqa: F722

    def __init__(
        self,
        qgrid: Float[torch.Tensor, "Q"],  # noqa: F821
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
    def __call__(
        self, batch: Batch
    ) -> Float[torch.Tensor, "N Q"]:  # noqa: F722
        """Return the bead-Debye sum I(q) per molecule."""

        device = batch.coord.device
        mask = batch.padding_mask()  # (N, M)
        qgrid = self._qgrid.to(device)
        ftable = self._fmag_table.to(device)
        # BF16 on Ampere+ for the O(m^2) distance-matrix build below (native
        # tensor cores there; T4 gets None -> stays fp32, see
        # autocast_dtype's docstring).
        dtype = autocast_dtype(device)

        preds: list[Float[torch.Tensor, "Q"]] = []  # noqa: F821
        hscan = HDBSCAN(min_samples=1, copy=False)  # type: ignore[arg-type]

        for n in range(batch.coord.shape[0]):
            coords_n = batch.coord[n][mask[n]]  # (m, 3)
            vocabs_n = batch.vocab[n][mask[n]]  # (m,)
            m = coords_n.shape[0]

            if m == 0:
                preds.append(torch.zeros(qgrid.shape[0], device=device))
                continue

            if m == 1:
                # HDBSCAN errors below 2 samples (see module design notes).
                # A single atom is trivially its own bead -- skip
                # clustering and let the shared bead-Debye code below
                # handle it exactly, same as any other singleton.
                cluster_id = torch.zeros(1, dtype=torch.long, device=device)
            else:
                # HDBSCAN needs a CPU numpy array, not a (possibly-GPU)
                # tensor.
                coords_n_np = coords_n.detach().cpu().numpy()
                labels_np = hscan.fit_predict(coords_n_np)
                labels = torch.as_tensor(labels_np, device=device)

                # give every noise point its own singleton cluster id,
                # placed after the real cluster ids, so the loop below can
                # treat every group uniformly -- see class docstring.
                noise = labels == -1
                n_real_clusters = (
                    int(labels[~noise].max().item()) + 1
                    if (~noise).any()
                    else 0
                )
                singleton_ids = n_real_clusters + torch.arange(
                    int(noise.sum().item()), device=device
                )
                cluster_id = labels.clone()
                cluster_id[noise] = singleton_ids

            n_clusters = int(cluster_id.max().item()) + 1

            f_atoms = ftable[vocabs_n]  # (m, Q) -- per-atom form factor

            # group-average each cluster into one bead: position is the
            # centroid, amplitude is the summed per-atom form factor. A
            # singleton (noise) "cluster" collapses to that atom's own
            # position/amplitude exactly, no information lost there.
            beads = []
            bead_amps = []
            for c in range(n_clusters):
                sel = cluster_id == c
                beads.append(coords_n[sel].mean(dim=0))
                bead_amps.append(f_atoms[sel].sum(dim=0))  # (Q,) -- Σf_i(q)

            beads = torch.stack(beads)  # (C, 3)
            bead_amps = torch.stack(bead_amps)  # (C, Q)

            # full Debye sum over beads, diagonal included: each bead
            # scatters with itself (self-distance 0, sinc(0) = 1 via the
            # qr.abs() < 1e-8 branch), same as an atom would in a plain
            # all-atom Debye sum. Unlike a two-level scheme, there is no
            # separate exact intra-cluster term already counting this, so
            # the diagonal must stay in.
            beads_c = beads if dtype is None else beads.to(dtype)
            diff = beads_c.unsqueeze(0) - beads_c.unsqueeze(1)  # (C, C, 3)
            dists = diff.norm(dim=-1).float()  # (C, C)

            i_q = torch.zeros(qgrid.shape[0], device=device)
            for a, b in q_chunks(qgrid.shape[0], n_clusters):
                ac = bead_amps[:, a:b].transpose(0, 1)  # (Qc, C)
                qr = qgrid[a:b].view(-1, 1, 1) * dists.unsqueeze(
                    0
                )  # (Qc, C, C)
                sinc = torch.where(
                    qr.abs() < 1e-8,
                    torch.ones_like(qr),
                    torch.sin(qr) / qr,
                )
                tmp = torch.bmm(sinc, ac.unsqueeze(2)).squeeze(2)  # (Qc, C)
                i_q[a:b] += (ac * tmp).sum(dim=1)  # full double sum

            preds.append(i_q)

        return torch.stack(preds)
