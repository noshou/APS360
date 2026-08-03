from collections import OrderedDict
from typing import Callable

import torch
import torch.nn.functional as F
from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import Float, jaxtyped
from numpy import floor, log2
from torch import nn

from ..batching import Batch
from ..utils.no_trilin_bilin import NoTrilinBilin
from .layer_head import LayerHead


class OutputHead(nn.Module):
    """
    Collapses per-atom contributions into a predicted I(q) curve.

    For each atom, combines its embedding with its form factor magnitude
    via a bilinear layer and passes the result through an MLP, which emits
    two per-atom, per-q channels (w, c). These are summed over atoms into
    the two limits of the Debye sum:

        I(q) = (sum_j w_j f_j)^2 + sum_j c_j f_j^2

    The squared term is the coherent limit: at q -> 0 every sinc(q r_ij)
    goes to 1 and the Debye double sum factorizes to (sum_j f_j)^2, which
    scales like M^2. The linear term is the incoherent limit: at high q the
    off-diagonal pairs oscillate away leaving sum_j f_j^2, which scales
    like M. Squaring a sum over atoms *is* the pairwise double sum, so the
    coherent term costs one O(M) pass, not O(M^2).

    Both channels stay O(1) at every q; the M^2 scaling comes from the
    squaring op rather than from learned magnitude. A single linear
    sum-of-f^2 head cannot reach (sum f)^2 at any parameter setting, and
    could only fake it by inflating its per-atom weights to ~M, which is
    what previously made low q converge far slower than high q.

    The forward pass is chunked to avoid storing the
    (N, M, Q, lambda_3) bilinear output tensor.
    """

    _bilinear: NoTrilinBilin
    _mlp: nn.Sequential
    _fwd_fn: Callable

    def __init__(
        self,
        lambda_1: int,
        lambda_3: int,
        lambda_4: int,
        out_chunk: int,
        compile: bool = False,
    ) -> None:
        """Build the bilinear layer and compression MLP.

        Parameters
        ----------
        lambda_1 : int
            Atom embedding dimension (bilinear input size).
        lambda_3 : int
            Hidden width of the bilinear output and MLP input.
        lambda_4 : int
            Number of halving steps in the MLP; must satisfy
            2**lambda_4 <= lambda_3.
        out_chunk : int
            Number of atoms processed per chunk in `forward`.
        compile : bool, optional
            If True, torch.compile `_forward_fn`. Default is False.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If `lambda_4` is not greater than 0, or if `lambda_4` exceeds
            floor(log2(lambda_3)) for the given `lambda_3`.
        """

        super().__init__()
        self._out_chunk = out_chunk

        if lambda_4 <= 0:
            raise ValueError("lambda_4 must be > 0")
        if lambda_4 > floor(log2(lambda_3)):
            verr1 = f"lambda_4 must be <= {int(floor(log2(lambda_3)))}"
            verr2 = f" for lambda_3={lambda_3}"
            verr  = verr1 + verr2
            raise ValueError(verr)
        self._bilinear = NoTrilinBilin(lambda_1, 1, lambda_3)

        dims = [lambda_3 // 2**i for i in range(lambda_4 + 1)]
        if dims[-1] == 1:
            dims[-1] = 2
        elif dims[-1] != 2:
            dims.append(2)

        ldicts = OrderedDict()
        for i in range(len(dims) - 1):
            ldicts[f"layer_{i}"] = nn.Linear(dims[i], dims[i + 1])
            if i < len(dims) - 2:
                ldicts[f"activation_{i}"] = nn.Mish()
        self._mlp = nn.Sequential(ldicts)

        # terminal layer emits (w, c);
        # softplus(0.5413) = 1.0 so both channels start
        # at unity, making I(0) = (Σf)² and high-q = Σf² correct at init.
        nn.init.constant_(self._mlp[-1].bias, 0.5413) # type: ignore

        self._fwd_fn = (
            torch.compile(self._forward_fn, dynamic=True, fullgraph=True)
            if compile
            else self._forward_fn
        )

    @staticmethod
    def _forward_fn( # no jaxtype/beartype cuz torch.compile will break
        bilinear: NoTrilinBilin,
        mlp: torch.nn.Sequential,
        emb_c: torch.Tensor,
        fmag_c: torch.Tensor,
        mask_c: torch.Tensor
    ) -> Tuple[
        Float[torch.Tensor, "N Q"],  # noqa: F722
        Float[torch.Tensor, "N Q"],  # noqa: F722
    ]:
        """Compute the coherent and incoherent partial sums for one chunk.

        Parameters
        ----------
        bilinear : NoTrilinBilin
            Bilinear layer combining atom embedding and form factor
            magnitude.
        mlp : torch.nn.Sequential
            Compression MLP mapping the bilinear output down to two
            scalars (w, c) per atom per q-point.
        emb_c : torch.Tensor
            Atom embeddings for this chunk, shape (N, Mc, Q, lambda_1).
        fmag_c : torch.Tensor
            Form factor magnitudes for this chunk, shape (N, Mc, Q, 1).
        mask_c : torch.Tensor
            Padding mask for this chunk, shape (N, Mc, 1).

        Returns
        -------
        torch.Tensor
            `coh_c`, coherent partial sum(w * f) over this chunk's atoms,
            shape (N, Q). NOT squared here; see `forward`.
        torch.Tensor
            `inc_c`, incoherent partial sum(c * f^2) over this chunk's
            atoms, shape (N, Q).
        """

        # (N,M,Q, λ₁) x (N,M,Q,1) -> (N,M,Q,λ₃)
        atomic = F.mish(bilinear(emb_c, fmag_c))

        # MLP compression -> (N,M,Q,2), split into the two channels
        chans = F.softplus(mlp(atomic))
        w, c = chans[..., 0], chans[..., 1]  # (N,M,Q) each

        # fmc**2 is the squared form factor and this reduces over M atoms.
        with torch.autocast(device_type=emb_c.device.type, enabled=False):
            fmc = fmag_c.squeeze(-1).float()  # (N, Mc, Q)
            maskf = mask_c.float()
            coh_c = (w.float() * fmc * maskf).sum(dim=1)  # f¹, (N, Q) fp32
            inc_c = (c.float() * fmc**2 * maskf).sum(dim=1)  # (N, Q) fp32
            return coh_c, inc_c

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        batch: Batch,
        msg_head: LayerHead,
    ) -> Tuple[
            Float[torch.Tensor, "N Q"],   # noqa: F722
            Float[torch.Tensor, "N Q"],   # noqa: F722
            Float[torch.Tensor, "N M Q"], # noqa: F722
            Float[torch.Tensor, "N M Q"], # noqa: F722
        ]:

        """Accumulate the two Debye limits over atom chunks.

        Returns the coherent sum UNSQUARED. Callers must reduce it over
        every atom of the molecule and only then call `combine`. See that
        method for why the square cannot happen here.

        Parameters
        ----------
        batch : Batch
            Input batch; used for `batch.padding_mask()`.
        msg_head : LayerHead
            Output of MessagePass; embeds (N, M, Q, lambda_1), f_mags and
            sigmas (N, M, Q, 1).

        Returns
        -------
        torch.Tensor
            `coh_accum`, coherent sum(w * f) over this call's atoms, shape
            (N, Q). NOT squared; see `combine`.
        torch.Tensor
            `inc_accum`, incoherent sum(c * f^2) over this call's atoms,
            shape (N, Q).
        torch.Tensor
            `f_mags`, per-atom form factor magnitudes, shape (N, M, Q).
        torch.Tensor
            `sigmas`, per-atom kernel bandwidth, shape (N, M, Q).
        """

        # 1. initialize values
        N, M, Q, _ = msg_head.embeds.shape
        mask = (
            batch.padding_mask().unsqueeze(-1).to(msg_head.embeds.dtype)
        )  # (N, M, 1)

        # accumulate both Debye limits over atom-chunks.
        coh_accum = torch.zeros(
            N, Q, device=msg_head.embeds.device, dtype=torch.float32
        )
        inc_accum = torch.zeros_like(coh_accum)

        # 2. Accumulate over chunks
        for mol1 in range(0, M, self._out_chunk):
            mol2 = min(mol1 + self._out_chunk, M)
            # .contiguous(): these views' stride/storage_offset
            # vary by mol1 (which chunk), and torch.compile guards on
            # stride()/storage_offset() in addition to shape causing
            # extra recompiles beyond the intended chunk-size guard.
            coh_c, inc_c = self._fwd_fn(
                self._bilinear,
                self._mlp,
                msg_head.embeds[:, mol1:mol2].contiguous(),
                msg_head.f_mags[:, mol1:mol2].contiguous(),
                mask[:, mol1:mol2].contiguous(),
            )
            coh_accum += coh_c
            inc_accum += inc_c

        # 3. return output head. coh_accum stays UNSQUARED: under tensor
        # parallelism this call only saw one rank's atom shard, so the
        # square has to wait until every shard has been reduced in.
        f_mags = msg_head.f_mags.squeeze(-1)
        sigmas = msg_head.sigmas.squeeze(-1)
        return coh_accum, inc_accum, f_mags, sigmas

    @staticmethod
    @jaxtyped(typechecker=beartype)
    def combine(
        coh: Float[torch.Tensor, "N Q"],  # noqa: F722
        inc: Float[torch.Tensor, "N Q"],  # noqa: F722
    ) -> Float[torch.Tensor, "N Q"]:  # noqa: F722
        """Square the coherent sum and add the incoherent one.

        `coh` MUST already cover every atom of each molecule before this
        is called. Squaring a partial sum computes sum_parts (sum_j)^2
        instead of (sum_parts sum_j)^2, which silently drops every atom
        pair split across the partition. That applies at both levels the
        atom dimension gets split:

        - the `_out_chunk` loop in `forward`, handled by accumulating
        over all chunks before returning
        - the TP atom shard in `ScatterNet.forward`, handled by
        all-reducing across ranks before calling this

        Neither failure raises. Both still train and still lower the
        loss, they just fit a systematically wrong low-q limit, so the
        chunk- and rank-invariance tests are the only things that catch
        them.

        Parameters
        ----------
        coh : torch.Tensor
            Fully reduced coherent sum(w * f), shape (N, Q).
        inc : torch.Tensor
            Fully reduced incoherent sum(c * f^2), shape (N, Q).

        Returns
        -------
        torch.Tensor
            Predicted I(q), shape (N, Q).
        """
        return coh**2 + inc
