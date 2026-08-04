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
from ..utils.pbid import PBId
from ..utils.sqrp import SqrP
from .layer_head import LayerHead

# Multiplier on the terminal Linear's default weight init. Small enough that
# `raw` does not disturb the w = c = 1 targets (std ~0.5 -> ~0.025), large
# enough that the backward through it is not degenerate. See the note at the
# call site in `OutputHead.__init__`.
_TERMINAL_INIT_GAIN = 0.05


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
    like M.

    Initialization
    --------------
    Both channels are offset so a zero MLP output gives w = c = 1, i.e.
    coh = sum_j f_j and inc = sum_j f_j^2 exactly. The head therefore
    emits the true Debye q->0 limit at step 0 and learns only the
    correction.

    The forward pass is chunked to avoid storing the
    (N, M, Q, lambda_3) bilinear output tensor.
    """

    _bilinear: NoTrilinBilin
    _mlp: nn.Sequential
    _pbid: PBId
    _sqrp: SqrP
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
            verr = verr1 + verr2
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
        self._pbid = PBId(1)
        self._sqrp = SqrP()

        # Terminal layer emits (w, c). Its bias is zeroed and its weight is
        # shrunk by _TERMINAL_INIT_GAIN so `raw` starts near 0 and the two
        # channels land on their w = c = 1 targets, i.e. on the true Debye
        # q->0 limit. Leaving the weight at its default put `raw` at an
        # O(0.5) random value that swamped the offsets (which are O(0.03) to
        # O(0.8)), gave I(0) a ~2x error before the model had learned
        # anything, and pushed E[raw_coh] negative, so the coherent channel
        # had to be dragged through coh = 0 where dI/dcoh = 2*coh = 0 is a
        # stationary point of the loss.
        terminal: nn.Linear = self._mlp[-1]  # type: ignore[assignment]
        with torch.no_grad():
            terminal.weight.mul_(_TERMINAL_INIT_GAIN)
            terminal.bias.zero_()

        self._fwd_fn = (
            torch.compile(self._forward_fn, dynamic=True, fullgraph=True)
            if compile
            else self._forward_fn
        )

    @staticmethod
    def _forward_fn(  # no jaxtype/beartype cuz torch.compile will break
        bilinear: NoTrilinBilin,
        mlp: torch.nn.Sequential,
        pbid: PBId,
        sqrp: SqrP,
        emb_c: torch.Tensor,
        fmag_c: torch.Tensor,
        mask_c: torch.Tensor,
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
        pbid : PBId
            Bent identity applied to the coherent channel, which must be
            able to take signed values.
        sqrp : SqrP
            Square-plus activation module applied to the incoherent channel.
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
            shape (N, Q).
        torch.Tensor
            `inc_c`, incoherent partial sum(c * f^2) over this chunk's
            atoms, shape (N, Q).
        """

        # (N,M,Q, λ₁) x (N,M,Q,1) -> (N,M,Q,λ₃)
        atomic = F.mish(bilinear(emb_c, fmag_c))

        # (N,M,Q,λ₃) -> (N,M,Q,λ₃//2) -> (N,M,Q,λ₃//4) -> ... -> (N,M,Q,2)
        # MLP compression splits input into incoherent and coherent channels.
        raw = mlp(atomic)

        # Both channels are offset so that a zero MLP output gives w = c = 1
        # exactly, i.e. coh = sum_j f_j and inc = sum_j f_j**2, which IS the
        # Debye q->0 limit. The head therefore starts on the physical answer
        # and only has to learn the correction. See the "Initialization"
        # section of the class docstring for why the target is 1 and not
        # 1/sqrt(M).
        #
        # The targets are constants, so there is no M here and nothing that
        # depends on the chunk. Both inverses stay differentiable in the
        # activations' own parameters, which is what keeps the "raw = 0 maps
        # to 1" property true as those parameters move.
        y = 1.0

        # 1. Coherent channel (w), signed, so it goes through PBId.
        # Solve PBId(x) = w*(sqrt(x^2+1)-1)/2 + x + b = y for x:
        #   x = 2[(2z + w) - w*sqrt(z^2 + w*z + 1)] / (4 - w^2),  z = y - b
        # PBId bounds |w| <= 1.9, so the 4 - w^2 denominator is >= 0.39 and
        # the radicand's discriminant (w^2 - 4) stays negative; both were
        # singular when w was unconstrained.
        w_param = pbid.weight
        b_param = pbid.bias if pbid.bias is not None else 0.0
        ymb = y - b_param
        numerator = 2.0*ymb+w_param-w_param*torch.sqrt(w_param*ymb+ymb**2+1.0)
        denominator = 2.0 * (1.0 - (w_param**2 / 4.0))
        target_input_coh = numerator / denominator
        coh_logits = raw[..., :1] + target_input_coh
        coh = pbid(coh_logits).squeeze(-1)

        # 2. Incoherent channel (c), strictly positive, so it goes through
        # SqrP. Solve SqrP(x) = (x + sqrt(x^2 + B)) / 2 = y for x:
        #   x = y - B / (4y)
        # At the default B = 4 and y = 1 this is exactly x = 0, where
        # SqrP'(0) = 1/2 is at its maximum. The old 1/sqrt(M) target put this
        # at x ~ -sqrt(M) instead, where SqrP' ~ 1/M (1.8e-4 at M = 5436) and
        # fp16's ulp at that magnitude quantized the channel to a constant.
        b_param_sqrp = sqrp.b
        if b_param_sqrp is None:
            b_param_sqrp = torch.full((), 4.0, device=raw.device)
        target_input_inc = y - (b_param_sqrp / (4.0 * y))
        inc_logits = raw[..., 1:] + target_input_inc
        c = sqrp(inc_logits).squeeze(-1)  # (N, Mc, Q)

        # Reduce over Mc atoms per chunk with autocast disabled for precision
        with torch.autocast(device_type=emb_c.device.type, enabled=False):
            fmc = fmag_c.squeeze(-1).float()                # (N, Mc, Q)
            maskf = mask_c.float()                          # (N, Mc, 1)
            coh_c = (coh.float() * fmc * maskf).sum(dim=1)  # f^1, (N, Q) fp32
            inc_c = (c.float() * fmc**2 * maskf).sum(dim=1) # f^2, (N, Q) fp32
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

        Returns the coherent sum unsquared. Callers must reduce it over
        every atom of the molecule and only then call `combine`.
        """

        # 1. initialize values
        N, M, Q, _ = msg_head.embeds.shape
        mask = batch.padding_mask().unsqueeze(-1).to(msg_head.embeds.dtype)

        # accumulate both Debye limits over atom-chunks.
        coh_accum = torch.zeros(
            N,
            Q,
            device=msg_head.embeds.device,
            dtype=torch.float32
        )
        inc_accum = torch.zeros_like(coh_accum)

        # 2. Accumulate over chunks
        for mol1 in range(0, M, self._out_chunk):
            mol2 = min(mol1 + self._out_chunk, M)
            coh_c, inc_c = self._fwd_fn(
                self._bilinear,
                self._mlp,
                self._pbid,
                self._sqrp,
                msg_head.embeds[:, mol1:mol2].contiguous(),
                msg_head.f_mags[:, mol1:mol2].contiguous(),
                mask[:, mol1:mol2].contiguous(),
            )
            coh_accum += coh_c
            inc_accum += inc_c

        # 3. return output head. coh_accum stays UNSQUARED
        f_mags = msg_head.f_mags.squeeze(-1)
        sigmas = msg_head.sigmas.squeeze(-1)
        return coh_accum, inc_accum, f_mags, sigmas

    @staticmethod
    @jaxtyped(typechecker=beartype)
    def combine(
        coh: Float[torch.Tensor, "N Q"],  # noqa: F722
        inc: Float[torch.Tensor, "N Q"],  # noqa: F722
    ) -> Float[torch.Tensor, "N Q"]:  # noqa: F722
        """Square the coherent sum and add the incoherent one."""
        return coh**2 + inc
