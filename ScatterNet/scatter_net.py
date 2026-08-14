import re
from typing import Callable, Tuple

import numpy as np
import torch
import xraydb
from torch import nn

from Preprocess import VOCAB

from .batching import Batch
from .model import Embed, MessagePass, OutputHead, SplineSmooth


class ScatterNet(nn.Module):
    """
    Full ScatterNet pipeline: Embed → MessagePass → OutputHead → SplineSmooth
    → I(q). Single-GPU only; there is no routing or sharding of any kind.

    1. Embed produces per-atom embeddings, form factor magnitudes, and
    sigmas from the batch.
    2. MessagePass runs lambda_2 rounds of RFF-based message passing on
    the full batch.
    3. OutputHead accumulates the coherent and incoherent I(q) sums from
    the message-passed atoms.
    4. SplineSmooth smooths the raw coherent/incoherent curves into the
    final predicted I(q).

    Also computes the training loss: Kratky MSLE plus a form-factor
    penalty. Sigma needs no penalty term; MessagePass bounds it
    structurally with Embed's arctan window on every round.

    Hyperparameters:
        lambda_1:  atom embedding dimension
        lambda_2:  number of message passing rounds
        lambda_3:  OutputHead hidden width
        lambda_4:  number of halving steps in OutputHead MLP
        lambda_5:  number of RFF features in MessagePass
        atm_chunk: atoms per M-chunk in MessagePass and OutputHead
        mol_chunk: molecules per N-chunk in MessagePass
        qgrid:     q-grid tensor, shape (Q,)
        energy:    X-ray energy (eV) for xraydb form factors
        eps_embd:  numerical floor for Embed
        eps_msgp:  numerical floor for MessagePass
        compile:   torch.compile checkpointed step functions

    Attributes
    ----------
    __fmag_table : torch.Tensor
        Reference form factor magnitudes per vocabulary entry per
        q-point, shape (V, Q) where V = len(VOCAB) + 1.
    __q_weights_ : torch.Tensor
        Kratky weighting (1 + q^2), shape (1, Q).
    """

    __emb: Embed
    __msg: MessagePass
    __out: OutputHead
    __spline: SplineSmooth
    __eps_embd: float
    __eps_msgp: float
    __fmag_table: torch.Tensor  # shape (V, Q)
    __q_weights_: torch.Tensor  # shape (1, Q)
    __fwd_fn: Callable

    def __init__(
        self,
        lambda_1: int,
        lambda_2: int,
        lambda_3: int,
        lambda_4: int,
        lambda_5: int,
        msg_seed: int,
        atm_chunk: int,
        mol_chunk: int,
        qgrid: torch.Tensor,  # shape (Q,)
        energy: float,
        eps_embd: float,
        eps_msgp: float,
        sigma_max: float = 100.0,
        sigma_floor: float = 0.5,
        sigma_init_gain: float = 1.0,
        compile: bool = False,
    ) -> None:
        """Construct the Embed, MessagePass, and OutputHead submodules.

        Parameters
        ----------
        lambda_1 : int
            Atom embedding dimension.
        lambda_2 : int
            Number of message passing rounds.
        lambda_3 : int
            OutputHead hidden width.
        lambda_4 : int
            Number of halving steps in the OutputHead MLP.
        lambda_5 : int
            Number of Random Fourier Features used in MessagePass.
        msg_seed : int
            RNG seed for MessagePass's fixed RFF frequency matrix.
        atm_chunk : int
            Atoms per M-chunk in MessagePass and OutputHead.
        mol_chunk : int
            Molecules per N-chunk in MessagePass (controls chem_env peak
            size).
        qgrid : torch.Tensor
            Q-grid points, shape (Q,).
        energy : float
            X-ray energy (eV) used to evaluate anomalous scattering
            factors f1/f2 via xraydb.
        eps_embd : float
            Numerical floor used inside Embed.
        eps_msgp : float
            Numerical floor for the kernel-weight sum at
            message_pass.py:614. The sigma floor it also used to serve now
            lives in Embed as `sigma_floor`.
        sigma_max : float, optional
            Cap on Embed's 1/q sigma envelope, in Angstroms. Default 100.0.
        sigma_floor : float, optional
            Lower clamp on sigma, in Angstroms. Default 0.5.
        sigma_init_gain : float, optional
            Multiplier on Embed's `_sigma` init. Default 1.0.
        compile : bool, optional
            If True, torch.compile Embed/MessagePass/OutputHead's
            checkpointed step functions. Default is False.

        Returns
        -------
        None
        """

        super().__init__()
        q_points = qgrid.shape[0]

        # torch._dynamo.config.cache_size_limit is a single global setting,
        # not scoped per module - Embed/OutputHead/LambdaHead/SplineSmooth
        # (via MessagePass's SplineSmooth) all compile their own functions
        # and draw from the same budget. The default (8) is too small: a
        # live run hit FailOnRecompileLimitHit in OutputHead during the
        # test-set pass, which has enough bucket-shape diversity to exceed
        # 8 recompiles even though train/val hadn't yet. Raised here, once,
        # for whichever submodule needs it, rather than in one submodule's
        # __init__ (previously lived in MessagePass's; broke silently for
        # every other module when MessagePass stopped compiling anything).
        if compile:
            torch._dynamo.config.cache_size_limit = max(
                torch._dynamo.config.cache_size_limit,
                8 * (lambda_2 + 1),
            )

        self.__msg = MessagePass(
            lambda_1,
            lambda_2,
            lambda_5,
            msg_seed,
            q_points,
            n_chunk=mol_chunk,
            m_chunk=atm_chunk,
            sigma_max=sigma_max,
            sigma_floor=sigma_floor,
            compile=compile,
        )
        self.__out = OutputHead(
            lambda_1, lambda_3, lambda_4, atm_chunk, compile=compile
        )
        delta_q = float(qgrid[1] - qgrid[0])
        self.__spline = SplineSmooth(q_points, delta_q, compile=compile)
        self.__eps_embd = eps_embd
        self.__eps_msgp = eps_msgp

        fmag_table = torch.zeros(len(VOCAB) + 1, len(qgrid))

        # convert q vector to s vector
        sgrid = (qgrid / (4 * torch.pi)).numpy()

        # special cases:
        # 1. ion is transuranic,  skip f1/f2
        # 2. ion is special case, map to appropriate base case
        # 3. ion is not in vocab, use ground state
        for idx, ion in enumerate(VOCAB.ions):
            key = ion.lower()
            if key in VOCAB.TRANSURANICS:
                f_mag = torch.tensor(xraydb.f0(ion, sgrid)).float()
            elif key in VOCAB.SPECIAL_CASES:
                resolved = VOCAB.SPECIAL_CASES[key]
                f_mag = torch.tensor(
                    np.hypot(
                        xraydb.f0(resolved, sgrid)
                        + xraydb.f1_chantler(resolved, energy),
                        xraydb.f2_chantler(resolved, energy),
                    )
                ).float()
            else:
                elem = re.sub(r"[0-9+\-]+$", "", key)
                f_mag = torch.tensor(
                    np.hypot(
                        xraydb.f0(ion, sgrid)
                        + xraydb.f1_chantler(elem, energy),
                        xraydb.f2_chantler(elem, energy),
                    )
                ).float()
            fmag_table[idx + 1] = f_mag

        self.register_buffer(
            "_ScatterNet__fmag_table", fmag_table, persistent=False
        )
        self.register_buffer(
            "_ScatterNet__q_weights_",
            (1 + qgrid**2).unsqueeze(0),
            persistent=False,
        )

        # Physical form-factor init for Embed. Untrained Embed emits
        # f_mag ~ 0.5 while the physical value is ~7, and d log I/d log f_mag
        # is exactly 2, so without this the model starts ~200x off in I(q)
        # and that single mismatch was 99.5% of the gradient norm at init.
        # Built here, after the table, so Embed is constructed last.
        #
        # Averaged over the ORGANIC elements, not over the whole vocab. The
        # vocab is 211 xraydb ions and is dominated by heavy ones (mean and
        # median |f|(0) both ~46.7), but the structures this trains on are
        # proteins and viral capsids, which are H/C/N/O/P/S: |f|(0) = 6.0
        # for C, 7.0 for N, 8.0 for O. Using the vocab mean overshoots the
        # typical atom by ~6.7x, and since I ~ f^2 that is a ~45x error in
        # I(q) pointing the WRONG way - measured I_pred/I_true = 30.3 at
        # init with the vocab mean, against 1.7 with the organic mean.
        organic = ("h", "c", "n", "o", "p", "s")
        rows = [
            idx + 1
            for idx, ion in enumerate(VOCAB.ions)
            if ion.lower() in organic
        ]
        f_init = fmag_table[rows or slice(1, None)].mean(dim=0)
        self.__emb = Embed(
            lambda_1,
            qgrid,
            sigma_max=sigma_max,
            sigma_floor=sigma_floor,
            sigma_init_gain=sigma_init_gain,
            f_init=f_init,
            compile=compile,
        )

        self.__fwd_fn = (
            torch.compile(self.__loss_fn, dynamic=True, fullgraph=True)
            if compile
            else self.__loss_fn
        )

    @staticmethod
    def __loss_fn(
        fmag_table: torch.Tensor,
        q_weights: torch.Tensor,
        output_head: torch.Tensor,
        coh: torch.Tensor,
        inc: torch.Tensor,
        f_mag_pred: torch.Tensor,
        iqval: torch.Tensor,
        vocab: torch.Tensor,
        mask: torch.Tensor,
        lambda_6: float,
        lambda_7: float,
    ) -> torch.Tensor:
        """Compute the total training loss.

        Sums the Kratky MSLE, the form-factor penalty, and a 2nd-
        derivative roughness penalty on the pre-smoothing coherent/
        incoherent curves, then averages over molecules and q-points.

        Parameters
        ----------
        fmag_table : torch.Tensor
            Reference form factor magnitudes per vocabulary entry per
            q-point, shape (V, Q).
        q_weights : torch.Tensor
            Kratky weighting (1 + q^2), shape (1, Q).
        output_head : torch.Tensor
            Predicted I(q), shape (N, Q).
        coh : torch.Tensor
            Raw accumulated coherent sum, pre-`SplineSmooth`, shape
            (N, Q). Penalized directly rather than post-smoothing so the
            penalty nudges `OutputHead` itself, instead of leaning on
            `SplineSmooth`'s Λ to compensate for a rough raw curve.
        inc : torch.Tensor
            Raw accumulated incoherent sum, pre-`SplineSmooth`, shape
            (N, Q).
        f_mag_pred : torch.Tensor
            Predicted form factor magnitudes, shape (N, M, Q).
        iqval : torch.Tensor
            Reference I(q), shape (N, Q).
        vocab : torch.Tensor
            Atom vocabulary indices, shape (N, M).
        mask : torch.Tensor
            Padding mask, shape (N, M); True marks a real (non-padding)
            atom.
        lambda_6 : float
            Weight on the form-factor penalty term.
        lambda_7 : float
            Weight on the 2nd-derivative roughness penalty term.

        Returns
        -------
        torch.Tensor
            Scalar total loss.
        """

        # Kratky-weighted MSLE: (1+q²) * (log1p(Î(q)) - log1p(I(q)))²
        residual = torch.log1p(output_head) - torch.log1p(iqval)
        msle_loss = q_weights * residual**2  # (N, Q)

        mask_2d = mask.unsqueeze(-1)  # (N, M, 1)
        n_atoms = mask.sum(dim=1, keepdim=True).float().clamp(min=1)  # (N, 1)

        # form-factor penalty.
        # log1p-normalized L2 vs xraydb reference, atom-count-normalized
        f_mag_real = torch.log1p(fmag_table[vocab])
        f_mag_pred = torch.log1p(f_mag_pred)
        ff_penalty = (
            (lambda_6 * ((f_mag_pred - f_mag_real) ** 2)) * mask_2d
        ).sum(dim=1) / n_atoms  # (N, Q)

        # 2nd-derivative roughness penalty, pre-smoothing coh/inc, in log
        # space like the terms above (raw coh/inc can be O(1e6)+ from
        # molecule size alone, which blew this term up to ~1e13 on a live
        # run). coh is signed (PBId), so log1p(coh**2) instead of log1p(coh).
        # Plain discrete 2nd difference, not a calculus-accurate derivative:
        # this is a roughness regularizer, not a physical quantity, so no
        # /delta_q**2 (that reintroduces the same scale problem via
        # lambda_7 instead) and no boundary stencil (dropped, like
        # PPSpline's D).
        log_coh = torch.log1p(coh**2)
        log_inc = torch.log1p(inc)
        d2_coh = log_coh[:, 2:] - 2.0 * log_coh[:, 1:-1] + log_coh[:, :-2]
        d2_inc = log_inc[:, 2:] - 2.0 * log_inc[:, 1:-1] + log_inc[:, :-2]
        smooth_penalty = lambda_7 * (d2_coh**2 + d2_inc**2)  # (N, Q-2)

        return (
            msle_loss.mean() + ff_penalty.mean() + smooth_penalty.mean()
        )

    def compute_loss(
        self,
        output_head: torch.Tensor,  # shape (N, Q)
        coh: torch.Tensor,  # shape (N, Q)
        inc: torch.Tensor,  # shape (N, Q)
        f_mag_pred: torch.Tensor,  # shape (N, M, Q)
        batch: Batch,
        lambda_6: float,
        lambda_7: float,
    ) -> torch.Tensor:  # scalar
        """Compute the total training loss.

        Sums the Kratky MSLE, the form-factor penalty, and a 2nd-
        derivative roughness penalty on the pre-smoothing coherent/
        incoherent curves, then averages over molecules and q-points.

        Parameters
        ----------
        output_head : torch.Tensor
            Predicted I(q), shape (N, Q).
        coh : torch.Tensor
            Raw accumulated coherent sum, pre-`SplineSmooth`, shape
            (N, Q).
        inc : torch.Tensor
            Raw accumulated incoherent sum, pre-`SplineSmooth`, shape
            (N, Q).
        f_mag_pred : torch.Tensor
            Predicted form factor magnitudes, shape (N, M, Q).
        batch : Batch
            Input batch with reference I(q), vocab, and padding mask.
        lambda_6 : float
            Weight on the form-factor penalty term.
        lambda_7 : float
            Weight on the 2nd-derivative roughness penalty term.

        Returns
        -------
        torch.Tensor
            Scalar total loss.
        """
        # .contiguous(): output_head/f_mag_pred/mask come from upstream
        # concatenation/squeeze ops whose stride can vary run to run, and
        # torch.compile guards on stride() in addition to shape - causing
        # avoidable recompiles.
        return self.__fwd_fn(
            self.__fmag_table,
            self.__q_weights_,
            output_head.contiguous(),
            coh.contiguous(),
            inc.contiguous(),
            f_mag_pred.contiguous(),
            batch.iqval,
            batch.vocab,
            batch.padding_mask(),
            lambda_6,
            lambda_7,
        )

    def forward(
        self, batch: Batch
    ) -> Tuple[
        torch.Tensor,  # shape (N, Q)
        torch.Tensor,  # shape (N, Q)
        torch.Tensor,  # shape (N, Q)
        torch.Tensor,  # shape (N, M, Q)
        torch.Tensor,  # shape (N, M, Q)
    ]:
        """Run the ScatterNet forward pass.

        Parameters
        ----------
        batch : Batch
            Input batch of molecules (atom vocabulary, coordinates, and
            target I(q) values).

        Returns
        -------
        torch.Tensor
            `iq`, predicted I(q), shape (N, Q).
        torch.Tensor
            `coh`, raw accumulated coherent sum, unsquared and
            pre-smoothing, shape (N, Q). For the 2nd-derivative loss.
        torch.Tensor
            `inc`, raw accumulated incoherent sum, pre-smoothing, shape
            (N, Q). For the 2nd-derivative loss.
        torch.Tensor
            `f_mags`, per-atom form factor magnitudes, shape (N, M, Q).
        torch.Tensor
            `sigmas`, per-atom Gaussian kernel bandwidth, shape (N, M, Q).
        """

        embed_head = self.__emb(batch, self.__eps_embd)
        msg_head = self.__msg(batch, embed_head, self.__eps_msgp)
        _, f_mags, sigmas = msg_head
        f_mags = f_mags.squeeze(-1)
        sigmas = sigmas.squeeze(-1)
        # whole molecule on one device, so coh already covers every atom
        coh, inc = self.__out(batch, msg_head)
        iq = self.__spline(sigmas, f_mags, coh, inc, batch)
        return iq, coh, inc, f_mags, sigmas
