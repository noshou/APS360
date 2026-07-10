import xraydb
import re
import numpy as np
import torch

from torch               import nn
from typing               import Callable
from jaxtyping           import jaxtyped, Float
from beartype            import beartype
from ScatterNet.batching import Batch
from Preprocess          import VOCAB

class Loss(nn.Module):

    """Training loss for ScatterNet: Kratky MSLE plus form-factor and sigma penalties.

    Attributes
    ----------
    _fmag_table : torch.Tensor
        Reference form factor magnitudes per vocabulary entry per
        q-point, shape (V, Q) where V = len(VOCAB) + 1.
    _q_weights_ : torch.Tensor
        Kratky weighting (1 + q^2), shape (1, Q).
    """

    _fmag_table: Float[torch.Tensor, "V Q"] # V = len(VOCAB) + 1
    _q_weights_: Float[torch.Tensor, "1 Q"] # kratky weighting
    _fwd_fn:     Callable # torch.compiled or plain _loss_fn, per the compile flag

    def __init__(self, qgrid, energy, compile: bool = False):

        """Precompute reference form factors and Kratky weights for the loss.

        Parameters
        ----------
        qgrid : torch.Tensor
            Q-grid points, shape (Q,).
        energy : float
            X-ray energy (eV) used to evaluate anomalous scattering
            factors f1/f2 via xraydb.
        compile : bool, optional
            If True, torch.compile `_loss_fn`. Default is False.

        Returns
        -------
        None
        """

        super().__init__()

        fmag_table = torch.zeros(len(VOCAB)+1, len(qgrid))

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
                        xraydb.f0(resolved, sgrid) +
                        xraydb.f1_chantler(resolved, energy),
                        xraydb.f2_chantler(resolved, energy)
                    )
                ).float()
            else:
                elem = re.sub(r'[0-9+\-]+$', '', key)
                f_mag = torch.tensor(
                    np.hypot(
                        xraydb.f0(ion, sgrid) +
                        xraydb.f1_chantler(elem, energy),
                        xraydb.f2_chantler(elem, energy)
                    )
                ).float()
            fmag_table[idx + 1] = f_mag

        self.register_buffer('_fmag_table', fmag_table)
        self.register_buffer('_q_weights_', (1 + qgrid**2).unsqueeze(0))
        # mode="reduce-overhead": see the matching comment in MessagePass - launch-bound
        # profile (Self CPU >> Self CUDA), CUDA graphs cut per-kernel dispatch overhead.
        self._fwd_fn = torch.compile(self._loss_fn, dynamic=True, fullgraph=True, mode="reduce-overhead") if compile else self._loss_fn

    @staticmethod
    def _loss_fn(
        fmag_table:  torch.Tensor,
        q_weights:   torch.Tensor,
        output_head: torch.Tensor,
        f_mag_pred:  torch.Tensor,
        sigma_pred:  torch.Tensor,
        iqval:       torch.Tensor,
        vocab:       torch.Tensor,
        mask:        torch.Tensor,
        lambda_6:    float,
        lambda_7:    float
    ) -> torch.Tensor:

        """Compute the total training loss.

        Sums the Kratky MSLE, form-factor penalty, and sigma penalty,
        then averages over molecules and q-points.

        Parameters
        ----------
        fmag_table : torch.Tensor
            Reference form factor magnitudes per vocabulary entry per
            q-point, shape (V, Q).
        q_weights : torch.Tensor
            Kratky weighting (1 + q^2), shape (1, Q).
        output_head : torch.Tensor
            Predicted I(q), shape (N, Q).
        f_mag_pred : torch.Tensor
            Predicted form factor magnitudes, shape (N, M, Q).
        sigma_pred : torch.Tensor
            Predicted sigma bandwidths, shape (N, M, Q).
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
            Weight on the sigma L2 penalty term.

        Returns
        -------
        torch.Tensor
            Scalar total loss.
        """

        # Kratky-weighted MSLE: (1+q²) * (log1p(Î(q)) - log1p(I(q)))²
        residual  = torch.log1p(output_head) - torch.log1p(iqval)
        msle_loss = q_weights * residual ** 2  # (N, Q)

        mask_2d = mask.unsqueeze(-1)                                     # (N, M, 1)
        n_atoms = mask.sum(dim=1, keepdim=True).float().clamp(min=1)     # (N, 1)

        # form-factor penalty: log1p-normalized L2 vs xraydb reference, atom-count-normalized
        f_mag_real = torch.log1p(fmag_table[vocab])
        f_mag_pred = torch.log1p(f_mag_pred)
        ff_penalty = ((lambda_6 * ((f_mag_pred - f_mag_real) ** 2)) * mask_2d).sum(dim=1) / n_atoms  # (N, Q)

        # sigma L2 penalty, atom-count-normalized
        sg_penalty = ((lambda_7 * torch.pow(sigma_pred, 2)) * mask_2d).sum(dim=1) / n_atoms  # (N, Q)

        return (msle_loss + ff_penalty + sg_penalty).mean()

    @jaxtyped(typechecker=beartype)
    def loss(
        self,
        output_head: Float[torch.Tensor, "N Q"],
        f_mag_pred:  Float[torch.Tensor, "N M Q"],
        sigma_pred:  Float[torch.Tensor, "N M Q"],
        batch: Batch,
        lambda_6: float,
        lambda_7: float
    ) -> Float[torch.Tensor, ""]:

        """Compute the total training loss.

        Sums the Kratky MSLE, form-factor penalty, and sigma penalty,
        then averages over molecules and q-points.

        Parameters
        ----------
        output_head : torch.Tensor
            Predicted I(q), shape (N, Q).
        f_mag_pred : torch.Tensor
            Predicted form factor magnitudes, shape (N, M, Q).
        sigma_pred : torch.Tensor
            Predicted sigma bandwidths, shape (N, M, Q).
        batch : Batch
            Input batch with reference I(q), vocab, and padding mask.
        lambda_6 : float
            Weight on the form-factor penalty term.
        lambda_7 : float
            Weight on the sigma L2 penalty term.

        Returns
        -------
        torch.Tensor
            Scalar total loss.
        """
        # .contiguous(): output_head/f_mag_pred/sigma_pred/mask come from upstream
        # concatenation/squeeze ops whose stride can vary run to run, and torch.compile
        # guards on stride() in addition to shape - causing avoidable recompiles.
        return self._fwd_fn(
            self._fmag_table,
            self._q_weights_,
            output_head.contiguous(),
            f_mag_pred.contiguous(),
            sigma_pred.contiguous(),
            batch.iqval,
            batch.vocab,
            batch.padding_mask(),
            lambda_6,
            lambda_7,
        )
