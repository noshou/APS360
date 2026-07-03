import xraydb
import re
import numpy as np
import torch

from torch               import nn
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

    def __init__(self, qgrid, energy):

        """Precompute reference form factors and Kratky weights for the loss.

        Parameters
        ----------
        qgrid : torch.Tensor
            Q-grid points, shape (Q,).
        energy : float
            X-ray energy (eV) used to evaluate anomalous scattering
            factors f1/f2 via xraydb.

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

    @jaxtyped(typechecker=beartype)
    def _kratky_MSLE(
        self,
        output_head: Float[torch.Tensor, "N Q"],
        batch: Batch,
        ) -> Float[torch.Tensor, "N Q"]:

        """Compute the Kratky-weighted mean squared log error.

        (1+q²) * (log1p(Î(q)) - log1p(I(q)))²: Kratky weighting emphasizes
        high-q structure.

        Parameters
        ----------
        output_head : torch.Tensor
            Predicted I(q), shape (N, Q).
        batch : Batch
            Input batch; uses `batch.iqval`, the reference I(q), shape
            (N, Q).

        Returns
        -------
        torch.Tensor
            Per-molecule, per-q-point Kratky MSLE, shape (N, Q).
        """
        residual = torch.log1p(output_head) - torch.log1p(batch.iqval)
        return self._q_weights_ * residual ** 2

    @jaxtyped(typechecker=beartype)
    def _ff_penalty(
        self,
        f_mag_pred: Float[torch.Tensor, "N M Q"],
        batch: Batch,
        lambda_6: float
        ) -> Float[torch.Tensor, "N Q"]:

        """Compute an atom-count-normalized form-factor penalty.

        log1p-normalized L2 between predicted and xraydb reference form
        factors, atom-count-normalized.

        Parameters
        ----------
        f_mag_pred : torch.Tensor
            Predicted form factor magnitudes, shape (N, M, Q).
        batch : Batch
            Input batch; uses `batch.vocab` to look up reference form
            factors and `batch.padding_mask()` to mask out padding atoms.
        lambda_6 : float
            Weight applied to the penalty.

        Returns
        -------
        torch.Tensor
            Per-molecule, per-q-point form-factor penalty, shape (N, Q).
        """
        f_mag_real = torch.log1p(self._fmag_table[batch.vocab])
        f_mag_pred = torch.log1p(f_mag_pred)
        n_atoms = batch.padding_mask().sum(dim=1, keepdim=True).float().clamp(min=1)  # (N, 1)
        mask = batch.padding_mask().unsqueeze(-1)  # (N, M, 1)
        return ((lambda_6*((f_mag_pred-f_mag_real)**2)*mask).sum(dim=1))/n_atoms

    @jaxtyped(typechecker=beartype)
    def _sg_penalty(
        self,
        sigmas:   Float[torch.Tensor, "N M Q"],
        lambda_7: float,
        batch:    Batch
    ) -> Float[torch.Tensor, "N Q"]:

        """Compute an atom-count-normalized L2 penalty on sigma bandwidths.

        Prevents RFF bandwidths from blowing up.

        Parameters
        ----------
        sigmas : torch.Tensor
            Predicted per-atom per-q sigma bandwidths, shape (N, M, Q).
        lambda_7 : float
            Weight applied to the penalty.
        batch : Batch
            Input batch; uses `batch.padding_mask()` to mask out padding
            atoms.

        Returns
        -------
        torch.Tensor
            Per-molecule, per-q-point sigma penalty, shape (N, Q).
        """
        mask    = batch.padding_mask().unsqueeze(-1)       # (N, M, 1)
        n_atoms = mask.sum(dim=1).clamp(min=1)             # (N, 1)
        pen_sig = (lambda_7 * torch.pow(sigmas, 2)) * mask # (N, M, Q)
        return pen_sig.sum(dim=1) / n_atoms                # (N, Q)

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
        msle_loss  = self._kratky_MSLE(output_head, batch)
        ff_penalty = self._ff_penalty(f_mag_pred, batch, lambda_6)
        sg_penalty = self._sg_penalty(sigma_pred, lambda_7, batch)

        return (msle_loss + ff_penalty + sg_penalty).mean()
