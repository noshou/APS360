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

    _fmag_table: Float[torch.Tensor, "V Q"] # V = len(VOCAB) + 1
    _q_weights_: Float[torch.Tensor, "1 Q"] # kratky weighting 
    
    def __init__(self, qgrid, energy):
        
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

        """
        Kratky-weighted MSLE between predicted and ground-truth I(q).

        Applies per-q weight (1+q^2) to emphasize high-q structure that would otherwise
        be dominated by the steep low-q signal. Uses log1p to handle the wide dynamic
        range of I(q) values.

        Computes (1+q_i^2) * (log1p(I_hat(q)) - log1p(I(q)))^2 per molecule and q-point.

        Args:
            output_head: predicted intensities, shape (N, Q)
            batch:       input batch; uses batch.iqval as ground truth, shape (N, Q)
        Returns:
            per-molecule per-q losses, shape (N, Q)
        """
        residual = torch.log1p(output_head) - torch.log1p(batch.iqval.float())
        return self._q_weights_ * residual ** 2

    @jaxtyped(typechecker=beartype)
    def _ff_penalty(
        self, 
        f_mag_pred: Float[torch.Tensor, "N M Q"], 
        batch: Batch, 
        lambda_6: float
        ) -> Float[torch.Tensor, "N Q"]:
        
        # retreive real form factor magnitudes, N x M x Q, log normalize them
        f_mag_real = torch.log1p(self._fmag_table[batch.vocab])
        f_mag_pred = torch.log1p(f_mag_pred)

        # since number of atoms can fluctuate depending on batch,
        # we normalize our penalization term to n_atoms
        n_atoms = batch.padding_mask().sum(dim=1, keepdim=True).float().clamp(min=1)  # (N, 1)
        
        # calculate l2 loss (log normalized), reduce to N x Q dimenions
        mask = batch.padding_mask().unsqueeze(-1)  # (N, M, 1)
        return ((lambda_6*((f_mag_pred-f_mag_real)**2)*mask).sum(dim=1))/n_atoms
    
    @jaxtyped(typechecker=beartype)
    def _sg_penalty(
        self,
        sigmas:   Float[torch.Tensor, "N M Q"],
        lambda_7: float,
        batch:    Batch,
        eps:      float
    ) -> Float[torch.Tensor, "N Q"]:

        mask    = batch.padding_mask().unsqueeze(-1)          # (N, M, 1)
        n_atoms = mask.sum(dim=1).clamp(min=1)               # (N, 1)
        inv_sig = (lambda_7 / (sigmas + eps)) * mask         # (N, M, Q)
        return inv_sig.sum(dim=1) / n_atoms                  # (N, Q)        
    
    @jaxtyped(typechecker=beartype)
    def loss(
        self,
        output_head: Float[torch.Tensor, "N Q"],
        f_mag_pred:  Float[torch.Tensor, "N M Q"],
        sigma_pred:  Float[torch.Tensor, "N M Q"],
        batch: Batch,
        lambda_6: float,
        lambda_7: float,
        eps: float
    ) -> Float[torch.Tensor, ""]:

        # cast to float32 before any loss computation: fp16 can overflow or produce
        # inf which then causes NaN in backward (inf*0, 0/0 patterns).
        output_head = output_head.float()
        f_mag_pred  = f_mag_pred.float()
        sigma_pred  = sigma_pred.float()

        msle_loss  = self._kratky_MSLE(output_head, batch)
        ff_penalty = self._ff_penalty(f_mag_pred, batch, lambda_6)
        sg_penalty = self._sg_penalty(sigma_pred, lambda_7, batch, eps)

        return (msle_loss + ff_penalty + sg_penalty).mean()
    