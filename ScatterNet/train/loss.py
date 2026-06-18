import xraydb
import re
import numpy as np
import torch

from jaxtyping           import jaxtyped, Float
from beartype            import beartype
from ScatterNet.batching import Batch
from Preprocess          import VOCAB

class Loss():

    _fmag_table: Float[torch.Tensor, "V Q"] # V = len(VOCAB) + 1
    
    def __init__(self, qgrid, energy):
        
        self._fmag_table = torch.zeros(len(VOCAB)+1, len(qgrid))
        
        # convert q vector to s vector
        sgrid = (qgrid / (4 * torch.pi)).numpy()
        
        # special cases: 
        # 1. ion is transuranic,  skip f1/f2
        # 2. ion is special case, map to appropriate base case
        # 3. ion is not in vocab, use ground state
        for idx, ion in enumerate(VOCAB.ions):
            if   ion in VOCAB.TRANSURANICS:
                f_mag = torch.tensor(xraydb.f0(ion, sgrid)).float()
            elif ion in VOCAB.SPECIAL_CASES.keys():
                f_mag = torch.tensor(
                    np.hypot(
                        xraydb.f0(VOCAB.SPECIAL_CASES[ion], sgrid) + 
                        xraydb.f1_chantler(VOCAB.SPECIAL_CASES[ion], energy),
                        xraydb.f2_chantler(VOCAB.SPECIAL_CASES[ion], energy)
                    )
                ).float()
            else:
                f_mag = torch.tensor(
                    np.hypot(
                        xraydb.f0(ion, sgrid) + 
                        xraydb.f1_chantler(re.sub(r'[0-9+\-]+$', '', ion), energy),
                        xraydb.f2_chantler(re.sub(r'[0-9+\-]+$', '', ion), energy)
                    )
                ).float() 
            self._fmag_table[idx + 1] = f_mag
        
        return 
    
    
    @jaxtyped(typechecker=beartype)
    def _MSLE(
        self,
        output_head: Float[torch.Tensor, "N Q"], 
        batch: Batch
        ) -> Float[torch.Tensor, "N Q"]: 
        
        """Mean squared log error between predicted and ground-truth I(q).

        Computes (log(1 + Î(q)) - log(1 + I(q)))² over all molecules and q-points.

        Args:
            output_head: predicted intensities from OutputHead, shape (N, Q)
            batch:       input batch; uses batch.iqval as ground truth, shape (N, Q)

        Returns:
            scalar loss tensor
        """
        preds = torch.log1p(output_head)
        truth = torch.log1p(batch.iqval)
        
        return (preds - truth) ** 2

    @jaxtyped(typechecker=beartype)
    def _ff_penalty(
        self, 
        f_mag_pred: Float[torch.Tensor, "N M Q"], 
        batch: Batch, 
        lambda_6: float
        ) -> Float[torch.Tensor, "N Q"]:
        
        # retreive real form factor magnitudes, N x M x Q, log normalize them
        f_mag_real = torch.log1p(self._fmag_table.to(batch.vocab.device)[batch.vocab])
        self._fmag_table = self._fmag_table.to(batch.vocab.device)
        f_mag_pred = torch.log1p(f_mag_pred)
        
        # since number of atoms can fluctuate depending on batch, 
        # we normalize our penalization term to n_atoms 
        n_atoms = batch.padding_mask().sum(dim=1, keepdim=True).float()  # (N, 1)
        
        # calculate l2 loss (log normalized), reduce to N x Q dimenions
        mask = batch.padding_mask().unsqueeze(-1)  # (N, M, 1)
        return ((lambda_6*((f_mag_pred-f_mag_real)**2)*mask).sum(dim=1))/n_atoms
    
    @jaxtyped(typechecker=beartype)
    def loss(
        self,
        output_head: Float[torch.Tensor, "N Q"],
        f_mag_pred: Float[torch.Tensor, "N M Q"], 
        batch: Batch, 
        lambda_6: float 
    ) -> Float[torch.Tensor, ""]:
        
        msle_loss  = self._MSLE(output_head, batch)
        ff_penalty = self._ff_penalty(f_mag_pred, batch, lambda_6)
        
        return (msle_loss + ff_penalty).mean()
    