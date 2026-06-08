import torch
import torch.nn.functional as F
from torch import nn
from jaxtyping import Float, jaxtyped
from beartype import beartype
from beartype.typing import Tuple
from .batch import Batch
from Preprocess import VOCAB

class BatchEmbed(nn.Module):

    _mbd:  nn.Embedding  # atom identity in learned space
    _f0f1: nn.Linear     # approximate real form factor at each q
    _f2:   nn.Linear     # approximate imaginary form factor at each q
    _trn:  nn.Bilinear   # estimated truncation radius per atom

    def __init__(self, lambda_1: int, qPoints: int) -> None:
        super().__init__()
        self._mbd  = nn.Embedding(len(VOCAB), lambda_1)
        self._f0f1 = nn.Linear(lambda_1, qPoints)
        self._f2   = nn.Linear(lambda_1, qPoints)
        self._trn  = nn.Bilinear(lambda_1, qPoints, 1)

    @jaxtyped(typechecker=beartype)
    def forward(self, batch: Batch) -> Tuple[
        Float[torch.Tensor, "N M H"],  # atom embeddings
        Float[torch.Tensor, "N M Q"],  # real form factor approximation
        Float[torch.Tensor, "N M 1"],  # truncation radius per atom
    ]:
        
        mbd_vecs = self._mbd(batch.vocab)                     # (N, M, H)
      
        f_re  = F.softplus(self._f0f1(mbd_vecs)) + 1e-7       # (N, M, Q)
        f_im  = self._f2(mbd_vecs)                            # (N, M, Q)
        ff_sq = f_re ** 2 + f_im ** 2                         # |f(q)|², (N, M, Q)

        trunc = F.softplus(self._trn(mbd_vecs, ff_sq)) + 1e-7 # (N, M, 1)

        return mbd_vecs, f_re, trunc
