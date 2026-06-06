import re
import torch
from torch import Tensor, nn
from beartype import beartype
from jaxtyping import Int
from vocab import VOCAB

# strips charge suffix from an ion string, handles sival/cval special cases
_CHARGE_RE = re.compile(r'[0-9]*[+\-]+$')
def _bare(_ion: str) -> str:
    ion = _ion.strip()
    if ion.lower() == 'cval':
        return 'c'
    elif ion.lower() == 'siva':
        return 'si'
    else:
        return _CHARGE_RE.sub('', ion).lower()


class BatchEncoding:
    """Encodes a batch of molecules into flat atom index and batch index tensors.

    Call once per batch at data loading time. The resulting tensors are passed
    directly to IonEmbed.forward and downstream modules.

    Attributes:
        flat_idx:  Int[Tensor, "total_atoms"]  xraydb vocabulary index per atom
        batch_idx: Int[Tensor, "total_atoms"]  molecule index per atom
    """

    @beartype
    def __init__(self, batch: list[list[str]]):
        flat: list[int] = []
        bidx: list[int] = []
        for mol_i, mol in enumerate(batch):
            for ion in mol:
                key = ion.lower().strip()
                if key not in VOCAB:
                    key = _bare(ion)
                if key not in VOCAB:
                    raise LookupError(f"{ion} not found in xraydb vocabulary")
                flat.append(VOCAB[key])
                bidx.append(mol_i)
        self.flat_idx:  Int[Tensor, "total_atoms"] = torch.tensor(flat)
        self.batch_idx: Int[Tensor, "total_atoms"] = torch.tensor(bidx)

class BatchEmbed(nn.Module):
    
    _mbd: nn.Embedding
    _dst: nn.Linear
    
    
    def __init__(self, lambda_1: int, qPoints: int):
    
        self._mbd = nn.Embedding(len(VOCAB), lambda_1)
        self._dst = nn.Linear(lambda_1, qPoints) 

    def forward(self, enc: BatchEncoding):
        vec = self._mbd(enc.flat_idx)