from jaxtyping import Bool, Float, Int, jaxtyped
from beartype import beartype
from beartype.typing import List
from dataclasses import dataclass
import torch
from torch.nn.utils.rnn import pad_sequence

@jaxtyped(typechecker=beartype)
@dataclass(frozen=True)
class Batch:
    
    """
    Batch of molecules.

    All tensors are zero-padded to the longest molecule in the batch (M atoms).
    Coordinates are centroid-subtracted Cartesian.

    Fields:
        vocab: VOCAB integer indices per atom (N, M)
        iqval: target I(q) intensities (N, Q)
        coord: centroid-subtracted Cartesian coordinates per atom (N, M, 3)
    """
    
    vocab: Int[torch.Tensor,   "N M"]    # N molecules, M max atoms (padded)
    iqval: Float[torch.Tensor, "N Q"]    # N molecules, Q q-points
    coord: Float[torch.Tensor, "N M 3"]  # Cartesian coordinates
    @jaxtyped(typechecker=beartype)
    def padding_mask(self) -> Bool[torch.Tensor, "N M"]:
        """Boolean mask. True for real atoms, False for padding (vocab == 0)."""
        return self.vocab != 0

    @classmethod
    def from_lists(
        cls,
        vocabs: List[Int[torch.Tensor,   "M"]],
        iqvals: List[Float[torch.Tensor, "Q"]],
        coords: List[Float[torch.Tensor, "M 3"]]
    ) -> "Batch":
        
        """
        Pad variable-length per-molecule tensors and construct a Batch.
        Forces all tensors to be exact same shape. Since vocabs are the 
        only ones guaranteed not to have collisions with padding value of 0, 
        users can only mask off of vocab.
        
        Args:
            vocabs: list of N vocab index tensors, each shape (M_i,)
            iqvals: list of N I(q) tensors, each shape (Q,)
            coords: list of N x,y,z coordinates, each shape (M, 3)
        """
        return cls(
            vocab=pad_sequence(vocabs, batch_first=True, padding_value=0),
            iqval=pad_sequence(iqvals, batch_first=True, padding_value=0).to(torch.float32),
            coord=pad_sequence(coords, batch_first=True, padding_value=0).to(torch.float32)
        )