from jaxtyping import Bool, Float, Int, jaxtyped
from beartype import beartype
from dataclasses import dataclass
import torch
from torch.nn.utils.rnn import pad_sequence

@jaxtyped(typechecker=beartype)
@dataclass(frozen=True)
class Batch:
    
    """
    Batch of molecules.

    All tensors are zero-padded to the longest molecule in the batch (M atoms).
    Coordinates are spherical, centroid-relative.

    Fields:
        vocab: VOCAB integer indices per atom (N, M)
        iqval: target I(q) intensities (N, Q)
        radii: radial distance r from centroid per atom (N, M)
        angle: [θ, φ] spherical angles from centroid per atom (N, M, 2)
    """
    
    vocab: Int[torch.Tensor,   "N M"]    # N molecules, M max atoms (padded)
    iqval: Float[torch.Tensor, "N Q"]    # N molecules, Q q-points
    radii: Float[torch.Tensor, "N M"]    # interatomic distances per atom
    angle: Float[torch.Tensor, "N M 2"]  # theta + phi per atom

    @jaxtyped(typechecker=beartype)
    def padding_mask(self) -> Bool[torch.Tensor, "N M"]:
        """Boolean mask. True for real atoms, False for padding (vocab == 0)."""
        return self.vocab != 0

    @classmethod
    def from_lists(
        cls,
        vocabs: list[Int[torch.Tensor,   "M"]],
        iqvals: list[Float[torch.Tensor, "Q"]],
        rads:   list[Float[torch.Tensor, "M"]],
        angles: list[Float[torch.Tensor, "M 2"]],
    ) -> "Batch":
        
        """
        Pad variable-length per-molecule tensors and construct a Batch.
        Forces all tensors to be exact same shape.
        
        Args:
            vocabs: list of N vocab index tensors, each shape (M_i,)
            iqvals: list of N I(q) tensors, each shape (Q,)
            rads:   list of N radii tensors, each shape (M_i,)
            angles: list of N angle tensors, each shape (M_i, 2)
        """
        return cls(
            vocab=pad_sequence(vocabs, batch_first=True, padding_value=0),
            iqval=pad_sequence(iqvals, batch_first=True, padding_value=0).to(torch.float32),
            radii=pad_sequence(rads,   batch_first=True, padding_value=0).to(torch.float32),
            angle=pad_sequence(angles, batch_first=True, padding_value=0).to(torch.float32),
        )