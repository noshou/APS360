from jaxtyping import Float, Int, jaxtyped
from beartype import beartype
from dataclasses import dataclass
import torch
from torch.nn.utils.rnn import pad_sequence

@jaxtyped(typechecker=beartype)
@dataclass(frozen=True)
class Batch:
    vocab: Int[torch.Tensor,   "N M"]    # N molecules, M max atoms (padded)
    iqval: Float[torch.Tensor, "N Q"]    # N molecules, Q q-points
    radii: Float[torch.Tensor, "N M"]    # interatomic distances per atom
    angle: Float[torch.Tensor, "N M 2"]  # theta + phi per atom

    @classmethod
    def from_lists(
        cls,
        vocabs: list[Int[torch.Tensor,   "M"]],
        iqvals: list[Float[torch.Tensor, "Q"]],
        rads:   list[Float[torch.Tensor, "M"]],
        angles: list[Float[torch.Tensor, "M 2"]],
    ) -> "Batch":
        return cls(
            vocab=pad_sequence(vocabs, batch_first=True),
            iqval=pad_sequence(iqvals, batch_first=True),
            radii=pad_sequence(rads,   batch_first=True),
            angle=pad_sequence(angles, batch_first=True),
        )