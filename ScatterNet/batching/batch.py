from dataclasses import dataclass

import torch
from beartype import beartype
from beartype.typing import List
from jaxtyping import Bool, Float, Int, jaxtyped
from torch.nn.utils.rnn import pad_sequence


@jaxtyped(typechecker=beartype)
@dataclass(frozen=True)
class Batch:
    """Batch of molecules.

    All tensors are zero-padded to the longest molecule in the batch (M atoms).
    Coordinates are centroid-subtracted Cartesian.

    Attributes
    ----------
    vocab : torch.Tensor
        VOCAB integer indices per atom, shape (N, M). Zero-padded; entries
        equal to 0 mark padding atoms.
    iqval : torch.Tensor
        Target I(q) intensities, shape (N, Q).
    coord : torch.Tensor
        Centroid-subtracted Cartesian coordinates per atom, shape (N, M, 3).
    """

    # N molecules, M max atoms (padded)
    vocab: Int[torch.Tensor, "N M"]  # noqa: F722

    # N molecules, Q q-points
    iqval: Float[torch.Tensor, "N Q"]  # noqa: F722

    # Cartesian coordinates
    coord: Float[torch.Tensor, "N M 3"] # noqa: F722

    @jaxtyped(typechecker=beartype)
    def padding_mask(self) -> Bool[torch.Tensor, "N M"]: # noqa: F722
        """Compute the boolean mask distinguishing real atoms from padding.

        Returns
        -------
        torch.Tensor
            Boolean mask of shape (N, M). True for real atoms, False for
            padding (where ``vocab == 0``).
        """
        return self.vocab != 0

    @jaxtyped(typechecker=beartype)
    def to(self, device: torch.device) -> "Batch":
        """Return a copy of this Batch with all tensors moved to ``device``.

        A no-op-cost move when the tensors are already on ``device``
        (``Tensor.to`` returns the same object).
        Used to run a whole batch on GPU without
        mutating this frozen instance.

        Parameters
        ----------
        device : torch.device
            Target device (e.g. ``torch.device("cuda")``).

        Returns
        -------
        Batch
            A new ``Batch`` with ``vocab``/``iqval``/``coord`` on ``device``.
        """
        return Batch(
            vocab=self.vocab.to(device),
            iqval=self.iqval.to(device),
            coord=self.coord.to(device),
        )

    @classmethod
    def from_lists(
        cls,
        vocabs: List[Int[torch.Tensor, "M"]],     # noqa: F821, F722
        iqvals: List[Float[torch.Tensor, "Q"]],   # noqa: F821
        coords: List[Float[torch.Tensor, "M 3"]], # noqa: F722
    ) -> "Batch":
        """Pad variable-length per-molecule tensors and construct a Batch.

        Forces all tensors to the exact same shape. Since vocabs are the
        only tensors guaranteed not to collide with the padding value of 0,
        users should only derive masks from ``vocab``.

        Parameters
        ----------
        vocabs : list of torch.Tensor
            List of N vocab index tensors, each of shape (M_i,).
        iqvals : list of torch.Tensor
            List of N I(q) tensors, each of shape (Q,).
        coords : list of torch.Tensor
            List of N x, y, z coordinate tensors, each of shape (M_i, 3).

        Returns
        -------
        Batch
            A new ``Batch`` with all tensors zero-padded to the longest
            molecule (M atoms) in the input lists.
        """
        return cls(
            vocab=pad_sequence(vocabs, batch_first=True, padding_value=0),
            iqval=pad_sequence(iqvals, batch_first=True, padding_value=0).to(
                torch.float32
            ),
            coord=pad_sequence(coords, batch_first=True, padding_value=0).to(
                torch.float32
            ),
        )
