from dataclasses import dataclass
from typing import List

import torch
from torch.nn.utils.rnn import pad_sequence


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
    vocab: torch.Tensor  # shape (N, M)

    # N molecules, Q q-points
    iqval: torch.Tensor  # shape (N, Q)

    # Cartesian coordinates
    coord: torch.Tensor  # shape (N, M, 3)

    def padding_mask(self) -> torch.Tensor:  # shape (N, M), bool
        """Compute the boolean mask distinguishing real atoms from padding.

        Returns
        -------
        torch.Tensor
            Boolean mask of shape (N, M). True for real atoms, False for
            padding (where ``vocab == 0``).
        """
        return self.vocab != 0

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
        vocabs: List[torch.Tensor],  # each shape (M_i,)
        iqvals: List[torch.Tensor],  # each shape (Q,)
        coords: List[torch.Tensor],  # each shape (M_i, 3)
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
