import h5py
import torch
from beartype.typing import List
from torch.utils.data import Dataset

from Preprocess import Encoding

from .batch import Batch


class BatchSet(Dataset):
    """A PyTorch Dataset where each item is a pre-built batch of molecules.

    Molecules vary enormously in atom count (2-atom QM9 molecules up to
    thousands-of-atom MOFs). A standard DataLoader with batch_size=N would
    group arbitrary molecules together, forcing heavy zero-padding that wastes
    GPU compute. Instead, Batcher pre-groups molecules by atom-count range into
    size-homogeneous batches, and BatchSet exposes those batches as a Dataset.
    Each __getitem__ returns a fully-padded Batch ready for the model.

    Because each item is already a batch of N molecules, use DataLoader with
    batch_size=1 and a passthrough collate function:

        from torch.utils.data import DataLoader

        train, val, test = Batcher(hdf5_db, enc, buckets).get_sets()

        train_loader = DataLoader(
            train,
            batch_size=1,
            shuffle=True,
            collate_fn=lambda x: x[0], # unwrap outer list added by DataLoader
        )

        for batch in train_loader:        # batch is a Batch, shape (N, ...)
            loss = model(batch)

    When averaging metrics across batches, weight by molecule count to
    avoid bias from variable batch sizes:

        total_loss, total_mols = 0, 0
        for batch in val_loader:
            n = batch.iqval.shape[0]
            total_loss += criterion(pred, batch.iqval).item() * n
            total_mols += n
        val_loss = total_loss / total_mols
    """

    def __init__(
        self,
        hdf5_db: str,
        enc: Encoding,
        batches: list,
    ):
        """Initialize the BatchSet with pre-built batches from a Batcher.

        Parameters
        ----------
        hdf5_db : str
            Path to raw HDF5 data.
        enc : Encoding
            Encoding instance used to fetch VOCAB_idx lazily, one batch at
            a time, in ``__getitem__`` -- so only the batch currently being
            materialized is ever resident in memory, not the whole dataset's
            encodings.
        batches : list
            Pre-built batches from ``Batcher``, each a list of
            (grp, stem, atoms) rows.
        """
        self.__batches = batches
        self.__enc = enc
        self.__db_path = hdf5_db

    @property
    def batches(self) -> list:
        """The list of batch buckets."""
        return self.__batches

    @property
    def enc(self) -> Encoding:
        """The vocabulary encoder."""
        return self.__enc

    @property
    def db_path(self) -> str:
        """Path to the source HDF5 database."""
        return self.__db_path

    def __len__(self) -> int:
        """Return the number of pre-built batches in this dataset.

        Returns
        -------
        int
            Number of batches.
        """
        return len(self.__batches)

    def __getitem__(self, i: int) -> Batch:
        """Load and return sub-batch ``i``
        as a typed, shape-validated ``Batch``.

        Parameters
        ----------
        i : int
            Index of the batch to load.

        Returns
        -------
        Batch
            The loaded batch, with vocab, I(q) intensities, and coordinates
            read from the HDF5 file and padded via ``Batch.from_lists``.
        """
        rows = self.__batches[i]
        vocab_by_key = self.__enc.get_vocab_for_keys(
            [(grp, stem) for grp, stem, _ in rows]
        )

        vocab: List = []
        iqval: List = []
        coord: List = []

        with h5py.File(self.__db_path, "r") as db:
            for grp, stem, _ in rows:
                mol = db[grp][stem]  # type: ignore
                vocab.append(torch.tensor(vocab_by_key[(grp, stem)]))
                iqval.append(torch.tensor(mol["I_q"][:]))  # type: ignore
                coord.append(torch.tensor(mol["coords"][:]))  # type: ignore

        return Batch.from_lists(vocab, iqval, coord)
