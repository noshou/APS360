import xraydb
import re
import numpy as np
import torch

from abc             import ABC, abstractmethod
from typing          import Iterable
from jaxtyping       import Float, jaxtyped
from beartype        import beartype
from ScatterNet.batching import Batch
from Preprocess          import VOCAB


@jaxtyped(typechecker=beartype)
def build_fmag_table(
    qgrid:  Float[torch.Tensor, "Q"],
    energy: float,
) -> Float[torch.Tensor, "V Q"]:
    """
    Build atomic form factor magnitude table for all VOCAB ions.
    Index 0 is the padding sentinel (all zeros).
    """
    table = torch.zeros(len(VOCAB) + 1, len(qgrid))
    sgrid = (qgrid / (4 * torch.pi)).numpy()

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
                    xraydb.f2_chantler(resolved, energy),
                )
            ).float()
        else:
            elem = re.sub(r'[0-9+\-]+$', '', key)
            f_mag = torch.tensor(
                np.hypot(
                    xraydb.f0(ion, sgrid) +
                    xraydb.f1_chantler(elem, energy),
                    xraydb.f2_chantler(elem, energy),
                )
            ).float()
        table[idx + 1] = f_mag

    return table


class Baseline(ABC):

    def fit(self, loader: Iterable[Batch]) -> "Baseline":
        return self

    @abstractmethod
    @jaxtyped(typechecker=beartype)
    def __call__(self, batch: Batch) -> Float[torch.Tensor, "N Q"]:
        ...
