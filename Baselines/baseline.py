import xraydb
import re
import time
import numpy as np
import torch

from abc                 import ABC, abstractmethod
from collections.abc     import Iterable
from jaxtyping           import Float, jaxtyped
from beartype            import beartype
from ScatterNet.batching import Batch
from Preprocess          import VOCAB

@jaxtyped(typechecker=beartype)
def build_fmag_table(
    qgrid:  Float[torch.Tensor, "Q"],
    energy: float,
) -> Float[torch.Tensor, "V Q"]:
    """Build atomic form factor magnitude table for all VOCAB ions.

    Index 0 is the padding sentinel (all zeros). For transuranic ions the
    tabulated f0 form factor is used directly; for special-case and regular
    ions f0 is combined with Chantler anomalous corrections f1/f2 via
    ``hypot`` to give the energy-dependent form factor magnitude.

    Parameters
    ----------
    qgrid : torch.Tensor
        q-point grid in inverse angstroms, shape ``(Q,)``.
    energy : float
        X-ray energy in eV, used for anomalous f1/f2 corrections.

    Returns
    -------
    torch.Tensor
        Form factor magnitude table of shape ``(V, Q)`` where ``V`` is
        ``len(VOCAB) + 1``. Row 0 is all zeros (padding sentinel).
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
    """Abstract base class for all baseline scattering-curve predictors.

    Subclasses implement ``__call__`` to predict I(q) for a batch of
    molecules. Baselines that need training statistics (e.g. per-element
    or per-atom-count means) override ``fit`` to accumulate those
    statistics from a training loader; baselines that are purely
    analytical (e.g. form-factor or Guinier based) can rely on the
    no-op default. ``timed_call`` wraps ``__call__`` with a CPU-time-per-atom
    measurement, so every baseline is directly comparable on cost without
    each subclass having to implement its own timing.
    """

    def fit(self, loader: Iterable[Batch]) -> "Baseline":
        """Fit the baseline to training data.

        Default no-op implementation for baselines that require no
        training (purely analytical baselines). Subclasses that need
        training statistics should override this method.

        Parameters
        ----------
        loader : Iterable[Batch]
            Iterable of training batches.

        Returns
        -------
        Baseline
            This instance, to allow chaining (e.g. ``baseline.fit(loader)``).
        """
        return self

    @abstractmethod
    @jaxtyped(typechecker=beartype)
    def __call__(self, batch: Batch) -> Float[torch.Tensor, "N Q"]:
        """Predict I(q) for every molecule in a batch.

        Parameters
        ----------
        batch : Batch
            Batch of molecules to predict scattering curves for.

        Returns
        -------
        torch.Tensor
            Predicted I(q) curves of shape ``(N, Q)``, one row per
            molecule in the batch.
        """
        ...

    def timed_call(
        self,
        batch: Batch,
    ) -> tuple[Float[torch.Tensor, "N Q"], float]:
        """Predict I(q) for a batch while measuring wall-clock time per atom.

        Wraps ``__call__`` so every baseline - existing and future, with no
        changes to the subclass itself - can be compared on cost per atom on
        equal footing.

        The clock is **wall-clock** (``time.perf_counter``), and when CUDA is
        available the call is bracketed with ``torch.cuda.synchronize()`` so
        asynchronous GPU work is fully counted. This replaces the old CPU-only
        ``time.process_time`` clock: once the heavy pairwise-sinc work runs on
        the GPU, process_time measured almost nothing (it does not see device
        compute), and a baseline's real cost now spans both host (type
        sampling, sklearn calls) and device (the O(m^2) sinc sums). Only an
        end-to-end, synchronized wall clock captures that combined cost.

        Because it is wall-clock, the per-atom figure depends on the machine,
        the device, and concurrent load, so - as before - only compare
        ``us_per_atom`` between baselines measured *in the same run on the same
        hardware*, never across machines or runs.

        Parameters
        ----------
        batch : Batch
            Batch of molecules to predict scattering curves for.

        Returns
        -------
        tuple of (torch.Tensor, float)
            ``(pred, wall_seconds_per_atom)``. ``pred`` is identical to calling
            this baseline directly, shape ``(N, Q)``. ``wall_seconds_per_atom``
            is the CUDA-synchronized wall time spent inside ``__call__`` divided
            by the total number of real, non-padding atoms across the batch.
            0.0 if the batch has no real atoms.
        """
        n_atoms = int(batch.padding_mask().sum().item())
        cuda    = torch.cuda.is_available()

        if cuda:
            torch.cuda.synchronize()
        start   = time.perf_counter()
        pred    = self(batch)
        if cuda:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        wall_seconds_per_atom = elapsed / n_atoms if n_atoms > 0 else 0.0
        return pred, wall_seconds_per_atom
