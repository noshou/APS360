import re
import time
from abc import ABC, abstractmethod
from collections.abc import Iterable

import numpy as np
import torch
import xraydb

from Preprocess import VOCAB
from ScatterNet.batching import Batch

_QCHUNK_BUDGET: int = (
    16_000_000  # target element count for a (Qc, S, S) intermediate
)


def q_chunks(nq: int, s: int):
    """Yield (a, b) q-index ranges so each (Qc, S, S) block stays <= budget.

    Every pairwise-sinc baseline builds a (q, atom, atom) intermediate,
    which is what dominates memory: it grows as S^2, so a single large
    molecule can ask for gigabytes in one allocation. Chunking over q
    bounds that intermediate no matter how big S gets.
    """
    qc = max(1, _QCHUNK_BUDGET // (s * s))
    for a in range(0, nq, qc):
        yield a, min(a + qc, nq)


def autocast_dtype(device: "torch.device | str") -> "torch.dtype | None":
    """BF16 on GPUs with native BF16 tensor cores, else None (stay fp32).

    Requires CUDA compute capability >= 8.0 (Ampere and newer: A100, L4,
    RTX Blackwell, etc). Turing (T4, capability 7.5) has no BF16
    tensor-core path -- its matmuls would run BF16 unaccelerated, likely a
    wash or a net loss rather than a speedup, so T4 is deliberately
    excluded rather than blanket-enabling BF16 on every CUDA device.
    CPU always returns None (autocast on CPU defaults to bf16 too, but
    these baselines' CPU path isn't the target of this optimization).

    Callers that get None back should skip any explicit dtype cast/
    `torch.autocast` entirely and run plain fp32, not pass `dtype=None` to
    `torch.autocast` (which would silently reinterpret as "use the
    default autocast dtype" rather than "do nothing").
    """
    device = torch.device(device)
    if device.type != "cuda":
        return None
    major, _minor = torch.cuda.get_device_capability(device)
    return torch.bfloat16 if major >= 8 else None


def build_fmag_table(
    qgrid: torch.Tensor,  # shape (Q,)
    energy: float,
) -> torch.Tensor:  # shape (V, Q)
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
                    xraydb.f0(resolved, sgrid)
                    + xraydb.f1_chantler(resolved, energy),
                    xraydb.f2_chantler(resolved, energy),
                )
            ).float()
        else:
            elem = re.sub(r"[0-9+\-]+$", "", key)
            f_mag = torch.tensor(
                np.hypot(
                    xraydb.f0(ion, sgrid) + xraydb.f1_chantler(elem, energy),
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
    def __call__(
        self, batch: Batch
    ) -> torch.Tensor:  # shape (N, Q)
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
    ) -> tuple[torch.Tensor, float]:  # (N, Q), seconds
        """Predict I(q) for a batch, measuring wall-clock time per atom.

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
            ``(pred, wall_seconds_per_atom)``. ``pred`` is identical to
            calling this baseline directly, shape ``(N, Q)``.
            ``wall_seconds_per_atom`` is the CUDA-synchronized wall time
            spent inside ``__call__`` divided by the total number of real,
            non-padding atoms across the batch. 0.0 if the batch has no
            real atoms.
        """
        n_atoms = int(batch.padding_mask().sum().item())
        cuda = torch.cuda.is_available()

        if cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        pred = self(batch)
        if cuda:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        wall_seconds_per_atom = elapsed / n_atoms if n_atoms > 0 else 0.0
        return pred, wall_seconds_per_atom
