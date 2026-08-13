import os
from typing import cast

import numpy as np
import numpy.typing as npt
import torch
from beartype import beartype

from .formfact import FormFactors
from .spherical_funcs import sphBess, sphHarm


def _detect_gpu() -> (
    tuple[torch.device, torch.dtype, torch.dtype | None, bool] | None
):
    """Return (device, float_dtype, complex_dtype_or_None, is_dml) or None.

    Set env var FORCE_CPU=1 to disable GPU detection entirely.
    DirectML (AMD/Intel via DirectX 12) does not support ComplexFloat, so we
    use a real-split path (is_dml=True, cdtype=None) and compute real/imaginary
    parts separately.
    CUDA/ROCm uses float64/complex128 for full precision.
    """
    if os.environ.get("FORCE_CPU"):
        return None
    try:  # Windows AMD/Intel GPU via DirectX 12
        import torch_directml  # pyright: ignore[reportMissingImports]

        return torch_directml.device(), torch.float32, None, True
    except ImportError:
        pass
    if torch.cuda.is_available():  # CUDA (Kaggle T4) or ROCm (native Linux)
        return torch.device("cuda"), torch.float64, torch.complex128, False
    return None


# GPU quad (device, fdtype, cdtype_or_None, is_dml), or None if CPU / FORCE_CPU
_GPU: tuple[torch.device, torch.dtype, torch.dtype | None, bool] | None = (
    _detect_gpu()
)

# Atoms below this threshold are faster on CPU. Above it, the (K, N) @ (N, Q)
# matmul is large enough to justify the round-trip.
# Empirically: ~1 ms PCIe transfer at N=1000; CPU compute at N=1000 ≈ 7 ms.
_GPU_THRESHOLD: int = 200


class StuhrmannMixin:
    """Mixin providing the Stuhrmann partial-wave decomposition of I(q).

    Intended to be mixed into a class that exposes ``elms``, ``theta``,
    ``phi``, and ``r`` attributes describing atomic positions (e.g.
    ``Molecule``).
    """

    # Provided by the host class (e.g. Molecule) this mixin is combined
    # with; declared here as properties (not plain annotations) so a host
    # class overriding them with its own @property doesn't trip pyright's
    # reportIncompatibleVariableOverride (a property override of a plain
    # variable declaration is flagged; property-over-property is not).
    @property
    def elms(self) -> list[str]: ...

    @property
    def theta(self) -> npt.NDArray[np.float64]: ...

    @property
    def phi(self) -> npt.NDArray[np.float64]: ...

    @property
    def r(self) -> npt.NDArray[np.float64]: ...

    @beartype
    def stuhrmann(
        self,
        ff: FormFactors,
        lMax: int,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.complex128]]:
        """Compute I(q) via the Stuhrmann decomposition.

        I(q) = 4π · Σ_{l=0}^{lMax} Σ_{m=-l}^{l} |B_lm(q)|²
        where B_lm(q) = Σ_i f_i(q) · j_l(q · r_i) · Y*_lm(θ_i, φ_i)

        Only m ≥ 0 modes are returned. For real-valued structures the -ve m
        modes are redundant: B_{l,-m} = (-1)^m · B*_{lm}. The contribution of
        the -m modes to I(q) is accounted for by doubling the m > 0 terms.

        Output index: k = l*(l+1)//2 + m  for l = 0..lMax, m = 0..l.
        Total modes: (lMax+1)*(lMax+2)//2.

        sphHarm and sphBess stay scipy/numpy — they are O(K·N) and O(L·Q·N)
        respectively and not the bottleneck.  The dominant cost is the inner
        matmul Y_l @ W.T which is O(K·N·Q) and runs on _DEVICE (GPU when
        available, CPU otherwise).

        Parameters
        ----------
        ff : FormFactors
            Precomputed form factors; must contain all elements in mol.elms.
        lMax : int
            Maximum l value.

        Returns
        -------
        I_q : ndarray of float64, shape (Q,)
            Scattering intensity at each q value.
        B_lm : ndarray of complex128, shape ((lMax+1)*(lMax+2)//2, Q)
            Partial wave amplitudes for m ≥ 0 only.

        Raises
        ------
        ValueError
            If lMax is negative, or if mol.elms contains ions not in ff.
        """
        if lMax < 0:
            raise ValueError("stuhrmann: lMax must be non-negative")
        missing = set(self.elms) - set(ff.ions) # type: ignore
        if missing:
            raise ValueError(
                f"stuhrmann: ions {missing} in molecule not found"
            )

        N = len(self.elms)

        use_gpu = _GPU is not None and N >= _GPU_THRESHOLD
        if use_gpu:
            try:
                return self._stuhrmann_compute(_GPU, ff, lMax, N)  # type: ignore[arg-type]
            except Exception:
                pass  # DirectML failed — fall through to CPU

        cpu_quad = (
            torch.device("cpu"),
            torch.float64,
            torch.complex128,
            False,
        )
        return self._stuhrmann_compute(cpu_quad, ff, lMax, N)

    def _stuhrmann_compute(
        self,
        quad: tuple,
        ff: FormFactors,
        lMax: int,
        N: int,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.complex128]]:
        device, fdtype, cdtype, is_dml = quad
        if is_dml:
            return self._stuhrmann_compute_dml(device, fdtype, ff, lMax, N)
        return self._stuhrmann_compute_complex(
            device, fdtype, cdtype, ff, lMax, N
        )

    def _stuhrmann_weights(
        self,
        lMax: int,
        fdtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """±m symmetry weights: m=0 → 1, m>0 → 2."""
        return torch.tensor(
            [
                2.0 - (m == 0)
                for deg in range(lMax + 1)
                for m in range(deg + 1)
            ],
            dtype=fdtype,
            device=device,
        )

    def _stuhrmann_compute_dml(
        self,
        device: torch.device,
        fdtype: torch.dtype,
        ff: FormFactors,
        lMax: int,
        N: int,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.complex128]]:
        """DirectML path: B_lm kept split into real/imaginary tensors.

        DirectML does not support ComplexFloat kernels, so we split B_lm
        into real and imaginary parts and perform two real matmuls per
        l-band. The math: (Y_r + i·Y_i) @ W.T = (Y_r @ W.T) + i·(Y_i @ W.T)
        |B_lm|² = B_r² + B_i²  — pure real, no complex ops needed on device.
        """
        N_reduced = (lMax + 1) * (lMax + 2) // 2
        Q = ff.qvals.size

        B_r = torch.zeros((N_reduced, Q), dtype=fdtype, device=device)
        B_i = torch.zeros((N_reduced, Q), dtype=fdtype, device=device)

        # Process atoms in chunks to bound peak GPU memory.
        # Peak per chunk on device (DML real-split, float32):
        #   Y_r + Y_i ~ 2×K×chunk×4B + j~(L+1)×Q×chunk×4B + f~chunk×Q×4B.
        # At chunk=5000, lMax=50, Q=51: ~290 MB — well within 8 GB VRAM.
        CHUNK = 5_000
        for start in range(0, N, CHUNK):
            sl = slice(start, min(start + CHUNK, N))

            Y = sphHarm(
                lMax, self.theta[sl], self.phi[sl]
            )  # (K, chunk), complex128
            j = sphBess(
                ff.qvals, lMax, self.r[sl]
            )  # (lMax+1, Q, chunk), float64
            f = np.stack(
                [ff.ff[ion] for ion in self.elms[sl]]
            )  # (chunk, Q), float64

            j_t = torch.as_tensor(j).to(
                device=device, dtype=fdtype
            )  # (lMax+1, Q, chunk)

            # Split f = f_r + i·f_i (f_r = f0+f1, f_i = f2) into
            # real tensors. Full expansion:
            # (f_r+i·f_i)·j·(Y_r+i·Y_i)
            #   B_r += j·(f_r·Y_r - f_i·Y_i)
            #   B_i += j·(f_r·Y_i + f_i·Y_r)
            Y_conj: npt.NDArray[np.complex128] = cast(
                npt.NDArray[np.complex128], np.conj(Y)
            )
            Y_r = torch.as_tensor(Y_conj.real).to(
                device=device, dtype=fdtype
            )  # (K, chunk)
            Y_i = torch.as_tensor(Y_conj.imag).to(
                device=device, dtype=fdtype
            )  # (K, chunk)
            f_rt = torch.as_tensor(np.real(f)).to(
                device=device, dtype=fdtype
            )  # (chunk, Q)
            f_it = torch.as_tensor(np.imag(f)).to(
                device=device, dtype=fdtype
            )  # (chunk, Q) f2
            for deg in range(lMax + 1):
                Wr = f_rt.T * j_t[deg]  # (Q, chunk)
                Wi = f_it.T * j_t[deg]  # (Q, chunk)
                k0 = deg * (deg + 1) // 2
                Yl_r = Y_r[k0 : k0 + deg + 1]  # (l+1, chunk)
                Yl_i = Y_i[k0 : k0 + deg + 1]  # (l+1, chunk)
                B_r[k0 : k0 + deg + 1] += (
                    Yl_r @ Wr.T - Yl_i @ Wi.T
                )  # (l+1, Q)
                B_i[k0 : k0 + deg + 1] += (
                    Yl_r @ Wi.T + Yl_i @ Wr.T
                )  # (l+1, Q)

        weights = self._stuhrmann_weights(lMax, fdtype, device)
        I_q = (weights[:, None] * (B_r**2 + B_i**2)).sum(dim=0) * (
            4 * np.pi
        )
        B_lm_np = (B_r.cpu().numpy() + 1j * B_i.cpu().numpy()).astype(
            np.complex128
        )
        return I_q.cpu().numpy().astype(np.float64), B_lm_np

    def _stuhrmann_compute_complex(
        self,
        device: torch.device,
        fdtype: torch.dtype,
        cdtype: torch.dtype,
        ff: FormFactors,
        lMax: int,
        N: int,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.complex128]]:
        """CUDA/ROCm/CPU path: B_lm kept fully complex."""
        N_reduced = (lMax + 1) * (lMax + 2) // 2
        Q = ff.qvals.size

        B_lm = torch.zeros((N_reduced, Q), dtype=cdtype, device=device)

        CHUNK = 5_000
        for start in range(0, N, CHUNK):
            sl = slice(start, min(start + CHUNK, N))

            Y = sphHarm(
                lMax, self.theta[sl], self.phi[sl]
            )  # (K, chunk), complex128
            j = sphBess(
                ff.qvals, lMax, self.r[sl]
            )  # (lMax+1, Q, chunk), float64
            f = np.stack(
                [ff.ff[ion] for ion in self.elms[sl]]
            )  # (chunk, Q), float64

            j_t = torch.as_tensor(j).to(
                device=device, dtype=fdtype
            )  # (lMax+1, Q, chunk)

            # Keep f fully complex so f2 flows through naturally.
            f_t = torch.as_tensor(f).to(
                device=device, dtype=cdtype
            )  # (chunk, Q)
            Y_t = torch.as_tensor(np.conj(Y)).to(
                device=device, dtype=cdtype
            )  # (K, chunk)
            for deg in range(lMax + 1):
                W = f_t.T * j_t[deg].to(dtype=cdtype)  # (Q, chunk)
                k0 = deg * (deg + 1) // 2
                Y_l = Y_t[k0 : k0 + deg + 1]  # (l+1, chunk)
                B_lm[k0 : k0 + deg + 1] += Y_l @ W.T  # (l+1, Q)

        weights = self._stuhrmann_weights(lMax, fdtype, device)
        I_q = (weights[:, None] * B_lm.abs() ** 2).sum(dim=0).real * (
            4 * np.pi
        )
        return I_q.cpu().numpy().astype(
            np.float64
        ), B_lm.cpu().numpy().astype(np.complex128)
