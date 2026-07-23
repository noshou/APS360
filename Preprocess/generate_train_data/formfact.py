import re
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from beartype import beartype
from xraydb import chantler_energies, f0, f1_chantler, f2_chantler

# Strip charge notation to get bare element symbol (e.g. Mn2+ → Mn, Na+ → Na)
# Used for xraydb functions that don't accept ionic notation
# (f1/f2, chantler_energies)
_CHARGE_RE = re.compile(r"[0-9]*[+\-]+$")


def _bare(ion: str) -> str:
    """Strip trailing charge notation from an ion symbol.

    Parameters
    ----------
    ion : str
        Ion symbol, possibly with charge notation.

    Returns
    -------
    str
        Bare element symbol with charge notation removed.
    """
    return _CHARGE_RE.sub("", ion)


@dataclass(frozen=True)
class FormFactors:
    """
    Container for precomputed complex atomic form factors over a q grid.

    The form factor convention is:
        f(q, E) = f0(s) + f'(E) + i·f''(E)

    where s = q/(4π) is the Cromer-Mann variable (sin(θ)/λ, Å⁻¹), f0 is the
    Thomson scattering term (Waasmaier/Cromer-Mann equals Z at s=0), and f'/f''
    are the real/imaginary Chantler anomalous corrections (xraydb f1_chantler /
    f2_chantler, which return the anomalous part only, not Z+f').

    Attributes
    ----------
    ions : list[str]
        Ion symbols in the order they were computed.
    qvals : ndarray of float64, shape (Q,)
        Momentum transfer grid in Å⁻¹.
    ff : dict[str, ndarray of complex128]
        Maps each ion symbol to its complex form factor array, shape (Q,).
    energy : float
        X-ray energy in eV used to compute the anomalous corrections.
    """

    ions: list[str]
    qvals: npt.NDArray[np.float64]
    ff: dict[str, npt.NDArray[np.complex128]]
    ionIdx: dict[str, int] = field(init=False)
    energy: float

    # Form factor convention (matches xraydb):
    #   f(q, E) = f0(s) + f'(E) + i·f''(E)
    # where s = q/(4π), f0 is the Cromer-Mann/Waasmaier T
    # homson term (→Z at s=0), and f'/f'' are the Chantler anomalous
    # corrections (f1_chantler/f2_chantler). xraydb.f1_chantler returns
    # f'(E) only so no subtraction of f0(0).

    def __post_init__(self) -> None:
        """Make the qvals and ff arrays read-only and build the ion index map.

        Returns
        -------
        None
        """
        self.qvals.flags.writeable = False
        for arr in self.ff.values():
            arr.flags.writeable = False
        object.__setattr__(
            self, "ionIdx", {ion: i for i, ion in enumerate(self.ions)}
        )

    @staticmethod
    def _validate_energy(ion: str, energy: float) -> None:
        """Validate that energy is within the
        Chantler tabulation range for an ion.

        Parameters
        ----------
        ion : str
            Ion symbol.
        energy : float
            X-ray energy in eV.

        Raises
        ------
        ValueError
            If no Chantler tabulation exists for the ion, or the energy is
            outside the tabulated range.
        """

        elem = _bare(ion)
        verr = f"No Chantler tabulation found for '{elem}' (from ion '{ion}')."
        try:
            valid = chantler_energies(elem, emin=0, emax=1e9)
        except Exception as exc:
            raise ValueError(verr) from exc
        if len(valid) == 0:
            raise ValueError(verr)
        emin, emax = min(valid), max(valid)
        if energy < emin or energy > emax:
            raise ValueError(
                f"Energy {energy} eV is outside the valid Chantler range "
                f"[{emin}, {emax}] eV for '{elem}'."
            )

    @staticmethod
    def _calc_formfact(
        ion: str,
        qvals: npt.NDArray[np.float64],
        energy: float,
        skip_anomalous: bool = False,
    ) -> npt.NDArray[np.complex128]:
        """Compute the complex atomic form factor
        f(q, E) = f0(s) + f'(E) + i·f''(E).

        Parameters
        ----------
        ion : str
            Ion symbol (e.g. ``'Fe'``, ``'Au3+'``).
        qvals : ndarray of float64, shape (N,)
            Momentum transfer values in Å⁻¹.
        energy : float
            X-ray energy in eV.
        skip_anomalous : bool, optional
            If ``True``, return just ``f0(s)`` (no anomalous corrections).

        Returns
        -------
        ndarray of complex128, shape (N,)
        """
        # f0 (Waasmaier/Cromer-Mann) accepts ionic notation;
        # fall back to bare element.
        # f'/f'' (Chantler) require bare element symbols.
        elem = _bare(ion)
        s = qvals / (4.0 * np.pi)
        try:
            f0_ = np.asarray(f0(ion, s), dtype=np.float64)
        except Exception:
            f0_ = np.asarray(f0(elem, s), dtype=np.float64)
        if skip_anomalous:
            return np.asarray(f0_, dtype=np.complex128)
        f1_ = float(np.asarray(f1_chantler(elem, energy)).squeeze())
        f2_ = float(np.asarray(f2_chantler(elem, energy)).squeeze())
        return np.asarray(f0_ + f1_ + 1j * f2_, dtype=np.complex128)

    @classmethod
    @beartype
    def fromIons(
        cls,
        ions: list[str],
        energy: float,
        qMin: int | float,
        qMax: int | float,
        step: float,
        log_path: str | None = None,
    ) -> "FormFactors":
        """
        Computes complex form factors for a list of ions over a q grid.

        Each ion is classified into one of three tiers:

        * **Dummy** - ``f0`` raises for the bare element (e.g. ``X1+``, ``M``).
        These sites carry no electron density and are skipped entirely.
        * **f0-only** - ``f0`` works but no Chantler tabulation exists for the
        element or the energy is outside the tabulated range (e.g. ``Pu``,
        ``Am``).  Only the Thomson scattering term ``f0(s)`` is returned as
        complex128 (no anomalous corrections).
        * **Full** - both ``f0`` and the Chantler f1/f2 corrections are
        available.  The complete form factor
        ``f(q,E) = f0(s) + f1(E) − f0(0) + i·f2(E)`` is computed.

        If ``log_path`` is given, skipped/fallback ions are written there, one
        per line, e.g.::

            DUMMY   X1+
            DUMMY   M
            F0-ONLY Pu
            F0-ONLY Am3+

        Parameters
        ----------
        ions : list[str]
            Ion symbols (e.g. ``['Fe', 'O']``).
        energy : float
            X-ray energy in eV.
        qMin : int or float
            Minimum momentum transfer in Å⁻¹.
        qMax : int or float
            Maximum momentum transfer in Å⁻¹.
        step : float
            Step size of the q grid in Å⁻¹.
        log_path : str or None, optional
            Path to a text file where dummy/f0-only ions are logged.  If
            ``None`` (default), no log file is written.

        Returns
        -------
        FormFactors

        Raises
        ------
        ValueError
            If bounds are invalid.
        """

        if (
            qMin < 0
            or qMax < 0
            or qMin >= qMax
            or step <= 0
            or (qMax - qMin) < step
        ):
            verr1 = "Invalid bounds or step size."
            verr2 = f" Ensure qMin={qMin} >= 0 and qMax={qMax} > 0, "
            verr3 = f"qMin < qMax, and step={step} "
            verr4 = "is smaller than or equal to the total range"
            verr  = verr1 + verr2 + verr3 + verr4
            raise ValueError(verr)

        num_elements = int(round((qMax - qMin) / step)) + 1
        qvals = np.linspace(qMin, qMax, num_elements, dtype=np.float64)

        log_lines: list[str] = []
        ff_dict: dict[str, npt.NDArray[np.complex128]] = {}
        for ion in dict.fromkeys(ions):
            elem = _bare(ion)

            # Tier 1 - dummy site: f0 raises for the bare element
            try:
                f0(elem, np.array([0.0]))
            except Exception:
                log_lines.append(f"DUMMY   {ion}")
                continue

            # Tier 2 - f0-only: Chantler tabulation missing/energy out of range
            try:
                cls._validate_energy(ion, energy)
            except ValueError:
                log_lines.append(f"F0-ONLY {ion}")
                ff_dict[ion] = cls._calc_formfact(
                    ion, qvals, energy, skip_anomalous=True
                )
                continue

            # Tier 3 - full form factor
            ff_dict[ion] = cls._calc_formfact(
                ion, qvals, energy, skip_anomalous=False
            )

        if log_path is not None and log_lines:
            with open(log_path, "w") as _fh:
                _fh.write("\n".join(log_lines) + "\n")

        if not ff_dict:
            verr1 = f"All {len(ions)} input ions were"
            verr2 = " classified as dummy sites (no electron density). "
            verr3 = "Check that ion symbols are valid element names."
            verr  = verr1 + verr2 + verr3
            raise ValueError(verr)
        return cls(
            ions=list(ff_dict.keys()), qvals=qvals, ff=ff_dict, energy=energy
        )
