import re
import numpy as np
from beartype import beartype
import numpy.typing as npt
from xraydb import f0, f1_chantler, f2_chantler, chantler_energies
from dataclasses import dataclass

# Strip charge notation to get bare element symbol (e.g. Mn2+ → Mn, Na+ → Na)
# Used for xraydb functions that don't accept ionic notation (f1/f2, chantler_energies)
_CHARGE_RE = re.compile(r'[0-9]*[+\-]+$')
def _bare(ion: str) -> str:
    return _CHARGE_RE.sub('', ion)

@dataclass(frozen=True)
class FormFactors:

    """
    Container for precomputed complex atomic form factors over a q grid.

    The form factor is defined as:
        f(q, E) = f0(s) + f1(E) - f0(0) + i·f2(E)

    where s = q / (4π) is the Cromer-Mann scattering variable (sin(θ)/λ in Å⁻¹),
    q = 4π sin(θ)/λ is the momentum transfer (Å⁻¹), and f1/f2 are the real and
    imaginary anomalous dispersion corrections (Chantler tabulation).

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

    ions:   list[str]
    qvals:  npt.NDArray[np.float64]
    ff:     dict[str, npt.NDArray[np.complex128]]
    energy: float

    def __post_init__(self) -> None:
        self.qvals.flags.writeable = False
        for arr in self.ff.values():
            arr.flags.writeable = False

    @staticmethod
    def _validate_energy(ion: str, energy: float) -> None:

        """Validate that energy is within the Chantler tabulation range for an ion.

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
        try:
            valid = chantler_energies(elem, emin=0, emax=1e9)
        except Exception:
            raise ValueError(f"No Chantler tabulation found for '{elem}' (from ion '{ion}').")
        if len(valid) == 0:
            raise ValueError(f"No Chantler tabulation found for '{elem}' (from ion '{ion}').")
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

        """Compute the complex atomic form factor f(q, E) for a given ion.

        The form factor is defined as:
            f(q, E) = f0(s) + f1(E) - f0(0) + i·f2(E)

        where s = q / (4π) converts from momentum transfer (Å⁻¹) to the
        Cromer-Mann scattering variable sin(θ)/λ expected by xraydb.f0.
        f1/f2 are the Chantler anomalous dispersion corrections at energy E.

        Parameters
        ----------
        ion : str
            Ion symbol (e.g. ``'Fe'``, ``'Au3+'``).
        qvals : ndarray of float64, shape (N,)
            Momentum transfer values in Å⁻¹.
        energy : float
            X-ray energy in eV.
        skip_anomalous : bool, optional
            If ``True``, skip the Chantler f1/f2 corrections and return just
            ``f0(s)`` as complex128.  Default is ``False``.

        Returns
        -------
        ndarray of complex128, shape (N,)
            Complex form factor evaluated at each q in ``qvals``.
        """

        # f1/f2 (Chantler) require bare element symbols.
        # f0 (Waasmaier/Cromer-Mann) accepts ionic notation when available and gives
        # physically different values — use ionic when possible, fall back to bare.
        # xraydb.f0 uses the Cromer-Mann variable s = sin(θ)/λ = q / (4π)
        elem = _bare(ion)
        s    = qvals / (4.0 * np.pi)
        try:
            f0_     = np.asarray(f0(ion, s),   dtype=np.float64)
            f0_zero = float(np.asarray(f0(ion, 0.0)).ravel()[0])
        except Exception:
            # ion not in Waasmaier table — fall back to neutral element
            f0_     = np.asarray(f0(elem, s),   dtype=np.float64)
            f0_zero = float(np.asarray(f0(elem, 0.0)).ravel()[0])
        if skip_anomalous:
            return np.asarray(f0_, dtype=np.complex128)
        f1_ = float(np.asarray(f1_chantler(elem, energy)).squeeze())
        f2_ = float(np.asarray(f2_chantler(elem, energy)).squeeze())
        return np.asarray((f0_ + f1_ - f0_zero) + 1j * f2_, dtype=np.complex128)

    @classmethod
    @beartype
    def fromIons(
        cls,
        ions:     list[str],
        energy:   float,
        qMin:     int | float,
        qMax:     int | float,
        step:     float,
        log_path: str | None = None,
    ) -> 'FormFactors':

        """
        Computes complex form factors for a list of ions over a q grid.

        Each ion is classified into one of three tiers:

        * **Dummy** — ``f0`` raises for the bare element (e.g. ``X1+``, ``M``).
          These sites carry no electron density and are skipped entirely.
        * **f0-only** — ``f0`` works but no Chantler tabulation exists for the
          element or the energy is outside the tabulated range (e.g. ``Pu``,
          ``Am``).  Only the Thomson scattering term ``f0(s)`` is returned as
          complex128 (no anomalous corrections).
        * **Full** — both ``f0`` and the Chantler f1/f2 corrections are
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

        if qMin < 0 or qMax < 0 or qMin >= qMax or step <= 0 or (qMax - qMin) < step:
            raise ValueError(
                f"Invalid bounds or step size. Ensure qMin={qMin} and qMax={qMax} are positive, "
                f"qMin < qMax, and step={step} is smaller than or equal to the total range."
            )

        num_elements = int(round((qMax - qMin) / step)) + 1
        qvals = np.linspace(qMin, qMax, num_elements, dtype=np.float64)

        log_lines: list[str] = []
        ff_dict: dict[str, npt.NDArray[np.complex128]] = {}
        for ion in dict.fromkeys(ions):
            elem = _bare(ion)

            # Tier 1 — dummy site: f0 raises for the bare element
            try:
                f0(elem, np.array([0.0]))
            except Exception:
                log_lines.append(f"DUMMY   {ion}")
                continue

            # Tier 2 — f0-only: Chantler tabulation missing or energy out of range
            try:
                cls._validate_energy(ion, energy)
            except ValueError:
                log_lines.append(f"F0-ONLY {ion}")
                ff_dict[ion] = cls._calc_formfact(ion, qvals, energy, skip_anomalous=True)
                continue

            # Tier 3 — full form factor
            ff_dict[ion] = cls._calc_formfact(ion, qvals, energy, skip_anomalous=False)

        if log_path is not None and log_lines:
            with open(log_path, 'w') as _fh:
                _fh.write('\n'.join(log_lines) + '\n')

        return cls(ions=list(ff_dict.keys()), qvals=qvals, ff=ff_dict, energy=energy)
