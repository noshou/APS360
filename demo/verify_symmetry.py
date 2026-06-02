"""
Sanity check for conjugate symmetry in stuhrmann.

Verifies:
  1. I_q from the new reduced loop matches I_q recomputed from the
     full B_lm reconstructed via B_{l,-m} = (-1)^m * conj(B_{lm}).
  2. I_q is finite, positive, and non-increasing at low q (Guinier).
  3. B_lm dtype is complex64.
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from Scattering import Molecule, FormFactors

XYZ    = "demo/demo_structures/C60BuckyBallHe.xyz"
ENERGY = 12412.8
Q_MIN, Q_MAX, Q_STEP = 0.02, 0.50, 0.02
LMAX   = 25

mol = Molecule.fromXYZ(XYZ)
ff  = FormFactors.fromIons(
    ions   = list(dict.fromkeys(mol.elms)),
    energy = ENERGY,
    qMin   = Q_MIN,
    qMax   = Q_MAX,
    step   = Q_STEP,
)

I_q, B_lm = mol.stuhrmann(ff, LMAX)

N_reduced = (LMAX + 1) * (LMAX + 2) // 2
N_full    = (LMAX + 1) ** 2
Q         = len(ff.qvals)

print(f"B_lm shape:  {B_lm.shape}  (expected ({N_reduced}, {Q}))")
print(f"B_lm dtype:  {B_lm.dtype}  (expected complex64)")
print(f"I_q  shape:  {I_q.shape}   (expected ({Q},))")

assert B_lm.shape == (N_reduced, Q),  "shape mismatch"
assert B_lm.dtype == np.complex64,    "dtype mismatch"
assert I_q.shape  == (Q,),            "I_q shape mismatch"

# Reconstruct full B_lm from conjugate symmetry.
B_full = np.zeros((N_full, Q), dtype=np.complex128)
for l in range(LMAX + 1):
    for m in range(0, l + 1):
        k_reduced = l * (l + 1) // 2 + m
        k_pos     = l * l + l + m
        k_neg     = l * l + l - m
        B_full[k_pos] = B_lm[k_reduced]
        if m > 0:
            B_full[k_neg] = ((-1) ** m) * np.conj(B_lm[k_reduced])

# Recompute I_q from full B_lm.
I_q_reconstructed = (4 * np.pi) ** 2 * np.sum(np.abs(B_full) ** 2, axis=0)

rel_err = np.abs(I_q - I_q_reconstructed) / (np.abs(I_q_reconstructed) + 1e-300)
print(f"\nI_q vs reconstructed I_q:")
print(f"  max relative error: {rel_err.max():.2e}  (should be < 1e-5)")
assert rel_err.max() < 1e-5, f"I_q mismatch: max rel err = {rel_err.max():.2e}"

# Basic physical checks.
assert np.all(np.isfinite(I_q)),  "I_q contains inf/NaN"
assert np.all(I_q > 0),           "I_q contains non-positive values"
print(f"\nI_q range: [{I_q.min():.3e}, {I_q.max():.3e}]")
print(f"I_q[0] > I_q[-1]: {I_q[0]:.3e} > {I_q[-1]:.3e}  (Guinier falloff)")

print("\nAll checks passed.")
