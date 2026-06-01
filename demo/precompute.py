"""
Precompute I_q and B_lm for all demo structures and save to precomputed/.
Run once from the APS360 directory:
    python demo/precompute.py
"""
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from Scattering import Molecule, FormFactors

STRUCT_DIR = Path(__file__).parent / "demo_structures"
OUTPUT_DIR = Path(__file__).parent / "precomputed"
OUTPUT_DIR.mkdir(exist_ok=True)

# q grid in Å⁻¹ (momentum transfer, physics convention: q = 4π sin(θ)/λ)
ENERGY = 12412.8   # eV (1 Å wavelength, standard SAXS)
Q_MIN  = 0.02
Q_MAX  = 0.50
Q_STEP = 0.02

lMax = 25

for xyz in sorted(STRUCT_DIR.glob("*.xyz")):
    print(f"Processing {xyz.stem} ...", end=" ", flush=True)
    t0  = time.time()
    mol = Molecule.fromXYZ(str(xyz))
    ff  = FormFactors.fromIons(
        ions   = list(dict.fromkeys(mol.elms)),
        energy = ENERGY,
        qMin   = Q_MIN,
        qMax   = Q_MAX,
        step   = Q_STEP,
    )
    I_q, B_lm = mol.stuhrmann(ff, lMax)
    np.savez(
        OUTPUT_DIR / f"{xyz.stem}.npz",
        I_q     = I_q,
        B_lm_re = np.real(B_lm),
        B_lm_im = np.imag(B_lm),
        qvals   = ff.qvals,
    )
    print(f"done ({time.time() - t0:.1f}s)")

print("All done.")
