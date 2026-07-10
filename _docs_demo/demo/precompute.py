"""
Precompute I_q and B_lm for all demo structures and save to precomputed/.
Run once from the APS360 directory:
    python _docs_demo/demo/precompute.py
"""
import sys
import time
import numpy as np

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from _docs_demo.demo.global_vals import *
from Scattering  import Molecule, FormFactors

OUTPUT_DIR.mkdir(exist_ok=True)

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
