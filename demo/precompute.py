"""
Precompute I_q and B_lm for all demo structures and save to precomputed/.
Run once from the APS630 directory:
    python demo/precompute.py
"""
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from Molecule import Molecule
from Stuhrmann import stuhrmann

STRUCT_DIR  = Path(__file__).parent / "demo_structures"
OUTPUT_DIR  = Path(__file__).parent / "precomputed"
OUTPUT_DIR.mkdir(exist_ok=True)

lMax = 25

for xyz in sorted(STRUCT_DIR.glob("*.xyz")):
    print(f"Processing {xyz.stem} ...", end=" ", flush=True)
    t0  = time.time()
    mol = Molecule.fromXYZ(str(xyz))
    I_q, B_lm = stuhrmann(mol.qVals, mol.elms, lMax, mol.theta, mol.phi, mol.r)
    np.savez(
        OUTPUT_DIR / f"{xyz.stem}.npz",
        I_q      = I_q,
        B_lm_re  = np.real(B_lm),
        B_lm_im  = np.imag(B_lm),
        qVals    = mol.qVals,
    )
    print(f"done ({time.time() - t0:.1f}s)")

print("All done.")
