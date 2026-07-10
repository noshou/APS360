"""Shared constants and precomputed grids for the SAXS demo pipeline.

Defines the q-grid / beam energy, spherical-harmonic mode count and
angular meshgrid, filesystem paths used by `precompute.py` and
`build_demo.py`, and per-structure display names and citation metadata
used when rendering the demo's `index.html`.
"""
import numpy as np
from pathlib import Path

# q grid in Å⁻¹ (momentum transfer, physics convention: q = 4π sin(θ)/λ)
ENERGY = 12412.8   # eV (1 Å wavelength, standard SAXS)
Q_MIN  = 0
Q_MAX  = 0.50
Q_STEP = 0.01

# number of modes
lMax = 50       

# meshgrid
resolution = 100
theta = np.linspace(0, np.pi,   resolution)
phi   = np.linspace(0, 2*np.pi, resolution)
THETA, PHI = np.meshgrid(theta, phi)

# paths
DEMO_DIR   = Path(__file__).parent
OUTPUT_DIR = DEMO_DIR / "precomputed"
PREC_DIR   = DEMO_DIR / "precomputed"
STRUCT_DIR = DEMO_DIR / "demo_structures"
OUT_HTML   = DEMO_DIR / "index.html"

# structures
DISPLAY_NAMES = {
    "6YNW":                                "ATP Synthase (Central Stalk/c-ring)",
    "7NK9":                                "ATP Synthase (F<sub>o</sub> Domain, State 1)",
    "C60BuckyBallHe":                      "He@C<sub>60</sub>",
    "COD_100010":                          "Octaphenyl-terphenyl",
    "COD_68580":                           "Au<sup>I</sup><sub>6</sub>Ag<sup>I</sup><sub>3</sub>Cu<sup>II</sup><sub>3</sub> Dodecanuclear Complex",
    "CoiledCoil":                          "Coiled-Coil",
    "NeonMOF":                             "Neon@MOF",
    "RhccCarborane":                       "RHCC complexed with <i>o</i>-Carborane",
    "core_PURSEC":                         "PURSEC (CoRE-MOF)",
    "core_RIVDIL":                         "RIVDIL (CoRE-MOF)",
    "hydration_shells_graphite_tip3p_n17": "Graphite Hydration Shell (TIP3P, n=17)",
    "qmof_qmof_1cf625b":                   "QMOF 1cf625b",
    "qmof_qmof_22103d2":                   "QMOF 22103d2",
}

CITATIONS = {
    "6YNW":                                ("https://doi.org/10.2210/pdb6YNW/pdb",       "PDB 6YNW"),
    "7NK9":                                ("https://doi.org/10.2210/pdb7NK9/pdb",       "PDB 7NK9"),
    "C60BuckyBallHe":                      ("https://doi.org/10.1038/ncomms2574",        "Bloodworth et al., Nat. Commun. 5, 3442 (2014)"),
    "COD_100010":                          ("https://doi.org/10.1107/S0108270111051900", "COD 100010"),
    "COD_68580":                           ("https://doi.org/10.1039/D1SC02497C",        "COD 68580"),
    "CoiledCoil":                          ("https://doi.org/10.2210/pdb8P4Y/pdb",       "PDB 8P4Y"),
    "NeonMOF":                             ("https://doi.org/10.1039/C6CC04808K",        "Tu et al., Chem. Commun. 52, 9307 (2016)"),
    "RhccCarborane":                       ("https://www.wwpdb.org/pdb?id=pdb_00007r6h", "PDB 7R6H"),
    "core_PURSEC":                         ("https://zenodo.org/records/3677685",        "Chung et al., Chem. Mater. 31, 5535 (2019)"),
    "core_RIVDIL":                         ("https://zenodo.org/records/3677685",        "Chung et al., Chem. Mater. 31, 5535 (2019)"),
    "hydration_shells_graphite_tip3p_n17": ("http://www-wales.ch.cam.ac.uk/",            "Gonzalez et al., J. Phys. Chem. C 112, 16497 (2008)"),
    "qmof_qmof_1cf625b":                   ("https://doi.org/10.1039/b917258k",          "QMOF Database v18 (Rosen et al., Matter 4, 1578 (2021))"),
    "qmof_qmof_22103d2":                   ("https://doi.org/10.1039/C7DT03625F",        "QMOF Database v18 (Rosen et al., Matter 4, 1578 (2021))"),
}
