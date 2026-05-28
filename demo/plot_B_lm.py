"""
Visualize B_lm(q) angular scattering distribution for a single structure.
Coefficients are scaled by sqrt(I_q) so frame size reflects scattering amplitude.
Run from the APS630 directory: python demo/plot_B_lm.py
"""
from Molecule import *
from Stuhrmann import *
import plotly.graph_objects as go
import numpy as np
import numpy.typing as npt
from beartype import beartype
from scipy.special import sph_harm_y

@beartype
def sh_proj(
    lMax: int,
    coeffs: npt.NDArray[np.complex128],
    thetaGrid: npt.NDArray[np.float64],
    phiGrid: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:

    """Projects complex B_lm coefficients onto a sphere using complex spherical harmonics.

    Reconstructs f(θ,φ) = Σ_lm B_lm * Y_lm(θ,φ), then uses r = 1 + Re(f) as the radius.

    Args:
        lMax:       Maximum harmonic degree
        coeffs:     Complex B_lm coefficients, shape ((lMax+1)²,), ordered
                    (0,0), (1,-1), (1,0), (1,1), (2,-2), ...
        thetaGrid:  Azimuthal angles (0 → 2π)
        phiGrid:    Polar angles (0 → π)

    Returns:
        (x, y, z): Cartesian coordinates of surface
    """

    if lMax < 0:
        raise ValueError("lMax must be >= 0")
    elif coeffs.size != (lMax + 1)**2:
        raise ValueError("size mismatch between coeffs and lMax")
    elif thetaGrid.shape != phiGrid.shape or thetaGrid.size == 0:
        raise ValueError("Invalid grid size")

    f = np.zeros(thetaGrid.shape, dtype=np.complex128)

    k = 0
    for l in range(lMax + 1):
        for m in range(-l, l + 1):
            f += coeffs[k] * sph_harm_y(l, m, phiGrid, thetaGrid)
            k += 1

    r = 1.0 + np.real(f)

    x = r * np.sin(phiGrid) * np.cos(thetaGrid)
    y = r * np.sin(phiGrid) * np.sin(thetaGrid)
    z = r * np.cos(phiGrid)
    return x, y, z


# setup lMax, resolution, theta/phi mesh
lMax = 25
resolution = 500
theta = np.linspace(0, 2*np.pi, resolution)
phi   = np.linspace(0, np.pi,   resolution)
THETA, PHI = np.meshgrid(theta, phi)

# calculate stuhrmann decomposition
fp = "demo_structures/C60BuckyBallHe.xyz"
CC = Molecule.fromXYZ(fp)
I_q, B_lm = stuhrmann(CC.qVals, CC.elms, lMax, CC.theta, CC.phi, CC.r)


# each q value has a specific "frame" associated with it
frames = []
for q_idx, q in enumerate(CC.qVals):
    coeffs = B_lm[:, q_idx]
    scale = np.abs(coeffs).max()
    if scale > 0:
        coeffs = coeffs / scale * np.sqrt(I_q[q_idx])
    x, y, z = sh_proj(lMax, coeffs, THETA, PHI)
    frames.append(go.Frame(
        data=[go.Surface(x=x, y=y, z=z, surfacecolor=z, colorscale="Brwnyl")],
        name=str(q_idx)
    ))

coeffs0 = B_lm[:, 0]
scale0 = np.abs(coeffs0).max()
if scale0 > 0:
    coeffs0 = coeffs0 / scale0 * np.sqrt(I_q[0])
x0, y0, z0 = sh_proj(lMax, coeffs0, THETA, PHI)

fig = go.Figure(
    data=[go.Surface(x=x0, y=y0, z=z0, surfacecolor=z0, colorscale="Brwnyl")],
    frames=frames
)

fig.update_layout(
    title="B_lm angular distribution",
    sliders=[{
        'steps': [
            {
                'args': [[str(i)], {'frame': {'duration': 0}, 'mode': 'immediate'}],
                'label': f'{q:.2f}',
                'method': 'animate'
            }
            for i, q in enumerate(CC.qVals)
        ],
        'currentvalue': {'prefix': 'q = ', 'suffix': ' Å⁻¹'},
    }]
)

fig.show()
