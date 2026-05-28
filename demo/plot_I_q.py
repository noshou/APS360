"""
Plot log-log I(q) vs q for a single structure.
Run from the APS630 directory: python demo/plot_I_q.py
"""
from Molecule import *
from Stuhrmann import *
import plotly.graph_objects as go

lMax = 25
fp   = "demo_structures/C60BuckyBallHe.xyz"
CC   = Molecule.fromXYZ(fp)
I_q, _ = stuhrmann(CC.qVals, CC.elms, lMax, CC.theta, CC.phi, CC.r)

fig = go.Figure()
fig.add_trace(go.Scatter(x=CC.qVals, y=I_q, mode='lines'))
fig.update_layout(
    title="I(q)",
    xaxis_title="q (Å⁻¹)",
    xaxis_type="log",
    yaxis_title="I(q)",
    yaxis_type="log",
)
fig.show()
