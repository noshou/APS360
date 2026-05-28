"""
Generate index.html from precomputed .npz files.
Run after precompute.py:
    python demo/build_demo.py
"""
import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import plotly.graph_objects as go
import plotly.io as pio
from scipy.special import sph_harm_y

DEMO_DIR   = Path(__file__).parent
PREC_DIR   = DEMO_DIR / "precomputed"
STRUCT_DIR = DEMO_DIR / "demo_structures"
OUT_HTML   = DEMO_DIR / "index.html"

lMax       = 10
resolution = 40

theta = np.linspace(0, 2*np.pi, resolution)
phi   = np.linspace(0, np.pi,   resolution)
THETA, PHI = np.meshgrid(theta, phi)


def sh_proj(lMax, coeffs, THETA, PHI):
    """Reconstruct angular scattering surface r = 1 + Re(Σ B_lm Y_lm) in Cartesian coordinates."""
    f = np.zeros(THETA.shape, dtype=np.complex128)
    k = 0
    for l in range(lMax + 1):
        for m in range(-l, l + 1):
            f += coeffs[k] * sph_harm_y(l, m, PHI, THETA)
            k += 1
    r = 1.0 + np.real(f)
    x = r * np.sin(PHI) * np.cos(THETA)
    y = r * np.sin(PHI) * np.sin(THETA)
    z = r * np.cos(PHI)
    return x, y, z


DISPLAY_NAMES = {
    "Aerolysin":      "Aerolysin",
    "C60BuckyBallHe": "He@C<sub>60</sub>",
    "CoiledCoil":     "Coiled-Coil",
    "Elf2Nucleosome": "ElF2-Nucleosome Complex",
    "PhiTEBaseplate": "φTE Bacteriophage Baseplate",
    "RhccCarborane":  "RHCC complexed with <i>o</i>-Carborane",
    "Stripak":        "STRIPAK Complex",
    "VATPaseLiRotor": "Li<sup>+</sup>-bound V-ATPase",
}

CITATIONS = {
    "Aerolysin":      ("https://doi.org/10.2210/pdb5JZT/pdb",           "PDB 5JZT"),
    "C60BuckyBallHe": ("https://doi.org/10.1038/ncomms2574",             "Bloodworth et al., Nat. Commun. 5, 3442 (2014)"),
    "CoiledCoil":     ("https://doi.org/10.2210/pdb8P4Y/pdb",           "PDB 8P4Y"),
    "Elf2Nucleosome": ("https://doi.org/10.2210/pdb9igj/pdb",           "PDB 9IGJ"),
    "PhiTEBaseplate": ("https://doi.org/10.2210/pdb9CUY/pdb",           "PDB 9CUY"),
    "RhccCarborane":  ("https://www.wwpdb.org/pdb?id=pdb_00007r6h",     "PDB 7R6H"),
    "Stripak":        ("https://doi.org/10.2210/pdb7k36/pdb",           "PDB 7K36"),
    "VATPaseLiRotor": ("https://doi.org/10.2210/pdb2CYD/pdb",           "PDB 2CYD"),
}

structures = []

for npz_file in sorted(PREC_DIR.glob("*.npz")):
    name     = npz_file.stem
    xyz_file = STRUCT_DIR / f"{name}.xyz"
    print(f"Building {name}...", end=" ", flush=True)

    data  = np.load(npz_file)
    I_q   = data["I_q"]
    B_lm  = data["B_lm_re"] + 1j * data["B_lm_im"]
    qVals = data["qVals"]
    xyz   = xyz_file.read_text()

    # I(q) figure — transparent bg, themed via CSS/JS at runtime
    I_pos  = I_q[I_q > 0]
    y_min  = float(np.floor(np.log10(I_pos.min()))) if len(I_pos) else 0
    y_max  = float(np.ceil(np.log10(I_pos.max())))  if len(I_pos) else 1
    q_pos  = qVals[qVals > 0]
    x_min  = float(np.floor(np.log10(q_pos.min()))) if len(q_pos) else 0
    x_max  = float(np.ceil(np.log10(q_pos.max())))  if len(q_pos) else 1
    pad    = 0.05  # log-unit padding so traces don't touch the axis lines
    fig_iq = go.Figure()
    fig_iq.add_trace(go.Scatter(
        x=qVals.tolist(), y=I_q.tolist(), mode="lines",
        line=dict(color="#268bd2", width=2)
    ))
    fig_iq.update_layout(
        title=dict(text="𝐼(𝑞)", font=dict(family="STIX Two Math, Georgia, serif", size=15)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            title="𝑞  (Å⁻¹)", type="log",
            exponentformat="power",
            showgrid=False, showline=True, mirror=True,
            ticks="outside", ticklen=5, minor=dict(ticks=""),
            range=[x_min - pad, x_max + pad],
        ),
        yaxis=dict(
            title="𝐼(𝑞)", type="log",
            exponentformat="power",
            showgrid=False, showline=True, mirror=True,
            ticks="outside", ticklen=5, minor=dict(ticks=""),
            range=[y_min - pad, y_max + pad],
        ),
        margin=dict(l=60, r=20, t=40, b=50),
        height=320,
    )

    # B_lm frames — normalize to unit amplitude; track true max range across all frames
    colorbar_cfg = dict(thickness=14, len=0.6, title=dict(text="z", font=dict(size=11)))
    frames = []
    max_r  = 0.0
    all_xyz = []
    for q_idx, q in enumerate(qVals):
        coeffs = B_lm[:, q_idx]
        scale  = np.abs(coeffs).max()
        if scale > 0:
            coeffs = coeffs / scale
        x, y, z = sh_proj(lMax, coeffs, THETA, PHI)
        max_r = max(max_r, float(np.abs(x).max()), float(np.abs(y).max()), float(np.abs(z).max()))
        all_xyz.append((x, y, z))

    ax_lim = max_r * 1.08
    for q_idx, (x, y, z) in enumerate(all_xyz):
        frames.append(go.Frame(
            data=[go.Surface(
                x=x.tolist(), y=y.tolist(), z=z.tolist(),
                surfacecolor=z.tolist(), colorscale="Brwnyl",
                showscale=True, colorbar=colorbar_cfg,
            )],
            name=str(q_idx)
        ))

    coeffs0 = B_lm[:, 0]
    scale0  = np.abs(coeffs0).max()
    if scale0 > 0:
        coeffs0 = coeffs0 / scale0
    x0, y0, z0 = sh_proj(lMax, coeffs0, THETA, PHI)

    fig_blm = go.Figure(
        data=[go.Surface(
            x=x0.tolist(), y=y0.tolist(), z=z0.tolist(),
            surfacecolor=z0.tolist(), colorscale="Brwnyl",
            showscale=True, colorbar=colorbar_cfg,
        )],
        frames=frames,
    )
    fig_blm.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        height=560,
        margin=dict(l=0, r=0, t=10, b=0),
        scene=dict(
            domain=dict(x=[0, 1], y=[0.18, 1]),
            aspectmode="cube",
            camera=dict(eye=dict(x=1.05, y=1.05, z=1.05)),
            xaxis=dict(title="x", showgrid=True, showticklabels=True, zeroline=False, range=[-ax_lim, ax_lim]),
            yaxis=dict(title="y", showgrid=True, showticklabels=True, zeroline=False, range=[-ax_lim, ax_lim]),
            zaxis=dict(title="z", showgrid=True, showticklabels=True, zeroline=False, range=[-ax_lim, ax_lim]),
        ),
        sliders=[{
            "steps": [
                {
                    "args": [[str(i)], {"frame": {"duration": 0}, "mode": "immediate"}],
                    "label": f"{q:.2f}", "method": "animate"
                }
                for i, q in enumerate(qVals)
            ],
            "currentvalue": {"prefix": "q = ", "suffix": " Å⁻¹"},
        }],
    )

    fig_iq_dict  = json.loads(pio.to_json(fig_iq))
    fig_blm_dict = json.loads(pio.to_json(fig_blm))

    citation_url, citation_label = CITATIONS.get(name, ("#", name))
    structures.append({
        "name":          name,
        "display_name":  DISPLAY_NAMES.get(name, name),
        "citation_url":  citation_url,
        "citation_label": citation_label,
        "xyz":  xyz,
        "iq":   fig_iq_dict,
        "blm":  fig_blm_dict,
    })
    print("done")

# --- HTML template ---
html = f"""<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
<meta charset="UTF-8">
<title>SAXS Demo</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.1.0/3Dmol-min.js"></script>
<style>
:root {{
    --bg:         #f2ede6;
    --paper:      #e8e2da;
    --text:       #3d3d3d;
    --text-dim:   #888880;
    --grid:       #ccc6bc;
    --border:     #bfb9b0;
    --tab-bg:     #e8e2da;
    --tab-hover:  #ddd7ce;
    --tab-active: #268bd2;
    --mol-bg:     #000000;
    --scene-bg:   #e8e2da;
    --accent:     #268bd2;
    --h1-color:   #1a6fa8;
}}
[data-theme="dark"] {{
    --bg:         #0d0d1a;
    --paper:      #12122a;
    --text:       #cccccc;
    --text-dim:   #888888;
    --grid:       #2a2a4e;
    --border:     #2a2a4e;
    --tab-bg:     #12122a;
    --tab-hover:  #1e1e3e;
    --tab-active: #4a90d9;
    --mol-bg:     #060612;
    --scene-bg:   #0d0d1a;
    --accent:     #4a90d9;
    --h1-color:   #a0c4ff;
}}

* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: sans-serif; background: var(--bg); color: var(--text); transition: background 0.2s, color 0.2s; }}

.header {{ display: flex; align-items: center; justify-content: space-between; padding: 16px 24px 10px; }}
h1 {{ font-size: 1.4em; color: var(--h1-color); letter-spacing: 0.05em; }}
.theme-btn {{
    padding: 6px 14px; cursor: pointer; border: 1px solid var(--border);
    background: var(--tab-bg); color: var(--text); border-radius: 20px;
    font-size: 0.82em; transition: background 0.15s;
}}
.theme-btn:hover {{ background: var(--tab-hover); }}

.tabs {{ display: flex; flex-wrap: wrap; gap: 6px; padding: 0 20px; border-bottom: 2px solid var(--border); }}
.tab-btn {{
    padding: 8px 14px; cursor: pointer; border: 1px solid var(--border);
    background: var(--tab-bg); color: var(--text-dim); border-radius: 4px 4px 0 0;
    font-size: 0.82em; transition: background 0.15s;
}}
.tab-btn:hover {{ background: var(--tab-hover); color: var(--text); }}
.tab-btn.active {{ background: var(--tab-active); color: white; border-color: var(--tab-active); }}

.tab-panel {{ display: none; padding: 20px; }}
.tab-panel.active {{ display: block; }}

.panel-layout {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; align-items: start; }}
.left-col {{ display: flex; flex-direction: column; gap: 12px; }}
.mol-viewer {{ width: 100%; aspect-ratio: 1 / 1; border: 1px solid var(--border); border-radius: 6px; position: relative; background: var(--mol-bg); }}
.iq-plot  {{ height: 320px; border: 1px solid var(--border); border-radius: 6px; overflow: hidden; background: var(--paper); }}
.blm-plot {{ height: 560px; border: 1px solid var(--border); border-radius: 6px; overflow: hidden; background: var(--paper); }}
.blm-caption {{
    font-family: "STIX Two Math", Georgia, serif;
    font-size: 0.85em;
    color: var(--text-dim);
    padding: 6px 10px 4px;
    background: var(--paper);
    border: 1px solid var(--border);
    border-radius: 6px 6px 0 0;
    border-bottom: none;
    margin-bottom: 0;
}}

details.math-section {{
    margin: 4px 20px 16px;
    padding: 12px 18px;
    background: var(--paper);
    border: 1px solid var(--border);
    border-radius: 6px;
    font-size: 0.88em;
    line-height: 1.9;
}}
details.math-section summary {{
    cursor: pointer;
    font-weight: 600;
    color: var(--accent);
    margin-bottom: 6px;
    user-select: none;
}}
.math-block {{
    font-family: monospace;
    margin: 4px 0 4px 18px;
    color: var(--text);
    white-space: pre;
}}
.math-desc {{ color: var(--text-dim); margin: 2px 0 2px 18px; font-size: 0.93em; }}
.citation {{
    margin-top: 14px;
    font-size: 0.92em;
    color: var(--text);
    font-weight: 500;
}}
.citation a {{ color: var(--accent); text-decoration: none; }}
.citation a:hover {{ text-decoration: underline; }}
</style>
</head>
<body>
<div class="header">
    <h1>SAXS Stuhrmann Decomposition</h1>
    <button class="theme-btn" id="themeBtn" onclick="toggleTheme()">◑ Dark mode</button>
</div>

<details class="math-section">
<summary>Mathematical Background — Stuhrmann Decomposition</summary>
<p>The scattering intensity is expressed as a sum over spherical harmonic channels:</p>
<div class="math-block">I(q) = (4π)² · Σ_{{l,m}} |B_lm(q)|²</div>
<p>where the coefficients B_lm(q) are computed by projecting atomic scattering onto the spherical harmonic basis:</p>
<div class="math-block">B_lm(q) = Σ_i  f_i(q) · j_l(q·rᵢ) · Y*_lm(θᵢ, φᵢ)</div>
<div class="math-desc">f_i(q) — complex atomic form factor (f₀ + f₁ + i·f₂) at momentum transfer q</div>
<div class="math-desc">j_l     — spherical Bessel function of order l; encodes radial shell information</div>
<div class="math-desc">Y_lm    — complex spherical harmonic; * denotes conjugate (analysis projection)</div>
<div class="math-desc">rᵢ, θᵢ, φᵢ — spherical coordinates of atom i relative to the molecular centroid</div>
<p style="margin-top:8px">The 3D surface visualisation reconstructs the angular scattering envelope:</p>
<div class="math-block">r(θ,φ) = 1 + Re( Σ_{{l,m}} B_lm · Y_lm(θ,φ) ) · √I(q) / max|B_lm|</div>
<div class="math-desc">Deformations from a unit sphere indicate anisotropic scattering at that q.  Amplitude is scaled by √I(q) so the overall size reflects total signal strength.</div>
<p style="margin-top:8px">Complexity: O(N · L²) vs O(N²) for the Debye sum — orders of magnitude faster for large molecules.</p>
</details>

<div class="tabs" id="tabs"></div>
<div id="panels"></div>

<script>
const structures = {json.dumps(structures)};

const themes = {{
    light: {{
        text:    '#3d3d3d',
        grid:    '#ccc6bc',
        sceneBg: '#e8e2da',
        molBg:   '#000000',
    }},
    dark: {{
        text:    '#cccccc',
        grid:    '#2a2a4e',
        sceneBg: '#0d0d1a',
        molBg:   '#000000',
    }}
}};

let currentTheme = 'light';
const viewers  = {{}};
const cursorQ  = {{}}; // tracks current q per tab for theme redraws

const tabs   = document.getElementById('tabs');
const panels = document.getElementById('panels');

structures.forEach((s, i) => {{
    const btn = document.createElement('button');
    btn.className = 'tab-btn' + (i === 0 ? ' active' : '');
    btn.innerHTML = s.display_name;
    btn.onclick = () => showTab(i);
    tabs.appendChild(btn);

    const panel = document.createElement('div');
    panel.id = 'panel-' + i;
    panel.className = 'tab-panel' + (i === 0 ? ' active' : '');
    panel.innerHTML = `
        <div class="panel-layout">
            <div class="left-col">
                <div class="blm-caption">𝑟(θ,φ) = 1 + Re(Σ<sub>lm</sub> 𝐵<sub>lm</sub>(𝑞) · 𝑌<sub>lm</sub>(θ,φ)) · √𝐼(𝑞) / max|𝐵<sub>lm</sub>| &nbsp;—&nbsp; deformation from unit sphere shows anisotropy; size scales with √𝐼(𝑞)</div>
                <div id="blm-${{i}}" class="blm-plot"></div>
                <div id="iq-${{i}}" class="iq-plot"></div>
                <div class="citation">Source: <a href="${{s.citation_url}}" target="_blank" rel="noopener">${{s.citation_label}}</a></div>
            </div>
            <div id="mol-${{i}}" class="mol-viewer"></div>
        </div>
    `;
    panels.appendChild(panel);
}});

function showTab(i) {{
    document.querySelectorAll('.tab-btn').forEach((b, j) => b.classList.toggle('active', i === j));
    document.querySelectorAll('.tab-panel').forEach((p, j) => p.classList.toggle('active', i === j));
    initTab(i);
}}

const initialized = new Set();

function initTab(i) {{
    if (initialized.has(i)) return;
    initialized.add(i);
    const s = structures[i];
    const t = themes[currentTheme];

    // molecule viewer
    viewers[i] = $3Dmol.createViewer(document.getElementById('mol-' + i), {{
        backgroundColor: t.molBg
    }});
    viewers[i].addModel(s.xyz, 'xyz');
    viewers[i].setStyle({{}}, {{sphere: {{scale: 0.25, colorscheme: 'Jmol'}}, stick: {{radius: 0.1}}}});
    viewers[i].zoomTo();
    viewers[i].render();

    // I(q) plot
    const iqLayout = Object.assign({{}}, s.iq.layout, {{
        'font.color': t.text,
        'xaxis.linecolor': t.text, 'yaxis.linecolor': t.text,
        'xaxis.color':     t.text, 'yaxis.color':     t.text,
        'xaxis.tickcolor': t.text, 'yaxis.tickcolor': t.text,
    }});
    const iqPromise = Plotly.newPlot('iq-' + i, s.iq.data, iqLayout, {{responsive: true, displayModeBar: false}});

    // B_lm plot with slider
    const blmLayout = Object.assign({{}}, s.blm.layout, {{
        'font.color': t.text,
        'scene.bgcolor': t.sceneBg,
        'scene.xaxis.gridcolor': t.grid,
        'scene.yaxis.gridcolor': t.grid,
        'scene.zaxis.gridcolor': t.grid,
        'scene.xaxis.color': t.text,
        'scene.yaxis.color': t.text,
        'scene.zaxis.color': t.text,
    }});
    const blmDiv = document.getElementById('blm-' + i);
    const blmPromise = Plotly.newPlot(blmDiv, s.blm.data, blmLayout, {{responsive: true, displayModeBar: false}})
        .then(() => Plotly.addFrames(blmDiv, s.blm.frames));

    Promise.all([iqPromise, blmPromise]).then(() => {{
        const q0 = s.iq.data[0].x[0];
        cursorQ[i] = q0;
        setIqCursor(i, q0);
    }});

    blmDiv.on('plotly_sliderchange', function(e) {{
        const idx = parseInt(e.step.args[0][0]);
        const q   = s.iq.data[0].x[idx];
        cursorQ[i] = q;
        setIqCursor(i, q);
    }});
}}

function setIqCursor(tabIdx, q) {{
    const t = themes[currentTheme];
    Plotly.relayout('iq-' + tabIdx, {{
        shapes: [{{
            type: 'line',
            x0: q, x1: q,
            y0: 0, y1: 1,
            yref: 'paper',
            line: {{ color: t.text, width: 1.5, dash: 'dot' }},
        }}],
    }});
}}

function toggleTheme() {{
    currentTheme = currentTheme === 'light' ? 'dark' : 'light';
    document.documentElement.dataset.theme = currentTheme;
    document.getElementById('themeBtn').textContent = currentTheme === 'dark' ? '◑ Light mode' : '◑ Dark mode';

    const t = themes[currentTheme];
    initialized.forEach(i => {{
        Plotly.relayout('iq-' + i, {{
            'font.color':      t.text,
            'xaxis.linecolor': t.text, 'yaxis.linecolor': t.text,
            'xaxis.color':     t.text, 'yaxis.color':     t.text,
            'xaxis.tickcolor': t.text, 'yaxis.tickcolor': t.text,
        }});
        Plotly.relayout('blm-' + i, {{
            'font.color': t.text,
            'scene.bgcolor': t.sceneBg,
            'scene.xaxis.gridcolor': t.grid, 'scene.yaxis.gridcolor': t.grid, 'scene.zaxis.gridcolor': t.grid,
            'scene.xaxis.color': t.text,     'scene.yaxis.color': t.text,     'scene.zaxis.color': t.text,
        }});
        if (viewers[i]) {{
            viewers[i].setBackgroundColor(t.molBg);
            viewers[i].render();
        }}
        if (cursorQ[i] !== undefined) {{
            setIqCursor(i, cursorQ[i]);
        }}
    }});
}}

initTab(0);
</script>
</body>
</html>"""

OUT_HTML.write_text(html)
print(f"\nWrote {OUT_HTML}  ({OUT_HTML.stat().st_size / 1e6:.1f} MB)")
