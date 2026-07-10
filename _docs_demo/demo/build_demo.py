"""
Generate index.html from precomputed .npz files.
Run after precompute.py:
    python _docs_demo/demo/build_demo.py
"""
import sys
import json
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import plotly.graph_objects as go
import plotly.io as pio
from scipy.special import sph_harm_y
from _docs_demo.demo.global_vals import *

def sh_proj(full_coeffs, Y_grid, THETA, PHI):
    """Reconstruct the angular scattering surface r = 1 + Re(Σ B_lm Y_lm) in Cartesian coordinates.

    Parameters
    ----------
    full_coeffs : ndarray of complex128, shape (K,)
        Spherical-harmonic coefficients in the full (l, m) layout, where
        k = l**2 + l + m.
    Y_grid : ndarray of complex128, shape (K, H, W)
        Precomputed spherical harmonics evaluated on the (THETA, PHI) meshgrid.
    THETA : ndarray of float, shape (H, W)
        Polar angle meshgrid.
    PHI : ndarray of float, shape (H, W)
        Azimuthal angle meshgrid.

    Returns
    -------
    tuple of ndarray
        `(x, y, z)` Cartesian coordinate arrays, each shape (H, W), of the
        reconstructed surface.
    """
    f = np.tensordot(full_coeffs, Y_grid, axes=([0], [0]))  # (H, W)
    r = 1.0 + np.real(f)
    x = r * np.sin(THETA) * np.cos(PHI)
    y = r * np.sin(THETA) * np.sin(PHI)
    z = r * np.cos(THETA)
    return x, y, z

# Precompute Y_lm on the fixed meshgrid once for all structures and all q-values.
K      = (lMax + 1) ** 2
Y_grid = np.zeros((K, *THETA.shape), dtype=np.complex128)
for l in range(lMax + 1):
    for m in range(-l, l + 1):
        Y_grid[l * l + l + m] = sph_harm_y(l, m, THETA, PHI)

# Build a (K,) index array mapping full k -> reduced k and sign for negative m.
_full_to_reduced = np.zeros(K, dtype=int)
_full_sign       = np.ones(K, dtype=int)
for l in range(lMax + 1):
    for m in range(-l, l + 1):
        k_full = l * l + l + m
        abs_m  = abs(m)
        _full_to_reduced[k_full] = l * (l + 1) // 2 + abs_m
        if m < 0:
            _full_sign[k_full] = (-1) ** abs_m

def expand_coeffs(B_lm_q):
    """Expand reduced m >= 0 coefficients to the full (l, m) layout.

    Parameters
    ----------
    B_lm_q : ndarray of complex128, shape (K_reduced,)
        Spherical-harmonic coefficients for a single q-value, stored only
        for m >= 0 using the reduced index `l*(l+1)//2 + abs(m)`.

    Returns
    -------
    ndarray of complex128, shape (K,)
        Coefficients expanded to the full (l, m) layout (k = l**2 + l + m),
        with negative-m entries filled in via the conjugate symmetry
        `B_l,-m = (-1)**m * conj(B_l,m)`.
    """
    full = B_lm_q[_full_to_reduced].copy()
    neg_mask = _full_sign < 0
    full[neg_mask] = _full_sign[neg_mask] * np.conj(full[neg_mask])
    return full

structures = []

for npz_file in sorted(PREC_DIR.glob("*.npz")):
    name     = npz_file.stem
    xyz_file = STRUCT_DIR / f"{name}.xyz"
    if not xyz_file.exists():
        print(f"skipped (no .xyz)")
        continue
    print(f"Building {name}...", end=" ", flush=True)

    data  = np.load(npz_file)
    I_q   = data["I_q"]
    B_lm  = data["B_lm_re"] + 1j * data["B_lm_im"]
    qVals = data["qvals"]
    xyz   = xyz_file.read_text(encoding="utf-8")

    # I(q) figure - transparent bg, themed via CSS/JS at runtime
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
            title="𝑞  (Å⁻¹)",
            showgrid=False, showline=True, mirror=True,
            ticks="outside", ticklen=5,
            range=[float(qVals.min()), float(qVals.max())],
        ),
        yaxis=dict(
            title="𝐼(𝑞)", type="log",
            exponentformat="power",
            showgrid=False, showline=True, mirror=True,
            ticks="outside", ticklen=5, minor=dict(ticks=""),
        ),
        margin=dict(l=60, r=20, t=30, b=40),
        height=250,
    )

    # B_lm frames - normalize to unit amplitude; track true max range across all frames
    colorbar_cfg = dict(thickness=14, len=0.6, title=dict(text="z", font=dict(size=11)))
    frames = []
    max_r  = 0.0
    all_xyz = []
    for q_idx, q in enumerate(qVals):
        coeffs = B_lm[:, q_idx]
        scale  = np.abs(coeffs).max()
        if scale > 0:
            coeffs = coeffs / scale
        x, y, z = sh_proj(expand_coeffs(coeffs), Y_grid, THETA, PHI)
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
    x0, y0, z0 = sh_proj(expand_coeffs(coeffs0), Y_grid, THETA, PHI)

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
        height=520,
        margin=dict(l=0, r=0, t=6, b=55),
        scene=dict(
            domain=dict(x=[0, 1], y=[0.24, 1]),
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
            "y": 0, "pad": {"t": 8, "b": 8},
        }],
    )

    data_dir = OUTPUT_DIR / "data"
    data_dir.mkdir(exist_ok=True)
    payload = {
        "iq":  json.loads(str(pio.to_json(fig_iq))),
        "blm": json.loads(str(pio.to_json(fig_blm))),
    }
    (data_dir / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")

    citation_url, citation_label = CITATIONS.get(name, ("#", name))
    structures.append({
        "name":          name,
        "display_name":  DISPLAY_NAMES.get(name, name),
        "citation_url":  citation_url,
        "citation_label": citation_label,
        "xyz":      xyz,
        "data_url": f"precomputed/data/{name}.json",
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

.header {{ display: flex; align-items: center; justify-content: space-between; padding: 10px 20px 6px; }}
h1 {{ font-size: 1.2em; color: var(--h1-color); letter-spacing: 0.05em; }}
.theme-btn {{
    padding: 4px 12px; cursor: pointer; border: 1px solid var(--border);
    background: var(--tab-bg); color: var(--text); border-radius: 20px;
    font-size: 0.78em; transition: background 0.15s;
}}
.theme-btn:hover {{ background: var(--tab-hover); }}

.tabs {{ display: flex; flex-wrap: wrap; gap: 4px; padding: 0 16px; border-bottom: 2px solid var(--border); }}
.tab-btn {{
    padding: 5px 11px; cursor: pointer; border: 1px solid var(--border);
    background: var(--tab-bg); color: var(--text-dim); border-radius: 4px 4px 0 0;
    font-size: 0.78em; transition: background 0.15s;
}}
.tab-btn:hover {{ background: var(--tab-hover); color: var(--text); }}
.tab-btn.active {{ background: var(--tab-active); color: white; border-color: var(--tab-active); }}

.tab-panel {{ display: none; padding: 10px 16px; }}
.tab-panel.active {{ display: block; }}

.panel-layout {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    grid-template-rows: auto clamp(380px, 54vh, 600px) clamp(180px, 24vh, 280px) auto;
    column-gap: 12px;
    row-gap: 8px;
}}
.blm-caption {{ grid-column: 1; grid-row: 1; }}
.blm-plot    {{ grid-column: 1; grid-row: 2; border: 1px solid var(--border); border-radius: 6px; overflow: hidden; background: var(--paper); }}
.iq-plot     {{ grid-column: 1; grid-row: 3; border: 1px solid var(--border); border-radius: 6px; overflow: hidden; background: var(--paper); }}
.citation    {{ grid-column: 1; grid-row: 4; }}
.mol-viewer  {{ grid-column: 2; grid-row: 2 / 4; width: 100%; border: 1px solid var(--border); border-radius: 6px; position: relative; background: var(--mol-bg); }}
.blm-caption {{
    font-family: "STIX Two Math", Georgia, serif;
    font-size: 0.78em;
    color: var(--text-dim);
    padding: 4px 8px 3px;
    background: var(--paper);
    border: 1px solid var(--border);
    border-radius: 6px 6px 0 0;
    border-bottom: none;
    margin-bottom: 0;
}}

details.math-section {{
    margin: 2px 16px 8px;
    padding: 8px 14px;
    background: var(--paper);
    border: 1px solid var(--border);
    border-radius: 6px;
    font-size: 0.82em;
    line-height: 1.8;
}}
details.math-section summary {{
    cursor: pointer;
    font-weight: 600;
    color: var(--accent);
    margin-bottom: 4px;
    user-select: none;
}}
.math-block {{
    font-family: monospace;
    margin: 4px 0 4px 18px;
    color: var(--text);
    white-space: pre;
}}
.math-desc {{ color: var(--text-dim); margin: 2px 0 2px 18px; font-size: 0.91em; }}
.citation {{
    margin-top: 6px;
    font-size: 0.85em;
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
<summary>Mathematical Background: Stuhrmann Decomposition</summary>
<p>The scattering intensity is expressed as a sum over spherical harmonic channels (truncated at l = 50, giving 2601 modes):</p>
<div class="math-block">I(q) = 4π · Σ_{{l=0}}^{{50}} Σ_{{m=-l}}^{{l}} |B_lm(q)|²</div>
<p>where the coefficients B_lm(q) are computed by projecting atomic scattering onto the spherical harmonic basis:</p>
<div class="math-block">B_lm(q) = Σ_i  f_i(q) · j_l(q·rᵢ) · Y*_lm(θᵢ, φᵢ)</div>
<div class="math-desc">f_i(q): complex atomic form factor (f₀ + f₁ + i·f₂) at momentum transfer q</div>
<div class="math-desc">j_l: spherical Bessel function of order l; encodes radial shell information</div>
<div class="math-desc">Y_lm: complex spherical harmonic; * denotes conjugate (analysis projection)</div>
<div class="math-desc">rᵢ, θᵢ, φᵢ: spherical coordinates of atom i relative to the molecular centroid</div>
<p style="margin-top:8px">The 3D surface visualises the angular scattering envelope:</p>
<div class="math-block">r(θ,φ) = 1 + Re( Σ_{{l,m}} B_lm(q) · Y_lm(θ,φ) ) / max|B_lm(q)|</div>
<div class="math-desc">Deformations from a unit sphere indicate anisotropic scattering at that q. Coefficients are normalised to unit amplitude so the shape reflects anisotropy, not absolute intensity.</div>
<p style="margin-top:8px">Complexity: O(N · L²) vs O(N²) for the Debye sum, orders of magnitude faster for large molecules.</p>
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
            <div class="blm-caption">𝑟(θ,φ) = 1 + Re(Σ<sub>lm</sub> 𝐵<sub>lm</sub>(𝑞) · 𝑌<sub>lm</sub>(θ,φ)) / max|𝐵<sub>lm</sub>|</div>
            <div id="blm-${{i}}" class="blm-plot"></div>
            <div id="iq-${{i}}" class="iq-plot"></div>
            <div class="citation">Source: <a href="${{s.citation_url}}" target="_blank" rel="noopener">${{s.citation_label}}</a></div>
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

    // molecule viewer - XYZ is small, inline
    viewers[i] = $3Dmol.createViewer(document.getElementById('mol-' + i), {{
        backgroundColor: t.molBg
    }});
    viewers[i].addModel(s.xyz, 'xyz');
    viewers[i].setStyle({{}}, {{sphere: {{scale: 0.25, colorscheme: 'Jmol'}}, stick: {{radius: 0.1}}}});
    viewers[i].zoomTo();
    viewers[i].render();

    // Plotly data is fetched on first tab open
    fetch(s.data_url)
        .then(r => r.json())
        .then(d => {{
            const iqLayout = Object.assign({{}}, d.iq.layout, {{
                'font.color': t.text,
                'xaxis.linecolor': t.text, 'yaxis.linecolor': t.text,
                'xaxis.color':     t.text, 'yaxis.color':     t.text,
                'xaxis.tickcolor': t.text, 'yaxis.tickcolor': t.text,
            }});
            const iqPromise = Plotly.newPlot('iq-' + i, d.iq.data, iqLayout, {{responsive: true, displayModeBar: false}});

            const blmLayout = Object.assign({{}}, d.blm.layout, {{
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
            const blmPromise = Plotly.newPlot(blmDiv, d.blm.data, blmLayout, {{responsive: true, displayModeBar: false}})
                .then(() => Plotly.addFrames(blmDiv, d.blm.frames));

            // store d so theme toggle and slider can reference it
            structures[i]._d = d;

            Promise.all([iqPromise, blmPromise]).then(() => {{
                Plotly.Plots.resize(document.getElementById('iq-'  + i));
                Plotly.Plots.resize(document.getElementById('blm-' + i));
                const q0 = d.iq.data[0].x[0];
                cursorQ[i] = q0;
                setIqCursor(i, q0);
            }});

            blmDiv.on('plotly_sliderchange', function(e) {{
                const idx = parseInt(e.step.args[0][0]);
                const q   = d.iq.data[0].x[idx];
                cursorQ[i] = q;
                setIqCursor(i, q);
            }});
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
        if (!structures[i]._d) return;  // still loading
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
