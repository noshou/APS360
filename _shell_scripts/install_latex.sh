#!/bin/bash
# LaTeX/JuliaMono toolchain for the diagnostic plots (Baselines/run/metrics.py
# renders text through real xelatex + fontspec, not classic latex+dvipng,
# since dvipng can't load an arbitrary system font). Same install as
# Baselines/run/colab_baselines.ipynb's setup cell.
#
# Best-effort: metrics.py already falls back to matplotlib's default
# mathtext if this is missing (loud warning, not a crash), so a flaky
# apt/network hiccup here shouldn't block training from starting.

# poppler-utils (pdftocairo) is a separate dependency from xelatex itself:
# matplotlib's pgf backend compiles text through xelatex into a PDF, then
# needs pdftocairo (or ghostscript) to rasterize that PDF into the PNG
# savefig() actually writes. Colab/Kaggle base images ship this already
# (which is why colab_baselines.ipynb's setup cell doesn't install it
# explicitly); the vast.ai template does not, and its absence crashed
# training with "RuntimeError: No suitable pdf to png renderer found" -
# past _configure_mpl()'s toolchain probe, since that probe only calls
# canvas.draw(), which never exercises the PDF->PNG conversion path.
if ! command -v xelatex >/dev/null 2>&1 || ! command -v pdftocairo >/dev/null 2>&1; then
  if apt-get update -q && apt-get install -y -q texlive-xetex texlive-latex-recommended texlive-fonts-recommended poppler-utils; then
    mkdir -p ~/.fonts
    if curl -fsSL https://github.com/cormullion/juliamono/releases/latest/download/JuliaMono-ttf.tar.gz | tar -xz -C ~/.fonts; then
      fc-cache -f ~/.fonts
    else
      echo "WARNING: JuliaMono font download failed - plots will render without it" >&2
    fi
  else
    echo "WARNING: texlive/poppler-utils install failed - plots will fall back to mathtext" >&2
  fi
fi
