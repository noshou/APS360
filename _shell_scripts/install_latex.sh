#!/bin/bash
# LaTeX/JuliaMono toolchain for the diagnostic plots (Baselines/run/metrics.py
# renders text through real xelatex + fontspec, not classic latex+dvipng,
# since dvipng can't load an arbitrary system font). Same install as
# Baselines/run/colab_baselines.ipynb's setup cell.
#
# Best-effort: metrics.py already falls back to matplotlib's default
# mathtext if this is missing (loud warning, not a crash), so a flaky
# apt/network hiccup here shouldn't block training from starting.

if ! command -v xelatex >/dev/null 2>&1; then
  if apt-get update -q && apt-get install -y -q texlive-xetex texlive-latex-recommended texlive-fonts-recommended; then
    mkdir -p ~/.fonts
    if curl -fsSL https://github.com/cormullion/juliamono/releases/latest/download/JuliaMono-ttf.tar.gz | tar -xz -C ~/.fonts; then
      fc-cache -f ~/.fonts
    else
      echo "WARNING: JuliaMono font download failed - plots will render without it" >&2
    fi
  else
    echo "WARNING: texlive install failed - plots will fall back to mathtext" >&2
  fi
fi
