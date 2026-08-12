#!/bin/bash
# Creates (or reuses) a venv on the large persistent /workspace disk and
# installs the project into it. Sourced, not executed - VENV_DIR and the
# activated venv need to persist into later steps (dataset download uses
# `hf`, supervisor's command= uses $VENV_DIR/bin/python3).
#
# Deliberately NOT the template's /venv/main: that lives on the container's
# small fixed-size root overlay filesystem, and installing torch + its
# CUDA-13 wheels there runs it out of space ("No space left on device")
# regardless of how big --disk is set at launch, since --disk only sizes
# the volume mounted at /workspace, not /.

VENV_DIR="${WORKSPACE:-/workspace}/venv"
wait_for "system python3" "command -v python3 >/dev/null"
if [ ! -f "$VENV_DIR/bin/activate" ]; then
  python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# --index-url forces the real PyPI, not whatever mirror/proxy pip defaults
# to - that mirror was missing numpy>=2.0 entirely.
pip install --no-cache-dir --index-url https://pypi.org/simple/ -r requirements.txt

# install the project itself (editable) so Preprocess/ScatterNet/Train are
# importable regardless of cwd - `python3 Train/train.py` only puts Train/'s
# own directory on sys.path, not the repo root, so `from Preprocess import
# Encoding` fails without this.
pip install --no-cache-dir --index-url https://pypi.org/simple/ -e .
