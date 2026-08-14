#!/bin/bash
# Actual entrypoint supervisor invokes (not train.py directly), so resume
# checkpoint selection re-runs on every process start - including a plain
# `supervisorctl restart` with no vast-provision.sh rerun, not just the
# first provisioning. Baking a --resume path into the supervisor conf file
# once, at provisioning time, would go stale the moment a newer checkpoint
# landed on Drive from a later run.
set -e

SCRIPT_DIR="${WORKSPACE:-/workspace}/APS360/_shell_scripts"
REPO_DIR="${WORKSPACE:-/workspace}/APS360"
# Matches setup_venv.sh's own derivation exactly - this script runs as a
# fresh process under supervisor, so nothing from the provisioning script's
# shell (including VENV_DIR itself) carries over.
VENV_DIR="${WORKSPACE:-/workspace}/venv"
cd "$REPO_DIR"

source "$SCRIPT_DIR/resume_checkpoint.sh"

RESUME_ARG=()
if [ -n "$RESUME_CKPT_PATH" ]; then
  RESUME_ARG=(--resume "$RESUME_CKPT_PATH")
fi

exec "$VENV_DIR/bin/python3" Train/train.py --config Train/train.yaml "${RESUME_ARG[@]}"
