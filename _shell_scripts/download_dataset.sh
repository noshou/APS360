#!/bin/bash
# Downloads the training dataset + encoding index from HuggingFace, unless
# already present (idempotent across re-runs on the same instance).
#
# `$VENV_DIR/bin/hf`, not bare `hf` - a bare `hf` resolving to a different
# (non-venv) install was the exact root cause of a download hanging at 0%
# that looked like HF-side throttling but wasn't (see README item 10).
# Calling the venv's own binary explicitly makes this correct regardless
# of shell PATH state, rather than depending on setup_venv.sh's `source
# activate` still being in effect in whatever shell runs this script.
#
# HF_TOKEN is not read explicitly here - `hf` picks it up straight from
# the environment on its own (huggingface_hub's standard behavior), so
# setting it as an instance env var (same field as RCLONE_CONFIG_B64/
# VAST_API_KEY) is all that is needed. Not fatal if unset: unauthenticated
# downloads work, just under stricter rate limits.

if [ -z "$HF_TOKEN" ]; then
  echo "WARNING: HF_TOKEN is not set - dataset download will be unauthenticated (stricter HuggingFace rate limits)."
fi

if [ ! -f "Preprocess/I(q)@L=50.h5" ] || [ ! -f "Preprocess/iq_train_set-ENCODING.sqlite3" ]; then
  "$VENV_DIR/bin/hf" download noshou/iq_train_set "I(q)@L=50.h5" "iq_train_set-ENCODING.sqlite3" \
    --repo-type dataset --local-dir Preprocess/
fi
