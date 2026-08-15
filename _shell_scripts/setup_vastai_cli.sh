#!/bin/bash
# Authenticates the `vastai` CLI (installed into the venv via
# requirements.txt), so Train/train.py can self-destroy this instance once
# training fully converges (see _destroy_vast_instance). Unlike
# setup_rclone.sh's RCLONE_CONFIG_B64, VAST_API_KEY missing is NOT fatal -
# auto-kill is a convenience, not something checkpoints depend on; training
# runs fine without it, it just won't tear the instance down for you at the
# end.

if [ -z "$VAST_API_KEY" ]; then
  echo "WARNING: VAST_API_KEY is not set - training will run normally, but won't be able to auto-destroy this instance on convergence."
else
  "$VENV_DIR/bin/vastai" set api-key "$VAST_API_KEY"
fi
