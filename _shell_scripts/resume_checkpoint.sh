#!/bin/bash
# Finds the best resume checkpoint on Drive (if any) and downloads it, so a
# restart (crash, supervisor restart, instance stop/start) doesn't silently
# start training from scratch. train.py itself only resumes when handed an
# explicit --resume <path>; it does not scan for or pick a checkpoint on its
# own, so that selection has to happen here, before it's launched.
#
# Checkpoints are deleted locally once pushed (Train/train.py:_rclone_push),
# so Drive is the only place a checkpoint can still be found after a restart.
# Sets RESUME_CKPT_PATH (empty if no checkpoint exists yet - a fresh run).
# Depends on VENV_DIR (setup_venv.sh) and rclone already being configured
# (setup_rclone.sh).

CKPT_RCLONE_DEST="$(
  grep '^ckpt_rclone_dest:' "$REPO_DIR/Train/train.yaml" | awk '{print $2}'
)"
CKPT_DIR="$(grep '^ckpt_dir:' "$REPO_DIR/Train/train.yaml" | awk '{print $2}')"
LOCAL_CKPT_DIR="$REPO_DIR/$CKPT_DIR"

RESUME_CKPT_PATH="$("$VENV_DIR/bin/python3" - "$CKPT_RCLONE_DEST" <<'PYEOF'
import re
import subprocess
import sys

dest = sys.argv[1]
result = subprocess.run(
    ["rclone", "lsf", dest], capture_output=True, text=True, check=False
)
if result.returncode != 0:
    # Dest doesn't exist yet (first-ever run) - not an error.
    sys.exit(0)

# "train" checkpoints keep the original 2-part name; val/test get a phase
# segment. See Train/train.py:_save_resume.
pattern = re.compile(r"^checkpoint_(\d+)_(?:(val|test)_)?(final|\d+)\.pt$")
_PHASE_RANK = {"train": 0, "val": 1, "test": 2}

best = None
best_key = None
for name in result.stdout.splitlines():
    m = pattern.match(name.strip())
    if not m:
        continue
    epoch, phase, batch_tag = m.groups()
    phase = phase or "train"
    batch_rank = float("inf") if batch_tag == "final" else int(batch_tag)
    key = (int(epoch), _PHASE_RANK[phase], batch_rank)
    if best_key is None or key > best_key:
        best_key = key
        best = name.strip()

if best:
    print(best)
PYEOF
)"

if [ -n "$RESUME_CKPT_PATH" ]; then
  mkdir -p "$LOCAL_CKPT_DIR"
  rclone copy "$CKPT_RCLONE_DEST/$RESUME_CKPT_PATH" "$LOCAL_CKPT_DIR/"
  RESUME_CKPT_PATH="$LOCAL_CKPT_DIR/$RESUME_CKPT_PATH"
  echo "resume_checkpoint.sh: found $RESUME_CKPT_PATH"
else
  echo "resume_checkpoint.sh: no checkpoint found, starting fresh"
fi
