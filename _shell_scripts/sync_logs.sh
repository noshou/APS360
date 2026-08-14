#!/bin/bash
# Periodically pushes the training process's stdout/stderr logs to Drive, so
# they survive an instance teardown the same way checkpoints/data already do
# (see setup_rclone.sh, Train/train.py:_rclone_push). Runs as its own
# supervisor program (see register_log_sync.sh), independent of
# scatternet-train itself, so logs keep syncing even if training crashes -
# that's often exactly when you need the tail of the log most.
#
# No `set -e`: a transient rclone failure (network hiccup) should not kill
# this loop permanently, just skip that interval and retry on the next one.

DEST="gdrive:ScatterNet_Train/logs"
INTERVAL_SEC=120

while true; do
  for f in /var/log/scatternet-train.log /var/log/scatternet-train.err.log; do
    [ -f "$f" ] && rclone copy "$f" "$DEST/" --no-traverse
  done
  sleep "$INTERVAL_SEC"
done
