#!/bin/bash
# Decodes and validates the rclone config. rclone is required, not optional:
# checkpoints are deleted locally once pushed (see
# Train/train.py:_rclone_push), so a broken/missing config must stop the run
# here rather than let training start and silently lose checkpoints later.
#
# No whitespace trimming on the input - a malformed value (e.g. embedded
# spaces from a bad copy/paste) should fail loudly here, not get silently
# "fixed" into something that may not be the value actually intended.
# Decoding successfully isn't enough either - garbage input can still
# produce a validly-parsed-but-broken config, so this also does a real
# connectivity check against the gdrive remote before proceeding.

if [ -z "$RCLONE_CONFIG_B64" ]; then
  echo "FATAL: RCLONE_CONFIG_B64 is not set - refusing to start training without durable checkpoint backup" >&2
  exit 1
fi
mkdir -p ~/.config/rclone
if ! echo "$RCLONE_CONFIG_B64" | base64 -d > ~/.config/rclone/rclone.conf; then
  echo "FATAL: RCLONE_CONFIG_B64 failed to decode" >&2
  exit 1
fi
if ! rclone lsd gdrive: >/dev/null 2>&1; then
  echo "FATAL: rclone config decoded but gdrive: remote is not reachable/valid" >&2
  exit 1
fi
