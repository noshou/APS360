#!/bin/bash
# Clones (or updates) the repo and cd's into it. Sourced, not executed -
# REPO_DIR and the cwd change need to persist into later steps.
#
# NOTE on invocation: this file lives inside the repo it clones, so the
# "repo doesn't exist yet" branch below can only run if vast-provision.sh
# itself was reached some other way (e.g. the runner does an initial
# `git clone` before invoking `bash APS360/_shell_scripts/vast-provision.sh`).
# In the normal case the repo already exists by the time this runs, and this
# step just pulls it to the latest `wip`.

REPO_DIR="${WORKSPACE:-/workspace}/APS360"

# git refuses to operate on a repo owned by a different user than whoever's
# running it (e.g. /workspace owned by the template's default `user`
# account, but we run as root) unless it's marked safe. Idempotent, cheap
# to run every time rather than requiring manual intervention per instance.
git config --global --add safe.directory "$REPO_DIR"

if [ ! -d "$REPO_DIR" ]; then
  git clone -b wip https://github.com/noshou/APS360.git "$REPO_DIR"
else
  git -C "$REPO_DIR" pull origin wip
fi
cd "$REPO_DIR"
