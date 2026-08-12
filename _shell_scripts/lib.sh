#!/bin/bash
# Shared helpers for the vast-provision.sh steps. Sourced, not executed.

# Poll `check` (a string eval'd repeatedly) until it succeeds or `timeout`
# seconds pass. Used to wait on things entrypoint.sh (the template's own
# setup, backgrounded alongside this script since it's unclear whether it
# ever returns control) is responsible for creating, so this script doesn't
# race it.
wait_for() {
  local desc="$1" check="$2" timeout="${3:-120}" waited=0
  until eval "$check"; do
    if [ "$waited" -ge "$timeout" ]; then
      echo "timed out after ${timeout}s waiting for $desc" >&2
      exit 1
    fi
    sleep 2
    waited=$((waited + 2))
  done
}
