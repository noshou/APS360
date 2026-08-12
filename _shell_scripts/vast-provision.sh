#!/bin/bash
set -e

####################
## vast.ai runner ##
####################

# Orchestrates the individual provisioning steps (sibling .sh files in this
# same directory), in order. Each step is sourced (not executed as a
# subprocess), since
# later steps depend on state earlier ones set: cwd (clone_repo.sh),
# the activated venv + VENV_DIR (setup_venv.sh), REPO_DIR, etc.
#
# Resolves its own directory via BASH_SOURCE, so this must be run as a real
# file (`bash .../vast-provision.sh`), not piped via `curl | bash` - a piped
# script has no on-disk path for its siblings to be found at.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# capture everything to a persistent log regardless of how/where this script
# is invoked from (onstart-cmd, manual SSH run, etc).
exec > >(tee -a /var/log/vast-provision.log) 2>&1
echo "=== vast-provision.sh started $(date -u) ==="

source "$SCRIPT_DIR/lib.sh"
source "$SCRIPT_DIR/clone_repo.sh"
source "$SCRIPT_DIR/install_latex.sh"
source "$SCRIPT_DIR/setup_venv.sh"
source "$SCRIPT_DIR/download_dataset.sh"
source "$SCRIPT_DIR/setup_rclone.sh"
source "$SCRIPT_DIR/register_supervisor.sh"
