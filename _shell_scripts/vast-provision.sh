#!/bin/bash

############################################################################################################
############################################## vast.ai runner ##############################################
############################################################################################################

set -e

# capture everything to a persistent log regardless of how/where this script
# is invoked from (onstart-cmd, manual SSH run, etc).
exec > >(tee -a /var/log/vast-provision.log) 2>&1
echo "=== vast-provision.sh started $(date -u) ==="

# computed directly rather than via BASH_SOURCE self-discovery, which
# resolved to empty in the actual vast.ai onstart-cmd invocation context
# (SCRIPT_DIR/lib.sh became just "/lib.sh"). We already know exactly where
# clone_repo.sh puts the repo, so just use that instead of inferring it.
SCRIPT_DIR="${WORKSPACE:-/workspace}/APS360/_shell_scripts"

source "$SCRIPT_DIR/lib.sh"
source "$SCRIPT_DIR/clone_repo.sh"
source "$SCRIPT_DIR/install_latex.sh"
source "$SCRIPT_DIR/setup_venv.sh"
source "$SCRIPT_DIR/download_dataset.sh"
source "$SCRIPT_DIR/setup_rclone.sh"
source "$SCRIPT_DIR/register_supervisor.sh"
source "$SCRIPT_DIR/register_log_sync.sh"
