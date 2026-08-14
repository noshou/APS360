#!/bin/bash
# Registers training as a Supervisor-managed service, so it survives an SSH
# disconnect and gets `supervisorctl status`/logs/restart for free. Depends
# on REPO_DIR (clone_repo.sh) and VENV_DIR (setup_venv.sh) already being set
# when sourced by vast-provision.sh; the := fallbacks below make a standalone
# invocation (e.g. `bash _shell_scripts/register_supervisor.sh` on its own,
# to re-register after a script change without a full reprovision) safe too -
# an unset SCRIPT_DIR previously wrote a broken command= path
# ("/run_train.sh") into the conf file with no error until supervisor tried
# to start it.
SCRIPT_DIR="${SCRIPT_DIR:-${WORKSPACE:-/workspace}/APS360/_shell_scripts}"
REPO_DIR="${REPO_DIR:-${WORKSPACE:-/workspace}/APS360}"
#
# command= points at run_train.sh, not train.py directly: resume checkpoint
# selection (resume_checkpoint.sh) needs to re-run on every process start,
# including a plain `supervisorctl restart` with no vast-provision.sh rerun.
# Baking a --resume path into this conf file once, here, would go stale the
# moment a newer checkpoint landed on Drive from a later run.
chmod +x "$SCRIPT_DIR/run_train.sh"

cat > /etc/supervisor/conf.d/scatternet-train.conf <<EOF
[program:scatternet-train]
directory=$REPO_DIR
command=$SCRIPT_DIR/run_train.sh
autostart=true
autorestart=false
stdout_logfile=/var/log/scatternet-train.log
stderr_logfile=/var/log/scatternet-train.err.log
EOF

wait_for "supervisorctl" "command -v supervisorctl >/dev/null"
supervisorctl reread
supervisorctl update
