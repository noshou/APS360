#!/bin/bash
# Registers training as a Supervisor-managed service, so it survives an SSH
# disconnect and gets `supervisorctl status`/logs/restart for free. Depends
# on REPO_DIR (clone_repo.sh) and VENV_DIR (setup_venv.sh) already being set.
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
