#!/bin/bash
# Registers training as a Supervisor-managed service, so it survives an SSH
# disconnect and gets `supervisorctl status`/logs/restart for free. Depends
# on REPO_DIR (clone_repo.sh) and VENV_DIR (setup_venv.sh) already being set.

cat > /etc/supervisor/conf.d/scatternet-train.conf <<EOF
[program:scatternet-train]
directory=$REPO_DIR
command=$VENV_DIR/bin/python3 Train/train.py --config Train/train.yaml
autostart=true
autorestart=false
stdout_logfile=/var/log/scatternet-train.log
stderr_logfile=/var/log/scatternet-train.err.log
EOF

wait_for "supervisorctl" "command -v supervisorctl >/dev/null"
supervisorctl reread
supervisorctl update
