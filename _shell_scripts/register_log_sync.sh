#!/bin/bash
# Registers sync_logs.sh as its own Supervisor-managed service, independent
# of scatternet-train (see that script's own comment for why). Same
# SCRIPT_DIR/REPO_DIR fallback pattern as register_supervisor.sh, for the
# same reason: safe to re-run standalone, not just via vast-provision.sh.
SCRIPT_DIR="${SCRIPT_DIR:-${WORKSPACE:-/workspace}/APS360/_shell_scripts}"
REPO_DIR="${REPO_DIR:-${WORKSPACE:-/workspace}/APS360}"
chmod +x "$SCRIPT_DIR/sync_logs.sh"

cat > /etc/supervisor/conf.d/scatternet-log-sync.conf <<EOF
[program:scatternet-log-sync]
directory=$REPO_DIR
command=$SCRIPT_DIR/sync_logs.sh
autostart=true
autorestart=true
stdout_logfile=/var/log/scatternet-log-sync.log
stderr_logfile=/var/log/scatternet-log-sync.err.log
EOF

wait_for "supervisorctl" "command -v supervisorctl >/dev/null"
supervisorctl reread
supervisorctl update
