#!/bin/bash
set -e

####################
## vast.ai runner ##
####################

# entrypoint.sh (the template's own setup) is backgrounded alongside this
# script rather than run before it, since it's unclear whether it ever
# returns control. wait_for polls for something entrypoint.sh is
# responsible for creating, so this script doesn't race it.
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

# clone repo - for now its WIP
REPO_DIR="${WORKSPACE:-/workspace}/APS360"
if [ ! -d "$REPO_DIR" ]; then
  git clone -b wip https://github.com/noshou/APS360.git "$REPO_DIR"
fi
cd "$REPO_DIR"

# activate and initialize venv - wait for entrypoint.sh to create it first
wait_for "venv activate script" "[ -f /venv/main/bin/activate ]"
source /venv/main/bin/activate
pip install --no-cache-dir -r requirements.txt

# download dataset + encodings
if [ ! -f "Preprocess/I(q)@L=50.h5" ] || [ ! -f "Preprocess/iq_train_set-ENCODING.sqlite3" ]; then
  hf download noshou/iq_train_set "I(q)@L=50.h5" "iq_train_set-ENCODING.sqlite3" \
    --repo-type dataset --local-dir Preprocess/
fi

# get rclone encoding
if [ -n "$RCLONE_CONFIG_B64" ]; then
  mkdir -p ~/.config/rclone
  echo "$RCLONE_CONFIG_B64" | base64 -d > ~/.config/rclone/rclone.conf
fi

# vast.ai configurations
cat > /etc/supervisor/conf.d/scatternet-train.conf <<EOF
[program:scatternet-train]
directory=$REPO_DIR
command=/venv/main/bin/python3 Train/train.py --config Train/train.yaml
autostart=true
autorestart=false
stdout_logfile=/var/log/scatternet-train.log
stderr_logfile=/var/log/scatternet-train.err.log
EOF

wait_for "supervisorctl" "command -v supervisorctl >/dev/null"
supervisorctl reread
supervisorctl update
