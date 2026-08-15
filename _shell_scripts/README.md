# Remote Scripting Guides

## Creating your RCLONE_CONFIG_B64 Environmen Variable

0. If you messed up or need to restart; that's okay! Make sure you're logged in and you can pretty much follow the exact same steps :D
1. Download `rclone`  and `sponge` on your box
2. run `rclone config`
3. select "New remote"
4. enter `gdrive` as the name. it will NOT work if it's not named gdrive!
5. enter `24` (or whatever is under `Google Drive`)
6. pause; do not exit the terminal
7. Google search "`console.cloud.google`" and click `https://console.cloud.google.com/)`
8. Navigate to `APIs & Services`
9. Click `Create project`
10. Enter `ScatterNet`
11. Click `Create`
12. Click `+ Enable APIs and servics` at the top
13. In `Search for APIs & Service`, type `drive`
14. Click `Google Drive API`
15. Click `Enable`
16. Go to APIs & Service

    * If you see the "`OAuth Consent Screen`" tab:
      * Navigate to "`OAuth Consent Screen`" and click `Get Started`
      * Enter an `App Name` as "ScatterNet"
      * Enter suppot email as your email address
      * Click `Next`
      * Set `Audience` as `External`
      * Click `Next`
      * Re-enter your email then Finish and Agree.
      * Click `Continue` then `Create`
      * In `OAuth Overview` click `Create OAuth Client`
    * If you **DO NOT** see the "`OAuth Consent Screen`" tab **OR** it dissapears when clicked:
      * Go to the `Clients` tab
      * At the top click `+ Create client`
17. In `Application` type select `Desktop App` and type whatever name you want

    * *Note: I don't actually know if application type matters, I don't think it does though!*
18. Click `Create` and **DO NOT CLICK AWAY** (for saftey, download the JSON if you wish :D)
19. Back to your terminal, you should see the client_id> prompt. Ctr+C it then Ctl+Shift+C it into your terminal and hit Enter

    * ```{bash}
      Option client_id.
      Google Application Client Id
      Setting your own is recommended.
      See https://rclone.org/drive/#making-your-own-client-id for how to create your own.
      If you leave this blank, it will use an internal key which is low performance.
      Enter a value. Press Enter to leave empty.
      client_id>
      ```
20. Next, you should see the client_secret> prompt. Ctl+C it then Ctl+Shift+C it into your terminal and hit Enter

    * ```{bash}
      Option client_secret.
      OAuth Client Secret.
      Leave blank normally.
      Enter a value. Press Enter to leave empty.
      client_secret>
      ```
21. Enter `1` (full read/write) and hit Enter
22. Hit Enter
23. Hit Enter
24. **DO NOT PRESS ENTER AGAIN**! Navigate to `https://console.cloud.google.com/auth/audience`
25. Under `Test Users` click `Add Users` and enter your email address when prompted and click `Save`
26. Go back to your terminal, it should look like what's below. Type `y` and hit Enter

    * ```{bash}
      Use web browser to automatically authenticate rclone with remote?
       * Say Y if the machine running rclone has a web browser you can use
       * Say N if running rclone on a (remote) machine without web browser access
      If not sure try Y. If Y failed, try N.

      y) Yes (default)
      n) No
      ```
27. A browser should pop up and it should say `Choose an account`. Click the account that you used to set this up. Ignore the big blue `Back to saftey` button and click the small `Continue` on the left.
28. Click `Continue` again, and it should get you to a page titled `Success!`
29. Back in your terminal, it should look like what's below. Type `n` and hit Enter.
30. Type 'y' and hit Enter.
31. Hit `q` to exit rclone (we're almost done!)
32. Copy and paste the command below into the terminal and hit enter:
    ``sed -n '/^\[gdrive\]/,/^\[/{ /^\[/ { /^\[gdrive\]/!d }; p }' ~/.config/rclone/rclone.conf | base64 -w0 | sponge ~/rclone_conf.txt``

    * *Note: `-w0` is required on Linux (GNU `base64` wraps output at 76 characters by default, which breaks the vast.ai `--env` parser if left in). On macOS, BSD `base64` doesn't wrap by default and doesn't accept `-w` at all, drop the flag if running this step there.*
33. You can now set your RCLONE_CONFIG_B64 environment varible to whatever gets outputted in step 32!
34. **Never share your .~/rclone_conf.txt on git or with an AI! If you do, you need to delete the API key *AND* remove access from the linked google account. Then you need to rerun all of the steps again. **

## vast.ai

1. Make sure you start with the "PyTorch (Vast)" template
2. Edit the file
3. Under Environment Varialbes, add RCLONE_CONFIG_B64 and set it to your base64 encoded rclone
   - Optionally also add `VAST_API_KEY` (from https://cloud.vast.ai/manage-keys/) so the instance can auto-destroy itself once training fully converges - see "Auto-kill on convergence" below. Without it, training runs the same, it just won't tear the instance down for you at the end.
   - Optionally also add `HF_TOKEN` (from https://huggingface.co/settings/tokens) so `download_dataset.sh`'s HuggingFace download is authenticated. Nothing to wire up beyond adding it here - `hf` reads it straight from the environment on its own, and it persists across every restart/reprovision this way (unlike `export HF_TOKEN=...` in one SSH session, which is gone the moment that shell closes). Without it, the download still works, just under HuggingFace's stricter unauthenticated rate limits.
4. **Make sure you've selected private so you don't leak your API token(s)!**
5. Under `Advanced Options`, copy below into `On-start Bash Commands`:

```shell
entrypoint.sh & \
curl -sSL https://raw.githubusercontent.com/noshou/APS360/wip/_shell_scripts/clone_repo.sh | bash && \
bash "${WORKSPACE:-/workspace}/APS360/_shell_scripts/vast-provision.sh"
```

5. One last time: ***Make sure you've selected private so you don't leak your API token!***
6. Click `save` and make sure you use it for all subsequent ScatterNet runs
7. Make sure you've set up SSH auth if you want to monitor the run (**highly reccomended**)
8. Good upload/download speeds are important; minimum of 34 GB of VRAM and 120GB of disk space is reccomended. Highly reccomend an A100 or better.
9. Helpful commands after `ssh`-ing into your instance:
   - `nvtop` shows live GPU utilization/memory.
   - `htop` shows live CPU/memory usage.
   - `watch -n 1 supervisorctl status` shows every supervisor-managed service (`scatternet-train`, `scatternet-log-sync`, etc.) and whether it's `RUNNING`/`STOPPED`/`FATAL`.
   - `tail -f /var/log/scatternet-train.log` streams training's stdout (batch/loss/checkpoint prints).
   - `tail -f /var/log/scatternet-train.err.log` streams training's stderr (where crashes/tracebacks show up).
   - `tail -f /var/log/scatternet-log-sync.log` / `.err.log` streams the log-sync service that pushes both logs above to `gdrive:ScatterNet_Train/logs/` every 2 minutes.
   - `: > /var/log/<log>` truncates a log file in place; use this before a restart if you want to watch fresh output only, not the previous run's history.
   - `supervisorctl restart scatternet-train` restarts training (auto-resumes from the latest checkpoint on Drive, see `run_train.sh`).
   - `bash _shell_scripts/vast-provision.sh` reruns the full provisioning chain; safe to rerun anytime, every step is idempotent.
   - `bash _shell_scripts/register_supervisor.sh` / `register_log_sync.sh` re-registers just one service after a script change, without a full reprovision.
   - `df -h /workspace` checks disk space.
10. `download_dataset.sh` now always calls `$VENV_DIR/bin/hf` explicitly (not bare `hf`), so it no longer depends on the invoking shell having the venv active. If you're running a HuggingFace download manually from your own SSH session and it hangs at `0%` for minutes, that same venv-vs-bare-`hf` mismatch is still the likely cause - confirm with `which hf`, then run it explicitly through the venv:
    ```shell
    source /workspace/venv/bin/activate
    /workspace/venv/bin/hf download noshou/iq_train_set "I(q)@L=50.h5" iq_train_set-ENCODING.sqlite3 --repo-type dataset --local-dir Preprocess/
    ```
    If it's still stuck after that, you're likely hitting unauthenticated rate limits - see `HF_TOKEN` under step 3 above (get a free one at https://huggingface.co/settings/tokens). For a one-off manual session, `export HF_TOKEN=hf_xxxxxxxxxxxx` works too, but only lasts that shell - prefer setting it as an instance env var so it survives restarts.
11. `supervisorctl status` also shows the vast.ai template's own default services (`caddy`, `cron`, `instance_portal`, `jupyter`, `syncthing`, `tensorboard`, `tunnel_manager`), alongside `scatternet-train`/`scatternet-log-sync`. None of these are used by this repo's workflow (SSH + `supervisorctl` + `rclone`, not the vast.ai web portal), so they're safe to `supervisorctl stop <name>` for tidiness if you want:
    - `jupyter`, `tensorboard`, `syncthing` - unused (metrics/plots go to Drive via `rclone`, not TensorBoard or syncthing).
    - `tunnel_manager` - manages public tunnel URLs for the vast.ai web portal; irrelevant if you're purely SSH-based.
    - `caddy`, `instance_portal` - serve the vast.ai web console UI itself; stopping these breaks the "Open" buttons in vast.ai's own dashboard for this instance, keep them if you ever want that fallback.
    - Leave `cron` alone - near-zero overhead, may handle internal template housekeeping unrelated to us.

    None of these compete with GPU training for resources (all lightweight; the only heavy process is `scatternet-train` itself), so stopping them is about tidiness, not performance. A stopped service comes back on the next reboot or `vast-provision.sh` rerun unless its own `autostart` is disabled in its conf file (template-managed, outside `_shell_scripts/`).
12. Starting a genuinely fresh run (new hyperparameters, archived the old Drive folder, etc.) needs more than just renaming/moving the old `ScatterNet_Train` folder on Drive and restarting. `Train/train.py` deletes each **checkpoint** locally right after it's pushed (see `_rclone_push` call sites), so a restart with no Drive checkpoints does start training from scratch correctly. But `data_dir` (`scatternet_data/` - the per-epoch metrics/plots directory) is *never* deleted locally, by design, so the loss-vs-epoch curve can be redrawn from disk on a real resume. If you archive the Drive folder away and restart without also clearing the local copy, the new run's first `_rclone_push(data_dir, ...)` re-uploads the old run's leftover `epoch_NNN/` directories straight into the freshly-created Drive folder - it looks like old data is somehow being "pulled back in", but it's actually just stale local state getting re-pushed, nothing cached or restored from Drive. Before restarting for a truly clean run:
    ```shell
    rm -rf /workspace/APS360/scatternet_data
    supervisorctl restart scatternet-train
    ```
13. **Auto-kill on convergence.** `Train/train.yaml` no longer sets `epochs` - training now runs until it's actually converged, not a fixed count. Concretely: smoothing (`SplineSmooth`'s Λ, `lambda_7`) starts OFF; when the raw model plateaus through `smoothing_lr_cut_trigger` (default 2) LR cuts with no escape, smoothing switches on and LR resets to its starting value (`lr`) so the model isn't stuck fine-tuning at whatever tiny LR it plateaued at. If the now-smoothed model plateaus the same way again, training stops - by that point every checkpoint/log/plot has already been pushed to Drive (the normal per-epoch `_rclone_push` calls), so the instance is safe to destroy, and it destroys itself automatically via the `vastai` CLI (installed by `requirements.txt`, authenticated by `setup_vastai_cli.sh` from `VAST_API_KEY`).
    - If `VAST_API_KEY` wasn't set, this is a no-op (a printed warning, not a crash) - the run still stops, you just have to destroy the instance yourself from the vast.ai console.
    - To go back to a fixed-length run instead (no convergence detection, no auto-kill), set `epochs: N` in `Train/train.yaml` like before - it becomes a hard cap regardless of convergence state and disables auto-destroy, since a manual cap implies you want to inspect the result yourself.
    - You can destroy an instance manually the same way training does, from your own machine or another instance: `vastai destroy instance $CONTAINER_ID` (needs `vastai set api-key ...` run once first, same as `setup_vastai_cli.sh` does automatically).