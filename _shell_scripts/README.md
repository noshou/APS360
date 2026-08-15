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
4. **Make sure you've selected private so you don't leak your API token!**
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