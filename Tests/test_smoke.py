"""
Training loop smoke test.

Runs 5 epochs on a tiny slice of real data (1-6 atom molecules, 50 batches/epoch)
and asserts that the loss falls over training — enough to confirm the full pipeline
(data loading → forward → loss → backward → checkpoint) is wired up correctly.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))

from ScatterNet.utils.config import RunConfig
from train             import main

CKPT_BEST   = "/tmp/smoke_best.pt"
CKPT_RESUME = "/tmp/smoke_resume.pt"
METRICS     = "/tmp/smoke_metrics.json"

cfg = RunConfig(
    epochs         = 5,
    buckets        = [(1, 3), (4, 6)],
    atom_size_ceil = 30,
    max_batches    = 50,          # 50 batches/epoch → fast, still exercises the loop
    num_workers    = 0,           # no multiprocessing overhead for local smoke test
    ckpt_best      = CKPT_BEST,
    ckpt_resume    = CKPT_RESUME,
    metrics        = METRICS,
)

main(cfg)

# assert loss actually fell over the 5 epochs
with open(METRICS) as f:
    history = json.load(f)["epochs"]

first_loss = history[0]["train_loss"]
last_loss  = history[-1]["train_loss"]
assert last_loss < first_loss, (
    f"loss did not decrease: epoch 1 = {first_loss:.4f}, epoch 5 = {last_loss:.4f}"
)
print(f"loss {first_loss:.4f} → {last_loss:.4f}  ✓")
print("smoke test passed")
