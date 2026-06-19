import h5py
import json
import torch

from collections.abc           import Iterator
from dataclasses               import replace as dc_replace
from torch.utils.data          import DataLoader
from Preprocess                import Encoding
from ScatterNet.batching       import Batcher, Batch
from Baselines                 import Baseline, MeanIQBaseline, AtomCountBaseline
from Baselines.chemistry_only  import CompositionBaseline, MeanFFBaseline
from Baselines.global_geometry import RgBaseline, PowerLawBaseline
from Baselines.local_geometry  import NNBaseline, PairPeakBaseline

# ── config ────────────────────────────────────────────────────────────────────

HDF5_PATH    = "I(q)@L=50.h5"
DB_NAME      = "scatternet"
RESULTS_PATH = "Baselines/baseline_results.json"
BATCHER_SEED = 0
ATOM_SIZE_CEIL: int = -1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BUCKETS: list[tuple[int, int]] = [
    (1,    30),
    (31,   100),
    (101,  400),
    (401,  2000),
    (2001, 10000),
]

# ── setup ─────────────────────────────────────────────────────────────────────

with h5py.File(HDF5_PATH, "r") as f:
    q_grid = torch.tensor(f["q_grid"][:]).float()   # type: ignore[index]
    energy = float(f["energy"][()])                  # type: ignore[index]

enc = Encoding(DB_NAME, HDF5_PATH)

batcher = Batcher(
    hdf5_db        = HDF5_PATH,
    enc            = enc,
    batches        = BUCKETS,
    seed           = BATCHER_SEED,
    atom_size_ceil = ATOM_SIZE_CEIL,
)
train_set, val_set, test_set = batcher.get_sets()

def _loader(dataset: object) -> DataLoader[Batch]:  # type: ignore[type-arg]
    return DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x[0])  # type: ignore[arg-type]

train_loader = _loader(train_set)
val_loader   = _loader(val_set)
test_loader  = _loader(test_set)

# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate_baseline(
    baseline: Baseline,
    loader:   DataLoader[Batch],  # type: ignore[type-arg]
) -> tuple[float, float]:
    """Return (msle, r2) on log1p scale; no Kratky weighting."""
    ss_res = 0.0
    sum_y  = 0.0
    sum_y2 = 0.0
    n_elem = 0

    with torch.no_grad():
        for batch in loader:
            batch = dc_replace(
                batch,
                vocab=batch.vocab.to(DEVICE),
                iqval=batch.iqval.to(DEVICE),
                coord=batch.coord.to(DEVICE),
            )
            pred       = baseline(batch)
            log_pred   = torch.log1p(pred.clamp(min=0))
            log_target = torch.log1p(batch.iqval)

            ss_res += ((log_pred - log_target) ** 2).sum().item()
            sum_y  += log_target.sum().item()
            sum_y2 += (log_target ** 2).sum().item()
            n_elem += log_target.numel()

    if n_elem == 0:
        return float("nan"), float("nan")

    msle   = ss_res / n_elem
    ss_tot = sum_y2 - sum_y ** 2 / n_elem
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return msle, r2


def run(
    name:      str,
    baseline:  Baseline,
    needs_fit: bool = False,
) -> dict[str, object]:
    """Fit if needed, evaluate on train/val/test, print and return results."""
    if needs_fit:
        print(f"  fitting {name}...")
        baseline.fit(train_loader)  # type: ignore[arg-type]

    print(f"  evaluating {name}...")
    train_msle, train_r2 = evaluate_baseline(baseline, train_loader)
    val_msle,   val_r2   = evaluate_baseline(baseline, val_loader)
    test_msle,  test_r2  = evaluate_baseline(baseline, test_loader)

    print(
        f"  {name}\n"
        f"    train  msle={train_msle:.4f}  r2={train_r2:.4f}\n"
        f"    val    msle={val_msle:.4f}  r2={val_r2:.4f}\n"
        f"    test   msle={test_msle:.4f}  r2={test_r2:.4f}"
    )

    return {
        "train": {"msle": train_msle, "r2": train_r2},
        "val":   {"msle": val_msle,   "r2": val_r2},
        "test":  {"msle": test_msle,  "r2": test_r2},
    }

# ── run all baselines ─────────────────────────────────────────────────────────

print("Running baseline harness...")

results: dict[str, object] = {}

results["mean_iq"]    = run("MeanIQBaseline",    MeanIQBaseline(),         needs_fit=True)
results["atom_count"] = run("AtomCountBaseline", AtomCountBaseline(),      needs_fit=True)
results["composition"]= run("CompositionBaseline",CompositionBaseline(),   needs_fit=True)
results["mean_ff"]    = run("MeanFFBaseline",    MeanFFBaseline(q_grid, energy))
results["rg"]         = run("RgBaseline",        RgBaseline(q_grid, energy))
results["power_law"]  = run("PowerLawBaseline",  PowerLawBaseline(q_grid), needs_fit=True)
results["nn_dist"]    = run("NNBaseline",        NNBaseline(q_grid, energy))
results["pair_peak"]  = run("PairPeakBaseline",  PairPeakBaseline(q_grid, energy))

# ── save ──────────────────────────────────────────────────────────────────────

with open(RESULTS_PATH, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {RESULTS_PATH}")
