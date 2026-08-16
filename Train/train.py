import argparse
import os
import random
import shutil
import subprocess
import sys
import time as _time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import replace as dc_replace
from itertools import count as _count
from time import perf_counter

import h5py
import torch
from torch.utils.data import DataLoader, Subset

from Preprocess import Encoding
from ScatterNet import ScatterNet
from ScatterNet.batching import Batcher, BatchSet
from ScatterNet.utils.config import RunConfig, load_config


def _first(x: list):
    """Collate function that unwraps the single-element list from batch_size=1.

    Parameters
    ----------
    x : list
        Single-element list produced by the DataLoader's default batching.

    Returns
    -------
    object
        The one element contained in `x`.
    """
    return x[0]


def _rclone_push(path: str, dest: str | None, delete_after: bool = False):
    """Copy a file or directory to a durable rclone remote so it survives a
    session timeout.

    The local filesystem is not reliably persisted across a crash/instance
    teardown, so checkpoints (and the plots directory) are also pushed to a
    remote (e.g. Drive) via rclone. `rclone copy` handles both cases: a file
    is copied into `dest`, a directory has its contents mirrored into `dest`
    (incrementally - files already present from an earlier push are
    skipped). No-op if `dest` is None or `path` is missing; never raises into
    the train loop.

    Parameters
    ----------
    path : str
        Local path of the file or directory to copy.
    dest : str or None
        rclone remote destination, or None to skip the push.
    delete_after : bool
        If True, remove the local copy once the push succeeds, so nothing
        accumulates on local disk. Only safe for paths never read back
        locally during the run (e.g. individual checkpoint files) - never
        pass True for `data_dir`, whose per-epoch metrics.json files are
        read back every epoch to rebuild the loss-per-epoch plot.

    Returns
    -------
    None
    """
    if not dest or not os.path.exists(path):
        return
    try:
        subprocess.run(
            ["rclone", "copy", path, dest],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except Exception as e:
        print(f"  [rclone] push of {path} -> {dest} failed: {e}")
        return
    if delete_after:
        try:
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
        except OSError as e:
            print(f"  [rclone] local cleanup of {path} failed: {e}")


def _destroy_vast_instance() -> None:
    """Destroy this vast.ai instance once training has fully converged.

    All local disk is wiped on destroy, but by this point every
    checkpoint/log/plot this run produced has already been pushed to
    Drive via the normal per-epoch `_rclone_push` calls, so nothing is
    lost. Uses the `vastai` CLI (installed into the venv by
    `_shell_scripts/setup_vastai_cli.sh`, authenticated via VAST_API_KEY)
    rather than the `vastai` Python SDK, matching this file's existing
    rclone-via-subprocess style. Resolved via `sys.executable`'s
    directory (not bare "vastai" on $PATH) since `run_train.sh` execs
    this file directly, not through an activated shell.

    No-op (with a print, never raises) if CONTAINER_ID isn't in the
    environment - e.g. running training locally, off vast.ai.

    Returns
    -------
    None
    """
    container_id = os.environ.get("CONTAINER_ID")
    if not container_id:
        print("  [auto-kill] CONTAINER_ID not set, skipping (not on vast.ai?)")
        return
    vastai_bin = os.path.join(os.path.dirname(sys.executable), "vastai")
    print(
        f"  [auto-kill] training converged, destroying instance {container_id}"
    )
    try:
        subprocess.run(
            [vastai_bin, "destroy", "instance", container_id],
            check=True,
        )
    except Exception as e:
        print(f"  [auto-kill] destroy of instance {container_id} failed: {e}")


# profiler
class _LoopProfiler:
    """Per-section wall-clock profiler for the training loop.

    Active only when cfg.profiler is set (otherwise every method is a
    near-zero no-op, so the normal training path pays nothing). Each
    section is CUDA-synced at both boundaries so async GPU kernels are
    charged to the section that launched them instead of leaking into
    the next one - without this, "data_wait" silently absorbs the
    previous batch's still-running backward.

    The main thing this is built to surface is data-loader stalls vs
    GPU compute (is __getitem__ starving the GPU?). A per-batch record
    is also kept so heavy buckets can be correlated with stalls, and
    per-batch peak CUDA memory is tracked (a free counter read - unlike
    torch.profiler's profile_memory, which OOMs) so you can see
    headroom for raising atm_chunk / mol_chunk.
    """

    def __init__(self, device: str, enabled: bool):
        """Initialize the profiler's timing and memory-tracking state.

        Parameters
        ----------
        device : str
            Torch device string (e.g. 'cuda:0' or 'cpu').
        enabled : bool
            Whether profiling is active. When False, all methods are
            near-zero no-ops.
        """
        self.enabled = enabled
        self.device = device
        self.cuda = enabled and device.startswith("cuda")
        self.totals: dict[str, float] = {}
        self.records: list[dict] = []
        self._prev_end: float | None = None
        self.peak_alloc_gb = (
            0.0  # max over all batches of per-batch peak live tensors
        )
        self.peak_resv_gb = (
            0.0  # max allocator-reserved (cache); OOM-relevant ceiling
        )
        self.peak_bi = -1  # which batch hit peak_alloc_gb

    def _sync(self):
        """Synchronize the current CUDA device, if profiling on CUDA.

        Returns
        -------
        None
        """
        if self.cuda:
            torch.cuda.synchronize()  # current device (set via set_device)

    @contextmanager
    def section(self, name: str, rec: dict | None = None):
        """Context manager that times a named code section.

        Parameters
        ----------
        name : str
            Name of the section, used as a key in `totals` and `rec`.
        rec : dict or None, optional
            Per-batch record dict to also store this section's elapsed
            time in, keyed by `name`.

        Yields
        ------
        None
        """
        if not self.enabled:
            yield
            return
        self._sync()
        t0 = perf_counter()
        try:
            yield
        finally:
            self._sync()
            dt = perf_counter() - t0
            self.totals[name] = self.totals.get(name, 0.0) + dt
            if rec is not None:
                rec[name] = dt

    def start_batch(self, rec: dict):
        """Charge time blocked in the DataLoader since the previous batch
        ended.

        Recorded under the 'data_wait' key. Call at the very top of the loop
        body, before any `.to()`.

        Parameters
        ----------
        rec : dict
            Per-batch record dict to store 'data_wait' time in.

        Returns
        -------
        None
        """
        if not self.enabled:
            return
        now = perf_counter()
        dt = 0.0 if self._prev_end is None else now - self._prev_end
        self.totals["data_wait"] = self.totals.get("data_wait", 0.0) + dt
        rec["data_wait"] = dt
        if self.cuda:
            # isolate this batch's peak so per-batch memory is attributable
            torch.cuda.reset_peak_memory_stats()

    def end_batch(self, rec: dict):
        """Finalize a batch's timing record and update peak-memory statistics.

        Parameters
        ----------
        rec : dict
            Per-batch record dict, appended to `records` after peak CUDA
            memory fields are attached.

        Returns
        -------
        None
        """
        if not self.enabled:
            return
        if self.cuda:
            pk = (
                torch.cuda.max_memory_allocated() / 1e9
            )  # peak LIVE tensors this batch (OOM driver)
            rv = (
                torch.cuda.memory_reserved() / 1e9
            )  # allocator cache high-water
            rec["peak_alloc_gb"] = pk
            rec["reserved_gb"] = rv
            if pk > self.peak_alloc_gb:
                self.peak_alloc_gb = pk
                self.peak_bi = rec["bi"]
            self.peak_resv_gb = max(self.peak_resv_gb, rv)
        self.records.append(rec)
        self._prev_end = perf_counter()

    # section order for the printed report (others appended as seen)
    _ORDER = [
        "data_wait",
        "h2d",
        "forward",
        "loss",
        "backward",
        "clip",
        "step",
    ]

    def report(self):
        """Print the accumulated per-section timing and memory report.

        Returns
        -------
        None
        """
        if not self.enabled or not self.records:
            return
        n = len(self.records)
        total = sum(self.totals.values())
        names = self._ORDER + [k for k in self.totals if k not in self._ORDER]
        tag = "[prof]"
        lines = [
            f"{tag} ---- section breakdown over {n} active batch(es) "
            f"(total {total:.3f}s, {total / n * 1e3:.1f} ms/batch) ----"
        ]
        for k in names:
            if k not in self.totals:
                continue
            sec = self.totals[k]
            lines.append(
                f"{tag}   {k:<16s} {sec:8.3f}s  {sec / total * 100:5.1f}%  "
                f"{sec / n * 1e3:8.2f} ms/batch"
            )
        if self.cuda:
            peak = max(
                (
                    r["peak_alloc_gb"]
                    for r in self.records
                    if "peak_alloc_gb" in r
                ),
                default=0.0,
            )
            lines.append(
                f"{tag}   peak CUDA mem: {peak:.2f}G  "
                f"(reserved high-water {self.peak_resv_gb:.2f}G)"
            )

        # heaviest batches by data_wait and by compute, with bucket geometry +
        # peak mem. label is padded to a fixed width so the "ms" field lines
        # up regardless of whether label is "data" or "compute" (different
        # lengths otherwise shift every field after it).
        def _top(key: str, label: str):
            """Print the 3 heaviest batches by a given record key.

            Parameters
            ----------
            key : str
                Record field to rank batches by (e.g. 'data_wait', 'compute').
            label : str
                Short label used in the printed line prefix.

            Returns
            -------
            None
            """
            rows = sorted(
                self.records, key=lambda r: r.get(key, 0.0), reverse=True
            )[:3]
            for r in rows:
                mem = (
                    f"peak_alloc={r['peak_alloc_gb']:6.2f}G"
                    if "peak_alloc_gb" in r
                    else ""
                )
                lines.append(
                    f"{tag}   heavy-{label:<8s}"
                    f"{r.get(key, 0.0) * 1e3:8.2f} ms  "
                    f"bi={r['bi']:<5d} mols={r['n_mols']:<6d} "
                    f"max_atoms={r['max_atoms']:<6d} "
                    f"real_atoms={r['real_atoms']:<6d} {mem}"
                )

        _top("data_wait", "data")
        _top("compute", "compute")

        # Per-group breakdown: the combined numbers above blend three
        # structurally
        # different bucket populations (compute-worst, memory-worst, median),
        # so they
        # aren't comparable across profiler runs whose group sizes differ, and
        # they're
        # not very informative even within one run. Report each group
        # separately so different groups' costs can be compared
        # like-for-like.
        _GROUP_LABEL = {
            "heavy_nm": "heaviest N*M",
            "heavy_m": "heaviest M",
            "regular": "median",
        }
        if any("group" in r for r in self.records):
            lines.append(f"{tag} ---- per-group breakdown ----")
            for g in ("heavy_nm", "heavy_m", "regular"):
                grp = [r for r in self.records if r.get("group") == g]
                if not grp:
                    continue
                gn = len(grp)

                def g_ms(k: str) -> float:
                    return sum(r.get(k, 0.0) for r in grp) / gn * 1e3

                g_peak = max(
                    (r.get("peak_alloc_gb", 0.0) for r in grp), default=0.0
                )
                lines.append(
                    f"{tag}   {_GROUP_LABEL[g]:<20s} n={gn:<3d} "
                    f"compute={g_ms('compute'):8.2f}ms/batch  "
                    f"forward={g_ms('forward'):7.2f}ms  "
                    f"backward={g_ms('backward'):7.2f}ms  "
                    f"peak_alloc={g_peak:.2f}G"
                )

        print("\n".join(lines))


# CLI
def _parse_args():
    """Parse CLI arguments.

    All model/training flags default to None so that `load_config` can
    distinguish 'not provided' from an explicit zero or false.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    p = argparse.ArgumentParser(
        description="Train ScatterNet. All flags override --config values."
    )
    p.add_argument("--config", default=None, help="path to YAML run config")

    # paths
    p.add_argument("--hdf5", default=None)
    p.add_argument("--encodings_sqlite3_path", default=None)
    p.add_argument("--ckpt_best", default=None)
    p.add_argument("--ckpt_dir", default=None)
    p.add_argument("--resume", default=None, help="path to resume checkpoint")

    # model
    p.add_argument("--lambda_1", type=int, default=None)
    p.add_argument("--lambda_2", type=int, default=None)
    p.add_argument("--lambda_3", type=int, default=None)
    p.add_argument("--lambda_4", type=int, default=None)
    p.add_argument("--lambda_5", type=int, default=None)
    p.add_argument("--msg_seed", type=int, default=None)
    p.add_argument("--atm_chunk", type=int, default=None)
    p.add_argument("--mol_chunk", type=int, default=None)
    p.add_argument("--compile", action="store_const", const=True, default=None)
    p.add_argument("--eps_embd", type=float, default=None)
    p.add_argument("--eps_msgp", type=float, default=None)

    # loss
    p.add_argument("--lambda_6", type=float, default=None)
    p.add_argument("--lambda_7", type=float, default=None)

    # training
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--lr_factor", type=float, default=None)
    p.add_argument("--lr_patience", type=int, default=None)
    p.add_argument("--lr_threshold", type=float, default=None)
    p.add_argument("--lr_min", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--grad_clip", type=float, default=None)
    p.add_argument("--adam_eps", type=float, default=None)
    p.add_argument("--smoothing_lr_cut_trigger", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batcher_seed", type=int, default=None)
    p.add_argument("--atom_size_ceil", type=int, default=None)
    p.add_argument(
        "--dataset_frac",
        type=float,
        default=None,
        help=(
            "fraction of each split's batches to use, (0.0, 1.0]; "
            "applies to train, val AND test"
        ),
    )
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument(
        "--verbosity", default=None, choices=["epoch", "batch", "diagnostic"]
    )
    p.add_argument(
        "--profiler", action="store_const", const=True, default=None
    )
    p.add_argument("--prof_warmup", type=int, default=None)
    p.add_argument("--prof_active", type=int, default=None)
    p.add_argument(
        "--data_dir",
        default=None,
        help="write per-epoch metrics + baseline-style diagnostic plots here",
    )

    return p.parse_args()


# eval helper
def evaluate(
    loader: torch.utils.data.DataLoader,
    model: ScatterNet,
    cfg: RunConfig,
    device: str,
    label: str = "eval",
    start_batch: int = 0,
    resume_state: dict | None = None,
    ckpt_cb: "Callable[[dict, int], None] | None" = None,
) -> tuple[float, float]:
    """Run one pass over `loader` without gradients and return (mean_loss, R2).

    Parameters
    ----------
    loader : torch.utils.data.DataLoader
        Loader yielding batches to evaluate over (val or test set).
    model : ScatterNet
        Model to evaluate; called in `eval()` mode.
    cfg : RunConfig
        Run configuration, used for `lambda_6`/`lambda_7`.
    device : str
        Torch device to move batch tensors to.
    label : str
        Split being walked ("val"/"test"), used in the progress line.
        Val and test hold one batch per bucket exactly like train, so
        this pass is thousands of batches long; it reports every 20,
        same as the train loop.
    start_batch : int
        Global batch index of the first batch `loader` will yield (0
        normally; > 0 on a mid-phase resume, where `loader`'s sampler
        already skips the batches before this index - see
        `_ResumableSequentialSampler`). Only used to label `_bi` in the
        progress line and checkpoint callback; the accumulators
        themselves are seeded from `resume_state`, not recomputed.
    resume_state : dict or None
        Accumulator state saved by an earlier `ckpt_cb` call
        (total_loss/total_mols/ss_res/sum_y/sum_y2/n_elem), folded in
        before this pass's own batches are added. None for a fresh pass.
    ckpt_cb : callable or None
        If given, called periodically (same cadence as the training
        loop's mid-epoch checkpoint) as `ckpt_cb(state, batch_idx)` with
        the current accumulator state and the last-processed global
        batch index, so the caller can write a resumable checkpoint.

    Returns
    -------
    tuple of (float, float)
        Mean loss over all molecules, and the R2 score computed in
        log1p space.
    """
    model.eval()
    total_loss = 0.0
    total_mols = 0.0
    ss_res = 0.0
    sum_y = 0.0
    sum_y2 = 0.0
    n_elem = 0.0
    if resume_state:
        total_loss = resume_state["total_loss"]
        total_mols = resume_state["total_mols"]
        ss_res = resume_state["ss_res"]
        sum_y = resume_state["sum_y"]
        sum_y2 = resume_state["sum_y2"]
        n_elem = resume_state["n_elem"]
    verbose = cfg.verbosity in ("batch", "diagnostic")
    n_batch = start_batch + len(loader)
    t0 = _time.time()
    last_ckpt = _time.time()
    with torch.no_grad():
        for _bi, batch in enumerate(loader, start=start_batch):
            batch = dc_replace(
                batch,
                vocab=batch.vocab.to(device),
                iqval=batch.iqval.to(device),
                coord=batch.coord.to(device),
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                iq, coh, inc, fmags, sigmas = model(batch)
                loss = model.compute_loss(
                    iq,
                    coh,
                    inc,
                    fmags,
                    batch,
                    cfg.lambda_6,
                    cfg.lambda_7,
                )
            iq = iq.float()  # R2/metrics accumulate in fp32
            n = batch.iqval.shape[0]
            b_loss = loss.item() * n
            b_mols = float(n)
            log_pred = torch.log1p(iq)
            log_target = torch.log1p(batch.iqval)
            b_ss_res = ((log_pred - log_target) ** 2).sum().item()
            b_sum_y = log_target.sum().item()
            b_sum_y2 = (log_target**2).sum().item()
            b_n_elem = float(log_target.numel())

            total_loss += b_loss
            total_mols += b_mols
            ss_res += b_ss_res
            sum_y += b_sum_y
            sum_y2 += b_sum_y2
            n_elem += b_n_elem
            del iq, coh, inc, fmags, sigmas, loss, log_pred, log_target

            if verbose and (_bi + 1) % 20 == 0:
                elapsed = _time.time() - t0
                rate = (_bi + 1 - start_batch) / elapsed
                print(
                    f"  [{label}] batch {_bi + 1:5d}/{n_batch}  "
                    f"loss {total_loss / max(total_mols, 1)}  "
                    f"{rate} batch/s",
                )

            # Crash safety: same cadence as the training loop's mid-epoch
            # checkpoint - val/test are thousands of batches long too, so
            # losing a whole pass to a late timeout is exactly the failure
            # mode this closes.
            if (
                ckpt_cb is not None
                and (_time.time() - last_ckpt) > cfg.ckpt_interval_sec
            ):
                ckpt_cb(
                    {
                        "total_loss": total_loss,
                        "total_mols": total_mols,
                        "ss_res": ss_res,
                        "sum_y": sum_y,
                        "sum_y2": sum_y2,
                        "n_elem": n_elem,
                    },
                    _bi,
                )
                last_ckpt = _time.time()
    mean_loss = total_loss / total_mols
    ss_tot = sum_y2 - sum_y**2 / n_elem
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return mean_loss, r2


# training worker
def _worker(cfg: RunConfig):
    """Run the training worker, in-process on the single available GPU.

    Parameters
    ----------
    cfg : RunConfig
        Full run configuration (paths, model hyperparameters, training
        schedule, and profiler settings).

    Returns
    -------
    None
    """

    if cfg.compile:
        os.environ.setdefault(
            "TORCHDYNAMO_VERBOSE", "1"
        )  # full Dynamo/Inductor traceback on compile failures

    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(cfg.batcher_seed)

    print(f"device={device}")
    print(
        f"lr={cfg.lr}  epochs={cfg.epochs}  λ1={cfg.lambda_1}  "
        f"λ2={cfg.lambda_2}  λ3={cfg.lambda_3}  λ4={cfg.lambda_4}  "
        f"λ5={cfg.lambda_5}  λ6={cfg.lambda_6}  λ7={cfg.lambda_7}"
    )

    # data

    with h5py.File(cfg.hdf5, "r") as f:
        q_grid = torch.tensor(f["q_grid"][:]).float()  # type: ignore[index]
    energy = 12_500.0
    q_points = len(q_grid)

    enc = Encoding(cfg.encodings_sqlite3_path, cfg.hdf5)

    batcher = Batcher(
        hdf5_db=cfg.hdf5,
        enc=enc,
        batches=cfg.buckets,
        seed=cfg.batcher_seed,
        atom_size_ceil=cfg.atom_size_ceil,
    )
    train_set, val_set, test_set = batcher.get_sets()

    if cfg.dataset_frac < 1.0:
        # Applies to all three splits. Batcher splits each bucket's MOLECULES,
        # so val and
        # test hold one batch per bucket just like train - thinning train alone
        # let
        # epoch-end eval dominate the very runs this knob exists to shorten.
        # Each split is
        # sampled independently (Batcher drops empty per-bucket splits, so the
        # three
        # .batches lists don't share an index space), deterministically off
        # batcher_seed
        # and fixed for the run. Rebuilds a BatchSet rather than wrapping in
        # Subset so
        # .batches stays intact - the profiler branch below reaches into it
        # directly.
        def _thin(bset: BatchSet, salt: int) -> tuple[BatchSet, int]:
            n = len(bset.batches)
            k = max(1, round(n * cfg.dataset_frac))
            idx = sorted(
                random.Random(cfg.batcher_seed + salt).sample(range(n), k)
            )
            return BatchSet(
                bset.db_path, bset.enc, [bset.batches[i] for i in idx]
            ), n

        (train_set, n_trn), (val_set, n_val), (test_set, n_tst) = (
            _thin(train_set, 0),
            _thin(val_set, 1),
            _thin(test_set, 2),
        )
        print(
            f"dataset_frac={cfg.dataset_frac}  "
            f"train={len(train_set)}/{n_trn}  "
            f"val={len(val_set)}/{n_val}  "
            f"test={len(test_set)}/{n_tst} batches",
        )

    pin = device != "cpu" and not str(device).startswith("privateuseone")
    pw = cfg.num_workers > 0

    class _ResumableSampler(torch.utils.data.Sampler):
        """RandomSampler equivalent, but able to start partway through the
        permutation. On a mid-epoch resume, iterating a shuffle=True
        DataLoader from index 0 and discarding batches via `if _bi < skip:
        continue` still pays the full load/collate cost for every skipped
        batch (real HDF5 reads through the dataset's __getitem__), which is
        silent and can take hours late in a long epoch. Slicing the
        permutation here means skipped indices are never handed to the
        DataLoader at all.
        """

        def __init__(self, data_source: BatchSet):
            self.data_source = data_source
            self.skip = 0

        def __len__(self) -> int:
            return len(self.data_source) - self.skip

        def __iter__(self) -> Iterator[int]:
            perm = torch.randperm(len(self.data_source)).tolist()
            return iter(perm[self.skip :])

    class _ResumableSequentialSampler(torch.utils.data.Sampler):
        """SequentialSampler equivalent, but able to start partway through.

        val/test loaders are unshuffled (order is already deterministic
        across runs), so a mid-pass resume just needs to skip the first
        `skip` indices - same rationale as `_ResumableSampler` above (skip
        indices never reach the DataLoader, so no wasted HDF5 reads for
        already-evaluated batches).
        """

        def __init__(self, data_source: BatchSet):
            self.data_source = data_source
            self.skip = 0

        def __len__(self) -> int:
            return len(self.data_source) - self.skip

        def __iter__(self) -> Iterator[int]:
            return iter(range(self.skip, len(self.data_source)))

    train_sampler = _ResumableSampler(train_set)
    train_loader = DataLoader(
        train_set,
        batch_size=1,
        sampler=train_sampler,
        collate_fn=_first,
        num_workers=cfg.num_workers,
        pin_memory=pin,
        persistent_workers=pw,
    )
    val_sampler = _ResumableSequentialSampler(val_set)
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        sampler=val_sampler,
        collate_fn=_first,
        num_workers=cfg.num_workers,
        pin_memory=pin,
        persistent_workers=pw,
    )
    test_sampler = _ResumableSequentialSampler(test_set)
    test_loader = DataLoader(
        test_set,
        batch_size=1,
        sampler=test_sampler,
        collate_fn=_first,
        num_workers=cfg.num_workers,
        pin_memory=pin,
        persistent_workers=pw,
    )

    # In profiler mode, pick which buckets to profile from metadata (atom
    # counts live in
    # each row (grp, stem, atoms), so no tensors are loaded). A single
    # N*M ranking
    # misses two things: (1) memory-risk buckets with large M but small N
    # (N*M is a
    # compute-time proxy dominated by N, so a huge-M/small-N bucket can rank
    # low on it even
    # though it has the biggest per-chunk activations), and (2) what a
    # "typical" batch even
    # looks like (a pure worst-case sample only ever shows outliers). So the
    # profiling budget
    # is split three ways:
    # 1. heaviest by N*M  - compute-time worst case (usually
    # huge-N/tiny-M).
    # 2. heaviest by raw M      - memory-risk worst case (large per-chunk RFF
    # tensors),
    #      invisible to the N*M ranking whenever N is small.
    # 3. a band around the median N*M - representative "regular" batches,
    # so the
    #      two worst-case groups have a baseline to compare against.
    # Worst-of-group-1 stays first (bi=0, the profiler's "wait" step) so it
    # gets a clean
    # peak_alloc reading with no torch-trace overhead, matching the original
    # single-ranking
    # behaviour for the compute-worst-case bucket.
    prof_loader = None
    worst: list = []
    prof_group_bounds = (
        0,
        0,
    )  # (end of heavy_nm, end of heavy_m) indices into `worst`
    if cfg.profiler:

        def _bucket_nm_proxy(i: int):
            """Compute the N*M compute-time proxy for bucket `i`.

            Parameters
            ----------
            i : int
                Index of the bucket in `train_set.batches`.

            Returns
            -------
            int
                Number of molecules in the bucket times the max atom count.
            """
            rows = train_set.batches[i]
            return len(rows) * max(r[2] for r in rows)

        def _bucket_max_m(i: int):
            """Return the raw maximum atom count M for bucket `i`.

            Parameters
            ----------
            i : int
                Index of the bucket in `train_set.batches`.

            Returns
            -------
            int
                Maximum atom count among molecules in the bucket.
            """
            return max(r[2] for r in train_set.batches[i])

        n_each = 1 + cfg.prof_warmup + cfg.prof_active
        n_group = max(1, n_each // 3)

        if cfg.prof_molecules:
            # hardcoded fixture list, build a lookup from (grp, stem) to
            # batch index
            key_to_bi: dict = {}
            for bi, batch in enumerate(train_set.batches):
                for grp, stem, _ in batch:
                    key_to_bi[(grp, stem)] = bi
            fixture_bis: list[int] = []
            for grp, stem in cfg.prof_molecules:
                bi = key_to_bi.get((grp, stem))
                if bi is not None and bi not in fixture_bis:
                    fixture_bis.append(bi)
            worst = fixture_bis
            n_hm = min(n_group, len(worst))
            prof_group_bounds = (n_hm, min(2 * n_group, len(worst)))
            mode = f"hardcoded fixtures ({len(worst)} batches)"
        else:
            by_nm = sorted(
                range(len(train_set)), key=_bucket_nm_proxy, reverse=True
            )
            by_m = sorted(
                range(len(train_set)), key=_bucket_max_m, reverse=True
            )

            heavy_nm = by_nm[:n_group]
            seen = set(heavy_nm)
            heavy_m = [i for i in by_m if i not in seen][:n_group]
            seen |= set(heavy_m)

            n_regular = max(0, n_each - len(heavy_nm) - len(heavy_m))
            mid = len(by_nm) // 2
            band_start = max(0, mid - n_regular)
            regular = [i for i in by_nm[band_start:] if i not in seen][
                :n_regular
            ]

            worst = heavy_nm + heavy_m + regular
            prof_group_bounds = (len(heavy_nm), len(heavy_nm) + len(heavy_m))
            mode = (
                f"{len(heavy_nm)} heaviest N*M + "
                f"{len(heavy_m)} heaviest M + {len(regular)} median"
            )
        prof_loader = DataLoader(
            Subset(train_set, worst),
            batch_size=1,
            shuffle=False,
            collate_fn=_first,
            num_workers=cfg.num_workers,
            pin_memory=pin,
        )

        def _describe(i: int):
            """Format a bucket's molecule count and max atom count for
            logging.

            Parameters
            ----------
            i : int
                Index of the bucket in `train_set.batches`.

            Returns
            -------
            str
                Human-readable description, e.g. '12 mols x 340 atoms'.
            """
            rows = train_set.batches[i]
            return f"{len(rows)} mols x {max(r[2] for r in rows)} atoms"

        print(f"[profiler] probing {mode}")
        n_hm, n_hmax = prof_group_bounds
        for label, group, proxy_fn in (
            ("worst N*M", worst[:n_hm], _bucket_nm_proxy),
            ("worst M", worst[n_hm:n_hmax], _bucket_max_m),
            ("median sample", worst[n_hmax:], _bucket_nm_proxy),
        ):
            if group:
                print(
                    f"  {label:<16s} "
                    f"proxy={proxy_fn(group[0]):<8d} "
                    f"{_describe(group[0])}",
                )

    train_mols = sum(len(b) for b in train_set.batches)
    val_mols = sum(len(b) for b in val_set.batches)
    test_mols = sum(len(b) for b in test_set.batches)
    print(
        f"batches/epoch: train={len(train_set)}  "
        f"val={len(val_loader)}  test={len(test_loader)}",
    )
    print(
        f"molecules:     train={train_mols}  val={val_mols}  "
        f"test={test_mols}",
    )

    # model

    model = ScatterNet(
        lambda_1=cfg.lambda_1,
        lambda_2=cfg.lambda_2,
        lambda_3=cfg.lambda_3,
        lambda_4=cfg.lambda_4,
        lambda_5=cfg.lambda_5,
        msg_seed=cfg.msg_seed,
        atm_chunk=cfg.atm_chunk,
        mol_chunk=cfg.mol_chunk,
        qgrid=q_grid,
        energy=energy,
        eps_embd=cfg.eps_embd,
        eps_msgp=cfg.eps_msgp,
        sigma_max=cfg.sigma_max,
        sigma_floor=cfg.sigma_floor,
        sigma_init_gain=cfg.sigma_init_gain,
        compile=cfg.compile,
    ).to(device)

    # Biases, norm/gain params, and PReLU/RFF-phase params are excluded from
    # weight decay: with decoupled_weight_decay=False (classic Adam+L2), decay
    # gets divided by each param's adaptive sqrt(exp_avg_sq) same as the loss
    # gradient, so any param with a small gradient-history norm (which these
    # low-dimensional gate/norm params always have) gets a disproportionately
    # large relative decay push. Once such a param starts shrinking, its
    # gradient history shrinks too, amplifying the decay further, a
    # self-reinforcing collapse to (subnormal) zero - this is exactly what
    # killed _msg._proj_agg/_rms_norm/_out._bilinear/_out._mlp.layer_0
    # by epoch 1 of the 2026-07-21 run. decoupled_weight_decay=True switches
    # to true AdamW-style decay (independent of the adaptive normalization),
    # and excluding these params from decay entirely is extra insurance
    # against the same class of parameter dying again.
    # `_mbd` also sits out weight decay. The embedding table is indexed, so
    # a vocab entry absent from the batch receives no gradient at all, but
    # decoupled decay applies every step regardless - the 2026-08-01 run had
    # 52 of 211 rows with exp_avg_sq == 0 (never once seen in the data)
    # shrinking monotonically toward zero. Decay is only meaningful for a
    # param the loss actually pushes back on.
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if (
            name.endswith(".bias")
            or "rms_norm" in name
            or "prelu" in name
            or "biasterm" in name
            or "_mbd" in name
        ):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.Adam(
        [
            {"params": decay_params, "weight_decay": cfg.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=cfg.lr,
        eps=cfg.adam_eps,
        decoupled_weight_decay=True,
    )

    # LR decay on validation plateau, stepped once per epoch with val loss.
    # Constructed here (before the resume block) so its state_dict can be
    # restored below alongside the optimizer. Unlike the previous
    # ExponentialLR, this schedule is path-dependent: `best` and
    # `num_bad_epochs` cannot be recomputed from the epoch number, so the
    # checkpoint's scheduler state is load-bearing, not an optimization.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg.lr_factor,
        patience=cfg.lr_patience,
        threshold=cfg.lr_threshold,
        threshold_mode="rel",
        min_lr=cfg.lr_min,
    )

    start_epoch = 1
    best_val = float("inf")
    # Delayed smoothing: off until smoothing_lr_cut_trigger LR cuts fail to
    # escape the plateau. target_lambda_7 preserves the configured value
    # while cfg.lambda_7 is forced to 0.0 during the pre-smoothing phase
    # (every read site - evaluate(), the training loop, save_epoch_plots -
    # just reads cfg.lambda_7 directly, so mutating it here is what makes
    # the switch take effect everywhere without threading a new parameter
    # through each call site).
    target_lambda_7 = cfg.lambda_7
    # CONSECUTIVE, not cumulative: a cut that gets followed by a new
    # best_val means the cut worked, so the streak resets to 0 (see the
    # "update best" block below). Without this, one early cut that
    # actually helped, plus one unrelated cut much later after a long
    # stretch of real progress, would count as "2 unproductive cuts" and
    # trigger smoothing on a run that was never actually stuck.
    lr_consecutive_fires = 0
    smoothing_triggered = False
    # Same "consecutive, resets on improvement" logic for the SAME kind of
    # stall once smoothing is on: smoothing_lr_cut_trigger more consecutive
    # unproductive cuts post-smoothing means the model is fully converged
    # (raw phase converged, then the smoothed phase converged too), which
    # is what ends the run and auto-kills the instance - see the
    # scheduler.step() block and _destroy_vast_instance.
    post_smoothing_lr_consecutive_fires = 0
    fully_converged = False
    # batches to skip in the first resumed epoch (exact mid-epoch resume)
    resume_skip = 0
    # (batch_idx, grad_norm) already recorded for the interrupted epoch,
    # carried across the resume so the distribution isn't fragmented by
    # restarts (this run checkpoints every ckpt_interval_sec).
    resume_grad_norms: list[tuple[int, float]] = []
    # which stage of start_epoch to resume into: None (fresh epoch, run
    # train → val → test as normal), or "val"/"test" to skip the stage(s)
    # that were already finished when the checkpoint was written.
    resume_phase: str | None = None
    resume_phase_skip = 0
    resume_eval_state: dict | None = None
    resume_train_loss = resume_train_r2 = None
    resume_val_loss = resume_val_r2 = None
    _PHASE_ORDER = ["train", "val", "test"]

    if cfg.resume:
        ckpt = torch.load(cfg.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        # ReduceLROnPlateau mutates param_groups["lr"] in place, so the true
        # current LR is captured in optimizer.state_dict() same as the
        # weights. The scheduler's OWN state (best, num_bad_epochs) is
        # path-dependent and restored separately just below - unlike the old
        # ExponentialLR, it cannot be recomputed from the epoch number.
        # The optimizer param-group layout has changed twice: a third group
        # for _out was added, then removed again. torch's error for a group
        # count mismatch is opaque, and the state is not remappable either,
        # since `state` is keyed by the param's index in the flattened group
        # order, so adding or removing a group shifts every index after it
        # and a "best effort" partial load would silently attach the wrong
        # Adam moments to the wrong tensors.
        n_ckpt = len(ckpt["optimizer"]["param_groups"])
        n_live = len(optimizer.param_groups)
        if n_ckpt != n_live:
            raise RuntimeError(
                f"optimizer param-group mismatch: checkpoint has {n_ckpt}, "
                f"this build has {n_live}. {cfg.resume} was written under a "
                "different param-group layout (OutputHead's separate group "
                "was added, then removed). Adam moments cannot be remapped "
                "across it. Either start a fresh run, or resume weights only "
                "by loading ckpt['model'] with cfg.resume unset."
            )
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        # "best_val" is the correct key going forward - it's the best
        # end-of-epoch validation loss seen so far, not a per-checkpoint
        # value (mid-epoch checkpoints don't run validation). Older
        # checkpoints wrote this same quantity under the misleading key
        # "val_loss", so fall back to that for backward compatibility.
        best_val = ckpt.get("best_val", ckpt.get("val_loss", float("inf")))
        # No fallback to the old cumulative "lr_cut_count"/
        # "post_smoothing_lr_cut_count" keys - those counted ANY cuts ever,
        # not consecutive-since-last-improvement, so an old value isn't a
        # valid starting point under the new semantics (it could easily be
        # stale-high, e.g. one cut that actually worked plus one unrelated
        # later cut). Starting at 0 on a checkpoint written before this
        # change is the correct behavior, not just a safe default: the
        # streak resets on any improvement, and every checkpoint that
        # predates this change happened while training was still (by
        # definition) making progress.
        lr_consecutive_fires = ckpt.get("lr_consecutive_fires", 0)
        smoothing_triggered = ckpt.get("smoothing_triggered", False)
        post_smoothing_lr_consecutive_fires = ckpt.get(
            "post_smoothing_lr_consecutive_fires", 0
        )
        saved_bi = ckpt.get("batch_idx", -1)
        # Older checkpoints (pre phase-machine) never wrote "phase" - they
        # only ever checkpointed mid-training, so "train" is the correct
        # default read for them too.
        saved_phase = ckpt.get("phase", "train")
        if saved_bi is not None and saved_bi >= 0:
            # mid-phase checkpoint → redo saved_phase this same epoch from
            # saved_bi+1. For "val"/"test" this also carries forward the
            # partial accumulator and any earlier phases' already-computed
            # scalar results, so those phases are not redone.
            start_epoch = ckpt["epoch"]
            resume_phase = saved_phase
            resume_phase_skip = saved_bi + 1
            resume_eval_state = ckpt.get("eval_state")
            resume_train_loss = ckpt.get("train_loss")
            resume_train_r2 = ckpt.get("train_r2")
            resume_val_loss = ckpt.get("val_loss")
            resume_val_r2 = ckpt.get("val_r2")
        else:
            # saved_phase fully completed as of this checkpoint.
            phase_i = _PHASE_ORDER.index(saved_phase)
            if phase_i == len(_PHASE_ORDER) - 1:  # "test" done → epoch done
                start_epoch = ckpt["epoch"] + 1
            else:
                start_epoch = ckpt["epoch"]
                resume_phase = _PHASE_ORDER[phase_i + 1]
                resume_train_loss = ckpt.get("train_loss")
                resume_train_r2 = ckpt.get("train_r2")
                resume_val_loss = ckpt.get("val_loss")
                resume_val_r2 = ckpt.get("val_r2")
        if resume_phase == "train":
            resume_skip = resume_phase_skip
            resume_phase = None  # training loop needs no phase-skip logic
        resume_grad_norms = list(ckpt.get("grad_norms", []))
        resumed_lr = optimizer.param_groups[0]["lr"]
        print(
            f"resumed from {cfg.resume} (epoch {ckpt['epoch']}, "
            f"phase {saved_phase}, batch_idx {saved_bi}, "
            f"best_val {best_val:.4f}, lr {resumed_lr:.4g})",
        )
        del ckpt

    # Delayed smoothing: off until smoothing_lr_cut_trigger LR cuts fail to
    # escape the plateau. Every read site (evaluate(), the training loop,
    # save_epoch_plots) just reads cfg.lambda_7 directly, so mutating it
    # here is what makes the switch take effect everywhere without
    # threading a new parameter through each call site. model.smoothing_
    # enabled is set the same way since it is not part of state_dict().
    if not smoothing_triggered:
        cfg.lambda_7 = 0.0
    model.smoothing_enabled = smoothing_triggered

    # profiler - diagnostic mode. Two decoupled layers:
    #
    #   * _LoopProfiler  - O(1)-memory wall-clock section timers. Runs the FULL
    # 1 + prof_warmup + prof_active window so the averages are
    # representative, and it costs ~nothing in RAM.
    #
    # * torch.profiler - kernel-level trace. This buffers every op (thousands
    # per
    # step here, given the checkpointed chunk loops) in HOST RAM and
    # materialises
    # them all at export, so it is memory-heavy and OOM-kills the process if
    # the
    # active window is long. It therefore samples only a few steady-state steps
    # (tb_active), and with_stack / profile_memory are OFF - with_stack stores
    # a
    #     full stack per event and is the main cause of the export RAM blow-up.
    #
    # The loop runs for the section-timer window; the torch trace stops
    # earlier.

    prof_warmup = cfg.prof_warmup
    prof_active = cfg.prof_active
    # The section timers profile the whole selected loader (both groups); break
    # on its
    # last batch. The memory-heavy torch trace still only samples the first few
    # steps.
    prof_stop_bi = (len(worst) - 1) if cfg.profiler else 0  # loop's last batch
    tb_active = min(
        prof_active, 3
    )  # keep the heavy torch trace to a few steps
    tb_stop_bi = prof_warmup + tb_active  # torch profiler's last step index

    loop_prof = _LoopProfiler(device, enabled=cfg.profiler)

    def _on_trace_ready(p: torch.profiler.profile):
        """Handle a completed torch.profiler trace: print a kernel table and
        save it.

        Parameters
        ----------
        p : torch.profiler.profile
            The profiler instance whose trace just completed.

        Returns
        -------
        None
        """
        # Print a kernel table straight to stdout - TensorBoard's inline
        # view hangs through Kaggle's proxy, but stdout always works.
        # Still write the trace file so it can be opened in Perfetto /
        # chrome://tracing if wanted.
        # This table is torch.profiler's own formatter (EventList.table),
        # not ours - we don't control its per-cell alignment, only these
        # call arguments. Its default max_name_column_width=55 plus 10
        # numeric columns makes each row ~180+ chars wide; a notebook
        # cell that soft-wraps long lines will make an otherwise-correct
        # fixed-width table look misaligned, since a wrapped continuation
        # line isn't column-aligned with the row below it. Shrinking the
        # name column keeps the whole table narrower so it's less likely
        # to wrap.
        print(
            f"\n[profiler] top GPU ops ({tb_active} active batch(es)):",
        )
        print(
            p.key_averages().table(
                sort_by="cuda_time_total",
                row_limit=25,
                max_name_column_width=30,
            ),
        )
        torch.profiler.tensorboard_trace_handler("./profiler_trace")(p)

    _prof = None
    if cfg.profiler:
        _prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=1, warmup=prof_warmup, active=tb_active, repeat=1
            ),
            on_trace_ready=_on_trace_ready,
            record_shapes=True,
            profile_memory=False,  # extra host RAM; not needed for speed
            with_stack=False,  # huge per-event RAM at export; keep off
        )
        _prof.start()
        print(
            "[profiler] started - section timers over "
            f"1+{prof_warmup}+{prof_active} batches, torch trace "
            f"over 1+{prof_warmup}+{tb_active}; "
            "traces -> ./profiler_trace/",
        )

    # training loop

    hparams = dict(
        lambda_1=cfg.lambda_1,
        lambda_2=cfg.lambda_2,
        lambda_3=cfg.lambda_3,
        lambda_4=cfg.lambda_4,
        lambda_5=cfg.lambda_5,
        msg_seed=cfg.msg_seed,
        q_points=q_points,
        eps_embd=cfg.eps_embd,
        eps_msgp=cfg.eps_msgp,
    )

    def _save_resume(
        ep: int,
        batch_idx: int,
        phase: str = "train",
        eval_state: dict | None = None,
        train_loss: float | None = None,
        train_r2: float | None = None,
        val_loss: float | None = None,
        val_r2: float | None = None,
    ):
        """Write the resume checkpoint (weights, optimizer, position) and
        push it off-box.

        Covers all three per-epoch stages, not just training: `phase`
        records which stage this checkpoint belongs to ("train", "val",
        "test"), so a resume can skip straight to wherever the run was
        actually interrupted instead of always redoing training. Val/test
        are thousands of batches long too (see `evaluate`/`save_epoch_plots`
        in Train/eval_plots.py), so losing a whole pass to a late timeout
        used to be as costly as losing the training tail - this closes that
        gap the same way the mid-epoch train checkpoint already does.

        Parameters
        ----------
        ep : int
            Current epoch number.
        batch_idx : int
            Batch index within `phase`; -1 marks `phase` as complete for
            this epoch (the phase-machine advances to the next phase, or to
            the next epoch if `phase` is "test"), >= 0 marks a mid-phase
            save at that batch.
        phase : str
            Which stage this checkpoint belongs to: "train", "val", or
            "test" (test also covers the diagnostic-plots pass).
        eval_state : dict or None
            Mid-phase accumulator state for `phase` in ("val", "test") -
            see `evaluate`'s `ckpt_cb` (val) or `save_epoch_plots`'s
            `ckpt_cb` (test). None when `phase` is "train", or when
            `batch_idx` is -1 (phase already folded into its scalar
            result below).
        train_loss, train_r2, val_loss, val_r2 : float or None
            This epoch's already-computed scalar results, carried forward
            so a resume into "val" or "test" phase doesn't need to redo
            earlier phases just to reconstruct the printed summary line
            and this epoch's metrics.json.

        Returns
        -------
        None
        """
        os.makedirs(cfg.ckpt_dir, exist_ok=True)
        batch_tag = "final" if batch_idx < 0 else str(batch_idx)
        # "train" keeps the original two-part name (pre-existing runs and
        # tooling already key off this format); only "val"/"test" - new
        # phases as of this checkpointing feature - get the phase segment.
        if phase == "train":
            ckpt_name = f"checkpoint_{ep}_{batch_tag}.pt"
        else:
            ckpt_name = f"checkpoint_{ep}_{phase}_{batch_tag}.pt"
        ckpt_path = os.path.join(cfg.ckpt_dir, ckpt_name)
        torch.save(
            {
                "epoch": ep,
                "batch_idx": batch_idx,
                "phase": phase,
                "eval_state": eval_state,
                "train_loss": train_loss,
                "train_r2": train_r2,
                "val_loss": val_loss,
                "val_r2": val_r2,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_val": best_val,
                "lr_consecutive_fires": lr_consecutive_fires,
                "smoothing_triggered": smoothing_triggered,
                "post_smoothing_lr_consecutive_fires": (
                    post_smoothing_lr_consecutive_fires
                ),
                "grad_norms": batch_grad_norms,
                "lr": optimizer.param_groups[0]["lr"],
                "q_grid": q_grid,
                "energy": energy,
                "hparams": hparams,
            },
            ckpt_path,
        )
        _rclone_push(ckpt_path, cfg.ckpt_rclone_dest, delete_after=True)

    # Dump the run's full config at the root of the data dir, so the per-epoch
    # metrics underneath it are self-describing. Written after the resume
    # block, so
    # a resumed run records the config it is actually continuing under.
    if cfg.data_dir:
        from Train.eval_plots import save_run_config_rtf

        save_run_config_rtf(cfg, cfg.data_dir)
        _rclone_push(cfg.data_dir, cfg.data_rclone_dest)

    _epoch_time_total = 0.0
    _epoch_count = 0
    # cfg.epochs is now an optional override cap, not the primary stopping
    # mechanism - default (None) is "run until fully converged" (raw phase
    # plateaus -> smoothing switches on -> smoothed phase plateaus -> stop
    # and auto-kill the instance, see the scheduler.step() block below and
    # _destroy_vast_instance). itertools.count gives the unbounded default;
    # the cap (if set) is checked as this iteration's break condition,
    # alongside fully_converged, at the end of the loop body.
    for epoch in _count(start_epoch):
        # Only the very first iteration of this loop can be a resume target -
        # every later epoch starts fresh regardless of what interrupted the
        # previous run.
        is_resumed_epoch = epoch == start_epoch
        skip_train_entirely = is_resumed_epoch and resume_phase in (
            "val",
            "test",
        )
        skip_val_entirely = is_resumed_epoch and resume_phase == "test"

        # Seed before iteration so train_loader shuffles identically across
        # resumes.
        torch.manual_seed(cfg.batcher_seed + epoch)

        model.train()
        train_loss_sum = 0.0
        train_mols = 0
        train_ss_res = 0.0
        train_sum_y = 0.0
        train_sum_y2 = 0.0
        train_n_elem = 0
        # (batch_idx, grad_norm) PRE-clip, for this epoch. Recorded because
        # grad_clip is only meaningful relative to the norms actually seen:
        # at grad_clip=1.0 against a measured |g| of 10-18 the clip fires on
        # every step, which is gradient normalisation rather than a safety
        # valve, and the clip factor then varies with molecule size (buckets
        # are size-sorted), systematically downweighting large molecules.
        # Set grad_clip off the p99 of this series.
        batch_grad_norms: list[tuple[int, float]] = (
            resume_grad_norms if is_resumed_epoch else []
        )
        resume_grad_norms = []

        torch.cuda.empty_cache()

        _t0 = _time.time()
        last_ckpt = _time.time()

        # On a mid-epoch resume, fast-forward over already-trained batches. The
        # per-epoch seed above makes the shuffle deterministic, so batch _bi
        # here
        # is the same molecule group as in the interrupted run. train_sampler
        # excludes the skipped indices outright (see _ResumableSampler), so
        # this costs nothing regardless of where in the epoch we resume.
        # skip_train_entirely means training already finished for this epoch
        # (the interrupted run got as far as val or test) - skipping every
        # index costs nothing for the same reason, and this epoch's actual
        # train_loss/train_r2 are carried forward from that earlier run's
        # checkpoint instead of being recomputed.
        skip = (
            len(train_set)
            if skip_train_entirely
            else (resume_skip if is_resumed_epoch else 0)
        )
        resume_skip = 0
        train_sampler.skip = skip

        for _bi, batch in enumerate(
            prof_loader if prof_loader is not None else train_loader,
            start=skip,
        ):
            if cfg.max_batches is not None and _bi >= cfg.max_batches:
                break

            rec: dict = {"bi": _bi}
            loop_prof.start_batch(rec)
            if cfg.profiler:
                # Bucket geometry, read off the still-on-CPU batch (no GPU
                # sync), logged next to per-batch timings.
                rec["n_mols"] = batch.vocab.shape[0]
                rec["max_atoms"] = batch.vocab.shape[1]
                rec["real_atoms"] = int(batch.padding_mask().sum().item())
                if _bi < prof_group_bounds[0]:
                    rec["group"] = "heavy_nm"
                elif _bi < prof_group_bounds[1]:
                    rec["group"] = "heavy_m"
                else:
                    rec["group"] = "regular"

            with loop_prof.section("h2d", rec):
                batch = dc_replace(
                    batch,
                    vocab=batch.vocab.to(device),
                    iqval=batch.iqval.to(device),
                    coord=batch.coord.to(device),
                )
            optimizer.zero_grad(set_to_none=True)

            with loop_prof.section("forward", rec):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    iq, coh, inc, fmags, sigmas = model(batch)
            with loop_prof.section("loss", rec):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = model.compute_loss(
                        iq,
                        coh,
                        inc,
                        fmags,
                        batch,
                        cfg.lambda_6,
                        cfg.lambda_7,
                    )

            if cfg.verbosity == "diagnostic":

                def _s(t: torch.Tensor):
                    """Format a tensor's NaN/Inf/min/max summary for debug
                    printing.

                    Parameters
                    ----------
                    t : torch.Tensor
                        Tensor to summarize.

                    Returns
                    -------
                    str
                        Summary string with nan/inf flags and min/max values.
                    """
                    return (
                        f"nan={t.isnan().any().item()} "
                        f"inf={t.isinf().any().item()} "
                        f"min={t.float().min().item()} "
                        f"max={t.float().max().item()}"
                    )

                if _bi < 10:
                    print(
                        f"  [debug] batch {_bi}  loss={loss.item()}  "
                        f"iq_nan={iq.isnan().any().item()}  "
                        f"iq_inf={iq.isinf().any().item()}",
                    )
                if loss.isnan() or loss.isinf():
                    print(f"  [NaN/Inf] batch {_bi}  loss={loss.item()}")
                    print(f"    iq:     {_s(iq)}")
                    print(f"    fmags:  {_s(fmags)}")
                    print(f"    sigmas: {_s(sigmas)}")
                    print(f"    iqval:  {_s(batch.iqval)}")
                    print(f"    coord:  {_s(batch.coord)}")
                    n_real = batch.padding_mask().sum().item()
                    print(
                        f"    vocab shape: {batch.vocab.shape}  "
                        f"n_real_atoms: {n_real}",
                    )

            with loop_prof.section("backward", rec):
                loss.backward()

            with loop_prof.section("clip", rec):
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.grad_clip
                )
            with loop_prof.section("step", rec):
                optimizer.step()

            # Grad-norm diagnostic, hoisted ABOVE the profiler branch so it
            # fires in
            # profiler mode too (that branch `continue`s past the rest of the
            # tail).
            if (
                cfg.verbosity == "diagnostic"
                and (_bi < 12 or not torch.isfinite(grad_norm))
            ):
                print(
                    f"  [grad] batch {_bi}  grad_norm={grad_norm.item()}",
                )

            if _prof is not None:
                _prof.step()
                if (
                    _bi >= tb_stop_bi
                ):  # stop the memory-heavy trace early; timers keep going
                    _prof.stop()
                    _prof = None
                    print(
                        "[profiler] torch trace written - see "
                        "./profiler_trace/",
                    )

            if cfg.profiler:
                rec["compute"] = sum(
                    rec.get(k, 0.0)
                    for k in (
                        "forward",
                        "loss",
                        "backward",
                        "clip",
                        "step",
                    )
                )
                del iq, coh, inc, fmags, sigmas, loss
                loop_prof.end_batch(rec)
                if _bi >= prof_stop_bi:  # loop's last profiled batch
                    loop_prof.report()
                    break
                continue  # skip the metrics/logging tail during profiling

            n = batch.iqval.shape[0]
            batch_loss = loss.item()
            train_loss_sum += batch_loss * n
            train_mols += n
            batch_grad_norms.append((_bi, grad_norm.item()))
            with torch.no_grad():
                log_pred = torch.log1p(iq)
                log_target = torch.log1p(batch.iqval)
                train_ss_res += ((log_pred - log_target) ** 2).sum().item()
                train_sum_y += log_target.sum().item()
                train_sum_y2 += (log_target**2).sum().item()
                train_n_elem += log_target.numel()

            del iq, coh, inc, fmags, sigmas, loss, log_pred, log_target

            if (
                cfg.verbosity in ("batch", "diagnostic")
                and (_bi + 1) % 20 == 0
            ):
                elapsed = _time.time() - _t0
                rate = (_bi + 1 - skip) / elapsed
                mem_pk = (
                    torch.cuda.max_memory_allocated() / 1e9
                )  # peak LIVE tensors (OOM-relevant)
                mem_rs = (
                    torch.cuda.memory_reserved() / 1e9
                )  # allocator cache (benign)
                print(
                    f"  ep {epoch}  batch {_bi + 1:5d}/{len(train_set)}"
                    f"  loss {train_loss_sum / max(train_mols, 1)}  "
                    f"{rate} batch/s  peak_alloc={mem_pk}G "
                    f"reserved={mem_rs}G",
                )
                torch.cuda.reset_peak_memory_stats()

            # Crash safety: every ckpt_interval_sec, save a mid-epoch resume
            # point and push it off-box.
            if (_time.time() - last_ckpt) > cfg.ckpt_interval_sec:
                _save_resume(epoch, _bi)
                last_ckpt = _time.time()
                if cfg.verbosity in ("batch", "diagnostic"):
                    print(
                        f"  [ckpt] saved mid-epoch resume @ epoch "
                        f"{epoch} batch {_bi}",
                    )

        if cfg.profiler:
            break  # diagnostic run: stop after profiling, no eval/checkpoint

        if skip_train_entirely:
            train_loss, train_r2 = resume_train_loss, resume_train_r2
        else:
            train_loss = train_loss_sum / train_mols
            train_ss_tot = train_sum_y2 - train_sum_y**2 / train_n_elem
            train_r2 = (
                1.0 - train_ss_res / train_ss_tot if train_ss_tot > 0 else 0.0
            )
            # Marks training done for this epoch, so a crash during val
            # or test resumes straight into that phase instead of
            # redoing the training loop.
            _save_resume(
                epoch,
                -1,
                phase="train",
                train_loss=train_loss,
                train_r2=train_r2,
            )

        if skip_val_entirely:
            val_loss, val_r2 = resume_val_loss, resume_val_r2
        else:
            val_start = (
                resume_phase_skip
                if (is_resumed_epoch and resume_phase == "val")
                else 0
            )
            val_state = (
                resume_eval_state
                if (is_resumed_epoch and resume_phase == "val")
                else None
            )
            val_sampler.skip = val_start
            val_loss, val_r2 = evaluate(
                val_loader,
                model,
                cfg,
                device,
                "val",
                start_batch=val_start,
                resume_state=val_state,
                ckpt_cb=(
                    lambda st, bi: _save_resume(
                        epoch,
                        bi,
                        phase="val",
                        eval_state=st,
                        train_loss=train_loss,
                        train_r2=train_r2,
                    )
                ),
            )
            torch.cuda.empty_cache()
            # Marks val done for this epoch, so a crash during test
            # resumes straight into test instead of redoing val.
            _save_resume(
                epoch,
                -1,
                phase="val",
                train_loss=train_loss,
                train_r2=train_r2,
                val_loss=val_loss,
                val_r2=val_r2,
            )

        # Test set is walked exactly once per epoch. When diagnostic plots are
        # on,
        # save_epoch_plots' single pass ALSO returns the test loss/R2, so we
        # skip the separate evaluate(test_loader) that used to double the test
        # cost. Baseline-style diagnostic plots (per-q R2, percent error,
        # Kratky
        # overlay, residual histogram, error-vs-atom-count) reuse
        # Baselines/run/metrics.py
        # so they stay directly comparable to Baselines/kaggle_baselines.ipynb.
        test_start = (
            resume_phase_skip
            if (is_resumed_epoch and resume_phase == "test")
            else 0
        )
        test_state = (
            resume_eval_state
            if (is_resumed_epoch and resume_phase == "test")
            else None
        )
        test_sampler.skip = test_start

        def _test_ckpt_cb(st: dict, bi: int) -> None:
            """Write a mid-test-phase resume checkpoint, carrying this
            epoch's already-known train/val results forward.
            """
            _save_resume(
                epoch,
                bi,
                phase="test",
                eval_state=st,
                train_loss=train_loss,
                train_r2=train_r2,
                val_loss=val_loss,
                val_r2=val_r2,
            )

        if cfg.data_dir:
            from Train.eval_plots import save_epoch_plots

            test_loss, test_r2 = save_epoch_plots(
                model,
                test_loader,
                q_grid,
                device,
                cfg.data_dir,
                epoch,
                compute_loss=True,
                lambda_6=cfg.lambda_6,
                lambda_7=cfg.lambda_7,
                verbose=cfg.verbosity in ("batch", "diagnostic"),
                start_batch=test_start,
                resume_state=test_state,
                ckpt_cb=_test_ckpt_cb,
                ckpt_interval_sec=cfg.ckpt_interval_sec,
            )
            torch.cuda.empty_cache()
        else:
            test_loss, test_r2 = evaluate(
                test_loader,
                model,
                cfg,
                device,
                "test",
                start_batch=test_start,
                resume_state=test_state,
                ckpt_cb=_test_ckpt_cb,
            )
            torch.cuda.empty_cache()

        _epoch_dt = _time.time() - _t0
        _epoch_time_total += _epoch_dt
        _epoch_count += 1
        avg_min = _epoch_time_total / _epoch_count / 60.0

        print(
            f"epoch {epoch:3d}"
            f"  train loss {train_loss:.4f}  r2 {train_r2:.4f}"
            f"  |  val loss {val_loss:.4f}  r2 {val_r2:.4f}"
            f"  |  test loss {test_loss:.4f}  r2 {test_r2:.4f}"
            f"  |  avg {avg_min:.1f} min/epoch"
        )

        # This epoch's numbers go in this epoch's own dir, one json per
        # epoch, and
        # the loss-vs-epoch curve is then drawn from the jsons READ BACK
        # OFF DISK
        # (not from an in-memory history list). That is what makes a resume
        # safe:
        # a resumed process has no memory of the epochs it did not run, so
        # a
        # single run-level metrics file would be rewritten with only the
        # post-resume epochs, silently truncating everything before it.
        # Concatenate
        # the per-epoch jsons after the run for the full history. Both
        # steps are
        # cheap (no forward passes), as is the off-box push (rclone copy is
        # incremental: earlier epochs' files are already uploaded and get
        # skipped).
        if cfg.data_dir:
            from Train.eval_plots import (
                load_epoch_history,
                save_epoch_loss_plot,
                save_epoch_metrics,
            )

            save_epoch_metrics(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_r2": train_r2,
                    "val_loss": val_loss,
                    "val_r2": val_r2,
                    "test_loss": test_loss,
                    "test_r2": test_r2,
                },
                cfg.data_dir,
                epoch,
            )
            save_epoch_loss_plot(
                load_epoch_history(cfg.data_dir), cfg.data_dir
            )
            _rclone_push(cfg.data_dir, cfg.data_rclone_dest)

        # update best BEFORE the resume save so the latter records the
        # current best_val. val_loss/val_r2 are always floats by this
        # point (either just computed, or carried forward from a
        # checkpoint written after val phase completed - never None).
        assert val_loss is not None
        if val_loss < best_val:
            best_val = val_loss
            # A cut that gets followed by real improvement worked - the
            # unproductive-cut streak breaks. Reset whichever counter is
            # currently active (the other is already 0, so this is a
            # no-op for it either way).
            lr_consecutive_fires = 0
            post_smoothing_lr_consecutive_fires = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "val_loss": val_loss,
                    "q_grid": q_grid,
                    "energy": energy,
                    "hparams": hparams,
                },
                cfg.ckpt_best,
            )
            _rclone_push(cfg.ckpt_best,cfg.ckpt_rclone_dest,delete_after=True)
            print(f"  saved best checkpoint (val {val_loss:.4f})")

        # phase="test", batch_idx=-1: test done -> whole epoch complete,
        # matching the phase-machine's terminal state (see the resume
        # block's _PHASE_ORDER walk).
        _save_resume(
            epoch,
            -1,
            phase="test",
            train_loss=train_loss,
            train_r2=train_r2,
            val_loss=val_loss,
            val_r2=val_r2,
        )

        # This epoch's resume-phase carryover is now fully consumed
        resume_phase = None
        resume_phase_skip = 0
        resume_eval_state = None
        resume_train_loss = resume_train_r2 = None
        resume_val_loss = resume_val_r2 = None

        # val_loss/val_r2 are always floats by this point (either just
        # computed, or carried forward from a checkpoint written after val
        # phase completed - never None).
        assert val_loss is not None
        prev_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr != prev_lr:
            print(f"  val loss plateaued, lr {prev_lr:.3g} -> {new_lr:.3g}")
            # lr_consecutive_fires/post_smoothing_lr_consecutive_fires are
            # NOT reset here on a cut - only a genuine improvement resets
            # them (see the "update best" block above). A cut simply
            # extends the current streak; whether that streak is actually
            # "consecutive and unproductive" is enforced by that reset,
            # not by anything in this block.
            if not smoothing_triggered:
                lr_consecutive_fires += 1
                if lr_consecutive_fires >= cfg.smoothing_lr_cut_trigger:
                    smoothing_triggered = True
                    model.smoothing_enabled = True
                    cfg.lambda_7 = target_lambda_7
                    for g in optimizer.param_groups:
                        g["lr"] = cfg.lr
                    scheduler._reset()  # clears best/num_bad_epochs/cooldown
                    print(
                        f"  {lr_consecutive_fires} consecutive lr cuts "
                        "without escaping the plateau - enabling "
                        f"smoothing (lambda_7={cfg.lambda_7:.3g}), lr "
                        f"reset to {cfg.lr:.3g}"
                    )
            else:
                # Same stall signature, now under smoothing: this many
                # consecutive cuts with no escape means the smoothed model
                # has itself converged, not just the raw one - training is
                # done.
                post_smoothing_lr_consecutive_fires += 1
                if (
                    post_smoothing_lr_consecutive_fires
                    >= cfg.smoothing_lr_cut_trigger
                ):
                    fully_converged = True
                    print(
                        f"  {post_smoothing_lr_consecutive_fires} "
                        "consecutive lr cuts post-smoothing without "
                        "escaping - model fully converged, ending run"
                    )

        if fully_converged:
            break
        if cfg.epochs is not None and epoch - start_epoch + 1 >= cfg.epochs:
            break

    if fully_converged:
        _destroy_vast_instance()


# entry point
def main(cfg: RunConfig | None = None):
    """Entry point for training.

    Pass a RunConfig directl or leave None to parse sys.argv.

    Parameters
    ----------
    cfg : RunConfig or None, optional
        Run configuration to use directly. If None, built by parsing
        CLI arguments and `load_config`.

    Returns
    -------
    None
    """

    if cfg is None:
        A = _parse_args()
        cfg = load_config(
            A.config,
            hdf5=A.hdf5,
            encodings_sqlite3_path=A.encodings_sqlite3_path,
            ckpt_best=A.ckpt_best,
            ckpt_dir=A.ckpt_dir,
            resume=A.resume,
            lambda_1=A.lambda_1,
            lambda_2=A.lambda_2,
            lambda_3=A.lambda_3,
            lambda_4=A.lambda_4,
            lambda_5=A.lambda_5,
            msg_seed=A.msg_seed,
            atm_chunk=A.atm_chunk,
            mol_chunk=A.mol_chunk,
            compile=A.compile,
            eps_embd=A.eps_embd,
            eps_msgp=A.eps_msgp,
            lambda_6=A.lambda_6,
            lambda_7=A.lambda_7,
            lr=A.lr,
            lr_factor=A.lr_factor,
            lr_patience=A.lr_patience,
            lr_threshold=A.lr_threshold,
            lr_min=A.lr_min,
            weight_decay=A.weight_decay,
            grad_clip=A.grad_clip,
            epochs=A.epochs,
            batcher_seed=A.batcher_seed,
            atom_size_ceil=A.atom_size_ceil,
            dataset_frac=A.dataset_frac,
            num_workers=A.num_workers,
            verbosity=A.verbosity,
            profiler=A.profiler,
            prof_warmup=A.prof_warmup,
            prof_active=A.prof_active,
            data_dir=A.data_dir,
        )

    assert cfg is not None

    _worker(cfg)


if __name__ == "__main__":
    main()
