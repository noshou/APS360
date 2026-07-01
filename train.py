import argparse
import h5py
import json
import os
import subprocess
import torch
import torch.distributed as dist

from contextlib import contextmanager
from time       import perf_counter

from torch.multiprocessing.spawn import spawn as mp_spawn  # type: ignore[attr-defined]
from dataclasses                 import replace as dc_replace
from torch.utils.data            import DataLoader, Subset
from Preprocess                  import Encoding
from ScatterNet                  import ScatterNet
from ScatterNet.batching         import Batcher
from ScatterNet.utils.config           import RunConfig, load_config
from ScatterNet.utils             import Loss

def _first(x):
    """collate_fn that unwraps the single-element list from batch_size=1."""
    return x[0]

def _rclone_push(path: str, dest: str | None):
    """Copy a checkpoint to a durable rclone remote (e.g. Drive) so it survives a
    Kaggle session timeout - /kaggle/working is not reliably persisted on a crash.
    No-op if dest is None or the file is missing; never raises into the train loop."""
    if not dest or not os.path.exists(path):
        return
    try:
        subprocess.run(
            ["rclone", "copy", path, dest], 
            check=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
    except Exception as e:
        print(f"  [rclone] push of {path} -> {dest} failed: {e}", flush=True)

# profiler
class _LoopProfiler:
    """Per-rank, per-section wall-clock profiler for the training loop.

    Active only when cfg.profiler is set (otherwise every method is a near-zero
    no-op, so the normal training path pays nothing). Each section is CUDA-synced
    at both boundaries so async GPU kernels are charged to the section that
    launched them instead of leaking into the next one - without this, "data_wait"
    silently absorbs the previous batch's still-running backward.

    Two things this is built to surface:
      * data-loader stalls vs GPU compute (is __getitem__ starving the GPU?), and
      * tensor-parallel rank skew - compare the same section across ranks; a rank
        that is fast in compute but slow in grad_allreduce is *waiting* on a
        slower peer (the cause of the NCCL ALLREDUCE timeout).
    A per-batch record is also kept so heavy buckets can be correlated with stalls,
    and per-batch peak CUDA memory is tracked (a free counter read - unlike
    torch.profiler's profile_memory, which OOMs) so you can see headroom for
    raising atm_chunk / mol_chunk.
    """

    def __init__(self, rank: int, device: str, enabled: bool):
        self.rank    = rank
        self.enabled = enabled
        self.device  = device
        self.cuda    = enabled and device.startswith("cuda")
        self.totals: dict[str, float] = {}
        self.records: list[dict]      = []
        self._prev_end: float | None  = None
        self.peak_alloc_gb = 0.0      # max over all batches of per-batch peak live tensors
        self.peak_resv_gb  = 0.0      # max allocator-reserved (cache); OOM-relevant ceiling
        self.peak_bi       = -1       # which batch hit peak_alloc_gb

    def _sync(self):
        if self.cuda:
            torch.cuda.synchronize()   # current device (set per-process via set_device)

    @contextmanager
    def section(self, name: str, rec: dict | None = None):
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
        """Charge time blocked in the DataLoader (since the previous batch ended)
        to 'data_wait'. Call at the very top of the loop body, before any .to()."""
        if not self.enabled:
            return
        now = perf_counter()
        dt  = 0.0 if self._prev_end is None else now - self._prev_end
        self.totals["data_wait"] = self.totals.get("data_wait", 0.0) + dt
        rec["data_wait"] = dt
        if self.cuda:
            # isolate this batch's peak so per-batch memory is attributable
            torch.cuda.reset_peak_memory_stats()

    def end_batch(self, rec: dict):
        if not self.enabled:
            return
        if self.cuda:
            pk = torch.cuda.max_memory_allocated() / 1e9   # peak LIVE tensors this batch (OOM driver)
            rv = torch.cuda.memory_reserved()      / 1e9   # allocator cache high-water
            rec["peak_alloc_gb"] = pk
            rec["reserved_gb"]   = rv
            if pk > self.peak_alloc_gb:
                self.peak_alloc_gb = pk
                self.peak_bi       = rec["bi"]
            self.peak_resv_gb = max(self.peak_resv_gb, rv)
        self.records.append(rec)
        self._prev_end = perf_counter()

    # section order for the printed report (others appended as seen)
    _ORDER = ["data_wait", "h2d", "forward", "loss", "backward",
              "grad_allreduce", "clip", "step"]

    def report(self):
        if not self.enabled or not self.records:
            return
        n      = len(self.records)
        total  = sum(self.totals.values())
        names  = self._ORDER + [k for k in self.totals if k not in self._ORDER]
        tag    = f"[prof r{self.rank}]"
        lines  = [
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
            peak = max((r["peak_alloc_gb"] for r in self.records if "peak_alloc_gb" in r), default=0.0)
            lines.append(
                f"{tag}   peak CUDA mem: {peak:.2f}G  "
                f"(reserved high-water {self.peak_resv_gb:.2f}G)"
            )
        # heaviest batches by data_wait and by compute, with bucket geometry + peak mem
        def _top(key, label):
            rows = sorted(self.records, key=lambda r: r.get(key, 0.0), reverse=True)[:3]
            for r in rows:
                mem = f" peak_alloc={r['peak_alloc_gb']:.2f}G" if "peak_alloc_gb" in r else ""
                lines.append(
                    f"{tag}   heavy-{label}: {r.get(key, 0.0) * 1e3:8.2f} ms  "
                    f"bi={r['bi']:<5d} mols={r['n_mols']:<5d} "
                    f"max_atoms={r['max_atoms']:<6d} real_atoms={r['real_atoms']}{mem}"
                )
        _top("data_wait", "data")
        _top("compute",   "compute")

        # Per-group breakdown: the combined numbers above blend three structurally
        # different bucket populations (compute-worst, memory-worst, median), so they
        # aren't comparable across profiler runs whose group sizes differ, and they're
        # not very informative even within one run. Report each group separately so
        # e.g. "did dp_atom_threshold help the heavy_nm group" is a like-for-like question.
        _GROUP_LABEL = {"heavy_nm": "heaviest N*M_shard", "heavy_m": "heaviest M", "regular": "median"}
        if any("group" in r for r in self.records):
            lines.append(f"{tag} ---- per-group breakdown ----")
            for g in ("heavy_nm", "heavy_m", "regular"):
                grp = [r for r in self.records if r.get("group") == g]
                if not grp:
                    continue
                gn      = len(grp)
                g_ms    = lambda k: sum(r.get(k, 0.0) for r in grp) / gn * 1e3
                g_peak  = max((r.get("peak_alloc_gb", 0.0) for r in grp), default=0.0)
                lines.append(
                    f"{tag}   {_GROUP_LABEL[g]:<20s} n={gn:<3d} "
                    f"compute={g_ms('compute'):8.2f}ms/batch  "
                    f"forward={g_ms('forward'):7.2f}ms  backward={g_ms('backward'):7.2f}ms  "
                    f"grad_allreduce={g_ms('grad_allreduce'):7.2f}ms  peak_alloc={g_peak:.2f}G"
                )

        print("\n".join(lines), flush=True)


# CLI
def _parse_args():
    """Parse CLI arguments. All model/training flags default to None so that
    load_config can distinguish 'not provided' from an explicit zero or false."""
    p = argparse.ArgumentParser(
        description="Train ScatterNet. All flags override --config values."
    )
    p.add_argument("--config",         default=None,  help="path to YAML run config")

    # paths
    p.add_argument("--hdf5",           default=None)
    p.add_argument("--db",             default=None)
    p.add_argument("--ckpt_best",      default=None)
    p.add_argument("--ckpt_resume",    default=None)
    p.add_argument("--metrics",        default=None)
    p.add_argument("--resume",         default=None,  help="path to resume checkpoint")

    # model
    p.add_argument("--lambda_1",       type=int,   default=None)
    p.add_argument("--lambda_2",       type=int,   default=None)
    p.add_argument("--lambda_3",       type=int,   default=None)
    p.add_argument("--lambda_4",       type=int,   default=None)
    p.add_argument("--lambda_5",       type=int,   default=None)
    p.add_argument("--msg_seed",       type=int,   default=None)
    p.add_argument("--atm_chunk",      type=int,   default=None)
    p.add_argument("--dp_atom_threshold", type=int, default=None)
    p.add_argument("--eps_embd",       type=float, default=None)
    p.add_argument("--eps_msgp",       type=float, default=None)

    # loss
    p.add_argument("--lambda_6",       type=float, default=None)
    p.add_argument("--lambda_7",       type=float, default=None)

    # training
    p.add_argument("--lr",             type=float, default=None)
    p.add_argument("--weight_decay",   type=float, default=None)
    p.add_argument("--grad_clip",      type=float, default=None)
    p.add_argument("--epochs",         type=int,   default=None)
    p.add_argument("--batcher_seed",   type=int,   default=None)
    p.add_argument("--atom_size_ceil", type=int,   default=None)
    p.add_argument("--num_workers",    type=int,   default=None)
    p.add_argument("--verbosity",      default=None, choices=["epoch", "batch", "diagnostic"])
    p.add_argument("--profiler",       action="store_const", const=True, default=None)
    p.add_argument("--prof_warmup",    type=int,   default=None)
    p.add_argument("--prof_active",    type=int,   default=None)

    return p.parse_args()

# eval helper
def evaluate(loader, model, criterion, cfg: RunConfig, device: str):
    """Run one pass over loader without gradients and return (mean_loss, R2).

    Both ranks call this with identical loaders (same seed, no shuffle), so
    the TP all_reduce/gather inside the model works correctly. Only rank 0
    uses the returned values for logging and checkpointing.
    """
    model.eval()
    total_loss = 0.0
    total_mols = 0
    ss_res = 0.0
    sum_y  = 0.0
    sum_y2 = 0.0
    n_elem = 0
    with torch.no_grad():
        for batch in loader:
            batch             = dc_replace(batch, vocab=batch.vocab.to(device), iqval=batch.iqval.to(device), coord=batch.coord.to(device))
            iq, fmags, sigmas, _, _ = model(batch)  # eval always TP-routes (model.eval()); local_batch==batch, loss_scale==1.0
            loss              = criterion.loss(iq, fmags, sigmas, batch, cfg.lambda_6, cfg.lambda_7)
            n = batch.iqval.shape[0]
            total_loss += loss.item() * n
            total_mols += n
            log_pred   = torch.log1p(iq)
            log_target = torch.log1p(batch.iqval)
            ss_res += ((log_pred - log_target) ** 2).sum().item()
            sum_y  += log_target.sum().item()
            sum_y2 += (log_target ** 2).sum().item()
            n_elem += log_target.numel()
            del iq, fmags, sigmas, loss, log_pred, log_target
    mean_loss = total_loss / total_mols
    ss_tot    = sum_y2 - sum_y ** 2 / n_elem
    r2        = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return mean_loss, r2

# distributed worker
def _worker(rank: int, cfg: RunConfig):
    """
    Per-process training worker. When launched with mp.spawn this runs on
    rank i with GPU cuda:i. When called directly (single GPU) rank is 0.

    Parallelism routing is handled inside ScatterNet.forward, per batch: by
    default atoms are sharded across ranks for tensor parallelism (TP) -
    MessagePass all_reduces between its two passes, and outputs are gathered
    before returning. If cfg.dp_atom_threshold > 0 and a training batch's
    padded atom count M falls below it, the batch is instead split by
    molecule across ranks (DP) with no in-model communication at all. This
    worker is responsible for syncing parameter gradients after each backward
    (all_reduce SUM), which DDP would otherwise handle; TP needs the full SUM
    (partial gradients per atom shard), and DP relies on ScatterNet.forward's
    loss_scale to make the same SUM reconstruct the correct global-mean
    gradient too - see the comment at the grad_allreduce call site below.
    """

    world_size = torch.cuda.device_count()
    is_dist    = world_size > 1

    if is_dist:
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "12355")
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    # Both ranks must see identical batch orderings each epoch.
    torch.manual_seed(cfg.batcher_seed)

    if rank == 0:
        print(f"world_size={world_size}  device={device}")
        print(f"lr={cfg.lr}  epochs={cfg.epochs}  λ1={cfg.lambda_1}  λ2={cfg.lambda_2}  λ3={cfg.lambda_3}  λ4={cfg.lambda_4}  λ5={cfg.lambda_5}  λ6={cfg.lambda_6}  λ7={cfg.lambda_7}")

    # data

    with h5py.File(cfg.hdf5, "r") as f:
        q_grid = torch.tensor(f["q_grid"][:]).float()   # type: ignore[index]
    energy   = 12_500.0
    q_points = len(q_grid)

    enc = Encoding(cfg.db, cfg.hdf5)

    batcher = Batcher(
        hdf5_db        = cfg.hdf5,
        enc            = enc,
        batches        = cfg.buckets,
        seed           = cfg.batcher_seed,
        atom_size_ceil = cfg.atom_size_ceil,
    )
    train_set, val_set, test_set = batcher.get_sets()

    pin = device != "cpu"
    pw  = cfg.num_workers > 0
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True,  collate_fn=_first, num_workers=cfg.num_workers, pin_memory=pin, persistent_workers=pw)
    val_loader   = DataLoader(val_set,   batch_size=1, shuffle=False, collate_fn=_first, num_workers=cfg.num_workers, pin_memory=pin, persistent_workers=pw)
    test_loader  = DataLoader(test_set,  batch_size=1, shuffle=False, collate_fn=_first, num_workers=cfg.num_workers, pin_memory=pin, persistent_workers=pw)

    # In profiler mode, pick which buckets to profile from metadata (atom counts live in
    # each row (grp, stem, atoms), so no tensors are loaded). A single N*M_shard ranking
    # misses two things: (1) memory-risk buckets with large M but small N (N*M_shard is a
    # compute-time proxy dominated by N, so a huge-M/small-N bucket can rank low on it even
    # though it has the biggest per-chunk activations), and (2) what a "typical" batch even
    # looks like (a pure worst-case sample only ever shows outliers). So the profiling budget
    # is split three ways:
    #   1. heaviest by N*M_shard  - compute-time worst case (usually huge-N/tiny-M; this is
    #      where TP's all-reduce cost piles up, see dp_atom_threshold).
    #   2. heaviest by raw M      - memory-risk worst case (large per-chunk RFF tensors),
    #      invisible to the N*M_shard ranking whenever N is small.
    #   3. a band around the median N*M_shard - representative "regular" batches, so the
    #      two worst-case groups have a baseline to compare against.
    # Worst-of-group-1 stays first (bi=0, the profiler's "wait" step) so it gets a clean
    # peak_alloc reading with no torch-trace overhead, matching the original single-ranking
    # behaviour for the compute-worst-case bucket.
    prof_loader = None
    worst: list = []
    prof_group_bounds = (0, 0)   # (end of heavy_nm, end of heavy_m) indices into `worst`
    if cfg.profiler:
        ws_proxy = max(world_size, 1)
        def _bucket_nm_proxy(i):
            rows = train_set._batches[i]
            return len(rows) * ((max(r[2] for r in rows) + ws_proxy - 1) // ws_proxy)
        def _bucket_max_m(i):
            return max(r[2] for r in train_set._batches[i])

        n_each  = 1 + cfg.prof_warmup + cfg.prof_active
        n_group = max(1, n_each // 3)

        by_nm = sorted(range(len(train_set)), key=_bucket_nm_proxy, reverse=True)
        by_m  = sorted(range(len(train_set)), key=_bucket_max_m,   reverse=True)

        heavy_nm = by_nm[:n_group]
        seen     = set(heavy_nm)
        heavy_m  = [i for i in by_m if i not in seen][:n_group]
        seen    |= set(heavy_m)

        n_regular  = max(0, n_each - len(heavy_nm) - len(heavy_m))
        mid        = len(by_nm) // 2
        band_start = max(0, mid - n_regular)
        regular    = [i for i in by_nm[band_start:] if i not in seen][:n_regular]

        worst = heavy_nm + heavy_m + regular
        prof_group_bounds = (len(heavy_nm), len(heavy_nm) + len(heavy_m))
        mode  = f"{len(heavy_nm)} heaviest N*M_shard + {len(heavy_m)} heaviest M + {len(regular)} median"
        prof_loader = DataLoader(Subset(train_set, worst), batch_size=1, shuffle=False,
                                 collate_fn=_first, num_workers=cfg.num_workers, pin_memory=pin)
        if rank == 0:
            def _describe(i):
                rows = train_set._batches[i]
                return f"{len(rows)} mols x {max(r[2] for r in rows)} atoms"
            print(f"[profiler] probing {mode}", flush=True)
            if heavy_nm:
                print(f"  worst N*M_shard≈{_bucket_nm_proxy(heavy_nm[0])} ({_describe(heavy_nm[0])})", flush=True)
            if heavy_m:
                print(f"  worst M          ({_describe(heavy_m[0])})", flush=True)
            if regular:
                print(f"  median sample    ({_describe(regular[0])})", flush=True)

    if rank == 0:
        train_mols = sum(len(b) for b in train_set._batches)
        val_mols   = sum(len(b) for b in val_set._batches)
        test_mols  = sum(len(b) for b in test_set._batches)
        print(
            f"batches/epoch: train={len(train_loader)}  val={len(val_loader)}  test={len(test_loader)}"
            f"  (buckets; same count by design - each bucket contributes one sub-batch per split)",
            flush=True,
        )
        print(f"molecules:     train={train_mols}  val={val_mols}  test={test_mols}", flush=True)

    # model

    model = ScatterNet(
        lambda_1  = cfg.lambda_1,
        lambda_2  = cfg.lambda_2,
        lambda_3  = cfg.lambda_3,
        lambda_4  = cfg.lambda_4,
        lambda_5  = cfg.lambda_5,
        msg_seed  = cfg.msg_seed,
        atm_chunk = cfg.atm_chunk,
        mol_chunk = cfg.mol_chunk,
        q_points  = q_points,
        eps_embd  = cfg.eps_embd,
        eps_msgp  = cfg.eps_msgp,
        dp_atom_threshold = cfg.dp_atom_threshold,
    ).to(device)

    criterion = Loss(q_grid, energy).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    start_epoch = 1
    history: list = []
    best_val = float("inf")
    resume_skip = 0      # batches to skip in the first resumed epoch (exact mid-epoch resume)

    if cfg.resume:
        ckpt = torch.load(cfg.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        best_val = ckpt.get("val_loss", float("inf"))
        saved_bi = ckpt.get("batch_idx", -1)
        if saved_bi is None or saved_bi < 0:        # checkpoint sat at an epoch boundary
            start_epoch = ckpt["epoch"] + 1
        else:                                        # mid-epoch checkpoint → redo this epoch from saved_bi+1
            start_epoch = ckpt["epoch"]
            resume_skip = saved_bi + 1
        if rank == 0:
            print(f"resumed from {cfg.resume} (epoch {ckpt['epoch']}, batch_idx {saved_bi}, val_loss {best_val:.4f})", flush=True)

    # profiler - diagnostic mode. Two decoupled layers, both on every rank:
    #
    #   * _LoopProfiler  - O(1)-memory wall-clock section timers. Runs the FULL
    #     1 + prof_warmup + prof_active window so per-rank averages are representative;
    #     this is what exposes tensor-parallel skew, and it costs ~nothing in RAM.
    #
    #   * torch.profiler - kernel-level trace. This buffers every op (thousands per
    #     step here, given the checkpointed chunk loops) in HOST RAM and materialises
    #     them all at export, so it is memory-heavy and OOM-kills the process if the
    #     active window is long. It therefore samples only a few steady-state steps
    #     (tb_active), and with_stack / profile_memory are OFF - with_stack stores a
    #     full stack per event and is the main cause of the export RAM blow-up.
    #
    # The loop runs for the section-timer window; the torch trace stops earlier.

    prof_warmup = cfg.prof_warmup
    prof_active = cfg.prof_active
    # The section timers profile the whole selected loader (both groups); break on its
    # last batch. The memory-heavy torch trace still only samples the first few steps.
    prof_stop_bi = (len(worst) - 1) if cfg.profiler else 0   # loop's last batch
    tb_active    = min(prof_active, 3)          # keep the heavy torch trace to a few steps
    tb_stop_bi   = prof_warmup + tb_active      # torch profiler's last step index

    loop_prof = _LoopProfiler(rank, device, enabled=cfg.profiler)

    def _on_trace_ready(p):
        # Print a kernel table straight to stdout - TensorBoard's inline view
        # hangs through Kaggle's proxy, but stdout always works. Still write the
        # trace file so it can be opened in Perfetto / chrome://tracing if wanted.
        if rank == 0:
            print(f"\n[profiler] top GPU ops (rank 0, {tb_active} active batch(es)):", flush=True)
            print(p.key_averages().table(sort_by="cuda_time_total", row_limit=25), flush=True)
        torch.profiler.tensorboard_trace_handler(f"./profiler_trace/rank{rank}")(p)

    _prof = None
    if cfg.profiler:
        _prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=1, warmup=prof_warmup, active=tb_active, repeat=1),
            on_trace_ready=_on_trace_ready,
            record_shapes=True,
            profile_memory=False,   # extra host RAM; not needed for a speed profile
            with_stack=False,       # huge per-event RAM at export → OOM kill; keep off
        )
        _prof.start()
        if rank == 0:
            print(
                f"[profiler] started on all ranks - section timers over 1+{prof_warmup}+{prof_active} "
                f"batches, torch trace over 1+{prof_warmup}+{tb_active}; traces -> ./profiler_trace/rank<r>/",
                flush=True,
            )

    # training loop

    hparams = dict(
        lambda_1=cfg.lambda_1, lambda_2=cfg.lambda_2,
        lambda_3=cfg.lambda_3, lambda_4=cfg.lambda_4,
        lambda_5=cfg.lambda_5, msg_seed=cfg.msg_seed,
        q_points=q_points,     eps_embd=cfg.eps_embd,
        eps_msgp=cfg.eps_msgp,
    )

    def _save_resume(ep, batch_idx):
        """Write the resume checkpoint (weights + optimizer + position) and push it
        off-box. batch_idx >= 0 marks a mid-epoch save; -1 marks an epoch boundary.
        Both TP ranks hold identical weights, so rank 0's state_dict is complete."""
        torch.save({
            "epoch":     ep,
            "batch_idx": batch_idx,
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_loss":  best_val,
            "q_grid":    q_grid,
            "energy":    energy,
            "hparams":   hparams,
        }, cfg.ckpt_resume)
        _rclone_push(cfg.ckpt_resume, cfg.ckpt_rclone_dest)

    for epoch in range(start_epoch, start_epoch + cfg.epochs):

        # Seed before iteration so all ranks shuffle train_loader identically.
        torch.manual_seed(cfg.batcher_seed + epoch)

        model.train()
        train_loss_sum = 0.0
        train_mols     = 0
        train_ss_res   = 0.0
        train_sum_y    = 0.0
        train_sum_y2   = 0.0
        train_n_elem   = 0

        torch.cuda.empty_cache()

        import time as _time
        _t0 = _time.time()
        last_ckpt = _time.time()

        # On a mid-epoch resume, fast-forward over already-trained batches. The
        # per-epoch seed above makes the shuffle deterministic, so batch _bi here
        # is the same molecule group as in the interrupted run.
        skip = resume_skip if epoch == start_epoch else 0
        resume_skip = 0

        for _bi, batch in enumerate(prof_loader if prof_loader is not None else train_loader):
            if _bi < skip:
                continue
            if cfg.max_batches is not None and _bi >= cfg.max_batches:
                break

            rec: dict = {"bi": _bi}
            loop_prof.start_batch(rec)
            if cfg.profiler:
                # Bucket geometry, read off the still-on-CPU batch (no GPU sync).
                # real_atoms drives the per-rank shard imbalance behind the
                # ALLREDUCE timeout, so it is logged next to per-batch timings.
                rec["n_mols"]     = batch.vocab.shape[0]
                rec["max_atoms"]  = batch.vocab.shape[1]
                rec["real_atoms"] = int(batch.padding_mask().sum().item())
                if _bi < prof_group_bounds[0]:
                    rec["group"] = "heavy_nm"
                elif _bi < prof_group_bounds[1]:
                    rec["group"] = "heavy_m"
                else:
                    rec["group"] = "regular"

            with loop_prof.section("h2d", rec):
                batch = dc_replace(batch, vocab=batch.vocab.to(device), iqval=batch.iqval.to(device), coord=batch.coord.to(device))
            optimizer.zero_grad(set_to_none=True)

            with loop_prof.section("forward", rec):
                iq, fmags, sigmas, local_batch, loss_scale = model(batch)
            with loop_prof.section("loss", rec):
                loss = criterion.loss(iq, fmags, sigmas, local_batch, cfg.lambda_6, cfg.lambda_7) * loss_scale

            if rank == 0 and cfg.verbosity == "diagnostic":
                def _s(t): return f"nan={t.isnan().any().item()} inf={t.isinf().any().item()} min={t.float().min().item()} max={t.float().max().item()}"
                if _bi < 10:
                    print(f"  [debug] batch {_bi}  loss={loss.item()}  iq_nan={iq.isnan().any().item()}  iq_inf={iq.isinf().any().item()}", flush=True)
                if loss.isnan() or loss.isinf():
                    print(f"  [NaN/Inf] batch {_bi}  loss={loss.item()}")
                    print(f"    iq:     {_s(iq)}")
                    print(f"    fmags:  {_s(fmags)}")
                    print(f"    sigmas: {_s(sigmas)}")
                    print(f"    iqval:  {_s(local_batch.iqval)}")
                    print(f"    coord:  {_s(local_batch.coord)}")
                    print(f"    vocab shape: {local_batch.vocab.shape}  n_real_atoms: {local_batch.padding_mask().sum().item()}", flush=True)

            with loop_prof.section("backward", rec):
                loss.backward()

            # With TP, each rank's parameter gradients are a partial sum over its
            # atom shard, so they must be summed across ranks (SUM, not average -
            # DDP averages, TP needs the full sum). With DP (small-M batches routed
            # by dp_atom_threshold), ScatterNet.forward's loss_scale = local_N/global_N
            # rescales each rank's local-mean loss before backward, so the same SUM
            # here reconstructs the correct global-mean gradient instead of double-
            # counting it - one all-reduce rule serves both routing modes. A rank
            # that is fast everywhere else but slow here is *waiting* on a laggard
            # peer - that's the skew to hunt with the per-rank section report.
            if is_dist:
                with loop_prof.section("grad_allreduce", rec):
                    for param in model.parameters():
                        if param.grad is not None:
                            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)

            with loop_prof.section("clip", rec):
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            with loop_prof.section("step", rec):
                optimizer.step()

            if _prof is not None:
                _prof.step()
                if _bi >= tb_stop_bi:   # stop the memory-heavy trace early; timers keep going
                    _prof.stop()
                    _prof = None
                    if rank == 0:
                        print("[profiler] torch trace written - see ./profiler_trace/rank<r>/", flush=True)

            if cfg.profiler:
                rec["compute"] = sum(
                    rec.get(k, 0.0)
                    for k in ("forward", "loss", "backward", "grad_allreduce", "clip", "step")
                )
                del iq, fmags, sigmas, loss
                loop_prof.end_batch(rec)
                if _bi >= prof_stop_bi:   # all ranks stop together to avoid NCCL deadlock
                    loop_prof.report()
                    break
                continue                  # skip the metrics/logging tail during profiling

            if rank == 0 and cfg.verbosity == "diagnostic" and (_bi < 12 or not torch.isfinite(grad_norm)):
                print(f"  [grad] batch {_bi}  grad_norm={grad_norm.item()}", flush=True)

            n = batch.iqval.shape[0]
            train_loss_sum += loss.item() * n
            train_mols     += n
            with torch.no_grad():
                log_pred      = torch.log1p(iq)
                log_target    = torch.log1p(batch.iqval)
                train_ss_res += ((log_pred - log_target) ** 2).sum().item()
                train_sum_y  += log_target.sum().item()
                train_sum_y2 += (log_target ** 2).sum().item()
                train_n_elem += log_target.numel()

            del iq, fmags, sigmas, loss, log_pred, log_target

            if rank == 0 and cfg.verbosity in ("batch", "diagnostic") and (_bi + 1) % 20 == 0:
                elapsed  = _time.time() - _t0
                rate     = (_bi + 1) / elapsed
                mem_pk   = torch.cuda.max_memory_allocated() / 1e9   # peak LIVE tensors (OOM-relevant)
                mem_rs   = torch.cuda.memory_reserved()    / 1e9     # allocator cache (benign)
                print(f"  ep {epoch}  batch {_bi+1:5d}/{len(train_loader)}  loss {train_loss_sum/max(train_mols,1)}  {rate} batch/s  peak_alloc={mem_pk}G reserved={mem_rs}G", flush=True)
                torch.cuda.reset_peak_memory_stats()

            # Crash safety: every ckpt_interval_sec, save a mid-epoch resume point
            # (rank 0 only; both ranks hold identical weights) and push it off-box.
            if rank == 0 and (_time.time() - last_ckpt) > cfg.ckpt_interval_sec:
                _save_resume(epoch, _bi)
                last_ckpt = _time.time()
                if cfg.verbosity in ("batch", "diagnostic"):
                    print(f"  [ckpt] saved mid-epoch resume @ epoch {epoch} batch {_bi}", flush=True)

        if cfg.profiler:
            break   # diagnostic run: stop after the profiling window, no eval/checkpoint

        train_loss   = train_loss_sum / train_mols
        train_ss_tot = train_sum_y2 - train_sum_y ** 2 / train_n_elem
        train_r2     = 1.0 - train_ss_res / train_ss_tot if train_ss_tot > 0 else 0.0

        val_loss,  val_r2  = evaluate(val_loader,  model, criterion, cfg, device)
        torch.cuda.empty_cache()
        test_loss, test_r2 = evaluate(test_loader, model, criterion, cfg, device)
        torch.cuda.empty_cache()

        if rank == 0:
            print(
                f"epoch {epoch:3d}"
                f"  train loss {train_loss:.4f}  r2 {train_r2:.4f}"
                f"  |  val loss {val_loss:.4f}  r2 {val_r2:.4f}"
                f"  |  test loss {test_loss:.4f}  r2 {test_r2:.4f}"
            )

            history.append({
                "epoch":      epoch,
                "train_loss": train_loss,
                "train_r2":   train_r2,
                "val_loss":   val_loss,
                "val_r2":     val_r2,
                "test_loss":  test_loss,
                "test_r2":    test_r2,
            })
            with open(cfg.metrics, "w") as fh:
                json.dump({"epochs": history}, fh, indent=2)
            _rclone_push(cfg.metrics, cfg.ckpt_rclone_dest)

            # update best BEFORE the resume save so the latter records the current best_val
            if val_loss < best_val:
                best_val = val_loss
                torch.save({
                    "epoch":    epoch,
                    "model":    model.state_dict(),
                    "val_loss": val_loss,
                    "q_grid":   q_grid,
                    "energy":   energy,
                    "hparams":  hparams,
                }, cfg.ckpt_best)
                _rclone_push(cfg.ckpt_best, cfg.ckpt_rclone_dest)
                print(f"  saved best checkpoint (val {val_loss:.4f})")

            _save_resume(epoch, -1)   # batch_idx=-1 marks the epoch as complete

    if is_dist:
        dist.destroy_process_group()

# entry point
def main(cfg: RunConfig | None = None):
    """
    Entry point. Pass a RunConfig directly (e.g. from a Kaggle notebook),
    or leave None to parse sys.argv.

    With multiple GPUs, launches one process per GPU via mp.spawn.
    Each process initialises NCCL and runs the full training loop with
    M-dimension tensor parallelism.
    """

    if cfg is None:
        A = _parse_args()
        cfg = load_config(
            A.config,
            hdf5           = A.hdf5,
            db             = A.db,
            ckpt_best      = A.ckpt_best,
            ckpt_resume    = A.ckpt_resume,
            metrics        = A.metrics,
            resume         = A.resume,
            lambda_1       = A.lambda_1,
            lambda_2       = A.lambda_2,
            lambda_3       = A.lambda_3,
            lambda_4       = A.lambda_4,
            lambda_5       = A.lambda_5,
            msg_seed       = A.msg_seed,
            atm_chunk      = A.atm_chunk,
            dp_atom_threshold = A.dp_atom_threshold,
            eps_embd       = A.eps_embd,
            eps_msgp       = A.eps_msgp,
            lambda_6       = A.lambda_6,
            lambda_7       = A.lambda_7,
            lr             = A.lr,
            weight_decay   = A.weight_decay,
            grad_clip      = A.grad_clip,
            epochs         = A.epochs,
            batcher_seed   = A.batcher_seed,
            atom_size_ceil = A.atom_size_ceil,
            num_workers    = A.num_workers,
            verbosity      = A.verbosity,
            profiler       = A.profiler,
            prof_warmup    = A.prof_warmup,
            prof_active    = A.prof_active,
        )

    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        mp_spawn(_worker, args=(cfg,), nprocs=n_gpus, join=True)
    else:
        _worker(0, cfg)

if __name__ == "__main__":
    main()
