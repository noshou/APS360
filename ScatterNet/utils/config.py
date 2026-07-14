from __future__  import annotations
from dataclasses import dataclass, field
from typing      import Optional

import yaml

# Default size buckets: each tuple is (min_atoms, max_atoms) for one batch group.
DEFAULT_BUCKETS: list[tuple[int, int]] = [
    (    1,    3),
    (    4,    6),
    (    7,   12),
    (   13,   14),
    (   15,   16),
    (   17,   17),
    (   18,   18),
    (   19,   19),
    (   20,   20),
    (   21,   21),
    (   22,   23),
    (   24,   26),
    (   27,   33),
    (   34,   40),
    (   41,   45),
    (   46,   50),
    (   51,   55),
    (   56,   60),
    (   61,   64),
    (   65,   69),
    (   70,   74),
    (   75,   80),
    (   81,   84),
    (   85,   90),
    (   91,   96),
    (   97,  102),
    (  103,  108),
    (  109,  116),
    (  117,  124),
    (  125,  132),
    (  133,  142),
    (  143,  152),
    (  153,  160),
    (  161,  170),
    (  171,  180),
    (  181,  192),
    (  193,  202),
    (  203,  216),
    (  217,  228),
    (  229,  242),
    (  243,  258),
    (  259,  276),
    (  277,  296),
    (  297,  316),
    (  317,  336),
    (  337,  364),
    (  365,  392),
    (  393,  428),
    (  429,  472),
    (  473,  524),
    (  525,  596),
    (  597,  696),
    (  697,  856),
    (  857, 1208),
    ( 1209, 3177),
    ( 3178, 4251),
    ( 4252, 6046)
    ]

@dataclass
class RunConfig:
    """
    Full configuration for a ScatterNet training run.

    Paths
    -----
    hdf5:                    Path to the raw HDF5 dataset containing I(q) curves and coordinates.
    encodings_sqlite3_path:  Path to the SQLite encoding database file. Built at this path
                             (via Preprocess) if it does not already exist.
    ckpt_best:      Where to save the checkpoint with the lowest validation loss.
    ckpt_resume:    Where to save the latest checkpoint for run resumption.
    resume:         Path to a checkpoint to resume from (None = train from scratch).

    Model
    -----
    lambda_1:       Atom embedding dimension. Larger = more expressive per-atom features.
    lambda_2:       Number of message passing rounds. Controls depth of neighbourhood aggregation.
    lambda_3:       OutputHead hidden width. Halved lambda_4 times before the final linear.
    lambda_4:       Number of halving steps in the OutputHead MLP (must satisfy 2^lambda_4 <= lambda_3).
    lambda_5:       Number of Random Fourier Features. Controls RBF kernel approximation quality.
    msg_seed:       RNG seed for the fixed RFF frequency matrix (reproducible across runs).
    atm_chunk:      Atoms processed per M-chunk in MessagePass and OutputHead. Controls peak size
                    of the RFF tensor (Nc, atm_chunk, Q, lambda_5) within one N-chunk.
    mol_chunk:      Molecules processed per N-chunk in MessagePass. Controls peak size of
                    chem_env (mol_chunk, Q, lambda_5, lambda_1); only one N-chunk's chem_env
                    exists at a time (freed after each N-chunk checkpoint recomputation).
    dp_atom_threshold: Training-only routing knob (only matters with >1 GPU). A batch's padded
                    atom count is M. If M < dp_atom_threshold AND the batch's molecule count
                    N >= 2*mol_chunk, it is routed through data-parallel splitting (molecules
                    divided across ranks, no in-model all-reduce) instead of the default
                    tensor-parallel atom sharding. Rationale: TP shards atoms across ranks and
                    all-reduces to reconstruct chem_env every round; for buckets with very few
                    atoms per molecule that all-reduce cost dwarfs the tiny amount of per-rank
                    compute it parallelises. The N >= 2*mol_chunk guard matters too: DP halves
                    the outer N-chunk loop but does NOT halve M before MessagePass's own
                    atm_chunk-loop runs (TP does, by sharding M first), so a DP-routed bucket
                    runs ~2x the inner M-chunk-loop launches TP would've had on the same
                    bucket - only worth it if halving N actually shrinks the outer loop. Without
                    this guard, buckets with small N but M just under the threshold get routed
                    to DP for zero benefit and a real launch-count cost, which is measurably
                    slower than plain TP (this bit a real run with dp_atom_threshold=5000).
                    0 (default) = always TP, matching pre-existing behaviour. Eval/test always
                    use TP regardless of this threshold (evaluate() relies on both ranks seeing
                    identical full-batch outputs).
    compile:        If True, torch.compile the checkpointed step functions in Embed, MessagePass,
                    OutputHead, and Loss (fullgraph=True, dynamic=True). False (default) runs them eager.
    amp:            If True, run forward/backward under fp16 autocast + GradScaler (CUDA only;
                    no-op on CPU). Halves activation memory and uses Turing/Ampere fp16 tensor
                    cores. MessagePass's RFF projection is kept in fp32 to avoid overflow.
                    False (default) trains in fp32.
    amp_init_scale: GradScaler starting loss scale when amp is on. Default 1024 (not torch's
                    65536): this model's activations sit near O(1) after mean-normalization, so
                    a lower start avoids the initial overflow/back-off thrash a 65536 scale
                    causes here. GradScaler still auto-grows/backs-off from this. Ignored when
                    amp is off.
    eps_embd:       Numerical floor in the Embed module (avoids division by zero).
    eps_msgp:       Numerical floor in MessagePass sigma clamping and aggregate denominator.

    Loss
    ----
    lambda_6:       Weight on the form-factor penalty term.
    lambda_7:       Weight on the sigma L2 penalty (prevents RFF bandwidth blowup). The
                    penalty is weighted by ~q^2 across the q-grid (normalised to mean 1, so
                    this value keeps the meaning it had under the old flat penalty), which
                    leaves the kernel range long where low q needs it. See ScatterNet/utils/loss.py.

    Training
    --------
    lr:             Adam learning rate.
    weight_decay:   Adam L2 weight decay coefficient.
    grad_clip:      Max gradient norm for gradient clipping (torch.nn.utils.clip_grad_norm_).
    epochs:         Number of full passes over all batches.
    batcher_seed:   RNG seed for the train/val/test molecule split (reproducible splits).
    atom_size_ceil: Maximum total atoms per batch; batches exceeding this are split via binary
                    search. -1 = auto (3x the largest molecule in the dataset).
    dataset_frac:   Fraction of the TRAIN split's batches to actually use, in (0.0, 1.0]. 1.0
                    (default) = use everything. At e.g. 0.1, only 10% of train's batches are
                    kept - the rest are dropped before the epoch loop ever sees them, not
                    skipped per-batch. val/test are always used at full size regardless of this
                    setting: val drives checkpoint selection and test feeds both the per-epoch
                    plots and any post-training comparison against
                    Baselines/kaggle_baselines.ipynb, and thinning either would make those
                    numbers noisier for no benefit - this knob only exists to cut training
                    compute. The kept train subset is fixed for the whole run (chosen once via
                    random.Random(batcher_seed)), so reruns with the same seed reproduce the
                    same subsample. For quick iteration/debugging, not for real training runs.
    num_workers:    DataLoader worker processes (0 = load in main process; use 0 for CPU debugging).
    max_batches:    Cap on batches per epoch (None = no limit; useful for quick sanity checks).
    ckpt_interval_sec: Seconds between mid-epoch resume-checkpoint saves. Crash safety: a session
                    timeout then costs at most this much work (resume picks up mid-epoch).
    ckpt_rclone_dest: rclone destination to copy checkpoints to after each save (e.g.
                    "gdrive:APS360/ckpts/"). None = no off-box copy. Needed on Kaggle, where
                    /kaggle/working is not reliably persisted through a session timeout.
    data_dir:       If set, write every per-epoch output the run produces under
                    `{data_dir}/epoch_{N:03d}/` (N zero-padded to 3 digits: epoch_001, ...):
                      - diagnostic plots on the test set (per-q R2, per-q percent error,
                        Kratky overlay, residual histogram, error-vs-atom-count, summary
                        bar chart). Reuses Baselines/metrics.py's evaluate()/run_all_plots()
                        unmodified (see Train/eval_plots.py), so these are directly comparable
                        to the plots Baselines/kaggle_baselines.ipynb produces for the
                        physics/learned baselines.
                      - metrics.json: that epoch's train/val/test loss and R2.
                      - loss_per_batch.png: this epoch's per-batch training loss.
                      - loss_per_epoch.png: train/val/test loss vs epoch, through this epoch
                        (rebuilt by reading every earlier epoch's metrics.json off disk).
                    Nothing is written outside an epoch dir except run_config.rtf (the run's
                    full RunConfig, dumped at `{data_dir}/run_config.rtf` when training starts),
                    so no epoch can overwrite another's numbers and a resumed run cannot
                    truncate the record of the epochs that preceded it. Concatenate the
                    per-epoch metrics.json files after the run for the whole history.
                    None (default) = disabled. The plots require the same LaTeX/JuliaMono
                    toolchain Baselines/kaggle_baselines.ipynb installs (matplotlib's "pgf"
                    backend via xelatex); everything else is cheap (written from data the loop
                    already collects, no extra forward passes).
    data_rclone_dest: rclone destination to copy the whole data_dir to after each epoch
                    (e.g. "gdrive:APS360/data/"). None = keep the data local only. The copy is
                    incremental (rclone skips files already uploaded), so re-pushing the dir
                    every epoch only sends the new epoch's files. Needed on Kaggle for the same
                    reason as ckpt_rclone_dest: /kaggle/working is not persisted on timeout.
    verbosity:      Logging level. "epoch" = one line per epoch only. "batch" = also print a
                    running-average loss every 50 batches. "diagnostic" = per-batch NaN/Inf check
                    with full tensor stats on the first 10 batches (use for debugging).
    profiler:       If True, run a short diagnostic instead of normal training: every rank
                    is wrapped in torch.profiler AND a lightweight per-section, per-rank
                    wall-clock timer (data-wait / H2D / forward / backward / grad-allreduce /
                    clip / step). The training loop stops after the profiling window and each
                    rank prints a section breakdown plus the heaviest batches (so rank-to-rank
                    skew and heavy buckets are visible). TensorBoard traces are written per
                    rank to ./profiler_trace/rank<r>/.
    prof_warmup:    Number of warmup batches in the profiler schedule (profiled but discarded,
                    so steady-state kernels are captured). Bump on Kaggle for a longer ramp.
    prof_active:    Number of active batches recorded by the lightweight section timers.
                    Larger = more representative stats (e.g. 50 to average over many buckets);
                    the loop runs 1 (wait) + prof_warmup + prof_active batches total. The
                    memory-heavy torch.profiler trace samples only min(prof_active, 3) of these
                    steps (a long torch-trace window exhausts host RAM and SIGKILLs the worker).

    Data
    ----
    buckets:        List of (min_atoms, max_atoms) size buckets to load. Molecules outside all
                    buckets are ignored. Override to restrict the atom-size range (e.g. smoke tests).
    """

    # paths
    hdf5:                    str   = "Preprocess/I(q)@L=50.h5"
    encodings_sqlite3_path:  str   = "Preprocess/scatternet-ENCODING.sqlite3"
    ckpt_best:      str            = "scatternet_best.pt"
    ckpt_resume:    str            = "scatternet_resume.pt"
    resume:         Optional[str]  = None

    # model
    lambda_1:       int            = 128
    lambda_2:       int            = 5
    lambda_3:       int            = 128
    lambda_4:       int            = 4
    lambda_5:       int            = 256
    msg_seed:       int            = 42
    atm_chunk:      int            = 1024
    mol_chunk:      int            = 32
    dp_atom_threshold: int         = 0         # 0 = always TP (old behaviour); see docstring above
    compile:        bool           = False     # torch.compile Embed/MessagePass/OutputHead/Loss's checkpointed step functions
    amp:            bool           = False     # fp16 autocast + GradScaler (CUDA only); RFF projection stays fp32
    amp_init_scale: float          = 1024.0    # GradScaler starting loss scale (amp only); lower than torch's 65536
    eps_embd:       float          = 1e-8
    eps_msgp:       float          = 1e-3

    # loss
    lambda_6:       float          = 0.1
    lambda_7:       float          = 0.1

    # training
    lr:             float          = 3e-4
    weight_decay:   float          = 1e-5
    grad_clip:      float          = 1.0
    epochs:         int            = 50
    batcher_seed:   int            = 0
    atom_size_ceil: int            = -1
    dataset_frac:   float          = 1.0       # fraction of TRAIN's batches to use, (0.0, 1.0]; deterministic subsample off batcher_seed; val/test always full
    num_workers:    int            = 4
    max_batches:    Optional[int]  = None
    ckpt_interval_sec: float       = 600.0     # mid-epoch checkpoint cadence (crash safety)
    ckpt_rclone_dest:  Optional[str] = None    # rclone dest for off-box checkpoint copies (None = off)
    data_dir:       Optional[str]  = None      # per-epoch metrics + baseline-style diagnostic plots dir (None = off)
    data_rclone_dest:  Optional[str] = None    # rclone dest for off-box copies of data_dir (None = off)
    verbosity:      str            = "epoch"   # "epoch" | "batch" | "diagnostic"
    profiler:       bool           = False     # diagnostic mode: per-rank torch.profiler + section timers; trace saved to ./profiler_trace/
    prof_warmup:    int            = 1         # profiler schedule warmup batches (profiled, discarded)
    prof_active:    int            = 3         # profiler schedule active batches (recorded); loop runs 1+warmup+active batches
    prof_molecules: Optional[list] = None      # hardcoded [(grp, stem), ...] fixture list; None = dynamic selection

    # data
    buckets: list[tuple[int, int]] = field(default_factory=lambda: DEFAULT_BUCKETS)


def load_config(config_path: Optional[str] = None, **cli_overrides) -> RunConfig:
    
    """Build a RunConfig by layering defaults, a YAML file, and CLI overrides.

    Precedence: dataclass defaults -> YAML file -> explicit CLI flags.

    Parameters
    ----------
    config_path : str, optional
        Path to a YAML file whose keys override the RunConfig defaults.
        None (default) skips this layer.
    **cli_overrides
        Explicit keyword overrides applied last (typically parsed from
        CLI flags). A value of None means "not provided on CLI" and is
        skipped.

    Returns
    -------
    RunConfig
        The fully resolved configuration.

    Raises
    ------
    ValueError
        If `config_path` or `cli_overrides` contains a key that is not a
        field of RunConfig.
    """

    cfg = RunConfig()

    if config_path is not None:
        with open(config_path) as f:
            data = yaml.safe_load(f) or {}
        for k, v in data.items():
            if not hasattr(cfg, k):
                raise ValueError(f"Unknown config key: {k!r}")
            setattr(cfg, k, v)

    for k, v in cli_overrides.items():
        if v is not None:           # None means "not provided on CLI"
            if not hasattr(cfg, k):
                raise ValueError(f"Unknown config key: {k!r}")
            setattr(cfg, k, v)

    return cfg
