from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import yaml

# Default size buckets:
# each tuple is (min_atoms, max_atoms) for one batch group.
DEFAULT_BUCKETS: list[tuple[int, int]] = [
    (1, 3),
    (4, 6),
    (7, 12),
    (13, 14),
    (15, 16),
    (17, 17),
    (18, 18),
    (19, 19),
    (20, 20),
    (21, 21),
    (22, 23),
    (24, 26),
    (27, 33),
    (34, 40),
    (41, 45),
    (46, 50),
    (51, 55),
    (56, 60),
    (61, 64),
    (65, 69),
    (70, 74),
    (75, 80),
    (81, 84),
    (85, 90),
    (91, 96),
    (97, 102),
    (103, 108),
    (109, 116),
    (117, 124),
    (125, 132),
    (133, 142),
    (143, 152),
    (153, 160),
    (161, 170),
    (171, 180),
    (181, 192),
    (193, 202),
    (203, 216),
    (217, 228),
    (229, 242),
    (243, 258),
    (259, 276),
    (277, 296),
    (297, 316),
    (317, 336),
    (337, 364),
    (365, 392),
    (393, 428),
    (429, 472),
    (473, 524),
    (525, 596),
    (597, 696),
    (697, 856),
    (857, 1208),
    (1209, 3177),
    (3178, 4251),
    (4252, 6046),
]


@dataclass
class RunConfig:
    """
    Full configuration for a ScatterNet training run.

    Attributes
    ----------
    Paths
    hdf5 : str
        Path to the raw HDF5 dataset containing I(q) curves and
        coordinates.
    encodings_sqlite3_path : str
        Path to the SQLite encoding database file. Built at this path
        (via Preprocess) if file does not exist.
    ckpt_best : str
        Checkpoint w/ the lowest raw validation loss.
    ckpt_best_ema : str
        Checkpoint w/ the lowest EMA-smoothed validation loss (see
        lr_ema_alpha); rollback source when smoothing triggers.
    ckpt_dir : str
        Directory to save numbered resume checkpoints into. Each
        mid-phase/phase-boundary save writes its own file
        (checkpoint_<epoch>_<phase>_<batch>.pt, with "final" for phase
        boundaries; phase is "train"/"val"/"test") instead of overwriting
        a single fixed path. See README's Checkpoint and Resume section.
    resume : str or None
        Path to a checkpoint to resume (None = train from scratch).

    Model
    lambda_1 : int
        Atom embedding dimension.
    lambda_2 : int
        Number of message passing rounds.
    lambda_3 : int
        OutputHead width. Halved lambda_4 times before final layer.
    lambda_4 : int
        Number of halving steps in the OutputHead MLP. Must satisfy
        2^lambda_4 <= lambda_3.
    lambda_5 : int
        Number of Random Fourier Features. Controls RBF kernel
        approximation quality.
    msg_seed : int
        RNG seed for the fixed RFF frequency matrix (reproducible
        across runs).
    atm_chunk : int
        Atoms processed per M-chunk in MessagePass and OutputHead.
        Controls peak size of the RFF tensor
        (Nc, atm_chunk, Q, lambda_5) within one N-chunk.
    mol_chunk : int
        Molecules processed per N-chunk in MessagePass. Controls peak
        size of chem_env (mol_chunk, Q, lambda_5, lambda_1); only one
        N-chunk's chem_env exists at a time (freed after each N-chunk
        checkpoint recomputation).
    compile : bool
        If True, torch.compile the checkpointed step functions in
        Embed, MessagePass, OutputHead, and Loss (fullgraph=True,
        dynamic=True). True (default) compiles them; set False to run
        eager.
    eps_embd : float
        Numerical floor in the Embed module (avoids division by
        zero).
    sigma_max : float
        Cap on Embed's 1/q sigma envelope, in Angstroms. Default 100.0.
    sigma_floor : float
        Lower clamp on sigma, in Angstroms (Embed). Default 0.5. Replaces the
        1e-3 floor `eps_msgp` applied at message_pass.py:426.
    eps_msgp : float
        Numerical floor in MessagePass sigma clamping and aggregate
        denominator.

    Loss
    lambda_6 : float
        Weight on the form-factor penalty term. See
        ScatterNet/scatter_net.py's ScatterNet.compute_loss.
    lambda_7 : float
        FINAL/target weight on the 2nd-derivative roughness penalty
        term, applied to the pre-`SplineSmooth` coherent/incoherent
        curves. Ramps from 0 to this value over smoothing_n_stages
        plateaus - see smoothing_lr_cut_trigger, smoothing_n_stages, and
        ScatterNet/scatter_net.py's ScatterNet.compute_loss.

    Training
    lr : float
        Adam learning rate (this is the starting value; see lr_factor).
    lr_factor : float
        Multiplier applied to the LR when validation loss plateaus
        (ReduceLROnPlateau, stepped once per epoch on val loss). 1.0
        disables decay.
    lr_patience : int
        Epochs of no improvement tolerated before the LR is cut.
    lr_threshold : float
        Relative improvement required to count as progress: an epoch
        counts as better only if val_loss < best * (1 - lr_threshold).
    lr_min : float
        Floor the LR is never reduced below.
    lr_ema_alpha : float
        Smoothing factor for the EMA of val_loss fed to
        scheduler.step(), so a single noisy-but-better epoch can't reset
        ReduceLROnPlateau's num_bad_epochs on its own. Does not affect
        best_val/scatternet_best.pt, which use raw val_loss.
    smoothing_lr_cut_trigger : int
        Number of CONSECUTIVE LR cuts (each one followed by no new
        best_val - a genuine improvement resets the streak) before each
        lambda_7 ramp step fires. See Train/train.py's scheduler.step()
        block.
    smoothing_n_stages : int
        Number of equal steps from lambda_7=0 up to its target value,
        one per smoothing_lr_cut_trigger-plateau. Once lambda_7 is at
        target and another plateau fires, the run is fully converged.

        Plateau decay rather than a horizon schedule because cfg.epochs is
        the epoch count for THIS invocation, not the run's total horizon (a
        resume runs cfg.epochs MORE epochs from wherever it left off), so
        cosine or step-to-a-target has no endpoint to key off. Unlike the
        previous ExponentialLR, the schedule is path-dependent and CANNOT be
        reconstructed from the epoch number, so a crash-resume (this run
        restarts often; see ckpt_interval_sec) relies on the scheduler
        state_dict saved in the checkpoint, which carries `best`,
        `num_bad_epochs`, and `cooldown_counter`.
    weight_decay : float
        Adam L2 weight decay coefficient. Applied to neither `_mbd`,
        biases, norm/gain params, nor the RFF phases; see the param-group
        split in Train/train.py.
    grad_clip_ema_alpha : float
        Smoothing factor for the EMA of raw (pre-clip) grad norms used
        as the adaptive clip threshold's baseline.
    grad_clip_multiplier : float
        Clip threshold = grad_clip_multiplier * EMA of recent grad norms.
    adam_eps : float
        Adam's denominator epsilon. Default 1e-12 rather than torch's
        1e-8, so parameters whose gradient RMS is small are not frozen by
        the epsilon dominating sqrt(v) in the update.
    epochs : int or None
        Hard cap on full passes over all batches THIS invocation. None
        (default) runs until fully converged instead - see
        smoothing_lr_cut_trigger and Train/train.py's
        _destroy_vast_instance.
    batcher_seed : int
        RNG seed for the train/val/test molecule split (reproducible
        splits).
    atom_size_ceil : int
        Maximum total atoms per batch; batches exceeding this are
        split via binary search. -1 = auto (3x the largest molecule in
        the dataset).
    dataset_frac : float
        Fraction of the TRAIN split's batches to actually use, in
        (0.0, 1.0]. 1.0 (default) = use everything.
    num_workers : int
        DataLoader worker processes (0 = load in main process; use 0
        for CPU debugging).
    max_batches : int or None
        Cap on batches per epoch (None = no limit; useful for quick
        sanity checks).
    ckpt_interval_sec : float
        Seconds between mid-epoch resume-checkpoint saves. Crash
        safety: a session timeout then costs at most this much work
        (resume picks up mid-epoch).
    ckpt_rclone_dest : str or None
        rclone destination to copy checkpoints to after each save
        (e.g. "gdrive:APS360/ckpts/"). None = no off-box copy. Needed
        on Kaggle, where /kaggle/working is not reliably persisted
        through a session timeout.
    data_dir : str or None
        If set, write every per-epoch output the run produces under
        `{data_dir}/epoch_{N:03d}/` (N zero-padded to 3 digits:
        epoch_001, ...):
    data_rclone_dest : str or None
        rclone destination to copy the whole data_dir to after each
        epoch. None = keep the data local only.
    verbosity : str
        Logging level. "epoch" = one line per epoch only. "batch" =
        also print a running-average loss every 50 batches.
        "diagnostic" = per-batch NaN/Inf check with full tensor stats
        on the first 10 batches (use for debugging).
    profiler : bool
        If True, run a short diagnostic instead of normal training:
        every rank is wrapped in torch.profiler AND a lightweight
        per-section, per-rank wall-clock timer (data-wait / H2D /
        forward / backward / grad-allreduce / clip / step). The
        training loop stops after the profiling window and each rank
        prints a section breakdown plus the heaviest batches (so
        rank-to-rank skew and heavy buckets are visible). TensorBoard
        traces are written per rank to ./profiler_trace/rank<r>/.
    prof_warmup : int
        Number of warmup batches in the profiler schedule (profiled
        but discarded, so steady-state kernels are captured). Bump on
        Kaggle for a longer ramp.
    prof_active : int
        Number of active batches recorded by the lightweight section
        timers. Larger = more representative stats (e.g. 50 to average
        over many buckets); the loop runs 1 (wait) + prof_warmup +
        prof_active batches total. The memory-heavy torch.profiler
        trace samples only min(prof_active, 3) of these steps (a long
        torch-trace window exhausts host RAM and SIGKILLs the worker).
    prof_molecules : list or None
        Hardcoded [(grp, stem), ...] fixture list for the profiler
        run; None = dynamic selection.

    Data
    buckets : list of tuple of int
        List of (min_atoms, max_atoms) size buckets to load. Molecules
        outside all buckets are ignored. Override to restrict the
        atom-size range (e.g. smoke tests).
    """

    # paths
    hdf5: str = "Preprocess/I(q)@L=50.h5"
    encodings_sqlite3_path: str = "Preprocess/scatternet-ENCODING.sqlite3"
    ckpt_best: str = "scatternet_best.pt"  # best raw val_loss
    ckpt_best_ema: str = "scatternet_best_ema.pt"  # best EMA; rollback src
    ckpt_dir: str = "checkpoints"
    resume: Optional[str] = None

    # model
    lambda_1: int = 128
    lambda_2: int = 4
    lambda_3: int = 256
    lambda_4: int = 4
    lambda_5: int = 128
    msg_seed: int = 42
    atm_chunk: int = 512
    mol_chunk: int = 256
    compile: bool = True  # torch.compile checkpointed step functions
    eps_embd: float = 1e-8
    eps_msgp: float = 1e-3
    sigma_max: float = 100.0
    sigma_floor: float = 0.5
    # 1.0, not 0.1: NoTrilinBilin now inits from the true fan-in in1*in2
    # rather than in1, which already divides the bound by sqrt(Q) = 7.14.
    sigma_init_gain: float = 1.0

    # loss
    lambda_6: float = 1.0
    # SplineSmooth always runs (LambdaHead needs live gradients from epoch
    # 0 to learn Λ, see ScatterNet.forward). This lambda_7 is the
    # FINAL/target roughness-penalty weight, not the starting one: it
    # ramps from 0 up to this value over smoothing_n_stages plateaus (see
    # smoothing_lr_cut_trigger, smoothing_n_stages below). Train/train.py's
    # lambda_7_stage tracks ramp progress.
    lambda_7: float = 0.25

    # training
    lr: float = 1.3e-4
    # ReduceLROnPlateau on val loss. Replaces ExponentialLR(gamma): an
    # unconditional per-epoch decay assumes an epoch is a fixed amount of
    # work, but dataset_frac scales its step count (at frac=0.05 an epoch is
    # ~1/20 the steps, so gamma=0.7/epoch decayed ~20x faster per step and
    # reached lr=5.2e-6 by epoch 11 whether or not the model had stopped
    # improving). Plateau decay keys off measured progress instead, so it
    # never decays while the model is still improving, and needs no known
    # horizon (cfg.epochs is a per-invocation count, not the run's total).
    # It is NOT fully scale-free though: lr_patience and lr_threshold are
    # counted in epochs, so a small dataset_frac shortens each epoch, puts
    # less progress in it, and makes a fixed relative threshold harder to
    # clear -> earlier cuts in step-terms. Loosen them when frac is small.
    lr_factor: float = 0.5
    lr_patience: int = 4
    lr_threshold: float = 1e-3  # relative; val loss must beat best by this
    lr_min: float = 1e-6
    # ReduceLROnPlateau.step() compares a SINGLE epoch's value against the
    # all-time best (torch source: `if is_better(current, best): best =
    # current; num_bad_epochs = 0` - no windowing/aggregation at all). At
    # dataset_frac=0.05, val loss is noisy enough that one barely-better
    # epoch can reset num_bad_epochs to 0 indefinitely even while the
    # aggregate trend across any patience-length window is flat or
    # worsening. scheduler.step() is fed an EMA of val_loss instead of the
    # raw value (this config knob is its smoothing factor - higher reacts
    # faster to genuine trend changes, lower is more resistant to
    # single-epoch noise), so a plateau decision reflects the trend, not
    # one lucky epoch. best_val/scatternet_best.pt (which checkpoint is
    # truly the best model) still use RAW val_loss, not the EMA - that is
    # a different question (which weights to keep) from whether the
    # trend has genuinely stalled (whether to cut LR).
    lr_ema_alpha: float = 0.3
    weight_decay: float = 0.1
    # No fixed grad_clip: per-epoch p99 grad norm varied 68x across this
    # run, so any static threshold was either always-firing or useless.
    # Clip at grad_clip_multiplier x an EMA of recent raw grad norms
    # instead - tracks the shifting distribution automatically.
    grad_clip_ema_alpha: float = 0.02
    grad_clip_multiplier: float = 4.0
    # Adam eps. 1e-12, not torch's 1e-8: the update is g/(sqrt(v) + eps), so
    # any parameter whose gradient RMS falls below eps has its step scaled by
    # g/eps instead of ~1 and is effectively frozen. Measured on the previous
    # tree, the sigma pathway ran at a per-element grad RMS of 9.4e-13 while
    # the form-factor heads ran at ~1e0, which froze 59% of the parameters.
    # The init fixes removed most of that imbalance; this is the insurance.
    adam_eps: float = 1e-12
    # Each ramp step (see lambda_7 above) fires after this many
    # CONSECUTIVE LR cuts fail to escape the plateau - a cut followed by
    # a genuine new best_val resets
    # the streak, so this is not a cumulative "N cuts ever" count, it is
    # "N cuts in a row with no real progress between them", i.e. more
    # learning rate genuinely isn't fixing it. At that point LR resets to
    # `lr`, weights/optimizer roll back to the best EMA checkpoint, and
    # lambda_7 steps up by target_lambda_7/smoothing_n_stages. See
    # Train/train.py's scheduler.step() block.
    smoothing_lr_cut_trigger: int = 2
    # Number of equal steps from lambda_7=0 up to its target value (one
    # step per smoothing_lr_cut_trigger-plateau). Once lambda_7 is at
    # target AND another plateau fires, the run is fully converged.
    smoothing_n_stages: int = 4
    # None (default): run until fully converged - lambda_7 ramps to its
    # target over smoothing_n_stages plateaus, then one more plateau at
    # full lambda_7 ends the run and auto-destroys the vast.ai instance
    # (see Train/train.py's _destroy_vast_instance). Set an int to
    # override with a hard cap on epochs run THIS invocation instead (a
    # resume still runs that many MORE from wherever it left off) - the
    # run stops at the cap regardless of convergence state, and does NOT
    # auto-destroy the instance (a manual cap implies you want to inspect
    # the result).
    epochs: Optional[int] = None
    batcher_seed: int = 0
    atom_size_ceil: int = -1
    dataset_frac: float = 1.0  # fraction of TRAIN's batches to use, (0.0, 1.0]
    num_workers: int = 4
    max_batches: Optional[int] = None
    ckpt_interval_sec: float = (
        600.0  # mid-epoch checkpoint cadence (crash safety)
    )
    ckpt_rclone_dest: Optional[str] = (
        None  # rclone dest for off-box checkpoint copies (None = off)
    )
    data_dir: Optional[str] = (
        None  # per-epoch metrics diagnostic plots dir (None = off)
    )
    data_rclone_dest: Optional[str] = (
        None  # rclone dest for off-box copies of data_dir (None = off)
    )
    verbosity: str = "epoch"  # "epoch" | "batch" | "diagnostic"
    profiler: bool = False  # diagnostic mode:
                            # per-rank torch.profiler/section timers;
                            # trace saved to ./profiler_trace/
    prof_warmup: int = (
        1  # profiler schedule warmup batches (profiled, discarded)
    )
    prof_active: int = 3    # profiler schedule active batches (recorded);
                            # loop runs 1+warmup+active batches
    prof_molecules: Optional[list] = (
        None  # hardcoded [(grp, stem), ...] list; None = dynamic selection
    )

    # data
    buckets: list[tuple[int, int]] = field(
        default_factory=lambda: DEFAULT_BUCKETS
    )

    def __post_init__(self) -> None:
        # dataclasses don't enforce field types at construction time, so an int
        # literal silently sails through here and only blows up later,
        # deep into a training run.
        for name in (
            "eps_embd",
            "eps_msgp",
            "sigma_max",
            "sigma_floor",
            "sigma_init_gain",
            "lambda_6",
            "lambda_7",
            "lr",
            "lr_factor",
            "lr_threshold",
            "lr_min",
            "weight_decay",
            "grad_clip_ema_alpha",
            "grad_clip_multiplier",
            "adam_eps",
            "dataset_frac",
            "ckpt_interval_sec",
        ):
            setattr(self, name, float(getattr(self, name)))


def load_config(
    config_path: Optional[str] = None, **cli_overrides  # noqa: ANN003
) -> RunConfig:
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
        if v is not None:  # None means "not provided on CLI"
            if not hasattr(cfg, k):
                raise ValueError(f"Unknown config key: {k!r}")
            setattr(cfg, k, v)

    return cfg
