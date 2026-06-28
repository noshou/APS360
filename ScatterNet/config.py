from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from typing      import Optional

# Default size buckets: each tuple is (min_atoms, max_atoms) for one batch group.
DEFAULT_BUCKETS: list[tuple[int, int]] = [
    (   1,    3), (   4,    6),
    (   7,   12), (  13,   14),
    (  15,   16), (  17,   17),
    (  18,   18), (  19,   19),
    (  20,   20), (  21,   21),
    (  22,   23), (  24,   26),
    (  27,   33), (  34,   40),
    (  41,   45), (  46,   50),
    (  51,   55), (  56,   60),
    (  61,   64), (  65,   69),
    (  70,   74), (  75,   80),
    (  81,   84), (  85,   90),
    (  91,   96), (  97,  102),
    ( 103,  108), ( 109,  116),
    ( 117,  124), ( 125,  132),
    ( 133,  142), ( 143,  152),
    ( 153,  160), ( 161,  170),
    ( 171,  180), ( 181,  192),
    ( 193,  202), ( 203,  216),
    ( 217,  228), ( 229,  242),
    ( 243,  258), ( 259,  276),
    ( 277,  296), ( 297,  316),
    ( 317,  336), ( 337,  364),
    ( 365,  392), ( 393,  428),
    ( 429,  472), ( 473,  524),
    ( 525,  596), ( 597,  696),
    ( 697,  856), ( 857, 1208),
    (1209, 3177), (3178, 4251),
    (4252, 6046), (6047, 78819),
    ]

@dataclass
class RunConfig:
    """
    Full configuration for a ScatterNet training run.

    Paths
    -----
    hdf5:           Path to the raw HDF5 dataset containing I(q) curves and coordinates.
    db:             Stem path for the SQLite encoding database (Preprocess writes <db>-ENCODING.sqlite3).
    ckpt_best:      Where to save the checkpoint with the lowest validation loss.
    ckpt_resume:    Where to save the latest checkpoint for run resumption.
    metrics:        JSON file that accumulates per-epoch train/val/test loss and R2.
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
    eps_embd:       Numerical floor in the Embed module (avoids division by zero).
    eps_msgp:       Numerical floor in MessagePass sigma clamping and aggregate denominator.

    Loss
    ----
    lambda_6:       Weight on the form-factor penalty term.
    lambda_7:       Weight on the sigma inverse-L1 regularisation (prevents RFF bandwidth collapse).
    eps_sigma:      Floor inside the sigma penalty (separate from eps_msgp).

    Training
    --------
    lr:             Adam learning rate.
    weight_decay:   Adam L2 weight decay.
    grad_clip:      Max gradient norm for gradient clipping (torch.nn.utils.clip_grad_norm_).
    epochs:         Number of full passes over all batches.
    batcher_seed:   RNG seed for the train/val/test molecule split (reproducible splits).
    atom_size_ceil: Maximum total atoms per batch; batches exceeding this are split via binary
                    search. -1 = auto (3x the largest molecule in the dataset).
    num_workers:    DataLoader worker processes (0 = load in main process; use 0 for CPU debugging).
    max_batches:    Cap on batches per epoch (None = no limit; useful for quick sanity checks).
    ckpt_interval_sec: Seconds between mid-epoch resume-checkpoint saves. Crash safety: a session
                    timeout then costs at most this much work (resume picks up mid-epoch).
    ckpt_rclone_dest: rclone destination to copy checkpoints to after each save (e.g.
                    "gdrive:APS360/ckpts/"). None = no off-box copy. Needed on Kaggle, where
                    /kaggle/working is not reliably persisted through a session timeout.
    verbosity:      Logging level. "epoch" = one line per epoch only. "batch" = also print a
                    running-average loss every 50 batches. "diagnostic" = per-batch NaN/Inf check
                    with full tensor stats on the first 10 batches (use for debugging).

    Data
    ----
    buckets:        List of (min_atoms, max_atoms) size buckets to load. Molecules outside all
                    buckets are ignored. Override to restrict the atom-size range (e.g. smoke tests).
    """

    # paths
    hdf5:           str            = "Preprocess/I(q)@L=50.h5"
    db:             str            = "Preprocess/scatternet"
    ckpt_best:      str            = "scatternet_best.pt"
    ckpt_resume:    str            = "scatternet_resume.pt"
    metrics:        str            = "scatternet_metrics.json"
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
    eps_embd:       float          = 1e-8
    eps_msgp:       float          = 1e-3

    # loss
    lambda_6:       float          = 0.1
    lambda_7:       float          = 0.1
    eps_sigma:      float          = 1e-4

    # training
    lr:             float          = 3e-4
    weight_decay:   float          = 1e-5
    grad_clip:      float          = 1.0
    epochs:         int            = 50
    batcher_seed:   int            = 0
    atom_size_ceil: int            = -1
    num_workers:    int            = 4
    max_batches:    Optional[int]  = None
    ckpt_interval_sec: float       = 600.0     # mid-epoch checkpoint cadence (crash safety)
    ckpt_rclone_dest:  Optional[str] = None    # rclone dest for off-box checkpoint copies (None = off)
    verbosity:      str            = "epoch"   # "epoch" | "batch" | "diagnostic"

    # data
    buckets: list[tuple[int, int]] = field(default_factory=lambda: DEFAULT_BUCKETS)


def load_config(config_path: Optional[str] = None, **cli_overrides) -> RunConfig:
    
    """Build RunConfig: dataclass defaults → YAML file → explicit CLI flags."""
    
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
