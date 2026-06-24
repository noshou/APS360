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
    
    # paths
    hdf5:         str            = "Preprocess/I(q)@L=50.h5"
    db:           str            = "Preprocess/scatternet"
    ckpt_best:    str            = "scatternet_best.pt"
    ckpt_resume:  str            = "scatternet_resume.pt"
    metrics:      str            = "scatternet_metrics.json"
    resume:       Optional[str]  = None
    
    # model
    lambda_1:     int            = 128
    lambda_2:     int            = 5
    lambda_3:     int            = 128
    lambda_4:     int            = 4
    lambda_5:     int            = 256
    msg_seed:     int            = 42
    eps_embd:     float          = 1e-8
    eps_msgp:     float          = 1e-3
    
    # loss
    lambda_6:     float          = 0.1
    
    # training
    lr:           float          = 3e-4
    weight_decay: float          = 1e-5
    grad_clip:    float          = 1.0
    epochs:       int            = 50
    batcher_seed: int            = 0
    atom_size_ceil: int          = -1
    
    # loss
    lambda_7:     float          = 0.1    # sigma inverse-L1 regularisation weight
    eps_sigma:    float          = 1e-4   # floor inside the sigma penalty (separate from eps_msgp)
    
    # data — override to restrict atom-size range (e.g. for smoke tests)
    buckets: list[tuple[int, int]] = field(default_factory=lambda: DEFAULT_BUCKETS)
    
    # cap batches per epoch (None = no limit; useful for quick sanity checks)
    max_batches: Optional[int]     = None
    
    # data loading workers (0 = main process; set to 0 for local CPU debugging)
    num_workers: int               = 4


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
