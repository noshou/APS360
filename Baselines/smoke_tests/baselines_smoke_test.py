"""CPU-only smoke test for the baseline evaluation pipeline, using synthetic data.

Generates a handful of fake molecules (valid VOCAB elements, random coordinates),
derives "ground truth" I(q) from the most detailed physics baseline
(BinnedDebyeBaseline) plus multiplicative log-normal noise, then runs every
baseline + Baselines/metrics.py's evaluate()/run_all_plots() against it. This
lets every metric and plot be checked for a hard crash or obviously-wrong
output *before* spending Kaggle GPU/session time on kaggle_baselines.ipynb --
it is not a check of prediction quality (the "ground truth" is synthetic).

Usage:
    python Baselines/smoke_tests/baselines_smoke_test.py --config Baselines/smoke_tests/baselines_smoke_test.yaml
"""

import argparse
import os
import random
import sys

import torch
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "Baselines", "physics-benchmarks"))
sys.path.insert(0, os.path.join(REPO, "Baselines", "learned-benchmarks"))

from ScatterNet.batching import Batch  # noqa: E402
from Preprocess           import VOCAB  # noqa: E402
from Baselines.metrics    import evaluate, run_all_plots  # noqa: E402

from rg                     import RgBaseline, GuinierPorodBaseline           # noqa: E402
from atom_count              import AtomCountBaseline                          # noqa: E402
from pair_peak               import BinnedDebyeBaseline                        # noqa: E402
from linsvm                  import Linsvm                                     # noqa: E402
from nearest_neighbour       import NNBaseline                                 # noqa: E402
from mlp                     import Mlp                                        # noqa: E402

_BASELINE_FACTORIES = {
    "rg":                     lambda cfg, q, e, train: RgBaseline(q, e),
    "guinier_porod":           lambda cfg, q, e, train: GuinierPorodBaseline(q, e),
    "atom_count":               lambda cfg, q, e, train: AtomCountBaseline().fit(train),
    "binned_debye":             lambda cfg, q, e, train: BinnedDebyeBaseline(q, e),
    # sklearn's MLPRegressor (vectorized BLAS/LBFGS), not TorchMlp -- on CPU (this
    # smoke test has no GPU) it beats a hand-rolled small-minibatch PyTorch loop,
    # whose overhead is mostly Python dispatch rather than compute. TorchMlp is
    # still what kaggle_baselines.ipynb uses, since Kaggle runs on GPU.
    "mlp":                      lambda cfg, q, e, train: Mlp(
        hidden_layer_sizes=tuple(cfg["mlp_hidden"]), max_iter=cfg["mlp_max_iter"],
    ).fit(train),
    "linear_svm":               lambda cfg, q, e, train: Linsvm(
        n_components=cfg["linsvm_n_components"]
    ).fit(train),
    "nearest_neighbour":        lambda cfg, q, e, train: NNBaseline(q, e).fit(train),
}

_DISPLAY_NAMES = {
    "rg":                     "Guinier (Rg)",
    "guinier_porod":           "Guinier-Porod",
    "atom_count":               "Atom Count",
    "binned_debye":             "Binned Debye",
    "mlp":                      "MLP",
    "linear_svm":               "Linear SVM",
    "nearest_neighbour":        "Nearest Neighbour",
}


def _make_synthetic_batch(cfg: dict, n_mols: int, q_grid: torch.Tensor, energy: float,
                           gen: torch.Generator, rng: random.Random) -> Batch:
    """Build n_mols synthetic molecules and derive noisy I(q) from BinnedDebyeBaseline.

    Coordinates are drawn from a Gaussian point cloud (not a real force field --
    only meant to give BinnedDebyeBaseline a plausible pairwise-distance
    distribution to work with).
    """
    vocabs, coords = [], []
    for _ in range(n_mols):
        n_atoms = rng.randint(cfg["atom_count_min"], cfg["atom_count_max"])
        elements = rng.choices(cfg["ions"], k=n_atoms)
        vocabs.append(torch.tensor([VOCAB[e] for e in elements], dtype=torch.long))
        coords.append(torch.randn(n_atoms, 3, generator=gen) * 2.0)

    zero_iqvals = [torch.zeros(len(q_grid)) for _ in vocabs]
    batch = Batch.from_lists(vocabs, zero_iqvals, coords)

    ground_truth_gen = BinnedDebyeBaseline(q_grid, energy)
    true_iq = ground_truth_gen(batch).clamp(min=0)
    noise = torch.randn(true_iq.shape, generator=gen) * cfg["noise_log_std"]
    observed = (true_iq * torch.exp(noise)).clamp(min=0)

    return Batch(vocab=batch.vocab, iqval=observed, coord=batch.coord)


def _chunk(batch: Batch, size: int) -> list:
    """Split one big Batch into a list of smaller Batches (mimics the real Batcher)."""
    n = batch.vocab.shape[0]
    return [
        Batch(vocab=batch.vocab[i:i + size], iqval=batch.iqval[i:i + size], coord=batch.coord[i:i + size])
        for i in range(0, n, size)
    ]


def main(config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg["seed"])
    gen = torch.Generator().manual_seed(cfg["seed"])
    rng = random.Random(cfg["seed"])

    q_grid = torch.linspace(cfg["q_min"], cfg["q_max"], cfg["q_points"])
    energy = float(cfg["energy"])

    print(f"Generating {cfg['n_train']} train + {cfg['n_test']} test synthetic molecules "
          f"({cfg['atom_count_min']}-{cfg['atom_count_max']} atoms, ions={cfg['ions']})...")
    train_full = _make_synthetic_batch(cfg, cfg["n_train"], q_grid, energy, gen, rng)
    test_full  = _make_synthetic_batch(cfg, cfg["n_test"],  q_grid, energy, gen, rng)

    train_batches = _chunk(train_full, size=4)
    test_batches  = _chunk(test_full,  size=4)

    results = []
    for key in cfg["baselines"]:
        if key not in _BASELINE_FACTORIES:
            raise ValueError(f"Unknown baseline {key!r} in config; known: {list(_BASELINE_FACTORIES)}")
        name = _DISPLAY_NAMES[key]
        print(f"Running {name}...")
        baseline = _BASELINE_FACTORIES[key](cfg, q_grid, energy, train_batches)
        result = evaluate(baseline, test_batches, q_grid, name)
        print(f"  {name:<26s}  MSLE={result.msle:.4f}  R²(raw)={result.r2_raw:.4f}  "
              f"R²(log1p)={result.r2_log1p:.4f}  {result.us_per_atom:.2f} μs/atom")
        results.append(result)

    print(f"\nWriting plots to {cfg['out_dir']}...")
    written = run_all_plots(results, q_grid, cfg["out_dir"])
    for path in written:
        print(f"  {path}")
    print("\nSmoke test OK -- every baseline and every plot ran without error.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "baselines_smoke_test.yaml"))
    args = p.parse_args()
    main(args.config)
