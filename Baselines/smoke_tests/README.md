# Baselines smoke test

CPU-only, synthetic-data smoke test for the baseline evaluation pipeline
(`Baselines/metrics.py` plus every baseline in `physics-benchmarks/` and
`learned-benchmarks/`).

## What it does

Generates a handful of fake molecules (valid VOCAB elements, random
coordinates), derives "ground truth" I(q) from the most detailed physics
baseline (`BinnedDebyeBaseline`) plus multiplicative log-normal noise, then runs
every baseline through `evaluate()` / `run_all_plots()` against it. This checks
each metric and plot for a hard crash or obviously-wrong output *before*
spending Colab GPU/session time on `colab_baselines.ipynb`.

## It only checks that things RUN

This is not a check of prediction quality. The "ground truth" is synthetic, and
the subset is tiny (about a dozen molecules), so the MSLE / R² / µs-per-atom
numbers it prints are not statistically meaningful. **Do not read them as
baseline performance.** Use `kaggle_baselines.ipynb` / `colab_baselines.ipynb`
on the real dataset for that.

## Run

```bash
python Baselines/smoke_tests/baselines_smoke_test.py --config Baselines/smoke_tests/baselines_smoke_test.yaml
```

Plots are written to `smoke_test_out/`. A run that ends with
`Smoke test OK` passed.

## Files

- `baselines_smoke_test.py` — the test driver.
- `baselines_smoke_test.yaml` — config: molecule counts, q-grid, elements, noise,
  which baselines to run, and cut-down learned-baseline hyperparameters.
- `smoke_test_out/` — generated plots (regenerated each run).
