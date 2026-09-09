# PG-SUI v1.8.4

PG-SUI v1.8.4 is a scientific-correctness and reproducibility patch release.

## Scientific correctness

- All four neural models and the `ImputeMostFrequent` and `ImputeRefAllele`
  baselines now use the same seeded train/validation/test sample split.
- Deterministic simulated-missingness generation now receives the configured
  seed explicitly, ensuring that every model evaluates the same masked cells.
- Every model writes an `evaluation_mask_test.npz` artifact containing exact
  test rows and evaluated coordinates for strict post-hoc validation.

## Multi-objective tuning

- Corrected Optuna result extraction for studies where scalar `best_value` and
  `best_trial` accessors are unavailable.
- Multi-objective studies select a reproducible representative Pareto trial by
  the first configured objective and report every objective value.

## Reproducibility

- Added a manifest-driven 50-task GPU/SLURM workflow, strict coordinate and
  support auditing, alternate-genotype summaries, reviewer-bundle generation,
  and AWS PCS transfer instructions for the PG-SUI/GTImputation benchmark.

## Distribution

- Includes the post-v1.8.3 ARM64 Docker correction that disables hanging Conda
  channel-notice retrieval under QEMU and supports exact-version republishing.
