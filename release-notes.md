# PG-SUI v1.8.2

PG-SUI v1.8.2 is a maintenance and robustness release for genotype
imputation, reporting, and distribution workflows.

## Scientific and runtime correctness

- Corrected NLPCA embedding snapshots and L1 regularization so optimization
  operates on module parameters consistently.
- Removed mutable neural-model defaults and normalized focal-loss gamma
  handling across supported devices.
- Added explicit population-label validation before population-specific mode
  imputation.
- Hardened haploid REF/ALT normalization, simulated-missingness sampling, and
  validation-array checks.
- Preserved error context while narrowing recoverable exception handling in
  deterministic and neural imputers.

## Reporting and analysis

- Made optional MultiQC reporting safe when the integration is unavailable.
- Improved model-scoped logging and diagnostic output for tuning, cleanup, and
  configuration failures.
- Added regression tests for metric-feature bootstrap analyses, validation
  runtime scaling, GTImputation VCF scoring, and CPU/GPU runtime comparisons.

## Dependencies and distribution

- Updated SNPio to `>=1.7.3` and retained the intentional scientific Python,
  PyTorch, Optuna, FastAPI, and Uvicorn dependency bounds.
- Enforced strict Conda channel priority without the `defaults` channel to
  prevent low-level library clobbering.
- Synchronized pip, Conda, and environment dependency declarations.
- Corrected setuptools package-discovery exclusions so Electron vendor modules
  are not bundled into the Python wheel.
- Pinned release Docker builds to the exact PG-SUI version published to PyPI.
- Made missing Docker Hub credentials fail the release workflow explicitly.
- Attached built source and wheel distributions to the GitHub release.

## Documentation and maintenance

- Updated pip, Conda, and Docker installation guidance.
- Corrected Sphinx version metadata and GitHub source links.
- Adopted Ruff for formatting, import sorting, and linting.
- Added Ruff checks to the tag and pull-request test workflow.
- Modernized type annotations and removed broad exception handling throughout
  the touched code paths.
