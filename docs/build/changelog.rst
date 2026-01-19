==========
Changelog
==========

An overview of changes to **PG-SUI** by release. This file mirrors the GitHub Markdown changelog and reflects the refactor-era docs: dataclass-first configs, presets, unified scikit-learn framework-based ``fit()/transform()`` methods, CLI precedence, and updated deterministic/supervised docs.

v1.7.2 - 2026-01-19
-------------------

Bug Fix - v1.7.2
^^^^^^^^^^^^^^^^

- Fixed issue where small n_samples would crash with tuning.
- The issue had to do with the trial suggestions for latent_dim.
- Fixed issue where ``latent_dim`` was not set correctly
- Fixed issue where ``_compute_hidden_layer_sizes`` was not called with consistently the same options among models.

v1.7.1 - 2026-01-17
-------------------

Features - v1.7.1
^^^^^^^^^^^^^^^^^

- Added ImputeNLPCA and ImputeUBP models with configs/presets, CLI integration, and new NLPCAModel/UBPModel architectures.
- Added Optuna tuning utilities (OptunaParamSpec validation and best-trial logging) to standardize objective params and study output.

Enhancements - v1.7.1
^^^^^^^^^^^^^^^^^^^^^

- Added ``plot.multiqc`` config support and propagated MultiQC-compatible plotting across deterministic and unsupervised models.
- Genotype distribution plots now compare imputed vs. original datasets with Jensen-Shannon distance plus MultiQC comparison panels.
- Plotting updates include multi-phase history handling, macro-only ROC/PR curves, and log-scaled confusion matrices.

Improvements - v1.7.1
^^^^^^^^^^^^^^^^^^^^^

- Refactored BaseNNImputer and unsupervised imputers for clearer mask handling, logging, and parameter saving.
- Updated Autoencoder/UBP training schedules (AdamW + warmup-to-cosine) and improved hyperparameter validation/logging.
- SimMissingTransformer gains ``mask_missing`` control to avoid simulated-missing overlaps with existing missingness.
- Optuna study database filenames now use UUIDs with clearer resume logging.

Bug Fixes - v1.7.1
^^^^^^^^^^^^^^^^^^

- Scorer ROC AUC/AP now handle undefined cases consistently and return 0.0 with clearer warnings.
- Focal cross-entropy loss now uses integer targets and a safe zero-loss early exit.

Documentation - v1.7.1
^^^^^^^^^^^^^^^^^^^^^^

- Added algorithm pages for Autoencoder, VAE, NLPCA, and UBP plus updated configs/tutorials and an Optuna tuning guide.

Tests - v1.7.1
^^^^^^^^^^^^^^

- Updated the scorer ROC AUC unit test fixture to a multi-class example.

v1.7.0 - 2026-01-08
-------------------

Breaking Changes - v1.7.0
^^^^^^^^^^^^^^^^^^^^^^^^^

- Removed ``ImputeNLPCA`` and ``ImputeUBP`` models/configs from the public API, CLI, and tests (Autoencoder/VAE remain).
- Dropped the ``--disable-simulate-missing`` CLI flag; use YAML overrides for ``sim.simulate_missing`` instead.
- Removed ``evaluate.*`` config sections from Autoencoder/VAE dataclasses; delete or migrate those keys in custom YAMLs.

Features - v1.7.0
^^^^^^^^^^^^^^^^^

- Added focal cross-entropy reconstruction loss support with optional reconstruction scaling for VAE training.
- Added KL-beta and focal-gamma scheduling options to VAE/Autoencoder training and tuning.
- Added Jaccard and Matthews correlation (MCC) metrics to tunable scorers; ImputeMostFrequent reporting now includes AP/Jaccard/MCC alongside core zygosity metrics.
- Added ``scripts/summarize_tuned_params.py`` to aggregate tuned-parameter JSONs and plot parameter distributions.

Enhancements - v1.7.0
^^^^^^^^^^^^^^^^^^^^^

- Expanded class-weight handling with ``weights_power``, ``weights_inverse``, ``weights_normalize``, and optional ``weights_max_ratio`` caps for imbalanced zygosity.
- Simplified tuning config by removing legacy knobs (fast/max_samples/max_loci/eval_interval/proxy batches) and expanding supported tuning metrics.
- Enforced canonical ordering for classification report plots (IUPAC classes first, averages last).

Improvements - v1.7.0
^^^^^^^^^^^^^^^^^^^^^

- Rebalanced training defaults (batch size, early stopping, max epochs, ``sim_prop``) and updated presets/config templates accordingly.
- Hidden-layer schedules now enforce strictly decreasing sizes for pyramid/linear layouts with clearer validation errors.
- Deterministic imputers now store 012 matrices as ``int8`` for lower memory; simulated-missing logging reduced to debug noise.
- Plot history output is standardized to Train/Validation curves; confusion-matrix prefixes no longer double-underscore.

Bug Fixes - v1.7.0
^^^^^^^^^^^^^^^^^^

- ``_one_hot_encode_012`` now preserves missing values as all ``-1`` vectors and validates out-of-range class encodings.
- ``decode_012`` normalization now handles ambiguous/byte/missing tokens more robustly to avoid accidental mis-decoding.
- ``validate_input_type`` now returns integer tensors for torch inputs to avoid dtype mismatches in loss functions.

Documentation - v1.7.0
^^^^^^^^^^^^^^^^^^^^^^

- Updated CLI/config/unsupervised docs to reflect the Autoencoder/VAE-only workflow and new weighting/tuning defaults.

v1.6.28 - 2025-12-24
--------------------

CI/Automation - v1.6.28
^^^^^^^^^^^^^^^^^^^^^^^

- Added disk cleanup and Docker pruning to the unit-tests workflow to reduce CI storage pressure.
- Disabled pip caching during unit tests and explicitly purged pip cache after installs.

v1.6.27 - 2025-12-23
--------------------

Bug Fixes - v1.6.27
^^^^^^^^^^^^^^^^^^^

- Preserved pre-initialized ``tree_parser`` instances in ``BaseNNImputer`` to support nonrandom simulated missingness.
- Normalized and validated tree-related CLI inputs (``--treefile``, ``--qmatrix``, ``--siterates``) when nonrandom strategies are requested.

Tests - v1.6.27
^^^^^^^^^^^^^^^

- Added unit coverage for all five ``SimMissingTransformer`` strategies.
- Added a file-based ``TreeParser`` integration test using Newick, IQ-TREE Q matrix, and site rates inputs.

v1.6.26 - 2025-12-22
--------------------

CI/Automation - v1.6.26
^^^^^^^^^^^^^^^^^^^^^^^

- Added a dedicated ``unit-tests`` GitHub Actions workflow that disables plotting during CI runs.
- Updated release workflows (PyPI, conda, Docker) to wait for the ``unit-tests`` workflow before building or uploading artifacts.
- Explicitly set ``MPLCONFIGDIR`` during CI installs/tests to avoid Matplotlib cache issues.

v1.6.25 - 2025-12-21
--------------------

Bug Fixes - v1.6.25
^^^^^^^^^^^^^^^^^^^

- Fixed haploid decoding in unsupervised imputers (Autoencoder, VAE, UBP, NLPCA) to emit REF/ALT bases rather than heterozygote codes.
- Normalized REF/ALT handling in ``decode_012`` to support ambiguous IUPAC inputs without injecting missing values.
- Ensured ImputeVAE persists ``best_parameters.json`` from the final fitted parameters (not just tuned params) for reliable best-params loading.
- Fixed CLI format aliases (``vcf.gz``, ``phy``, ``gen``, ``structure``) and added STRUCTURE-specific parsing flags.

Enhancements - v1.6.25
^^^^^^^^^^^^^^^^^^^^^^

- Normalized plot-format aliases (e.g., ``jpeg`` -> ``jpg``) and expanded CLI plot format support to include ``svg`` where supported.

Documentation - v1.6.25
^^^^^^^^^^^^^^^^^^^^^^^

- Clarified IUPAC outputs in docs and examples.
- Updated supervised imputer docstrings to match IUPAC return types.
- Refined configuration and tutorial docs to align presets/defaults with ``containers.py``, corrected YAML examples, and clarified simulated-missingness behavior for deterministic/unsupervised models.

v1.6.24 - 2025-12-17
--------------------

Bug Fixes - v1.6.24
^^^^^^^^^^^^^^^^^^^

Refactor imputation methods to unify decoding logic and enhance missing data simulation

- Introduced a new `decode_012` method in `ImputeRefAllele` and `BaseNNImputer` to standardize the decoding of 012 and 0-9 integer encodings to IUPAC nucleotides across different imputer classes.
- Updated all imputer classes (`Autoencoder`, `NLPCA`, `UBP`, `VAE`) to utilize the new decoding method, ensuring consistency and reducing code duplication.
- Refactored missing data simulation logic into a common method `sim_missing_transform` to streamline the process of preparing data for imputation.
- Enhanced error handling and logging for missing data scenarios, ensuring clearer messages when required data is not available.
- Removed redundant code related to float genotype caching and simulation mask handling, simplifying the overall structure of the imputer classes.
- Fixed critical bug where SNPio's decode_012 was injecting missing values into the decoded data.

v1.6.23 - 2025-12-17
--------------------

Bug Fixes - v1.6.23
^^^^^^^^^^^^^^^^^^^

- Fixed critical bug where the outputs from ``transform()`` in all methods still had missing genotypes.
  - Was a simple fix, was just not a robust enough detection step for missing values. Now handles any negative value instead of just "-1" or "-9".

v1.6.22 - 2025-12-14
--------------------

Bug Fixes - v1.6.22
^^^^^^^^^^^^^^^^^^^

- CLI config precedence with ``--load-best-params`` (ImputeUBP, ImputeNLPCA):

  - ``--load-best-params`` now strictly disables hyperparameter tuning for tune-capable models (including ImputeUBP and ImputeNLPCA) and cannot be overridden by:

    - presets,
    - YAML config values (e.g., ``tune.enabled: true``),
    - ``--tune`` / ``--tune-n-trials``,
    - ``--set tune.*=...``,
    - or keys embedded in the loaded best-parameters JSON.

  - Added a strict "force tuning off" enforcement that is applied:

    - after loading best parameters,
    - after applying CLI overrides,
    - and again after ``--set`` overrides (final guarantee).

  - If ``--tune``, ``--tune-n-trials``, or ``--set tune.*=...`` is provided alongside ``--load-best-params``, the CLI logs warnings and proceeds with tuning disabled.

- Prefix resolution correctness affecting best-params lookup (ImputeUBP, ImputeNLPCA):

  - Fixed ``io.prefix`` inference to prefer ``--input`` (not only legacy ``--vcf``) when deriving the default prefix.
  - Ensured the resolved ``prefix`` is written back into ``args`` so subsequent config building and best-parameter lookup use a consistent prefix value.
- Resolved an edge case where explicit ``--prefix`` could be ignored when ``--input`` was also provided.
- Updated docs to clarify ``--load-best-params`` behavior and precedence.
- Added unit tests covering best-parameter loading behavior with various CLI override combinations.
- Code cleanup and documentation improvements throughout all modules.
- Bumped version to v1.6.22.

Enhancements - v1.6.22
^^^^^^^^^^^^^^^^^^^^^^

- Validation-loss support and stability improvements (ImputeUBP, ImputeNLPCA):

  - Implemented explicit validation evaluation compatible with simulated-missing scoring:

    - added support for ``eval_mask_override`` to score only intended entries (e.g., simulated-missing mask),
    - added explicit ``GT_val`` handling to ensure metrics are computed against the true (pre-mask) genotypes.

  - Added pruning/evaluation helpers to keep validation logic aligned and safe:

    - ``_resolve_prune_eval_mask_and_gt()`` resolves a mask and matching GT matrix aligned to the current ``X_val``,
    - ``_eval_for_pruning()`` centralizes latent inference plus evaluation for Optuna pruning.

  - Added schema-aware caching of validation latents to reduce redundant inference and prevent cross-shape reuse:

    - cache keys include latent dimension (``z``), loci width (``L``), and number of classes (``K``).

  - Hardened shape and consistency checks to fail fast (with clear errors) when ``X_val``, masks, or GT matrices do not align.

- Validation-loss integration into the training loop (ImputeUBP, ImputeNLPCA):

  - Integrated periodic validation evaluation during training (via ``eval_interval``) with optional latent inference:

    - supports ``eval_latent_steps``, ``eval_latent_lr``, and ``eval_latent_weight_decay``,
    - enables Optuna pruning based on a chosen validation metric during the final phase.

  - Ensured evaluation respects haploid vs diploid class semantics and avoids scoring against masked ``-1`` values.
  - Added logging of validation loss to the training history for post-hoc analysis and plotting.
- Improved CLI logging clarity around tuning vs best-parameter loading states.
- Updated docs to clarify ``--load-best-params`` behavior and precedence.
- Refactored ImputeUBP and ImputeNLPCA training and validation loops for clarity and maintainability.
- Added unit tests covering the new validation logic, pruning, and best-parameter loading behavior.
- Code cleanup and documentation improvements throughout all modules.
- Bumped version to v1.6.22.

1.6.21 - 2025-12-09
-------------------

Changes - v1.6.21
^^^^^^^^^^^^^^^^^

- Standardized genotype classification across all imputers:
  - **Diploid:** 3-class categorical output (**REF, HET, ALT**).
  - **Haploid:** 2-class categorical output (**REF, ALT**) with explicit **012→01** collapse (treat `2` as `1`).
- Unified model “head” semantics with scoring semantics (output channel count now matches the evaluation class count).

Bug Fixes v1.6.21
^^^^^^^^^^^^^^^^^

- Corrected training history / loss tracking so history objects are consistently **`dict[str, list[float]]`** (e.g., `{"Train":[...], "Val":[...]}`), preventing nested dict artifacts and restoring plot compatibility.
- Repaired validation loss logging and shape alignment:
  - Validation loss is computed against the **aligned ground-truth matrix** (not model inputs / undefined variables).
  - Masking rules now consistently exclude true missing (`-1`) and honor optional evaluation masks (e.g., simulated-missing test masks).
- Hardened logits reshaping and dimensional checks so decoder outputs reliably normalize to **(B, L, K)** and error clearly on mismatch (prevents silent mis-scoring when tuning subsets change locus counts).

Improvements - v1.6.21
^^^^^^^^^^^^^^^^^^^^^^

- Added/standardized numeric stability guards (finite checks, gradient checks, clipping) across training and validation loops to reduce NaN/Inf cascades during optimization.
- Ensured class-weight handling is device-consistent and reused across training/validation without dtype/device drift.
- Made history and evaluation outputs more deterministic and easier to serialize/debug (no `defaultdict` leakage; explicit casting/conversion where needed).

v1.6.14 to v1.6.20
------------------

Ignore these releases. Was experimenting with some new algorithms that didn't work out. Should have just had them on a separate branch.

1.6.14-alpha - 2025-12-05
-------------------------

Enhancements - v1.6.14-alpha
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Support for conda installation via the btmartin721 channel has been added to the installation instructions.
- Support for a Docker installation method has been included, providing users with an alternative way to run PG-SUI in containerized environments.
- The MacOS GUI add-on section has been updated to include instructions for installing the optional Electron wrapper, which provides a point-and-click interface for PG-SUI users on MacOS.
- Updated the README.md file to reflect the latest changes and enhancements in PG-SUI.
- Updated CI/CD workflows to streamline the release process and ensure smooth deployment of new versions. Now automatically bumps version numbers and pushes to TestPyPi and PyPi, conda channel, and Docker Hub on newly tagged pushes.

1.6.12 - 2025-11-28
-------------------

Bug Fixes - v1.6.12
^^^^^^^^^^^^^^^^^^^

- Fixed bugs where ``SimMissingTransformer`` was not accessing the ``tree`` attribute from the correct object. It was supposed to access a SNPio ``TreeParser`` object but was trying to get the ``tree`` attribute from a ``GenotypeData`` object.
- Fixed bugs with ``sim_strategy='nonrandom'`` and ``sim_strategy='nonrandom_weighted'``.

Enhancements - v1.6.12
^^^^^^^^^^^^^^^^^^^^^^

- Improved ImputeUBP performance marginally by eliminating a python loop and using a vectorized ``torch.Tensor`` solution.
- Updated and optimized preset configurations to reduce potential wasted computational efforts.

1.6.11 - 2025-11-21
-------------------

Enhancements - v1.6.11
^^^^^^^^^^^^^^^^^^^^^^

- Added execution time for each model to the CLI logging output. Each model run now logs its total execution time upon completion, providing users with insights into performance and efficiency.

Bug Fixes - v1.6.11
^^^^^^^^^^^^^^^^^^^

- Corrected an issue in all unsupervised imputers where the latent seed was not being set properly during evaluation. This fix ensures consistent and reproducible results.
- Fixed a bug where some CLI settings were not being applied correctly when using the ``--set`` flag. Specifically, the ``eval_latent_steps`` and ``tune.infer_eval`` options were not respecting user overrides correctly in ImputeUBP and ImputeNLPCA.

1.6.10 - 2025-11-20
-------------------

Bug Fixes - v1.6.10
^^^^^^^^^^^^^^^^^^^

- Fixed a bug with GPU training in ImputeVAE and ImputeAutoencoder. This bug was introduced in v1.6.9 where the gamma and beta parameters were incorrectly converted from tensors to numpy arrays in these models and failed only when using GPU. CPU training was unaffected.

1.6.9 - 2025-10-29
------------------

Enhancements
^^^^^^^^^^^^

- Added explicit CLI flags for simulated-missingness control: ``--sim-strategy``, ``--sim-prop``, and ``--disable-simulate-missing`` so users can align evaluation masks across models without modifying YAML configs.

Documentation
^^^^^^^^^^^^^

- README plus key RST pages (Tutorial, About, Unsupervised, Supervised, Deterministic, Implement New Models) now describe the new CLI switches with updated command snippets covering common workflows.

1.6.8 - 2025-10-27
------------------

Enhancements - 1.6.8
^^^^^^^^^^^^^^^^^^^^

- GitHub Actions release workflows added to auto bump PG-SUI version numbers and push to TestPyPi and PyPi.

1.6.4 - 2025-10-26
------------------

Bug Fixes - 1.6.4
^^^^^^^^^^^^^^^^^

- Fix for import bug where none of the PG-SUI modules could be imported. Incorrect path was specified in ``pyproject.toml``.

Features - 1.6.4
^^^^^^^^^^^^^^^^

- New Docker image for use with workflow managers like ``Nextflow``.
- New conda recipe
- Fixed issues with ``pyproject.toml`` with invalid classifiers when building.

1.6.3 - 2025-10-25
------------------

Changes - v1.6.3
^^^^^^^^^^^^^^^^

- Fixed typo in ``ImputeRefAllele`` docstring.
- Added Dockerfile and entrypoint script.
- Added electron app for GUI usage.
- Improved startup script for better error handling.
- Enhanced Dockerfile for multi-stage builds.
- Updated documentation throughout for clarity and accuracy.

1.6.2 — 2025-10-04
------------------

Changes
^^^^^^^

- Updated dependencies in ``pyproject.toml`` and ``conf.py``
- Version bump

1.6.1 — 2025-10-04
------------------

Highlights
^^^^^^^^^^

- **Dataclass-based configuration** across models (e.g., ``VAEConfig``, ``UBPConfig``, ``NLPCAConfig``, ``MostFrequentConfig``, ``RefAlleleConfig``, **``HGBConfig``**, **``RFConfig``**).
- **Presets** for all configs: ``fast``, ``balanced``, ``thorough``.
- **Unified contract:** models receive ``GenotypeData`` at construction; call ``fit()`` then ``transform()`` (no arguments).
- **CLI precedence:** ``code defaults < --preset < --config < explicit flags < --set k=v``.
- **New visualizations:** cross-model radar, improved PR curves, zygosity bars, confusion matrices, training curves.

Features
^^^^^^^^

- Added ``HGBConfig`` and ``RFConfig`` dataclasses for supervised imputers (Histogram-based Gradient Boosting and Random Forest) with presets and YAML support.
- Deterministic imputers (``ImputeMostFrequent``, ``ImputeRefAllele``) upgraded to the dataclass/YAML pipeline and unified plotting/metrics output.

Enhancements
^^^^^^^^^^^^

- Consistent nested config sections across all families: ``io``, ``model``, ``train``, ``tune``, ``evaluate``/``split``, ``plot``, plus ``algo``/``imputer``/``sim`` where applicable.
- Optuna tuning flow streamlined (proxy batch option, warm-up pruning, latent-inference hooks for NLPCA-like decoders).
- Improved logging and directory structure: ``{prefix}_output/{Family}/{plots,metrics,models,optimize}/{Model}/``.

Docs
^^^^

- **About PG-SUI**: rewritten with author–year citations; clarifies supervised vs unsupervised vs deterministic.
- **Deterministic Imputers** page: refactor-aligned; examples, YAML usage, CLI overrides, dataclass API.
- **Supervised Imputers** page: ``RFConfig``/``HGBConfig`` added; clarified ``IterativeImputer`` integration and evaluation protocol.
- **Unsupervised Imputers** page: expanded workflow overview, config summaries, and usage examples tied to the shared BaseNNImputer stack.
- **Tutorials**: “Implementing New Models” updated to dataclass + wrapper patterns (NLPCA/UBP/decoder-first examples).
- **Tutorial** page: quick-start code, YAML sample, and CLI overrides now mirror the current config attributes and IUPAC outputs.
- Fixed Sphinx issues (e.g., ``:noindex:`` typos, math blocks, section headings).

Breaking Changes
^^^^^^^^^^^^^^^^

- ``fit(X, y=None)`` and ``transform(X)`` signatures removed; new pattern:

  .. code-block:: python

     model = SomeImputer(genotype_data=gd, config=SomeConfig.from_preset("balanced"))
     model.fit()
     X_imputed = model.transform()

- CLI flags harmonized; prefer dot-path overrides such as ``--set model.latent_dim=16``.

1.6.0 — 2025-09-23
------------------

Features
^^^^^^^^

- Core imputation families validated end-to-end:

  - Unsupervised: ``ImputeAutoencoder``, ``ImputeNLPCA``, ``ImputeUBP``, ``ImputeVAE``.
  - Supervised: ``ImputeRandomForest``, ``ImputeHistGradientBoosting``.

- Shared plotting stack and classification reports (zygosity and IUPAC-10).

1.5.2 — 2025-03-01
------------------

Features
^^^^^^^^

- Added unsupervised models:
  ``ImputeAutoencoder``, ``ImputeNLPCA``, ``ImputeUBP``, ``ImputeVAE``.

Changes
^^^^^^^

- ``BaseNNImputer`` extended to standardize training loops, evaluation, and plotting.
- Documentation: new tutorials and examples for extending/implementing models.

1.5.1 — 2025-02-07
------------------

Bug Fixes
^^^^^^^^^

- Fixed ``ImputeAutoencoder`` missing ``self`` error.
- Various stability fixes in supervised pipelines.

Features
^^^^^^^^

- New simulation strategies for training-time missingness.

Changes
^^^^^^^

- ``SimGenotypeDataTransformer`` expanded; tutorials refreshed.

1.5 — 2025-01-28
----------------

Features
^^^^^^^^

- **Optuna** parameter optimization integrated for deep models.
- Performance improvements across DL implementations; modular architecture for easier research iteration.

Changed
^^^^^^^

- Moved to **PyTorch** (from TensorFlow) for deep learning.
- Unified on ``GenotypeData`` as the core data container.
- Replaced Grid/GASearchCV with Optuna.

1.0.2.1 — 2023-09-11
--------------------

Bug Fixes
^^^^^^^^^

- Resolved duplicated ``self`` in supervised imputers.
- Corrected ``ImputeNLPCA`` incorrectly dispatching to ``ImputeUBP``.
- Fixed ``gt_probability`` heatmap (now ``simulated_genotypes`` plot).
- Ensured plot directories are created.
- Non-ML imputers now decode integer genotypes correctly.
- Supervised default ``prefix`` matches unsupervised (``imputer``).
- Fixed ``ImputeKNN`` and ``ImputeRandomForest`` execution errors.
- Pinned pandas to avoid future warnings; added ``warnings.simplefilter`` for ``FutureWarning``.

Changed
^^^^^^^

- New plotting for ``test.py``.

1.0.2 — 2023-08-28
------------------

Bug Fix
^^^^^^^

- Use ``GenotypeData.copy()`` internally to work around pysam Cython ``VariantHeader`` behavior.

1.0 — 2023-07-29
----------------

Changed
^^^^^^^

- First full (non-beta) release.

0.3.0 — 2023-07-26
------------------

Features
^^^^^^^^

- Unsupervised models: moved from 0/1/2 to nucleotide multi-label encoding (4-class), improving metrics via reduced class imbalance.
- Faster unsupervised grid searches by pruning redundant scorer work.

Changed
^^^^^^^

- Docs clearer on argument purposes.
- Refactors in ``estimators.py``, ``scorers.py`` for modularity/maintainability.

Removed
^^^^^^^

- 0/1/2 inputs for unsupervised (superseded by nucleotide multi-label).

0.2.4 — 2023-07-24
------------------

Features
^^^^^^^^

- Initial public release:
  four unsupervised neural models, three supervised ``IterativeImputer``-based models, and four deterministic imputers.
