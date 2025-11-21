==========
Changelog
==========

An overview of changes to **PG-SUI** by release. This file mirrors the GitHub Markdown changelog and reflects the refactor-era docs: dataclass-first configs, presets, unified scikit-learn framework-based ``fit()/transform()`` methods, CLI precedence, and updated deterministic/supervised docs.

1.6.10 - 2025-11-20
-------------------

Bug Fixes - v1.6.10
^^^^^^^^^^^^^^^^^^^

- Fixed a bug with GPU training in ImputeVAE and ImputeAutoencoder. This bug was introduced in v1.6.9 where the gamma and beta parameters were incorrectly converted from tensors to numpy arrays in these models and failed only when using GPU. CPU training was unaffected.

1.6.9 - 2025-10-29
------------------

Enhancements
^^^^^^^^^^^^

- Added explicit CLI flags for simulated-missingness control: ``--sim-strategy``, ``--sim-prop``, and ``--simulate-missing`` (store-false) so users can align evaluation masks across models without modifying YAML configs.

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
