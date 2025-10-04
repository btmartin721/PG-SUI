==========
Changelog
==========

An overview of changes to **PG-SUI** by release. This file mirrors the GitHub Markdown changelog and reflects the refactor-era docs: dataclass-first configs, presets, unified scikit-learn framework-based ``fit()/transform()`` methods, CLI precedence, and updated deterministic/supervised docs.

Unreleased
----------

- HTML summary report that stitches plots, key metrics, and logs into a single artifact.
- Add end-to-end CLI examples for multi-model radar (macro-F1, macro-PR, overall accuracy, HET-F1).
- Expand config presets with GPU/MPS-aware training defaults.

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

- Consistent nested config sections across all families: ``io``, ``training``/``train``, ``tuning``/``tune``, ``evaluate``, ``plot``, ``algorithm``/``algo``.
- Optuna tuning flow streamlined (proxy batch option, warm-up pruning, latent-inference hooks for NLPCA-like decoders).
- Improved logging and directory structure: ``{prefix}_output/{Family}/{plots,metrics,models,optimize}/{Model}/``.

Docs
^^^^

- **About PG-SUI**: rewritten with author–year citations; clarifies supervised vs unsupervised vs deterministic.
- **Deterministic Imputers** page: refactor-aligned; examples, YAML usage, CLI overrides, dataclass API.
- **Supervised Imputers** page: ``RFConfig``/``HGBConfig`` added; clarified ``IterativeImputer`` integration and evaluation protocol.
- **Tutorials**: “Implementing New Models” updated to dataclass + wrapper patterns (NLPCA/UBP/decoder-first examples).
- Fixed Sphinx issues (e.g., ``:noindex:`` typos, math blocks, section headings).

Breaking Changes
^^^^^^^^^^^^^^^^

- ``fit(X, y=None)`` and ``transform(X)`` signatures removed; new pattern:

  .. code-block:: python

     model = SomeImputer(genotype_data=gd, config=SomeConfig.from_preset("balanced"))
     model.fit()
     X_imputed = model.transform()

- CLI flags harmonized; prefer dot-path overrides such as ``--set training.model_latent_dim=16``.

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
