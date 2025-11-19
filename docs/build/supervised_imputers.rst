Supervised Imputers
===================

Overview
--------

PG-SUI's supervised imputers frame genotype imputation as **multiclass prediction** (0 = REF, 1 = HET, 2 = ALT for diploids; haploids collapse to two classes). They use typed ``*Config`` dataclasses with presets (``fast``, ``balanced``, ``thorough``), optional YAML, and a consistent **instantiate → fit() → transform()** workflow. Under the hood each model wraps :class:`sklearn.impute.IterativeImputer` with a tree-based estimator, evaluates on simulated missingness, and reports both 0/1/2 and IUPAC metrics.

.. code-block:: python

   from snpio import VCFReader
   from pgsui import ImputeRandomForest, RFConfig 

   gd = VCFReader("data.vcf.gz", popmapfile="pops.popmap", prefix="demo")

   cfg = RFConfig.from_preset("balanced")
   cfg.io.prefix = "rf_demo"

   model = ImputeRandomForest(genotype_data=gd, config=cfg)
   model.fit()
   X012_imputed = model.transform()

Shared Arguments
----------------

The supervised configs expose the following sections (see the specific ``*Config`` definitions below for field-level defaults):

- **``io``** - run prefix, logging toggles, seeds, and job counts for sklearn.
- **``model``** - estimator hyperparameters (e.g., tree depth, number of estimators, learning rate).
- **``train``** - validation split size used when carving out the hold-out set.
- **``imputer``** - IterativeImputer settings such as ``max_iter`` and ``n_nearest_features``.
- **``sim``** - controls the :class:`pgsui.data_processing.transformers.SimGenotypeDataTransformer` used to mask additional sites for evaluation (proportion, strategy, missing code).
- **``plot``** - figure export format, DPI, font size, despine toggle, and interactive display.
- **``tune``** - retained for API parity; presets disable Optuna for tree models but the structure accepts future tuning hooks.

Evaluation artefacts (metrics, plots, tuned parameters) land under ``{prefix}_output/Supervised/{plots,metrics,parameters}/`` with one folder per model.

Notes
^^^^^

- Inputs are taken from SNPio's ``GenotypeData`` at **instantiation** and encoded to 0/1/2 with ``-9/-1`` treated as missing.
- ``fit()`` splits with ``train.validation_split``, trains an IterativeImputer that wraps the chosen estimator, simulates extra missingness via ``sim`` settings, and writes macro metrics plus plots.
- ``transform()`` imputes the full cohort, coerces any residual negatives/NaNs to ``-9``, and returns IUPAC strings in addition to cached plots.
- Class imbalance can be managed via estimator ``class_weight`` plus the macro-averaged scoring suite reported for every run.

Config Dataclasses (API)
------------------------

Random Forest (Config)
^^^^^^^^^^^^^^^^^^^^^^

Captures :class:`sklearn.ensemble.RandomForestClassifier` knobs plus IterativeImputer and simulation settings used by :class:`pgsui.impute.supervised.imputers.random_forest.ImputeRandomForest`.

.. autoclass:: pgsui.data_processing.containers.RFConfig
   :members:
   :show-inheritance:
   :inherited-members:
   :noindex:

HistGradientBoosting (Config)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Defines the histogram-based gradient boosting estimator parameters together with IterativeImputer and simulation envelopes consumed by :class:`pgsui.impute.supervised.imputers.hist_gradient_boosting.ImputeHistGradientBoosting`.

.. autoclass:: pgsui.data_processing.containers.HGBConfig
   :members:
   :show-inheritance:
   :inherited-members:
   :noindex:

Supervised Imputer Models
-------------------------

Random Forest
^^^^^^^^^^^^^

Wraps :class:`sklearn.impute.IterativeImputer` around a RandomForestClassifier to iteratively fill masked loci; metrics are reported on both 0/1/2 and decoded IUPAC targets using the simulated validation mask.

.. autoclass:: pgsui.impute.supervised.imputers.random_forest.ImputeRandomForest
   :members:
   :show-inheritance:
   :inherited-members:
   :noindex:

HistGradientBoosting
^^^^^^^^^^^^^^^^^^^^

Uses :class:`sklearn.ensemble.HistGradientBoostingClassifier` as the IterativeImputer estimator, enabling faster training on wide genotype matrices while sharing the evaluation and plotting stack with the Random Forest variant.

.. autoclass:: pgsui.impute.supervised.imputers.hist_gradient_boosting.ImputeHistGradientBoosting
   :members:
   :show-inheritance:
   :inherited-members:
   :noindex:

CLI Examples
------------

Run both supervised models with the ``balanced`` preset and a shared prefix:

.. code-block:: bash

   pg-sui \
      --vcf data.vcf.gz \
      --popmap pops.popmap \
      --models ImputeRandomForest ImputeHistGradientBoosting \
      --preset balanced \
      --sim-strategy random_weighted_inv \
      --sim-prop 0.35 \
      --set io.prefix=supervised_demo

YAML + overrides:

.. code-block:: bash

   pg-sui \
      --vcf data.vcf.gz \
      --popmap pops.popmap \
      --models ImputeHistGradientBoosting \
      --preset thorough \
      --config hgb.yaml \
      --set io.prefix=hgb_thorough \
      --set imputer.max_iter=12 \
      --sim-prop 0.40

Use ``--simulate-missing`` to temporarily disable simulated masking for diagnostics or ablation studies; omit it to honour the preset/YAML defaults.

Outputs
-------

- Plots: ``{prefix}_output/Supervised/plots/{Model}/``
- Metrics (CSV/JSON): ``{prefix}_output/Supervised/metrics/{Model}/``
- Cross-model radar summary compares macro-F1, macro-PR, accuracy, and HET-F1.
