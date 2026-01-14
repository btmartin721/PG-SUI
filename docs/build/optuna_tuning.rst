Optuna Hyperparameter Tuning
============================

.. _optuna_tuning:

Overview
--------

PG-SUI automates hyperparameter optimization with `Optuna <https://optuna.org/>`__. Tuning is available for the unsupervised neural network models (ImputeAutoencoder, ImputeVAE, ImputeUBP, ImputeNLPCA) and is also used by supervised tree models when enabled. Each model defines its own search space, while the Optuna framework handles sampling, pruning, and best-parameter persistence.

Workflow
--------

1. Enable tuning via ``tune.enabled`` or the CLI ``--tune`` flag.
2. PG-SUI creates an Optuna study using a TPE sampler and Hyperband-based
   pruning with patience to stop weak trials early.
3. Each trial samples a model-specific set of hyperparameters, trains the
   model, and evaluates a validation metric (or metrics) defined by
   ``tune.metrics``.
4. The best trial parameters are persisted and then used for the final
   full-data fit.

For unsupervised models, tuning uses the simulated-missing evaluation mask so
trials are scored against known truth.

Configuration controls
----------------------

Key fields in :class:`pgsui.data_processing.containers.TuneConfig`:

- ``tune.enabled``: turn tuning on or off.
- ``tune.metrics``: metric name or a list of metrics for multi-objective tuning
  (for example, ``["f1", "mcc"]``).
- ``tune.n_trials``: number of Optuna trials to run.
- ``tune.epochs`` / ``tune.batch_size``: optional per-trial training limits
  used by model-specific tuning loops (when supported).
- ``tune.patience``: model-specific patience setting used during tuning (when supported).
- ``tune.save_db`` / ``tune.resume``: persist and optionally resume an Optuna
  SQLite database.

Parallelism is controlled by ``io.n_jobs``. Use ``--n-jobs`` on the CLI or set
``io.n_jobs`` in YAML to increase Optuna worker count.

Artifacts and outputs
---------------------

PG-SUI stores tuning artifacts under ``<prefix>_output/<Family>/optimize/``:

- ``parameters/best_tuned_parameters.json``: Optuna-selected parameters.
- ``plots/``: optional Optuna study visualizations.
- ``study_database/``: SQLite databases when ``tune.save_db`` is enabled.

After the final fit, the effective parameters are saved in
``<prefix>_output/<Family>/parameters/best_parameters.json``.

Examples
--------

YAML configuration:

.. code-block:: yaml

   tune:
     enabled: true
     metrics: ["f1", "mcc"]
     n_trials: 50
     save_db: true
     resume: false

CLI usage:

.. code-block:: bash

   pg-sui \
     --input cohort.vcf.gz \
     --models ImputeNLPCA ImputeUBP \
     --preset balanced \
     --tune \
     --tune-n-trials 50 \
     --n-jobs 4

Notes
-----

- ``--load-best-params`` disables tuning and uses the saved parameters from a
  previous run. This keeps runs reproducible and avoids conflicting overrides.
