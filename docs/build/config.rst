Config Reference
================

This page summarizes the configuration surface exposed through the ``containers.py`` dataclasses that back the CLI and Python APIs. Every block below lists defaults and the preset bundles applied by ``from_preset(...)`` helpers. Values are shown as declared in code; all options are ASCII-safe and can be overridden via YAML or dot-keys on the CLI.

Common Blocks (Unsupervised NN)
-------------------------------

The two deep imputer families (Autoencoder, VAE) share these structures.

.. list-table:: IOConfig
   :header-rows: 1

   * - Field
     - Default
     - Description
   * - ``prefix``
     - ``"pgsui"``
     - Run/output prefix used for directories and logging.
   * - ``ploidy``
     - ``2``
     - Ploidy level (``1`` for haploid, ``2`` for diploid).
   * - ``verbose`` / ``debug``
     - ``False`` / ``False``
     - Logging verbosity.
   * - ``seed``
     - ``None``
     - RNG seed.
   * - ``n_jobs``
     - ``1``
     - Parallel jobs for Optuna.
   * - ``scoring_averaging``
     - ``"macro"``
     - Averaging mode for metrics (``macro``, ``micro``, ``weighted``).

.. list-table:: ModelConfig
   :header-rows: 1

   * - Field
     - Default
     - Description
   * - ``latent_init``
     - ``"random"``
     - Latent init (``"random"`` or ``"pca"``).
   * - ``latent_dim``
     - ``2``
     - Latent width.
   * - ``dropout_rate``
     - ``0.2``
     - Dropout applied to hidden layers.
   * - ``num_hidden_layers``
     - ``2``
     - Count of hidden layers.
   * - ``activation``
     - ``"relu"``
     - Hidden non-linearity (``relu``, ``elu``, ``selu``, ``leaky_relu``).
   * - ``layer_scaling_factor``
     - ``5.0``
     - Scales hidden widths.
   * - ``layer_schedule``
     - ``"pyramid"``
     - Width layout (``"pyramid"``, ``"linear"``).

.. list-table:: TrainConfig
   :header-rows: 1

   * - Field
     - Default
     - Description
   * - ``batch_size``
     - ``64``
     - Mini-batch size.
   * - ``learning_rate``
     - ``1e-3``
     - Base LR.
   * - ``l1_penalty``
     - ``0.0``
     - L1 regularization.
   * - ``early_stop_gen``
     - ``25``
     - Early-stop patience (epochs).
   * - ``min_epochs`` / ``max_epochs``
     - ``100`` / ``2000``
     - Epoch bounds.
   * - ``validation_split``
     - ``0.2``
     - Holdout fraction.
   * - ``weights_max_ratio``
     - ``None``
     - Cap on class-weight ratio.
   * - ``weights_power``
     - ``1.0``
     - Power scaling for class weights.
   * - ``weights_normalize``
     - ``True``
     - Whether to normalize weights.
   * - ``weights_inverse``
     - ``False``
     - Whether to invert weights.
   * - ``gamma``
     - ``0.0``
     - Focal-loss gamma.
   * - ``gamma_schedule``
     - ``False``
     - Whether to anneal gamma during training.
   * - ``device``
     - ``"cpu"``
     - ``"cpu"``, ``"gpu"``, or ``"mps"``.

.. list-table:: TuneConfig
   :header-rows: 1

   * - Field
     - Default
     - Description
   * - ``enabled``
     - ``False``
     - Turn on Optuna.
   * - ``metric``
     - ``"f1"``
     - Objective metric (``f1``, ``accuracy``, ``pr_macro``, ``average_precision``, ``roc_auc``, etc.).
   * - ``n_trials``
     - ``100``
     - Number of trials.
   * - ``resume`` / ``save_db``
     - ``False`` / ``False``
     - Reuse or persist Optuna DB.
   * - ``epochs`` / ``batch_size``
     - ``500`` / ``64``
     - Training envelope during tuning.
   * - ``patience``
     - ``10``
     - Number of evaluation intervals with no improvement before pruning a trial.

.. list-table:: PlotConfig
   :header-rows: 1

   * - Field
     - Default
     - Description
   * - ``fmt``
     - ``"pdf"``
     - Output format.
   * - ``dpi``
     - ``300``
     - Resolution.
   * - ``fontsize``
     - ``18``
     - Base font size.
   * - ``despine``
     - ``True``
     - Remove top/right spines.
   * - ``show``
     - ``True``
     - Interactive display toggle.

.. list-table:: SimConfig
   :header-rows: 1

   * - Field
     - Default
     - Description
   * - ``simulate_missing``
     - ``False``
     - Whether to simulate missingness for eval (required for unsupervised models).
   * - ``sim_strategy``
     - ``"random"``
     - ``random``, ``random_weighted``, ``random_weighted_inv``, ``nonrandom``, ``nonrandom_weighted``.
   * - ``sim_prop``
     - ``0.20``
     - Proportion to mask.
   * - ``sim_kwargs``
     - ``None``
     - Extra args forwarded to ``SimMissingTransformer``.

Field-by-field notes
--------------------

- **IOConfig**
  - ``ploidy``: Set to ``1`` for haploids; controls class count and decoding.
  - ``n_jobs``: Controls Optuna parallelism.

- **ModelConfig**
  - ``latent_dim``: Governs compression strength; higher values retain more signal at the cost of capacity/overfit.
  - ``layer_schedule``: ``pyramid`` shrinks toward the bottleneck; ``linear`` walks widths linearly.

- **TrainConfig**
  - ``weights_power``: Adjusts the aggression of class weighting (e.g., 0.5 for sqrt, 1.0 for standard inverse frequency).
  - ``gamma``: Controls focal loss behavior. Use ``gamma_schedule=True`` to anneal it.

- **SimConfig**
  - ``sim_strategy``: ``nonrandom`` strategies require a tree parser.

Unsupervised NN presets
-----------------------

Each model exposes ``from_preset("fast" | "balanced" | "thorough")`` to seed a baseline, then allows overrides.

AutoencoderConfig
~~~~~~~~~~~~~~~~~

Preset baseline (all presets):
- ``io``: ``verbose=False``, ``ploidy=2``.
- ``train``: ``validation_split=0.20``, ``weights_max_ratio=None``, ``weights_power=1.0``, ``weights_normalize=True``, ``gamma=0.0``.
- ``model``: ``activation="relu"``, ``layer_schedule="pyramid"``, ``layer_scaling_factor=2.0``.
- ``sim``: ``simulate_missing=True``, ``sim_strategy="random"``, ``sim_prop=0.2``.
- ``tune``: ``enabled=False``, ``n_trials=100``.

Preset overrides:

- **fast**
  - ``model``: ``latent_dim=4``, ``num_hidden_layers=1``, ``dropout_rate=0.10``.
  - ``train``: ``batch_size=128``, ``learning_rate=2e-3``, ``early_stop_gen=15``, ``max_epochs=200``.
  - ``tune``: ``patience=15``.

- **balanced**
  - ``model``: ``latent_dim=8``, ``num_hidden_layers=2``, ``dropout_rate=0.20``.
  - ``train``: ``batch_size=64``, ``learning_rate=1e-3``, ``early_stop_gen=25``, ``max_epochs=500``.
  - ``tune``: ``patience=25``.

- **thorough**
  - ``model``: ``latent_dim=16``, ``num_hidden_layers=3``, ``dropout_rate=0.30``.
  - ``train``: ``batch_size=64``, ``learning_rate=5e-4``, ``early_stop_gen=50``, ``max_epochs=1000``.
  - ``tune``: ``patience=50``.

VAEConfig + VAEExtraConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~

Inherits structure from Autoencoder but adds `VAEExtraConfig`.

.. list-table:: VAEExtraConfig
   :header-rows: 1

   * - Field
     - Default
     - Description
   * - ``kl_beta``
     - ``1.0``
     - Final KL weight.
   * - ``kl_beta_schedule``
     - ``False``
     - Whether to anneal KL beta.

Preset overrides:

- **fast**
  - ``model``: ``latent_dim=4``, ``num_hidden_layers=2``, ``dropout_rate=0.10``.
  - ``train``: ``batch_size=128``, ``learning_rate=2e-3``, ``early_stop_gen=15``, ``max_epochs=200``.
  - ``vae``: ``kl_beta=0.5``.
  - ``tune``: ``patience=15``.

- **balanced**
  - ``model``: ``latent_dim=8``, ``num_hidden_layers=4``, ``dropout_rate=0.20``.
  - ``train``: ``batch_size=64``, ``learning_rate=1e-3``, ``early_stop_gen=25``, ``max_epochs=500``.
  - ``vae``: ``kl_beta=1.0``.
  - ``tune``: ``patience=25``.

- **thorough**
  - ``model``: ``latent_dim=16``, ``num_hidden_layers=8``, ``dropout_rate=0.30``.
  - ``train``: ``batch_size=64``, ``learning_rate=5e-4``, ``early_stop_gen=50``, ``max_epochs=1000``.
  - ``vae``: ``kl_beta=1.0``.
  - ``tune``: ``patience=50``.

Deterministic Imputers
----------------------

These configurations use simpler blocks and do not use Neural Network specific settings like `ModelConfig`.

MostFrequentConfig / RefAlleleConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Common Fields:
- ``split.test_size``: Default ``0.2``.
- ``sim.simulate_missing``: Default ``False`` (enabled in presets).
- ``algo.missing``: Default ``-1``.

**MostFrequentAlgoConfig Extra Fields:**
- ``by_populations``: ``False``.
- ``default``: ``0``.

Supervised Wrappers (RF / HistGB)
---------------------------------

Supervised models use distinct config classes ending in ``ConfigSupervised``.

.. list-table:: IOConfigSupervised
   :header-rows: 1

   * - Field
     - Default
     - Description
   * - ``prefix``
     - ``"pgsui"``
     - Run identity.
   * - ``n_jobs``
     - ``1``
     - Parallel jobs.
   * - ``seed``
     - ``None``
     - RNG seed.

.. list-table:: TuningConfigSupervised
   :header-rows: 1

   * - Field
     - Default
     - Description
   * - ``enabled``
     - ``True``
     - Master toggle.
   * - ``metric``
     - ``"pr_macro"``
     - Tuning metric.
   * - ``n_trials``
     - ``100``
     - Trial count.
   * - ``n_jobs``
     - ``8``
     - Parallel jobs for tuning.
   * - ``fast``
     - ``True``
     - Whether to use faster settings (subsampling etc.).

.. list-table:: RFModelConfig
   :header-rows: 1

   * - Field
     - Default
     - Description
   * - ``n_estimators``
     - ``100``
     - Forest size.
   * - ``max_depth``
     - ``None``
     - Depth cap.
   * - ``class_weight``
     - ``"balanced"``
     - Class weighting strategy.

.. list-table:: HGBModelConfig
   :header-rows: 1

   * - Field
     - Default
     - Description
   * - ``n_estimators``
     - ``100``
     - Boosting iterations.
   * - ``learning_rate``
     - ``0.1``
     - Step size.
   * - ``n_iter_no_change``
     - ``10``
     - Early stopping patience.

Presets (Supervised)
~~~~~~~~~~~~~~~~~~~~

**RandomForest (RFConfig):**

- **fast**: ``n_estimators=50``, ``max_iter=5``, ``tune.enabled=False``.
- **balanced**: ``n_estimators=200``, ``max_iter=10``, ``tune.enabled=False``.
- **thorough**: ``n_estimators=500``, ``max_depth=50``, ``max_iter=20``, ``tune.enabled=False``.

**HistGradientBoosting (HGBConfig):**

- **fast**: ``n_estimators=50``, ``learning_rate=0.2``, ``max_iter=5``.
- **balanced**: ``n_estimators=150``, ``learning_rate=0.1``, ``max_iter=10``.
- **thorough**: ``n_estimators=500``, ``learning_rate=0.05``, ``max_iter=20``, ``n_iter_no_change=20``.