Config Reference
================

This page summarizes the configuration surface exposed through the ``containers.py`` dataclasses that back the CLI and Python APIs. Every block below lists defaults and the preset bundles applied by ``from_preset(...)`` helpers. Values are shown as declared in code; all options are ASCII-safe and can be overridden via YAML or dot-keys on the CLI.

Common Blocks (unsupervised NN)
-------------------------------

The four deep imputer families (UBP, NLPCA, Autoencoder, VAE) share these structures:

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
     - Averaging mode for metrics.

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
     - Hidden non-linearity.
   * - ``layer_scaling_factor``
     - ``5.0``
     - Scales hidden widths.
   * - ``layer_schedule``
     - ``"pyramid"``
     - Width layout (``"pyramid"``, ``"constant"``, ``"linear"``).
   * - ``gamma``
     - ``2.0``
     - Focal-loss gamma.

.. list-table:: TrainConfig
   :header-rows: 1

   * - Field
     - Default
     - Description
   * - ``batch_size``
     - ``32``
     - Mini-batch size.
   * - ``learning_rate``
     - ``1e-3``
     - Base LR.
   * - ``lr_input_factor``
     - ``1.0``
     - Multiplier for first layer LR (UBP).
   * - ``l1_penalty``
     - ``0.0``
     - L1 regularization.
   * - ``early_stop_gen``
     - ``20``
     - Early-stop patience (epochs).
   * - ``min_epochs`` / ``max_epochs``
     - ``100`` / ``5000``
     - Epoch bounds.
   * - ``validation_split``
     - ``0.2``
     - Holdout fraction.
   * - ``weights_beta``
     - ``0.9999``
     - Class-weight EMA beta.
   * - ``weights_max_ratio``
     - ``1.0``
     - Cap on class-weight ratio.
   * - ``device``
     - ``"cpu"``
     - ``"cpu"``, ``"gpu"``, or ``"mps"``.

.. list-table:: TuneConfig
   :header-rows: 1

   * - Field
     - Default
     - Description
   * - ``enabled`` / ``fast``
     - ``False`` / ``True``
     - Turn on Optuna; fast uses subsampling.
   * - ``metric``
     - ``"f1"``
     - Objective metric (``f1``, ``accuracy``, ``pr_macro``, ``average_precision``, ``roc_auc``, ``precision``, ``recall``).
   * - ``n_trials``
     - ``100``
     - Number of trials.
   * - ``resume`` / ``save_db``
     - ``False`` / ``False``
     - Reuse or persist Optuna DB.
   * - ``max_samples`` / ``max_loci``
     - ``512`` / ``0``
     - Subsampling caps (0 = all).
   * - ``epochs`` / ``batch_size``
     - ``500`` / ``64``
     - Training envelope during tuning.
   * - ``eval_interval`` / ``patience``
     - ``20`` / ``10``
     - Eval cadence and prune patience.
   * - ``infer_epochs``
     - ``100``
     - Latent refinement steps for inference during tuning.
   * - ``proxy_metric_batch``
     - ``0``
     - If >0, limit validation rows per metric call.

.. list-table:: EvalConfig
   :header-rows: 1

   * - Field
     - Default
     - Description
   * - ``eval_latent_steps``
     - ``50``
     - Gradient steps to refine latent codes at eval.
   * - ``eval_latent_lr``
     - ``1e-2``
     - LR for latent refinement.
   * - ``eval_latent_weight_decay``
     - ``0.0``
     - Weight decay on latent refinement.

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
     - ``0.10``
     - Proportion to mask.
   * - ``sim_kwargs``
     - ``None``
     - Extra args forwarded to ``SimMissingTransformer``.

Field-by-field notes
--------------------

This section adds narrative detail on when to adjust each option and how blocks interact.

- **IOConfig**

  - ``prefix``: Use to isolate outputs per experiment.
  - ``ploidy``: Set to ``1`` for haploids; controls class count and decoding.
  - ``n_jobs``: Controls Optuna parallelism (``1`` keeps deterministic logging).
  - ``seed``: Honored across data splits and numpy/torch RNG where available.
  - ``scoring_averaging``: Metric averaging strategy for class-imbalanced reports.

- **ModelConfig**

  - ``latent_dim``: Governs compression strength; higher values retain more signal at the cost of capacity/overfit.
  - ``layer_schedule``/``layer_scaling_factor``: Shape hidden widths; ``pyramid`` shrinks toward the bottleneck, ``linear`` walks widths linearly, ``constant`` keeps widths fixed.
  - ``activation``: ``relu``, ``elu``, ``selu``, or ``leaky_relu``.
  - ``latent_init="pca"``: Seeds latents from PCA on the training set (UBP only) for quicker convergence.

- **TrainConfig**

  - ``lr_input_factor``: Scales the first layer LR (UBP) to stabilize early training.
  - ``weights_beta``/``weights_max_ratio``: Smooth per-class weights to avoid extreme focal weighting; increase ``weights_max_ratio`` for strongly imbalanced data.
  - ``early_stop_gen``/``min_epochs``: Patience-based early stopping with a warm-up floor.

- **TuneConfig**

  - ``fast``/``max_samples``/``max_loci``: Subsample rows/loci to keep sweeps quick.
  - ``proxy_metric_batch``: Limits validation rows during pruning for very wide matrices.
  - ``infer_epochs``: Controls latent refinement steps during tuning (``0`` disables).
  - ``eval_interval``/``patience``: Gate Optuna pruning frequency.

- **EvalConfig**

  - ``eval_latent_steps``: Set to ``0`` to disable gradient-based refinement.
  - ``eval_latent_lr``/``eval_latent_weight_decay``: Optimizer settings for latent refinement during evaluation only.

- **PlotConfig**

  - ``fmt``: ``pdf/png/jpg/jpeg/svg``.
  - ``despine``: Toggles seaborn-style axes cleanup.
  - ``show``: Default ``True``; set ``False`` on headless environments (e.g., Read the Docs, CI).

- **SimConfig**

  - ``simulate_missing``: Required for current unsupervised training; applies during evaluation/tuning to mimic missingness.
  - ``sim_strategy``: ``nonrandom``/``nonrandom_weighted`` require a tree parser; weighted variants bias masks toward common (or rare, inv) genotypes or longer branches.
  - ``sim_prop``: Masks a proportion of observed cells; the transformer prevents fully-masked rows/cols.
  - ``sim_kwargs``: Forwards advanced knobs (e.g., ``het_boost``).

Unsupervised NN presets
-----------------------

Each model exposes ``from_preset("fast" | "balanced" | "thorough")`` to seed a baseline, then allows overrides.

UBPConfig
~~~~~~~~~

Preset baseline (all presets):

- ``io``: ``verbose=False``, ``ploidy=2``.
- ``model``: ``activation="relu"``, ``layer_schedule="pyramid"``, ``latent_init="random"``.
- ``sim``: ``simulate_missing=True``, ``sim_strategy="random"``, ``sim_prop=0.2``.

Preset overrides:

- ``fast``

  - ``model``: ``latent_dim=4``, ``num_hidden_layers=1``, ``layer_scaling_factor=2.0``, ``dropout_rate=0.10``, ``gamma=1.5``.
  - ``train``: ``batch_size=256``, ``learning_rate=2e-3``, ``early_stop_gen=5``, ``min_epochs=10``, ``max_epochs=150``, ``weights_beta=0.999``, ``weights_max_ratio=5.0``.
  - ``tune``: ``enabled=True``, ``fast=True``, ``n_trials=20``, ``epochs=150``, ``batch_size=256``, ``max_samples=512``, ``eval_interval=20``, ``infer_epochs=20``, ``patience=5``.
  - ``evaluate``: ``eval_latent_steps=20``.

- ``balanced``

  - ``model``: ``latent_dim=8``, ``num_hidden_layers=2``, ``layer_scaling_factor=3.0``, ``dropout_rate=0.20``, ``gamma=2.0``.
  - ``train``: ``batch_size=128``, ``learning_rate=1e-3``, ``early_stop_gen=15``, ``min_epochs=50``, ``max_epochs=600``, ``weights_beta=0.9999``, ``weights_max_ratio=5.0``.
  - ``tune``: ``enabled=True``, ``fast=False``, ``n_trials=60``, ``epochs=200``, ``batch_size=128``, ``max_samples=2048``, ``eval_interval=10``, ``infer_epochs=50``, ``patience=10``.
  - ``evaluate``: ``eval_latent_steps=40``.

- ``thorough``

  - ``model``: ``latent_dim=16``, ``num_hidden_layers=3``, ``layer_scaling_factor=5.0``, ``dropout_rate=0.30``, ``gamma=2.5``.
  - ``train``: ``batch_size=64``, ``learning_rate=5e-4``, ``early_stop_gen=30``, ``min_epochs=100``, ``max_epochs=2000``, ``weights_beta=0.9999``, ``weights_max_ratio=5.0``.
  - ``tune``: ``enabled=True``, ``fast=False``, ``n_trials=100``, ``epochs=600``, ``batch_size=64``, ``max_samples=0`` (all), ``eval_interval=10``, ``infer_epochs=80``, ``patience=20``.
  - ``evaluate``: ``eval_latent_steps=100``.

NLPCAConfig
~~~~~~~~~~~

Preset baseline (all presets):

- ``io``: ``verbose=False``, ``ploidy=2``.
- ``train``: ``validation_split=0.20``.
- ``model``: ``activation="relu"``, ``layer_schedule="pyramid"``, ``latent_init="random"``.
- ``evaluate``: ``eval_latent_lr=1e-2``, ``eval_latent_weight_decay=0.0``.
- ``sim``: ``simulate_missing=True``, ``sim_strategy="random"``, ``sim_prop=0.2``.

Preset overrides:

- ``fast``

  - ``model``: ``latent_dim=4``, ``num_hidden_layers=1``, ``layer_scaling_factor=2.0``, ``dropout_rate=0.10``, ``gamma=1.5``.
  - ``train``: ``batch_size=256``, ``learning_rate=2e-3``, ``early_stop_gen=5``, ``min_epochs=10``, ``max_epochs=150``, ``weights_beta=0.999``, ``weights_max_ratio=5.0``.
  - ``tune``: ``enabled=True``, ``fast=True``, ``n_trials=20``, ``epochs=150``, ``batch_size=256``, ``max_samples=512``, ``eval_interval=20``, ``infer_epochs=20``, ``patience=5``.
  - ``evaluate``: ``eval_latent_steps=20``.

- ``balanced``

  - ``model``: ``latent_dim=8``, ``num_hidden_layers=2``, ``layer_scaling_factor=3.0``, ``dropout_rate=0.20``, ``gamma=2.0``.
  - ``train``: ``batch_size=128``, ``learning_rate=1e-3``, ``early_stop_gen=15``, ``min_epochs=50``, ``max_epochs=600``, ``weights_beta=0.9999``, ``weights_max_ratio=5.0``.
  - ``tune``: ``enabled=True``, ``fast=False``, ``n_trials=60``, ``epochs=200``, ``batch_size=128``, ``max_samples=2048``, ``eval_interval=10``, ``infer_epochs=50``, ``patience=10``.
  - ``evaluate``: ``eval_latent_steps=40``.

- ``thorough``

  - ``model``: ``latent_dim=16``, ``num_hidden_layers=3``, ``layer_scaling_factor=5.0``, ``dropout_rate=0.30``, ``gamma=2.5``.
  - ``train``: ``batch_size=64``, ``learning_rate=5e-4``, ``early_stop_gen=30``, ``min_epochs=100``, ``max_epochs=2000``, ``weights_beta=0.9999``, ``weights_max_ratio=5.0``.
  - ``tune``: ``enabled=True``, ``fast=False``, ``n_trials=100``, ``epochs=600``, ``batch_size=64``, ``max_samples=0``, ``eval_interval=10``, ``infer_epochs=80``, ``patience=20``.
  - ``evaluate``: ``eval_latent_steps=100``.

AutoencoderConfig
~~~~~~~~~~~~~~~~~

Preset baseline (all presets):

- ``io``: ``verbose=False``, ``ploidy=2``.
- ``train``: ``validation_split=0.20``.
- ``model``: ``activation="relu"``, ``layer_schedule="pyramid"``.
- ``evaluate``: ``eval_latent_steps=0``, ``eval_latent_lr=0``, ``eval_latent_weight_decay=0``.
- ``sim``: ``simulate_missing=True``, ``sim_strategy="random"``, ``sim_prop=0.2``.

Preset overrides:

- ``fast``

  - ``model``: ``latent_dim=4``, ``num_hidden_layers=1``, ``layer_scaling_factor=2.0``, ``dropout_rate=0.10``, ``gamma=1.5``.
  - ``train``: ``batch_size=256``, ``learning_rate=2e-3``, ``early_stop_gen=5``, ``min_epochs=10``, ``max_epochs=150``, ``weights_beta=0.999``, ``weights_max_ratio=5.0``.
  - ``tune``: ``enabled=True``, ``fast=True``, ``n_trials=20``, ``epochs=150``, ``batch_size=256``, ``max_samples=512``, ``eval_interval=20``, ``patience=5``, ``infer_epochs=0``.

- ``balanced``

  - ``model``: ``latent_dim=8``, ``num_hidden_layers=2``, ``layer_scaling_factor=3.0``, ``dropout_rate=0.20``, ``gamma=2.0``.
  - ``train``: ``batch_size=128``, ``learning_rate=1e-3``, ``early_stop_gen=15``, ``min_epochs=50``, ``max_epochs=600``, ``weights_beta=0.9999``, ``weights_max_ratio=5.0``.
  - ``tune``: ``enabled=True``, ``fast=False``, ``n_trials=60``, ``epochs=200``, ``batch_size=128``, ``max_samples=2048``, ``eval_interval=10``, ``patience=10``, ``infer_epochs=0``.

- ``thorough``

  - ``model``: ``latent_dim=16``, ``num_hidden_layers=3``, ``layer_scaling_factor=5.0``, ``dropout_rate=0.30``, ``gamma=2.5``.
  - ``train``: ``batch_size=64``, ``learning_rate=5e-4``, ``early_stop_gen=30``, ``min_epochs=100``, ``max_epochs=2000``, ``weights_beta=0.9999``, ``weights_max_ratio=5.0``.
  - ``tune``: ``enabled=True``, ``fast=False``, ``n_trials=100``, ``epochs=600``, ``batch_size=64``, ``max_samples=0``, ``eval_interval=10``, ``patience=20``, ``infer_epochs=0``.

VAEConfig + VAEExtraConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~

Common blocks align with Autoencoder (no latent refinement at eval). Extra block:

.. list-table:: VAEExtraConfig
   :header-rows: 1

   * - Field
     - Default
     - Description
   * - ``kl_beta``
     - ``1.0``
     - Final KL weight.

Presets:

Preset baseline (all presets):

- ``io``: ``verbose=False``, ``ploidy=2``.
- ``train``: ``validation_split=0.20``.
- ``model``: ``activation="relu"``, ``layer_schedule="pyramid"``.
- ``evaluate``: ``eval_latent_steps=0``, ``eval_latent_lr=0``, ``eval_latent_weight_decay=0``.
- ``sim``: ``simulate_missing=True``, ``sim_strategy="random"``, ``sim_prop=0.2``.

Preset overrides:

- ``fast``

  - ``model``: ``latent_dim=4``, ``num_hidden_layers=1``, ``layer_scaling_factor=2.0``, ``dropout_rate=0.10``, ``gamma=1.5``.
  - ``vae``: ``kl_beta=0.5``.
  - ``train``: ``batch_size=256``, ``learning_rate=2e-3``, ``early_stop_gen=5``, ``min_epochs=10``, ``max_epochs=150``, ``weights_beta=0.999``, ``weights_max_ratio=5.0``.
  - ``tune``: ``enabled=True``, ``fast=True``, ``n_trials=20``, ``epochs=150``, ``batch_size=256``, ``max_samples=512``, ``eval_interval=20``, ``patience=5``, ``infer_epochs=0``.

- ``balanced``

  - ``model``: ``latent_dim=8``, ``num_hidden_layers=2``, ``layer_scaling_factor=3.0``, ``dropout_rate=0.20``, ``gamma=2.0``.
  - ``vae``: ``kl_beta=1.0``.
  - ``train``: ``batch_size=128``, ``learning_rate=1e-3``, ``early_stop_gen=15``, ``min_epochs=50``, ``max_epochs=600``, ``weights_beta=0.9999``, ``weights_max_ratio=5.0``.
  - ``tune``: ``enabled=True``, ``fast=False``, ``n_trials=60``, ``epochs=200``, ``batch_size=128``, ``max_samples=2048``, ``eval_interval=10``, ``patience=10``, ``infer_epochs=0``.

- ``thorough``

  - ``model``: ``latent_dim=16``, ``num_hidden_layers=3``, ``layer_scaling_factor=5.0``, ``dropout_rate=0.30``, ``gamma=2.5``.
  - ``vae``: ``kl_beta=1.0``.
  - ``train``: ``batch_size=64``, ``learning_rate=5e-4``, ``early_stop_gen=30``, ``min_epochs=100``, ``max_epochs=2000``, ``weights_beta=0.9999``, ``weights_max_ratio=5.0``.
  - ``tune``: ``enabled=True``, ``fast=False``, ``n_trials=100``, ``epochs=600``, ``batch_size=64``, ``max_samples=0``, ``eval_interval=10``, ``patience=20``, ``infer_epochs=0``.

Deterministic imputers
----------------------

These mirror the NN config ergonomics but with simpler blocks.

MostFrequentConfig
~~~~~~~~~~~~~~~~~~

Uses ``IOConfig``, ``PlotConfig``, ``SimConfig``, and a deterministic split.

.. list-table:: MostFrequent blocks
   :header-rows: 1

   * - Block
     - Fields (defaults)
     - Notes
   * - ``split``
     - ``test_size=0.2``, ``test_indices=None``
     - Sample-level holdout.
   * - ``algo``
     - ``by_populations=False``, ``default=0``, ``missing=-1``
     - Compute modes globally or per population.

Preset ``from_preset("fast"|"balanced"|"thorough")`` aligns logging + split + simulated missing: ``simulate_missing=True``, ``sim_strategy="random"``, ``sim_prop=0.2``.

RefAlleleConfig
~~~~~~~~~~~~~~~

Same structure as MostFrequent but ``algo`` only includes ``missing=-1``. Presets mirror MostFrequent (logging + split + simulated missing toggles).

Supervised wrappers (RF / HistGB)
---------------------------------

Supervised configs live at the bottom of ``containers.py`` and control classical models used in wrappers.

.. list-table:: Shared blocks
   :header-rows: 1

   * - Block
     - Fields (defaults)
     - Description
   * - ``IOConfigSupervised``
     - ``prefix="pgsui"``, ``seed=None``, ``n_jobs=1``, ``verbose=False``, ``debug=False``
     - Run identity and logging.
   * - ``PlotConfigSupervised``
     - ``fmt="pdf"``, ``dpi=300``, ``fontsize=18``, ``despine=True``, ``show=False``
     - Figure styling.
   * - ``TrainConfigSupervised``
     - ``validation_split=0.20``
     - Sample-level split (validated in ``__post_init__``).
   * - ``ImputerConfigSupervised``
     - ``n_nearest_features=10``, ``max_iter=10``
     - IterativeImputer scaffolding.
   * - ``SimConfigSupervised``
     - ``prop_missing=0.5``, ``strategy="random_inv_genotype"``, ``het_boost=2.0``, ``missing_val=-1``
     - Simulated missingness for evaluation.
   * - ``TuningConfigSupervised``
     - ``enabled=True``, ``n_trials=100``, ``metric="pr_macro"``, ``n_jobs=8``, ``fast=True``
     - Optuna knobs for classical models.

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
   * - ``min_samples_split`` / ``min_samples_leaf``
     - ``2`` / ``1``
     - Splitting controls.
   * - ``max_features``
     - ``"sqrt"``
     - Features per split.
   * - ``criterion``
     - ``"gini"``
     - Split metric.
   * - ``class_weight``
     - ``"balanced"``
     - Class weighting.

.. list-table:: HGBModelConfig
   :header-rows: 1

   * - Field
     - Default
     - Description
   * - ``n_estimators``
     - ``100`` (maps to ``max_iter``)
     - Boosting rounds.
   * - ``learning_rate``
     - ``0.1``
     - Step size.
   * - ``max_depth``
     - ``None``
     - Depth cap.
   * - ``min_samples_leaf``
     - ``1``
     - Leaf size.
   * - ``max_features``
     - ``1.0``
     - Fraction of features.
   * - ``n_iter_no_change`` / ``tol``
     - ``10`` / ``1e-7``
     - Early-stop patience and tolerance.

Presets:

- RF ``fast``

  - ``model``: ``n_estimators=50``, ``max_depth=None``.
  - ``imputer``: ``max_iter=5``.
  - ``io``: ``n_jobs=1``.
  - ``tune``: ``enabled=False``.
- RF ``balanced``

  - ``model``: ``n_estimators=200``, ``max_depth=None``.
  - ``imputer``: ``max_iter=10``.
  - ``io``: ``n_jobs=1``.
  - ``tune``: ``enabled=False``, ``n_trials=100``.
- RF ``thorough``

  - ``model``: ``n_estimators=500``, ``max_depth=50``.
  - ``imputer``: ``max_iter=20``.
  - ``io``: ``n_jobs=1``.
  - ``tune``: ``enabled=False``, ``n_trials=250``.

- HGB ``fast``

  - ``model``: ``n_estimators=50``, ``learning_rate=0.2``, ``max_depth=None``.
  - ``imputer``: ``max_iter=5``.
  - ``tune``: ``enabled=False``, ``n_trials=50``.
- HGB ``balanced``

  - ``model``: ``n_estimators=150``, ``learning_rate=0.1``, ``max_depth=None``.
  - ``imputer``: ``max_iter=10``.
  - ``tune``: ``enabled=False``, ``n_trials=100``.
- HGB ``thorough``

  - ``model``: ``n_estimators=500``, ``learning_rate=0.05``, ``n_iter_no_change=20``, ``max_depth=None``.
  - ``imputer``: ``max_iter=20``.
  - ``tune``: ``enabled=False``, ``n_trials=250``.

Quick override tips
-------------------

- YAML: supply nested keys matching the dataclasses; environment variables in YAML are expanded (``${VAR}`` or ``${VAR:default}``).
- Dict or CLI dot-keys: use ``apply_dot_overrides`` semantics, e.g. ``model.latent_dim=16`` or ``tune.fast=False``.
- Presets: call ``Config.from_preset("fast")`` (or choose ``balanced`` / ``thorough``) before applying overrides to keep intention explicit.
