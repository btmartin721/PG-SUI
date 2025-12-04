PG-SUI Tutorial
===============

Overview
--------

**PG-SUI** (Population Genomic Supervised & Unsupervised Imputation) performs missing-data imputation on SNP genotype matrices using **Deterministic**, **Unsupervised**, and **Supervised** models. It integrates tightly with `SNPio <https://github.com/btmartin721/SNPio>`_ for reading, filtering, and encoding genotypes, and emphasizes **unsupervised deep learning** methods tuned for genomic class imbalance and diploid/variable ploidy data. Unsupervised neural approaches (e.g., non-linear PCA and autoencoding families) are inspired by prior work on representation learning and generative modeling (Scholz et al., 2005; Hinton & Salakhutdinov, 2006; Kingma & Welling, 2013), while the Unsupervised Backpropagation approach follows the imputation framing of Gashler et al. (2016).

What's new
----------

- **Dataclass-based configuration** at the API level. Each imputer is configured with a typed ``*Config`` dataclass (e.g., ``VAEConfig``, ``UBPConfig``, ``NLPCAConfig``) instead of ad-hoc kwargs.
- **Presets** (``fast``, ``balanced``, ``thorough``) available via ``*.from_preset("...")`` and the CLI ``--preset`` flag.
- **YAML configuration files** (``--config path.yaml``) to persist experiments; YAML merges with presets and can be partially specified.
- **Refactored CLI** with a clear precedence model:

  ``code defaults  <  preset (--preset)  <  YAML (--config)  <  explicit CLI flags  <  --set k=v``

  where ``--set`` applies deep dot-path overrides (e.g., ``--set model.latent_dim=16``).

- **New visualizations**, including a cross-model **radar (spider) plot** summarizing macro-F1, macro-PR, accuracy, and HET-F1, plus updated confusion matrices, per-class PR curves, zygosity bars, and training curves.
- **Unified I/O and plotting** via nested config sections (``io``, ``training``, ``tuning``, ``plot``) across all imputers.
- **Consistent fit/transform contract**: pass a ``GenotypeData`` to the model **at instantiation**, then call ``fit()`` followed by ``transform()`` (no arguments).

How to run PG-SUI
-----------------

- **Command line** — run ``pg-sui`` directly for scripted, reproducible workflows. This tutorial includes a copy/paste CLI quickstart and the full precedence model.
- **Desktop GUI (Electron)** — launch ``pgsui-gui`` for a point-and-click wrapper around the CLI. The GUI writes the exact CLI command it executes; see :doc:`gui` for a guided walkthrough.

Model Families
--------------

- **Deterministic (Baselines)**
  - ``ImputeMostFrequent`` — per-locus mode imputation (global or population-aware).
  - ``ImputeRefAllele`` — fills missing diploid genotypes with the REF (0) state.

- **Unsupervised (Deep Learning)**
  - ``ImputeNLPCA`` — Non-linear PCA with optional latent optimization (Scholz et al., 2005).
  - ``ImputeUBP`` — Unsupervised Backpropagation with joint latent + weight training (Gashler et al., 2016).
  - ``ImputeAutoencoder`` — Standard autoencoder reconstruction (Hinton & Salakhutdinov, 2006).
  - ``ImputeVAE`` — Variational Autoencoder with KL regularization (Kingma & Welling, 2013).

- **Supervised (Machine Learning)**
  - ``ImputeHistGradientBoosting`` — histogram-based gradient boosting classifier.
  - ``ImputeRandomForest`` — random forest classifier.

.. note::

   **API change:** ``fit()`` and ``transform()`` do **not** accept inputs. Each model is constructed with a ``genotype_data`` object and a typed ``*Config``. Then call ``fit()`` and ``transform()`` in sequence.

Installation
------------

Install PG-SUI with pip (ideally in a fresh virtual environment). Add the GUI extras if you want the Electron desktop app:

.. code-block:: bash

    pip install pg-sui            # CLI + Python API
    pip install "pg-sui[gui]"     # optional desktop GUI
    pgsui-gui-setup               # one-time Electron dependency install

See :doc:`install` for more detail. PG-SUI expects genotype I/O via **SNPio**. See SNPio docs for VCF/PHYLIP/STRUCTURE/GENEPOP readers and Docker/conda installs.

Data Input, Encodings, and Conventions
--------------------------------------

PG-SUI uses SNPio's ``GenotypeData`` object:

- **Inputs**: VCF, PHYLIP, STRUCTURE, or GENEPOP, plus optional ``popmap`` (recommended).
- **Working encoding**: **0/1/2** for diploids (REF/HET/ALT) with **-9** for missing.
- **Evaluation labels**: ``["REF", "HET", "ALT"]`` for diploids; haploids collapse to two classes.
- **IUPAC handling**: decoding/encoding utilities are provided by SNPio.

CLI Quickstart (copy/paste)
---------------------------

1. Place your VCF/PHYLIP/STRUCTURE/GENEPOP file and optional popmap in a working directory.
2. Run a preset with two deep models (edit paths/models as needed):

.. code-block:: bash

    pg-sui \
      --vcf pgsui/example_data/vcf_files/phylogen_nomx.vcf.gz \
      --popmap pgsui/example_data/popmaps/test.popmap \
      --models ImputeUBP ImputeVAE \
      --preset balanced \
      --prefix demo

3. Optional: add a YAML config (``--config vae_balanced.yaml``), apply quick tweaks with ``--set key=value``, or disable simulated missingness with ``--simulate-missing``.
4. Outputs and plots are written under ``<prefix>_output/``. The CLI prints paths as it runs.

Prefer a visual workflow? Launch the desktop app with ``pgsui-gui`` and follow :doc:`gui`.

.. _simulated_missingness:

Simulated missingness strategies
--------------------------------

PG-SUI evaluates imputers on a **simulated masking** of observed genotypes so every model sees the same held-out entries. The mask is created once per run and reused across selected models. Control it via YAML (``sim`` section) or the CLI flags ``--sim-strategy`` and ``--sim-prop``; disable entirely with ``--simulate-missing``.

- ``random`` — Uniform masking across eligible cells until the target proportion is reached (baseline).
- ``random_weighted`` — Column-wise masking weighted by observed genotype frequencies; common genotypes are masked more often.
- ``random_weighted_inv`` — Column-wise masking weighted inversely by observed genotype frequencies; rarer genotypes are masked more often.
- ``nonrandom`` — Phylogenetically clustered masking. Requires a SNPio genotype tree; clades are sampled uniformly (tips/internal) and masked together.
- ``nonrandom_weighted`` — Phylogenetically clustered masking with clades sampled proportional to branch length (emphasizes longer, more diverged branches).

Use ``--sim-prop`` (0–1) to set the proportion of observed entries to hide. For diagnostics that rely only on organically missing calls, pass ``--simulate-missing`` (store-false flag) to skip simulated masking.

Summary table
^^^^^^^^^^^^^

.. list-table:: Summary of missing data simulation strategies
   :header-rows: 1
   :widths: auto

   * - Strategy
     - Selection logic
     - Biologically mimics
     - Expected Difficulty
   * - Random
     - Uniform coin flip per cell
     - Random sequencing errors; random read-depth fluctuations
     - Easy
   * - Random weighted
     - Probability proportional to genotype frequency (masks common)
     - Reference bias: common allele captured more; technical artifacts
     - Moderate
   * - Random weighted inv
     - Probability inversely proportional to genotype frequency (masks rare)
     - Allelic dropout / minor-allele loss; ascertainment bias
     - Hard
   * - Nonrandom
     - Pick a random clade, mask the locus in that clade (tree required)
     - PCR failure from primer-site mutation within a clade
     - Hard
   * - Nonrandom weighted
     - Pick clade proportional to branch length (tree required)
     - Divergence-driven dropout on long, isolated branches
     - Very hard

Quick Start (End-to-End, Dataclass API)
---------------------------------------

.. code-block:: python

    # SNPio: load genotype data
    from snpio import VCFReader

    # PG-SUI: top-level imports for configs and models
    from pgsui import VAEConfig, ImputeVAE

    gd = VCFReader(
        filename="pgsui/example_data/vcf_files/phylogen_nomx.vcf.gz",
        popmapfile="pgsui/example_data/popmaps/test.popmap",
        prefix="pgsui_demo",
        force_popmap=True,
        plot_format="pdf",
    )

    # Start from a preset, then customize a few fields
    cfg = VAEConfig.from_preset("balanced")
    cfg.io.prefix = "pgsui_demo"
    cfg.model.latent_dim = 16
    cfg.tune.enabled = True
    cfg.tune.n_trials = 100
    cfg.tune.metric = "pr_macro"
    cfg.plot.show = False
    cfg.vae.kl_beta = 1.0  # VAE-specific (Kingma & Welling, 2013)

    model = ImputeVAE(genotype_data=gd, config=cfg)
    model.fit()
    genotypes_iupac = model.transform()  # returns decoded IUPAC strings

Using Presets Programmatically
------------------------------

All ``*Config`` dataclasses provide ``fast``, ``balanced``, and ``thorough`` presets:

.. code-block:: python

    from pgsui import NLPCAConfig, ImputeNLPCA, UBPConfig, ImputeUBP

    nlpca_cfg = NLPCAConfig.from_preset("fast")       # prioritizes speed
    ubp_cfg   = UBPConfig.from_preset("thorough")     # prioritizes performance

    # Override selected fields after preset expansion
    ubp_cfg.model.num_hidden_layers = 3
    ubp_cfg.io.prefix = "ubp_run1"

    # Instantiate and run
    nlpca = ImputeNLPCA(genotype_data=gd, config=nlpca_cfg)
    ubp   = ImputeUBP(genotype_data=gd, config=ubp_cfg)

    nlpca.fit()
    X_nlpca = nlpca.transform()

    ubp.fit()
    X_ubp   = ubp.transform()

YAML Configuration Files
------------------------

You can store experiments in YAML and load them from the CLI or Python. YAML merges with presets and only needs to include fields you want to override.

**Example YAML (``vae_balanced.yaml``)**

.. code-block:: yaml

    io:
      prefix: "vae_demo"
      plot_format: "pdf"

    model:
      latent_dim: 16
      num_hidden_layers: 3
      layer_scaling_factor: 4.0
      dropout_rate: 0.20
      hidden_activation: "relu"

    train:
      learning_rate: 0.0008
      early_stop_gen: 15
      min_epochs: 50
      max_epochs: 1000
      validation_split: 0.20
      device: "cpu"
      seed: 42

    vae:
      kl_beta: 1.0
      kl_warmup: 30
      kl_ramp: 150

    tune:
      enabled: true
      metric: "pr_macro"
      n_trials: 100
      fast: true
      max_samples: 1024
      patience: 10

    plot:
      show: false
      dpi: 300

Loading YAML in Python:

.. code-block:: python

    from pgsui import VAEConfig, ImputeVAE
    from pgsui.data_processing.config import load_yaml_to_dataclass

    cfg = load_yaml_to_dataclass("vae_balanced.yaml", VAEConfig)
    model = ImputeVAE(genotype_data=gd, config=cfg)
    model.fit()
    X_vae = model.transform()

Command-Line Interface (CLI)
----------------------------

Use the CLI for automation or to mirror runs configured in the GUI (the desktop app assembles these same flags under the hood). The ``pg-sui`` CLI supports running one or more models with the same dataset and a shared precedence rule set.

**Precedence model** (highest last):

``code defaults  <  preset (--preset)  <  YAML (--config)  <  explicit CLI flags  <  --set k=v``

- ``--preset`` selects a baseline preset.
- ``--config`` applies YAML on top of the preset.
- Explicit CLI flags (if provided) override YAML.
- ``--set`` applies deep dot-path overrides for final tweaks.

**Simulation controls** (see :ref:`simulated_missingness` for option details)

- ``--sim-strategy``: choose how simulated masking selects loci.
- ``--sim-prop``: set the proportion of observed entries to hide when creating the evaluation mask.
- ``--simulate-missing``: disable simulated masking entirely for the run (store-false flag). Leave it unset to inherit the preset/YAML choice or force a value via ``--set sim.simulate_missing=True``.

**Typical CLI usage**

.. code-block:: bash

    # Minimal run with a preset
    pg-sui \
      --vcf pgsui/example_data/vcf_files/phylogen_nomx.vcf.gz \
      --popmap pgsui/example_data/popmaps/test.popmap \
      --models ImputeUBP ImputeVAE \
      --preset balanced \
      --prefix demo

    # Use a YAML config and override a couple fields
    pg-sui \
      --vcf data.vcf.gz \
      --popmap pops.popmap \
      --models ImputeVAE \
      --preset thorough \
      --config vae_balanced.yaml \
      --set io.prefix=vae_vs_ubp \
      --set model.latent_dim=32

    # Deterministic baselines for a quick yardstick
    pg-sui \
      --vcf data.vcf.gz \
      --popmap pops.popmap \
      --models ImputeMostFrequent ImputeRefAllele \
      --preset fast \
      --prefix baselines

    # Override simulated-missingness globally from the CLI
    pg-sui \
      --vcf cohort.vcf.gz \
      --popmap pops.popmap \
      --models ImputeUBP ImputeNLPCA \
      --preset balanced \
      --sim-strategy random_weighted_inv \
      --sim-prop 0.30 \
      --set io.prefix=balanced_sim_override

    # Temporarily disable simulated masking (store_false flag)
    pg-sui \
      --vcf cohort.vcf.gz \
      --models ImputeVAE \
      --simulate-missing \
      --set io.prefix=vae_observed_only

Deterministic Models (Configs)
------------------------------

**ImputeMostFrequent**

.. code-block:: python

    from pgsui import MostFrequentConfig, ImputeMostFrequent

    cfg = MostFrequentConfig.from_preset("fast")
    cfg.io.prefix = "mode_imp"
    cfg.algo.by_populations = True  # pop-aware if popmap provided

    model = ImputeMostFrequent(genotype_data=gd, config=cfg)
    model.fit()
    X_mode = model.transform()

**ImputeRefAllele**

.. code-block:: python

    from pgsui import RefAlleleConfig, ImputeRefAllele

    cfg = RefAlleleConfig.from_preset("fast")
    cfg.io.prefix = "ref_imp"

    model = ImputeRefAllele(genotype_data=gd, config=cfg)
    model.fit()
    X_ref = model.transform()

Unsupervised Deep Learning (Configs)
------------------------------------

**Non-linear PCA (ImputeNLPCA)** *(Scholz et al., 2005)*

.. code-block:: python

    from pgsui import NLPCAConfig, ImputeNLPCA

    cfg = NLPCAConfig.from_preset("balanced")
    cfg.io.prefix = "nlpca_run"
    model = ImputeNLPCA(genotype_data=gd, config=cfg)
    model.fit()
    X_nlpca = model.transform()

**Unsupervised Backpropagation (ImputeUBP)** *(Gashler et al., 2016)*

.. code-block:: python

    from pgsui import UBPConfig, ImputeUBP

    cfg = UBPConfig.from_preset("thorough")
    cfg.io.prefix = "ubp_run"
    model = ImputeUBP(genotype_data=gd, config=cfg)
    model.fit()
    X_ubp = model.transform()

**Standard Autoencoder (ImputeAutoencoder)** *(Hinton & Salakhutdinov, 2006)*

.. code-block:: python

    from pgsui import AutoencoderConfig, ImputeAutoencoder

    cfg = AutoencoderConfig.from_preset("balanced")
    cfg.io.prefix = "ae_run"
    cfg.model.dropout_rate = 0.15
    model = ImputeAutoencoder(genotype_data=gd, config=cfg)
    model.fit()
    X_ae = model.transform()

**Variational Autoencoder (ImputeVAE)** *(Kingma & Welling, 2013)*

.. code-block:: python

    from pgsui import VAEConfig, ImputeVAE

    cfg = VAEConfig.from_preset("balanced")
    cfg.io.prefix = "vae_run"
    cfg.vae.kl_beta = 1.0
    model = ImputeVAE(genotype_data=gd, config=cfg)
    model.fit()
    X_vae = model.transform()

Supervised Models (Configs)
---------------------------

**ImputeHistGradientBoosting**

.. code-block:: python

    from pgsui import HGBConfig, ImputeHistGradientBoosting

    cfg = HGBConfig.from_preset("balanced")
    cfg.io.prefix = "hgb_run"
    cfg.imputer.max_iter = 12
    cfg.sim.prop_missing = 0.35

    model = ImputeHistGradientBoosting(genotype_data=gd, config=cfg)
    model.fit()
    X_hgb = model.transform()

**ImputeRandomForest**

.. code-block:: python

    from pgsui import RFConfig, ImputeRandomForest

    cfg = RFConfig.from_preset("balanced")
    cfg.io.prefix = "rf_run"
    cfg.model.n_estimators = 300
    cfg.imputer.n_nearest_features = 64

    model = ImputeRandomForest(genotype_data=gd, config=cfg)
    model.fit()
    X_rf = model.transform()

Common Config Sections (Fields at a Glance)
-------------------------------------------

All ``*Config`` dataclasses expose nested sections (names vary a little by family). The essentials:

- **io** -- run prefix, logging verbosity, random seeds, output format.
- **model** -- estimator architecture (latent dimension, layer schedule, tree counts, etc.); deterministic configs instead expose ``algo``.
- **train** -- optimisation knobs for neural models (batch size, learning rate, early stopping, validation split, device, class-weight limits).
- **tune** -- Optuna envelope for the neural stack; retained for supervised configs for API compatibility.
- **evaluate** / **split** -- latent optimisation controls for unsupervised models or held-out split definitions for deterministic ones.
- **imputer** and **sim** -- IterativeImputer and simulated-missingness settings unique to supervised models.
- **plot** -- export format, DPI, fonts, and whether to display figures interactively.
- **sim** -- simulated-missingness controls (strategy, proportion, enable/disable).

Visualization & Reports
-----------------------

After ``fit()``, each model writes plots and metrics under:

``{prefix}_output/{Family}/plots/{Model}/`` and ``{prefix}_output/{Family}/metrics/{Model}/``

Key figures
^^^^^^^^^^^

- **Radar (spider) summary** across models: macro-F1, macro-PR, accuracy, HET-F1.
- **Confusion matrices** (overall and per-zygosity).
- **Per-class precision-recall curves** and macro-averaged PR.
- **Zygosity bar charts** (REF/HET/ALT) for error composition.
- **Training curves** (loss/metric vs. epoch) for deep models.
- **Feature importances** for supervised tree-based models (if enabled).

Common Evaluation
-----------------

Metrics are stratified by zygosity (REF/HET/ALT for diploids; binary for haploids) and can also be summarized under 10-base IUPAC encodings. Macro-averaged F1 and macro-PR are emphasized to handle class imbalance. Summary CSV/JSON files accompany figures to support downstream comparison and aggregation.

Tips for Performance & Reproducibility
--------------------------------------

- Enable Optuna with ``tune.enabled = True`` and increase ``tune.n_trials`` for more robust hyperparameters.
- Use ``train.device="gpu"`` (CUDA) or ``"mps"`` (Apple Silicon) when available.
- Prefer ``tune.metric="pr_macro"`` on imbalanced datasets.
- Set ``train.seed`` for reproducible splits, latent initialisation, and Optuna sampling.

Typical Workflow
----------------

1. **Read + filter + encode** with SNPio (``GenotypeData``; optional ``GenotypeEncoder`` for decoding).
2. **Baseline** with ``ImputeMostFrequent`` or ``ImputeRefAllele`` to establish a floor.
3. **Unsupervised model** (e.g., ``ImputeVAE`` or ``ImputeUBP``) with tuning enabled.
4. **Optional supervised models** (HGB/RF) to benchmark against deep models.
5. **Compare reports** (radar summary, macro-PR/F1, zygosity, confusion matrices).
6. **Decode/Export** final matrices to IUPAC or downstream formats as needed.

Minimal API Reference
---------------------

All imputers follow the same high-level pattern:

.. code-block:: python

    model = SomeImputer(genotype_data=gd, config=SomeConfig.from_preset("balanced"))
    model.fit()                      # trains; writes plots/reports
    X_imputed = model.transform()    # imputes missing alleles and returns IUPAC strings

References
----------

Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *Journal of Artificial Intelligence Research*, 16, 321-357.

Gashler, M. S., Smith, M. R., Morris, R., & Martinez, T. (2016). Missing value imputation with unsupervised backpropagation. *Computational Intelligence*, 32(2), 196-215.

Hinton, G. E., & Salakhutdinov, R. R. (2006). Reducing the dimensionality of data with neural networks. *Science*, 313(5786), 504-507.

Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes. *arXiv preprint* arXiv:1312.6114.

Scholz, M., Kaplan, F., Guy, C. L., Kopka, J., & Selbig, J. (2005). Non-linear PCA: a missing data approach. *Bioinformatics*, 21(20), 3887-3895.
