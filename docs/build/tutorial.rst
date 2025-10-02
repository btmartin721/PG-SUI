PG-SUI Tutorial
===============

Overview
--------

**PG-SUI** (Population Genomic Supervised & Unsupervised Imputation) performs missing-data imputation on SNP genotype matrices using **Deterministic**, **Unsupervised**, and **Supervised** models. It integrates tightly with `SNPio <https://github.com/btmartin721/SNPio>`_ for reading, filtering, and encoding genotypes, and emphasizes **unsupervised deep learning** methods designed for genomic class imbalance and diploid/variable ploidy data.

What's new
----------

- **Dataclass-based configuration** at the API level. Each imputer is configured with a typed ``*Config`` dataclass (e.g., ``VAEConfig``, ``UBPConfig``, ``NLPCAConfig``) instead of arbitrary keyword arguments.
- **Presets** (``fast``, ``balanced``, ``thorough``) available via ``*.from_preset("...")`` and the CLI ``--preset`` flag.
- **YAML configuration files** (``--config path.yaml``) to persist experiments; YAML merges with presets and can be partially specified.
- **Refactored CLI** with a clear precedence model:
    - ``code defaults  <  preset (--preset)  <  YAML (--config)  <  explicit CLI flags  <  --set k=v``
    - where ``--set`` applies deep dot-path overrides (e.g., ``--set training.model_latent_dim=16``).

- **New visualizations**, including a cross-model **radar plot** (spider chart) summarizing key metrics (macro-F1, macro-PR, accuracy, HET-F1), alongside updated confusion matrices, per-class PR curves, zygosity bars, and progress logs.
- **Unified I/O and plotting** via nested config sections (``io``, ``training``, ``tuning``, ``plot``) across all imputers.
- **Consistent fit/transform contract**: models use the provided ``GenotypeData`` at instantiation; call ``fit()`` then ``transform()`` (no arguments).

Model Families
--------------

- **Deterministic (Baselines)**
    - ``ImputeMostFrequent`` — per-locus mode imputation (global or population-aware).
    - ``ImputeRefAllele`` — fill missing genotypes with the REF genotype (0 in 0/1/2).

- **Unsupervised (Deep Learning)**
    - ``ImputeNLPCA`` — Non-linear PCA with optional latent optimization.
    - ``ImputeUBP`` — Unsupervised Backpropagation with joint latent + weight training.
    - ``ImputeAutoencoder`` — Standard autoencoder reconstruction.
    - ``ImputeVAE`` — Variational Autoencoder with KL regularization.

- **Supervised (Machine Learning)**
    - ``ImputeHistGradientBoosting`` — histogram-based gradient boosting classifier.
    - ``ImputeRandomForest`` — random forest classifier.

.. note::

   **API change:** ``fit()`` and ``transform()`` no longer accept inputs. Each model receives a ``genotype_data`` object **at instantiation** and then exposes ``fit()`` followed by ``transform()`` (no arguments).

Installation
------------

Install PG-SUI with pip (ideally in a fresh virtual environment):

.. code-block:: bash

    pip install pg-sui

PG-SUI expects genotype I/O via **SNPio**. See SNPio docs for VCF/PHYLIP/STRUCTURE/GENEPOP readers and Docker/conda installs.

Data In, Encodings, and Conventions
-----------------------------------

PG-SUI uses SNPio's ``GenotypeData`` object:

- **Inputs**: VCF, PHYLIP, STRUCTURE, or GENEPOP plus optional ``popmap`` (recommended).
- **Working encoding**: **0/1/2** for diploids (REF/HET/ALT) with **-9** for missing.
- **Evaluation labels**: ``["REF", "HET", "ALT"]`` for diploids; haploids collapse to 2 classes.
- **IUPAC handling**: decoding/encoding utilities are provided by SNPio.

Quick Start (End-to-End, Dataclass API)
---------------------------------------

.. code-block:: python

    # --- SNPio: load genotype data ---
    from snpio import VCFReader

    # --- PG-SUI: dataclass-configured model (example: VAE) ---
    from pgsui import VAEConfig  # dataclass
    from pgsui import ImputeVAE # imputer

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
    cfg.training.model_latent_dim = 16
    cfg.tuning.tune = True
    cfg.tuning.n_trials = 100
    cfg.tuning.metric = "pr_macro"
    cfg.plot.show_plots = False
    cfg.vae.kl_beta = 1.0  # VAE-specific

    vae = ImputeVAE(genotype_data=gd, config=cfg)
    vae.fit()
    X012_imputed = vae.transform()  # returns 0/1/2 genotype numpy array

Using Presets Programmatically
------------------------------

All ``*Config`` dataclasses provide presets with sensible defaults:

.. code-block:: python

    from pgsui import NLPCAConfig, ImputeNLPCA, UBPConfig, ImputeUBP

    nlpca_cfg = NLPCAConfig.from_preset("fast")          # smallest, quickest
    ubp_cfg   = UBPConfig.from_preset("thorough")        # strongest, slower

    # You can override selected fields after preset expansion:
    ubp_cfg.model.num_hidden_layers = 3
    ubp_cfg.io.prefix = "ubp_run1"  # output prefix 

    # Instantiate and run
    nlpca = ImputeNLPCA(genotype_data=gd, config=nlpca_cfg)
    ubp   = ImputeUBP(genotype_data=gd, config=ubp_cfg)
    nlpca.fit(); X_nlpca = nlpca.transform()
    ubp.fit();   X_ubp   = ubp.transform()

YAML Configuration Files
------------------------

You can store experiments in YAML and load them from the CLI or Python. YAML merges with presets and only needs to include fields you want to override.

**Example YAML (``vae_balanced.yaml``)**

.. code-block:: yaml

    # Model-specific section (example: VAEConfig)
    io:
      prefix: "vae_demo"
      plot_format: "pdf"
    training:
      model_latent_dim: 16
      model_num_hidden_layers: 3
      model_hidden_layer_sizes: [256, 128, 64]
      model_learning_rate: 0.0001
      model_beta: 1.0        # VAE only
      model_early_stop_gen: 20
      model_min_epochs: 20
      model_validation_split: 0.20
      device: "cpu"
      seed: 42
    tuning:
      tune: true
      n_trials: 100
      metric: "pr_macro"
      n_jobs: 8
      weights_temperature: 3.0
      weights_alpha: 1.0
      weights_normalize: true
    plot:
      show_plots: false
      dpi: 300

Loading YAML in Python:

.. code-block:: python

    from pgsui import VAEConfig, load_yaml_to_dataclass
    cfg = load_yaml_to_dataclass("vae_balanced.yaml", VAEConfig)
    vae = ImputeVAE(genotype_data=gd, config=cfg)
    vae.fit()
    X_vae = vae.transform()

Command-Line Interface (CLI)
----------------------------

The ``pg-sui`` CLI supports running one or more models with the same dataset and shared precedence rules.

**Precedence model** (highest last):

``code defaults  <  preset (--preset)  <  YAML (--config)  <  explicit CLI flags  <  --set k=v``

- ``--preset`` chooses a baseline preset.
- ``--config`` applies YAML on top of the preset.
- Explicit CLI flags (if provided) override YAML.
- ``--set`` applies deep dot-path overrides for final tweaks.

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
        --set training.model_latent_dim=32

    # Deterministic baseline for a quick yardstick
    pg-sui \
        --vcf data.vcf.gz \
        --popmap pops.popmap \
        --models ImputeMostFrequent ImputeRefAllele \
        --preset fast \
        --prefix baselines

Deterministic Models (Configs)
------------------------------

**ImputeMostFrequent**

.. code-block:: python

    from pgsui import MostFrequentConfig, ImputeMostFrequent

    cfg = MostFrequentConfig.from_preset("fast")
    cfg.io.prefix = "mode_imp"
    cfg.algorithm.by_population = True  # make pop-aware if popmap is provided

    model = ImputeMostFrequent(genotype_data=gd, config=cfg)
    model.fit()
    X_mode = model.transform()

**ImputeRefAllele**

.. code-block:: python

    from pgsui.data_processing.config import RefAlleleConfig
    from pgsui.impute.deterministic import ImputeRefAllele

    cfg = RefAlleleConfig.from_preset("fast")
    cfg.io.prefix = "ref_imp"

    model = ImputeRefAllele(genotype_data=gd, config=cfg)
    model.fit()
    X_ref = model.transform()

Unsupervised Deep Learning (Configs)
------------------------------------

**Non-linear PCA (ImputeNLPCA)**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from pgsui.data_processing.config import NLPCAConfig
    from pgsui.impute.unsupervised.imputers import ImputeNLPCA

    cfg = NLPCAConfig.from_preset("balanced")
    cfg.io.prefix = "nlpca_run"
    model = ImputeNLPCA(genotype_data=gd, config=cfg)
    model.fit(); X_nlpca = model.transform()

**Unsupervised Backpropagation (ImputeUBP)**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from pgsui.data_processing.config import UBPConfig
    from pgsui.impute.unsupervised.imputers import ImputeUBP

    cfg = UBPConfig.from_preset("thorough")
    cfg.io.prefix = "ubp_run"
    model = ImputeUBP(genotype_data=gd, config=cfg)
    model.fit(); X_ubp = model.transform()

**Standard Autoencoder (ImputeAutoencoder)**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from pgsui.data_processing.config import SAEConfig
    from pgsui.impute.unsupervised.imputers import ImputeAutoencoder

    cfg = SAEConfig.from_preset("balanced")
    cfg.io.prefix = "sae_run"
    model = ImputeAutoencoder(genotype_data=gd, config=cfg)
    model.fit(); X_sae = model.transform()

**Variational Autoencoder (ImputeVAE)**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from pgsui.data_processing.config import VAEConfig
    from pgsui.impute.unsupervised.imputers import ImputeVAE

    cfg = VAEConfig.from_preset("balanced")
    cfg.io.prefix = "vae_run"
    cfg.training.model_beta = 1.0
    model = ImputeVAE(genotype_data=gd, config=cfg)
    model.fit(); X_vae = model.transform()

Supervised Models (Configs)
---------------------------

**ImputeHistGradientBoosting**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from pgsui import HGBConfig, ImputeHistGradientBoosting

    cfg = HGBConfig.from_preset("balanced")
    cfg.io.prefix = "hgb_run"
    cfg.tuning.tune = True
    cfg.tuning.n_trials = 100
    cfg.tuning.metric = "pr_macro"

    model = ImputeHistGradientBoosting(genotype_data=gd, config=cfg)
    model.fit(); X_hgb = model.transform()

**ImputeRandomForest**
^^^^^^^^^^^^^^^^^^^^^^          

.. code-block:: python

    from pgsui import RFConfig, ImputeRandomForest

    cfg = RFConfig.from_preset("balanced")
    cfg.io.prefix = "rf_run"
    cfg.tuning.tune = True
    cfg.tuning.n_trials = 100
    cfg.tuning.metric = "pr_macro"

    model = ImputeRandomForest(genotype_data=gd, config=cfg)
    model.fit()
    X_rf = model.transform()

Common Config Sections (Fields at a Glance)
-------------------------------------------

All ``*Config`` dataclasses share a common structure with nested sections (names may vary slightly by imputer):

- ``io``: ``prefix``, ``plot_format``, output directories, artifact toggles.
- ``training``: architecture and optimization (e.g., ``model_latent_dim``, hidden sizes, learning rate, early stopping, validation split, device, seed).
- ``tuning``: ``tune``, ``n_trials``, ``metric``, ``n_jobs``, class-weight shaping (``weights_temperature``, ``weights_alpha``, ``weights_normalize``).
- ``plot``: ``show_plots``, ``dpi``, per-plot toggles.
- ``algorithm``: method-specific flags (e.g., ``by_population`` for MostFrequent).

Visualization & Reports
-----------------------

After ``fit()``, each model writes plots and metrics under:

``{prefix}_output/{Family}/plots/{Model}/`` and ``{prefix}_output/{Family}/metrics/{Model}/``

Key figures
^^^^^^^^^^^

- **Radar (spider) summary** across models: macro-F1, macro-PR, overall accuracy, HET-F1.
- **Confusion matrices** (overall and per-zygosity).
- **Per-class precision–recall curves** and macro-averaged PR.
- **Zygosity bar charts** (REF/HET/ALT) for error composition.
- **Training curves** (loss/metric vs. epoch) for deep models.
- **Feature importances** for supervised tree-based models (if enabled).

Common Evaluation
-----------------

Metrics are stratified by zygosity, **REF/HET/ALT** (diploid) or binary classes (haploid), as well as by 10-base IUPAC encodings. Macro-averaged F1 or Precision-Recall are emphasized due to class imbalance. Summary CSV/JSON files accompany figures to facilitate downstream comparison.

Tips for Performance & Reproducibility
--------------------------------------

- Enable Optuna with ``tuning.tune = True`` and increase ``tuning.n_trials`` for more robust hyperparameters.
- Use ``training.device`` set to ``"gpu"`` (CUDA) or ``"mps"`` (Apple Silicon) when available.
- Prefer ``tuning.metric = "pr_macro"`` on imbalanced datasets.
- Set ``training.seed`` for reproducibility of splits, latent init, and tuner sampling.

Typical Workflow
----------------

1. **Read + filter + encode** with SNPio (``GenotypeData``; optional ``GenotypeEncoder`` for decoding).
2. **Baseline** with ``ImputeMostFrequent`` or ``ImputeRefAllele`` to establish a floor.
3. **Unsupervised model** (e.g., ``ImputeVAE`` or ``ImputeUBP``) with tuning enabled.
4. **Optional supervised models** (HGB/RF) to benchmark against deep models.
5. **Compare reports** (radar summary, macro-PR/F1, zygosity, confusion matrices).
6. **Decode/Export** final matrices to IUPAC or desired downstream formats.

Minimal API Reference
---------------------

All imputers follow the same high-level pattern:

.. code-block:: python

    model = SomeImputer(genotype_data=gd, config=SomeConfig.from_preset("balanced"))
    model.fit()                 # trains; writes plots/reports
    X_imputed = model.transform()  # imputes missing alleles

Citations & Background
----------------------

.. [1] Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes. arXiv:1312.6114.
.. [2] Gashler, M. S., Smith, M. R., Morris, R., & Martinez, T. (2016). Missing value imputation with unsupervised backpropagation. *Computational Intelligence*, 32(2), 196-215.
.. [3] Hinton, G. E., & Salakhutdinov, R. R. (2006). Reducing the dimensionality of data with neural networks. *Science*, 313(5786), 504-507.
.. [4] Scholz, M., Kaplan, F., Guy, C. L., Kopka, J., & Selbig, J. (2005). Non-linear PCA: a missing data approach. *Bioinformatics*, 21(20), 3887-3895.
