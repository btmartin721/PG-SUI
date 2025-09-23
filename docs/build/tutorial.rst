PG-SUI Tutorial
===============

Overview
--------

**PG-SUI** (Population Genomic Supervised & Unsupervised Imputation) performs missing-data imputation on SNP genotype matrices using a suite of **Deterministic**, **Unsupervised**, and **Supervised** models. It integrates tightly with `SNPio <https://github.com/btmartin721/SNPio>`_ for reading, filtering, and encoding genotypes, and emphasizes **unsupervised deep learning** methods designed for genomic class imbalance and diploid/variable ploidy data.

**Model Families**
^^^^^^^^^^^^^^^^^^

- **Deterministic (Baselines)**
    - ``ImputeMostFrequent`` — per-locus mode imputation (global or population-aware).
    - ``ImputeRefAllele`` — fill missing genotypes with the REF genotype (0 for 0/1/2 encoding).

- **Unsupervised (Deep Learning)**
    - ``ImputeNLPCA`` — Non-linear PCA with latent optimization.
    - ``ImputeUBP`` — Unsupervised Backpropagation with latent + weight training phases.
    - ``ImputeAutoencoder`` — Standard autoencoder reconstruction without latent inference.
    - ``ImputeVAE`` — Variational Autoencoder with KL regularization.

- **Supervised (Machine Learning)**
    - ``ImputeHistGradientBoosting`` — Histogram-based gradient boosting classifier.
    - ``ImputeRandomForest`` — Random forest classifier.

.. note::

   **API change:** ``fit()`` and ``transform()`` no longer accept arrays as inputs. Each model uses the ``genotype_data`` object supplied **at instantiation**.
    Call ``model.fit()`` and then ``model.transform()`` (no arguments).

Installation
------------

Install PG-SUI with pip (recommended in a fresh virtual environment):

.. code-block:: bash

    pip install pg-sui

PG-SUI expects genotype I/O via **SNPio**. See the SNPio docs for installation options (pip/conda/Docker) and for supported readers (VCF/PHYLIP/STRUCTURE/GENEPOP).

Data In, Encodings, and Conventions
-----------------------------------

PG-SUI uses SNPio's ``GenotypeData`` object to read input genotypes:

- **Inputs**: VCF, PHYLIP, STRUCTURE, or GENEPOP plus optional ``popmap`` (recommended).
- **Working encoding**: **0/1/2** for diploids (REF/HET/ALT) with **-1** for missing.
- **Evaluation labels**: ``["REF", "HET", "ALT"]`` for diploids; haploids collapse to 2 classes.
- **IUPAC handling**: decoding/encoding utilities are provided by SNPio.

Quick Start (End-to-End)
------------------------

.. code-block:: python

    # --- SNPio: load genotype data ---
    from snpio import VCFReader

    gd = VCFReader(
        filename="pgsui/example_data/vcf_files/phylogen_nomx.vcf.gz",
        popmapfile="pgsui/example_data/popmaps/test.popmap",
        prefix="pgsui_demo",
        force_popmap=True,
        plot_format="pdf",
    )

    # Shared training kwargs across models (safe defaults; tune as desired)
    common = dict(
        tune=True,                   # use Optuna for hyperparameter search
        tune_n_trials=100,           # 100-1000 is typical
        tune_metric="pr_macro",      # handles class imbalance well
        weights_temperature=3.0,     # class weight shaping
        weights_alpha=1.0,
        weights_normalize=True,
        model_early_stop_gen=20,
        model_min_epochs=20,
        model_validation_split=0.20, # stratified train/val split
        model_learning_rate=1e-4,
        model_latent_dim=8,          # unsupervised models only
        model_num_hidden_layers=3,
        model_hidden_layer_sizes=[256, 128, 64],
        model_gamma=2.0,             # focal-loss gamma
        device="cpu",                # "gpu" or "mps" if available
        n_jobs=8,                    # CPU workers for tuning
        verbose=True,
        seed=42,
    )

    # Choose a model and run fit() -> transform() (no inputs)
    from pgsui import ImputeVAE

    vae = ImputeVAE(genotype_data=gd, **common)
    vae.fit()
    X012_imputed = vae.transform()   # returns 0/1/2 with -1 filled

Deterministic Models
--------------------

Deterministic baselines are fast and give useful yardsticks.

**ImputeMostFrequent (Mode)**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This fills missing genotypes with the most frequent genotype per locus, either globally or per population if a ``popmap`` is provided and the ``by_population`` argument is set to ``True``.

.. code-block:: python

    from pgsui import ImputeMostFrequent

    mode_imp = ImputeMostFrequent(
        genotype_data=gd,
        by_population=False,  # modes per population if popmap available
        prefix="pgsui_demo",
        verbose=True,
        seed=42,
    )
    mode_imp.fit()
    X012_mode = mode_imp.transform()

**ImputeRefAllele (Fill missing with REF=0)**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from pgsui import ImputeRefAllele

    ref_imp = ImputeRefAllele(
        genotype_data=gd,
        prefix="pgsui_demo",
        verbose=True,
        seed=42,
    )
    ref_imp.fit()
    X012_ref = ref_imp.transform()

Unsupervised Deep Learning
--------------------------

All unsupervised models share the same high-level API and plotting/reporting hooks. They differ in their latent modeling and training dynamics. The ``common`` dict above contains typical keyword arguments for unsupervised models.

**Non-linear PCA (ImputeNLPCA)**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Learns per-sample latent vectors that are **optimized directly**.
- Good balance of flexibility and interpretability.

.. code-block:: python

    from pgsui import ImputeNLPCA

    nlpca = ImputeNLPCA(genotype_data=gd, **common)
    nlpca.fit()
    X012_nlpca = nlpca.transform()

**Unsupervised Backpropagation (ImputeUBP)**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Three phases: (1) shallow mapping, (2) MLP refinement, (3) joint latent + weight optimization.
- More flexible than NLPCA; potentially stronger reconstructions.

.. code-block:: python

    from pgsui import ImputeUBP

    ubp = ImputeUBP(genotype_data=gd, **common)
    ubp.fit()
    X012_ubp = ubp.transform()

**Standard Autoencoder (ImputeAutoencoder)**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Deterministic encoder/decoder; no latent optimization at inference.
- Simple, fast, and often competitive.

.. code-block:: python

    from pgsui import ImputeAutoencoder

    sae = ImputeAutoencoder(genotype_data=gd, **common)
    sae.fit()
    X012_sae = sae.transform()

**Variational Autoencoder (ImputeVAE)**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Probabilistic latent space with KL regularization (``model_beta``).
- Robust and well-tuned for class imbalance and sparse ALT/HET classes.

.. code-block:: python

    from pgsui import ImputeVAE

    vae = ImputeVAE(genotype_data=gd, **{**common, "model_beta": 1.0})
    vae.fit()
    X012_vae = vae.transform()

Supervised Models
-----------------

Supervised models require labeled training genotypes (observed entries) and learn to predict missing values using correlated loci/features. PG-SUI uses correlation-guided feature selection (MICE-inspired chaining) internally.

**ImputeHistGradientBoosting**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from pgsui import ImputeHistGradientBoosting

    hgb = ImputeHistGradientBoosting(
        genotype_data=gd,
        prefix="pgsui_demo",
        tune=True,
        tune_n_trials=100,
        tune_metric="pr_macro",
        model_validation_split=0.2,
        n_jobs=8,
        verbose=True,
        seed=42,
    )
    hgb.fit()
    X012_hgb = hgb.transform()

**ImputeRandomForest**
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from pgsui import ImputeRandomForest

    rf = ImputeRandomForest(
        genotype_data=gd,
        prefix="pgsui_demo",
        tune=True,
        tune_n_trials=100,
        tune_metric="pr_macro",
        model_validation_split=0.2,
        device="cpu",
        n_jobs=8,
        verbose=True,
        seed=42,
    )
    rf.fit()
    X012_rf = rf.transform()

Common Parameters
-----------------

Below are the most commonly used parameters across PG-SUI models. If a parameter is not applicable to a family, it is ignored.

**General (all models)**
^^^^^^^^^^^^^^^^^^^^^^^^

- ``prefix``: String used to name output folders/files. Default: model-dependent (e.g., ``"pgsui"``).
- ``verbose``: ``True/False`` for detailed logs. Default: ``False``.
- ``seed``: Integer seed for reproducibility (splits, initializations, Optuna). Default: ``None``.
- ``n_jobs``: Parallel workers for CPU-bound tasks (e.g., tuning). Default: ``1`` or model-dependent.
- ``device``: ``"cpu"``, ``"gpu"``, or ``"mps"``. Default: ``"cpu"``; use GPU/MPS if available.

**Tuning / Class-Imbalance (all ML/DL models)**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``tune``: Enable Optuna hyperparameter search. Default: ``False``.
- ``tune_n_trials``: Number of Optuna trials. Typical: ``100-1000``.
- ``tune_metric``: Optimization objective (e.g., ``"pr_macro"``, ``"f1"``, ``"accuracy"``). Prefer ``"pr_macro"`` for imbalance.
- ``weights_temperature``: Temperature scaling for class weights (higher → more emphasis on rare classes). Typical: ``2-5``.
- ``weights_alpha``: Additional scaling factor for class weights. Default: ``1.0``.
- ``weights_normalize``: Normalize class weights to sum to 1. Default: ``True``.

**Training / Early Stopping (most models)**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``model_validation_split``: Fraction of samples for validation (stratified if possible). Typical: ``0.2``.
- ``model_min_epochs``: Minimum epochs before early stopping can trigger. Typical: ``20-50``.
- ``model_early_stop_gen``: Patience in epochs with no improvement. Typical: ``10-40``.
- ``model_learning_rate``: Base LR for optimizers. Typical: ``1e-4 - 3e-4`` for DL; higher for tree models.
- ``model_gamma``: Focal-loss gamma (DL classification heads). Typical: ``1-3``.

**Latent-Space / Architecture (Unsupervised only: NLPCA, UBP, VAE, Autoencoder)**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``model_latent_dim``: Dimensionality of latent space (compression strength). Start with ``8-32``.
- ``model_num_hidden_layers``: Depth of decoder/MLP. Typical: ``2-4``.
- ``model_hidden_layer_sizes``: Explicit layer sizes (e.g., ``[256, 128, 64]``). Use powers of two for large matrices.
- ``model_beta`` (VAE only): KL-divergence weight (β-VAE). ``0.5-2.0`` recommended.
- ``model_dropout_rate`` (if available): Regularization; ``0.0-0.5`` typical.

**Supervised specifics**
^^^^^^^^^^^^^^^^^^^^^^^^

- ``feature_selection_k`` (if exposed): Number of correlated loci to use as predictors per target locus. Increase for denser LD.
- ``iterative_rounds`` (if exposed): MICE-like chained imputation passes; more rounds can improve convergence.

**Plotting / Output**
^^^^^^^^^^^^^^^^^^^^^

- Directory structure: ``{prefix}_output/{Family}/plots/{Model}/`` and ``{prefix}_output/{Family}/metrics/{Model}/``.
- ``plot_format``: ``"pdf"`` (default), ``"png"``, ``"jpg"``.
- ``plot_dpi``: Resolution for rasterized outputs. Default: ``300``.
- ``plot_show_plots``: ``True/False`` to display interactively (e.g., notebooks). Default: ``False``.

Common Evaluation & Plots
-------------------------

Every imputer exposes consistent evaluation utilities (classification reports, per-class PR/F1, confusion matrices, and zygosity breakdowns) and writes artifacts under the paths listed above after ``fit()``. Metrics are stratified by **REF/HET/ALT** (diploid) or binary classes (haploid), with macro-PR/F1 emphasized for imbalance.

Tips for Performance & Reproducibility
--------------------------------------

- Use the automated ``tune=True`` option to enable Optuna hyperparameter search.
- Use ``device="gpu"`` (CUDA) or ``"mps"`` (Apple Silicon) when available for speed-ups on large matrices.
- Increase ``tune_n_trials`` (e.g., 100-1000) for more robust hyperparameters.
- Prefer ``tune_metric="f1"`` or ``"pr_macro"`` on imbalanced REF/HET/ALT datasets.
- Set ``seed`` to an integer for reproducibility of splits, latent init, and Optuna sampling.

Typical Workflow
----------------

1. **Read + filter + encode** with SNPio (``GenotypeData``; optionally ``GenotypeEncoder`` for decoding).
2. **Choose a baseline** (``ImputeMostFrequent`` or ``ImputeRefAllele``) to establish a floor.
3. **Run an unsupervised model** (``ImputeVAE`` or ``ImputeUBP``) with tuning enabled.
4. **Optionally run supervised models** (HGB/RF).
5. **Compare reports** (macro-PR/F1, zygosity plots, confusion matrices).
6. **Decode** to IUPAC (if desired) and export final matrices for downstream analyses.

Minimal API Reference (At a Glance)
-----------------------------------

All PG-SUI imputers follow the same high-level pattern:

.. code-block:: python

    model = SomeImputer(genotype_data=gd, **kwargs)
    model.fit()  # trains the model; writes plots/reports
    X_imputed = model.transform()  # imputes missing alleles

Citations & Background
----------------------

.. [1] Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes. arXiv:1312.6114.
.. [2] Gashler, M. S., Smith, M. R., Morris, R., & Martinez, T. (2016). Missing value imputation with unsupervised backpropagation. *Computational Intelligence*, 32(2), 196-215.
.. [3] Hinton, G. E., & Salakhutdinov, R. R. (2006). Reducing the dimensionality of data with neural networks. *Science*, 313(5786), 504-507.
.. [4] Scholz, M., Kaplan, F., Guy, C. L., Kopka, J., & Selbig, J. (2005). Non-linear PCA: a missing data approach. *Bioinformatics*, 21(20), 3887-3895.
