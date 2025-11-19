About PG-SUI
============

**PG-SUI**: Population Genomic Supervised & Unsupervised Imputation

PG-SUI Philosophy
-----------------

PG-SUI is a Python 3 toolkit for imputing missing genotypes in population genomic SNP matrices using **deterministic**, **unsupervised**, and **supervised** approaches. It integrates with SNPio for I/O and encoding, emphasizes robust handling of class imbalance, and follows a refactored design with **typed dataclass configurations**, **presets**, optional **YAML** configs, and a consistent **instantiate → fit() → transform()** workflow. Unsupervised deep models build on representation learning and generative modeling ideas (Hinton & Salakhutdinov, 2006; Kingma & Welling, 2013; Scholz et al., 2005; Gashler et al., 2016).

Key Design (at a glance)
------------------------

- **Typed configs**: each imputer uses a ``*Config`` dataclass (e.g., ``VAEConfig``, ``NLPCAConfig``, ``UBPConfig``) with presets (``fast``, ``balanced``, ``thorough``).
- **Workflow**: pass a SNPio ``GenotypeData`` at construction → ``fit()`` → ``transform()`` (no arguments).
- **Overrides**: presets ⇢ YAML (optional) ⇢ explicit overrides via dot-keys; CLI mirrors the same precedence.
- **CLI overrides**: ``pg-sui`` exposes ``--sim-strategy``, ``--sim-prop``, and ``--simulate-missing`` so you can globally control missing-data simulation per run without editing YAML.
- **Evaluation**: macro-F1 and macro-PR with zygosity-aware summaries to address genomic class imbalance.

Unsupervised Imputation Methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Unsupervised models in PG-SUI are purpose-built for genomic data:

- **Variational Autoencoder (VAE)** (Kingma & Welling, 2013) — latent probabilistic modeling with KL regularization.
- **Standard Autoencoder (SAE)** (Hinton & Salakhutdinov, 2006) — deterministic encoder-decoder reconstruction.
- **Non-linear PCA (NLPCA)** (Scholz et al., 2005) — decoder-style network with latent optimization.
- **Unsupervised Backpropagation (UBP)** (Gashler et al., 2016) — joint training of latent vectors and decoder weights.

These models learn structure from observed entries and then infer true missing genotypes:

1. **Train on observed values**: real missings are masked; optional simulated masking during training improves robustness.
2. **Predict true missings**: after training, the model predicts the masked cells to yield a complete matrix.

Detailed Neural Network Imputation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Autoencoder (SAE)**: compresses loci into a low-dimensional latent space and reconstructs the 0/1/2 matrix through a decoder (Hinton & Salakhutdinov, 2006).
- **NLPCA**: initializes reduced-dimensional inputs and iteratively refines them by minimizing reconstruction loss on observed entries (Scholz et al., 2005).
- **VAE**: learns a distribution over the latent space (mean/variance), sampling latents during training for a regularized decoder (Kingma & Welling, 2013).
- **UBP**: three-phase training that alternates/refines latent vectors and MLP weights for improved imputation on sparse, imbalanced data (Gashler et al., 2016).

Supervised Imputation Methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Supervised baselines frame imputation as multiclass genotype prediction per locus using tree-based models:

- **Random Forest** (``ImputeRandomForest``)
- **Histogram-based Gradient Boosting** (``ImputeHistGradientBoosting``)

These models learn from observed genotypes to predict missing states and can be tuned with the same Optuna-driven machinery as the unsupervised models. They provide strong, interpretable comparisons alongside the deep models.

Deterministic (Non-ML) Methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Deterministic baselines are simple, fast yardsticks:

- **Per-population mode per SNP** (population-aware majority class when a popmap is available).
- **Global mode per SNP** (dataset-wide majority class).
- **Reference-allele fill** (fills with REF genotype under the working 0/1/2 scheme).

Why Choose PG-SUI?
^^^^^^^^^^^^^^^^^^

PG-SUI combines classical baselines with modern unsupervised and supervised learners tailored to population genomics. The API is **consistent**, **extensible**, and **reproducible** (typed configs, presets, seeds), with evaluation and plotting designed for **class-imbalanced** diploid/haploid data. Users can select quick deterministic baselines, interpretable supervised trees, or higher-capacity unsupervised neural models depending on data scale and goals.

References
----------

Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *Journal of Artificial Intelligence Research*, 16, 321-357.

Gashler, M. S., Smith, M. R., Morris, R., & Martinez, T. (2016). Missing value imputation with unsupervised backpropagation. *Computational Intelligence*, 32(2), 196-215.

Hinton, G. E., & Salakhutdinov, R. R. (2006). Reducing the dimensionality of data with neural networks. *Science*, 313(5786), 504-507.

Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes. *arXiv preprint* arXiv:1312.6114.

Scholz, M., Kaplan, F., Guy, C. L., Kopka, J., & Selbig, J. (2005). Non-linear PCA: a missing data approach. *Bioinformatics*, 21(20), 3887-3895.
