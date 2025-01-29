Tutorial
========

PG-SUI Overview
---------------

PG-SUI (Population Genomic Supervised and Unsupervised Imputation) performs missing data imputation on SNP datasets using a combination of **deep learning, machine learning, and non-machine learning** approaches. The package provides multiple imputation strategies, with particular emphasis on **unsupervised deep learning methods**, which are currently the most developed and fully functional.

The currently supported algorithms include:

Unsupervised Deep Learning Methods (Fully/Partially Functional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

PG-SUI offers several **unsupervised deep learning** approaches for SNP genotype imputation. These methods do not require labeled training data, making them well-suited for genomic datasets with missing values.

+ Variational AutoEncoder (VAE) [1]_  *(Fully Functional)*
+ Unsupervised Backpropagation (UBP) [2]_ *(Coming Soon)*
+ Standard AutoEncoder (SAE) [3]_
+ Non-linear Principal Component Analysis (NLPCA) [4]_

**Variational AutoEncoder (VAE) – Fully Functional**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- The **ImputeVAE** model is currently the most robust and optimized deep learning approach available in PG-SUI.
- The model encodes the SNP dataset into a **low-dimensional latent space**, where each encoded sample is drawn from a probability distribution (mean and variance).
- After training, the decoder reconstructs the SNP matrix while **inferring missing genotypes** based on learned patterns.

.. code-block:: python

        vae = ImputeVAE(
        genotype_data=data, 
        tune=True, # Tune model parameters with Optuna.
        tune_n_trials=100, # Recommended: 100-1000.
        tune_metric="pr_macro", # Deals well with class imbalance.
        weights_temperature=3.0, # For adjusting class weights.
        weights_alpha=1.0,
        weights_normalize=True,
        model_early_stop_gen=20, # Model parameters.
        model_min_epochs=20,
        model_validation_split=0.21, # Split into train/ val/ test sets.
        model_learning_rate=0.0001, # Tunable parameters from here to model_beta.
        model_latent_dim=2, 
        model_num_hidden_layers=2, 
        model_hidden_layer_sizes=[128, 64], 
        model_gamma=2.0, # For focal loss. 
        model_beta=1.0, # For KL divergence.
        device="cpu", 
        n_jobs=8, # Number of CPUs to use with Optuna parameter tuning.
        verbose=1, 
        seed=42, # For reproducibility.
    )

There are also other arguments. Please see the API documentation for more details.

**Unsupervised Backpropagation (UBP) – Coming Soon**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- UBP is an **enhanced version of NLPCA**, introducing a **three-phase training process**:
  1. **Phase 1:** A single-layer perceptron learns an initial mapping from a randomly generated latent space.
  2. **Phase 2:** A multi-layer perceptron (MLP) refines the learned representations.
  3. **Phase 3:** Both the latent space and network weights are optimized together for **improved imputation accuracy**.
- This method will offer a more flexible and powerful alternative to NLPCA once functional.

.. code-block:: python

    ubp = ImputeUBP(genotype_data=data)  # Feature in development

**Standard AutoEncoder (SAE)**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::

    The SAE model is currently under active development and optimization. It might not be fully functional in the current release.

- SAE reduces the dataset into a **compressed latent representation** and then reconstructs the original SNP matrix.
- Unlike VAE, SAE does not model a probability distribution; it simply learns a **deterministic encoding-decoding function**.

.. code-block:: python

        sae = ImputeStandardAutoEncoder(
        genotype_data=data, 
        tune=True, # Tune parameters with Optuna.
        tune_n_trials=100, # Recommended: 100-1000.
        tune_metric="pr_macro", # Deals well with class imbalance.
        weights_temperature=3.0, # For adjusting class weights.
        weights_alpha=1.0,
        weights_normalize=True,
        model_early_stop_gen=20, # Model parameters.
        model_min_epochs=20,
        model_validation_split=0.21, # Split into train/ val/ test sets.
        model_learning_rate=0.0001, # Tunable parameters from here to model_beta.
        model_latent_dim=2, 
        model_num_hidden_layers=2, 
        model_hidden_layer_sizes=[128, 64], 
        model_gamma=2.0, # For focal loss. 
        device="cpu", 
        n_jobs=8, # Number of CPUs to use with Optuna parameter tuning.
        verbose=1, 
        seed=42, # For reproducibility.
    )

**Non-Linear Principal Component Analysis (NLPCA)**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- NLPCA initializes a randomly generated **low-dimensional representation** of the dataset.
- This reduced-dimensional input is refined over multiple backpropagation iterations until it **accurately reconstructs the original data**.

.. note::
    
    NLPCA is currently under active development and optimization. It might not be fully functional in the current release.

.. code-block:: python

    nlpca = ImputeNLPCA(
        genotype_data=data, 
        tune=True, # Tune parameters with Optuna.
        tune_n_trials=100, # Recommended: 100-1000.
        tune_metric="pr_macro", # Deals well with class imbalance.
        weights_temperature=3.0, # For adjusting class weights.
        weights_alpha=1.0,
        weights_normalize=True,
        model_early_stop_gen=20, # Model parameters.
        model_min_epochs=20,
        model_validation_split=0.21, # Split into train/ val/ test sets.
        model_learning_rate=0.0001, # Tunable parameters from here to model_beta.
        model_latent_dim=2, 
        model_num_hidden_layers=2, 
        model_hidden_layer_sizes=[128, 64], 
        model_gamma=2.0, # For focal loss. 
        device="cpu", 
        n_jobs=8, # Number of CPUs to use with Optuna parameter tuning.
        verbose=1, 
        seed=42, # For reproducibility.
    )

---

Supervised Machine Learning Methods (Under Active Development)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

PG-SUI also includes **supervised learning approaches**, which require labeled training data. These models leverage **neighboring SNPs and population structure** to predict missing values.

Currently, supervised methods are still **under active development**, and their performance is being optimized. The supported supervised classifiers include:

+ **XGBoost**
+ **Random Forest (or Extra Trees)**
+ **K-Nearest Neighbors (KNN)**

Supervised methods work by identifying the **N-nearest informative loci** based on absolute correlation with the missing data and iteratively imputing SNPs. This process is inspired by the **MICE (Multivariate Imputation by Chained Equations) algorithm** [5]_.

.. code-block:: python

    knn = ImputeKNN(genotype_data=data)  # K-Nearest Neighbors
    rf = ImputeRandomForest(genotype_data=data)  # Random Forest
    xgb = ImputeXGBoost(genotype_data=data)  # XGBoost

These classifiers will be further refined in upcoming releases.

---

Installing PG-SUI
-----------------

The easiest way to install PG-SUI is via pip:

.. code-block:: bash

    pip install pg-sui

For manual installation instructions and dependencies, see the :doc:`Installation <install>` page.

---

Input Data
----------

PG-SUI uses the **GenotypeData** class from the `SNPio package <https://github.com/btmartin721/SNPio>`_ to load and preprocess data. Supported input formats include:

- **VCF**
- **STRUCTURE**
- **PHYLIP**

A population map (popmap) file is required, and **phylogenetic tree and rate matrix files** are optional for **phylogeny-aware imputation**.

**Example: Loading Input Data**

.. code-block:: python

    from snpio import VCFReader

    gd = VCFReader(
        filename="pgsui/example_data/phylip_files/test_n100.phy",
        popmapfile="pgsui/example_data/popmaps/test.popmap",
        guidetree="pgsui/example_data/trees/test.tre",
        qmatrix="pgsui/example_data/trees/test.qmat",
        siterates="pgsui/example_data/trees/test_siterates_n100.txt",
        prefix="test_imputer",
        force_popmap=True,
        plot_format="pdf",
    )

    vae = ImputeVAE(genotype_data=gd, **kwargs)
    ubp = ImputeUBP(genotype_data=gd, **kwargs)

---

Non-Machine Learning Imputation Methods
---------------------------------------

For comparison and baseline performance assessment, PG-SUI also includes **non-machine learning imputation strategies**, which impute missing genotypes based on simple heuristics.

**Supported Non-ML Methods:**
- **Phylogeny-informed imputation** (uses evolutionary relationships)
- **Per-population allele frequency imputation**
- **Global allele frequency imputation**
- **Reference allele imputation**
- **Matrix factorization-based imputation**

.. code-block:: python

    # Phylogeny-based imputation
    phylo = ImputePhylo(genotype_data=data, **kwargs)

    # Allele frequency imputation (by population or globally)
    pop_af = ImputeAlleleFreq(genotype_data=data, by_populations=True, **kwargs)
    global_af = ImputeAlleleFreq(genotype_data=data, by_populations=False, **kwargs)

    # Non-negative Matrix Factorization
    mf = ImputeMF(genotype_data=data, **kwargs)

---

References
----------

.. [1] Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes. arXiv preprint arXiv:1312.6114.
.. [2] Gashler, M. S., Smith, M. R., Morris, R., & Martinez, T. (2016). Missing value imputation with unsupervised backpropagation. Computational Intelligence, 32(2), 196-215.
.. [3] Hinton, G.E., & Salakhutdinov, R.R. (2006). Reducing the dimensionality of data with neural networks. Science, 313(5786), 504-507.
.. [4] Scholz, M., Kaplan, F., Guy, C. L., Kopka, J., & Selbig, J. (2005). Non-linear PCA: a missing data approach. Bioinformatics, 21(20), 3887-3895.
.. [5] Stef van Buuren, Karin Groothuis-Oudshoorn (2011). mice: Multivariate Imputation by Chained Equations in R. Journal of Statistical Software 45: 1-67.