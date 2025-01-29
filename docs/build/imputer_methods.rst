Imputation Methods in PG-SUI
============================

This page describes the mathematical formulations and methodologies behind the imputation models implemented in PG-SUI, including **deep learning-based methods (ImputeVAE and ImputeUBP)** and **traditional machine learning methods (IterativeImputer and the MICE algorithm)**.

Variational Autoencoder (VAE) for SNP Imputation
------------------------------------------------

The **Variational Autoencoder (VAE)** is a generative deep learning model that learns a **probabilistic latent space representation** of the input SNP data and reconstructs missing values by generating plausible genotypes. The **custom MaskedFocalLoss criterion** ensures that real missing values (encoded as -1) are **excluded from the loss computation**, enabling robust training on incomplete datasets.

### Mathematical Formulation

A **VAE** consists of two key components:

1. **Encoder**: Maps the input SNP data \( X \) to a lower-dimensional latent space \( Z \).
2. **Decoder**: Reconstructs \( X' \) from the latent representation \( Z \).

The encoder learns a probability distribution over the latent space:

.. math::

    q_{\phi}(z|X) = \mathcal{N}(z; \mu(X), \sigma^2(X))

where:

- \( \mu(X) \) and \( \sigma^2(X) \) are the mean and variance of the learned latent distribution.
- \( z \) is the latent variable sampled using the **reparameterization trick**:

.. math::

    z = \mu(X) + \sigma(X) \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)

The decoder then reconstructs the input SNP matrix by generating \( X' \):

.. math::

    p_{\theta}(X'|z) = \mathcal{N}(X'; f_{\theta}(z), I)

where \( f_{\theta}(z) \) is a neural network that maps \( z \) back to the original SNP space.

The **loss function** used for training combines:
1. A **reconstruction loss** that calculates the focal loss only on known (non-masked) values.
2. A **Kullback-Leibler (KL) divergence** to ensure latent space regularization.

Custom MaskedFocalLoss Criterion
--------------------------------

The reconstruction loss specifically excludes real missing values \( M \) (encoded as -1):

.. math::

    \mathcal{L}_{\text{reconstruction}} = \sum_{i=1}^{N} \sum_{j=1}^{p} \text{FocalLoss}(X_{ij}, X'_{ij}) \cdot \mathbb{I}[M_{ij} \neq -1]

where:

- \( \mathbb{I}[M_{ij} \neq -1] \) is the indicator function masking missing values.
- Focal loss is defined as:

.. math::

    \text{FocalLoss}(x, x') = -\alpha (1 - \hat{p})^\gamma \log(\hat{p})

    \quad \text{where} \quad \hat{p} = \text{sigmoid}(x')

Here, \( \alpha \) is the weighting factor, and \( \gamma \) controls the focus on hard-to-predict samples.

The combined loss function is:

.. math::

    \mathcal{L}(\theta, \phi) = \mathcal{L}_{\text{reconstruction}} - D_{KL}(q_{\phi}(z|X) || p(z))

where \( D_{KL} \) is the KL divergence that ensures \( q_{\phi}(z|X) \) remains close to a standard normal prior \( p(z) = \mathcal{N}(0, I) \).

The focal loss and KL divergence are weighted by hyperparameters \( \alpha \) and \( \beta \) to balance the reconstruction and regularization objectives. The **weights_temperature** parameter adjusts the class weights, while **model_gamma** and **model_beta** control the focal loss and KL divergence scaling, respectively.

Focal loss can be particularly effective for imputation tasks with **class imbalance**, as it focuses on hard-to-predict samples.

Implementation in PG-SUI
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    vae = ImputeVAE(
        genotype_data=data, 
        tune=True, # Tune parameters with Optuna.
        tune_n_trials=100, # Recommended: 100-1000.
        tune_metric="pr_macro", # Deals well with class imbalance.
        weights_temperature=3.0, # For adjusting class weights.
        weights_alpha=1.0,
        weights_normalize=True,
        model_early_stop_gen=20, # Model parameters.
        model_min_epochs=20,
        model_validation_split=0.21,
        model_learning_rate=0.0001,
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

Unsupervised Backpropagation (UBP) for SNP Imputation
-----------------------------------------------------

**Unsupervised Backpropagation (UBP)** is a non-linear, iterative method that refines missing values through a **three-phase training process**. Like VAE, it uses the **MaskedFocalLoss criterion** to mask real missing values during optimization.

Mathematical Formulation
~~~~~~~~~~~~~~~~~~~~~~~~

UBP starts with an **initial randomly generated reduced-dimensional representation** \( Z_0 \), which is refined across three stages:

1. **Phase 1: Single-layer perceptron refinement**
    - A single-layer neural network transforms the initial representation \( Z_0 \) into \( Z_1 \).
    - The optimization minimizes the **MaskedFocalLoss** for known SNP values.

.. math::

    Z_1 = W_1 Z_0 + b_1

where \( W_1 \) and \( b_1 \) are the network weights and biases.

1. **Phase 2: Multi-layer perceptron (MLP) training**
    - A deeper neural network is trained using the refined latent space \( Z_1 \) and only non-masked SNP values:

.. math::

    X' = f_{\theta}(Z_1)

3. **Phase 3: Joint optimization**
   - Both \( Z \) and \( f_{\theta} \) are optimized **simultaneously** to further improve the imputed values:

.. math::

    \mathcal{L}_{\text{total}} = \mathcal{L}_{\text{reconstruction}} + \mathcal{L}_{\text{refinement}}

The final reconstructed SNP dataset is:

.. math::

    X' = f_{\theta}(Z^*)

where \( Z^* \) is the final optimized latent space representation.

Implementation in PG-SUI
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ubp = ImputeUBP(
        genotype_data=data, 
        tune=True, # Tune parameters with Optuna.
        tune_n_trials=100, # Recommended: 100-1000.
        tune_metric="pr_macro", # Deals well with class imbalance.
        weights_temperature=3.0, # For adjusting class weights.
        weights_alpha=1.0,
        weights_normalize=True,
        model_early_stop_gen=20, # Model parameters.
        model_min_epochs=20,
        model_validation_split=0.21,
        model_learning_rate=0.0001,
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

IterativeImputer and the MICE Algorithm
---------------------------------------

**IterativeImputer** in `scikit-learn` is a general-purpose multiple imputation strategy that iteratively models each SNP column based on the most correlated loci.

### Multivariate Imputation by Chained Equations (MICE)
MICE performs **sequential regression-based imputation**, where each missing value is predicted iteratively based on other features.

Let:

- \( X = (X_1, X_2, ..., X_p) \) be the SNP dataset with missing values.
- \( X_{-j} \) be all columns except the \( j \)th one.

For each column \( X_j \):

1. **Initialize** missing values using a simple strategy (e.g., mean imputation).
2. **Train a regression model** \( f_j \) predicting \( X_j \) using \( X_{-j} \):

.. math::

    X_j = f_j(X_{-j}) + \epsilon

   where \( f_j \) is a regression model (e.g., **Random Forest**, **XGBoost**, **KNN**).

3. **Predict missing values** in \( X_j \) using \( f_j \).
4. **Repeat for all columns** and cycle multiple times (controlled by `max_iter`).

The process stops when convergence is reached (i.e., imputed values stabilize across iterations).

Coming Soon in PG-SUI
----------------------

.. code-block:: python

    knn = ImputeKNN(genotype_data=data, **kwargs)
    rf = ImputeRandomForest(genotype_data=data, **kwargs)
    xgb = ImputeXGBoost(genotype_data=data, **kwargs)

Further Reading
---------------

For additional details, refer to the **scikit-learn documentation**:
`https://scikit-learn.org/stable/modules/impute.html#iterative-imputer`

Conclusion
----------

PG-SUI provides a range of **deep learning, machine learning, and statistical methods** for SNP imputation. While **ImputeVAE is fully functional**, **ImputeUBP is in development**, and the **IterativeImputer framework** enables regression-based imputation.

For more details, see the `PG-SUI API documentation <pgsui.impute>`_.
