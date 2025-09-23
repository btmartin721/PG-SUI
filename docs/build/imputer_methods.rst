Imputation Methods Implemented in PG-SUI
========================================

This page describes the mathematical formulations and methodologies behind the imputation models implemented in PG-SUI, including **deep learning-based methods (ImputeNLPCA, ImputeUBP, ImputeAutoencoder, and ImputeVAE)** and **traditional machine learning methods (IterativeImputer and the MICE algorithm)**.

Unsupervised Backpropagation (UBP) for SNP Imputation
-----------------------------------------------------

**Unsupervised Backpropagation (UBP)** is a non-linear, iterative method that refines missing values through a **three-phase training process**. Like VAE, it uses the **MaskedFocalLoss criterion** to mask real missing values during optimization.

Model Overview
~~~~~~~~~~~~~~

The UBP model is structured as follows:

1. **Phase 1:** Uses a linear decoder to transform an initial latent input into refined feature representations. The output from this phase is a better representation of the inputs, which will be used in later phases.
2. **Phase 2:** Uses a deeper, flexible neural network to refine only the weights while keeping the refined inputs fixed.
3. **Phase 3:** Further refines both the input representations and model weights using the same deep architecture as in Phase 2.

Mathematical Formulation
~~~~~~~~~~~~~~~~~~~~~~~~

The UBP model is designed to impute missing values in genotype data using unsupervised backpropagation. The model operates in three phases, each of which refines the model differently to improve the accuracy of imputation. This section describes the mathematical formulation of the UBP model.

Phase 1: Linear Decoder
^^^^^^^^^^^^^^^^^^^^^^^

The model in Phase 1 employs a simple linear mapping from the latent space to the output space:

.. math::
    \mathbf{Y}_{1} = \mathbf{W}_{1} \mathbf{Z} + \mathbf{b}_{1}

where:
- :math:`\mathbf{Z} \in \mathbb{R}^{d}` is the latent input of dimension :math:`d`.
- :math:`\mathbf{W}_{1}` is the weight matrix of the linear decoder.
- :math:`\mathbf{b}_{1}` is the bias vector.
- :math:`\mathbf{Y}_{1} \in \mathbb{R}^{n \times c}` is the output, where :math:`n` is the number of features and :math:`c` is the number of classes.

Phases 2 and 3: Deep Network
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Phases 2 and 3 use a deeper neural network to refine the imputed data. The network consists of multiple hidden layers, each with weights, biases, activation functions, and dropout for regularization:

.. math::
    \mathbf{H}_{i} = \sigma(\mathbf{W}_{i} \mathbf{H}_{i-1} + \mathbf{b}_{i}) \quad \text{for } i = 2, \dots, L

where:
- :math:`\mathbf{H}_{0} = \mathbf{Z}` is the initial input.
- :math:`\sigma` is an activation function, such as ReLU or ELU.
- :math:`\mathbf{W}_{i}` and :math:`\mathbf{b}_{i}` are the weights and biases of the :math:`i`-th layer.
- :math:`L` is the number of layers.

The final layer produces the output:

.. math::
    \mathbf{Y}_{2} = \mathbf{W}_{L} \mathbf{H}_{L-1} + \mathbf{b}_{L}

Loss Function
^^^^^^^^^^^^^

The model minimizes a masked focal loss to handle class imbalance and focus on difficult-to-predict samples:

.. math::
    \mathcal{L}_{\text{focal}}(p_t) = - \alpha_t (1 - p_t)^{\gamma} \log(p_t)

where:
- :math:`p_t` is the probability of the correct class.
- :math:`\alpha_t` is a weighting factor for class balance.
- :math:`\gamma` controls the focus on difficult examples.

The overall loss is computed as:

.. math::
    \mathcal{L} = \frac{1}{|M|} \sum_{(i,j) \in M} \mathcal{L}_{\text{focal}}(p_t)

where:
- :math:`M` is the set of valid (unmasked) samples.
- :math:`|M|` is the number of valid samples.

Training Procedure
~~~~~~~~~~~~~~~~~~

The training procedure involves three distinct phases:
1. **Phase 1:** Train the linear decoder to refine inputs.
2. **Phase 2:** Train the deeper network to refine only weights.
3. **Phase 3:** Train the deeper network to refine both inputs and weights.

Each phase optimizes its parameters using the Adam optimizer with a cosine annealing learning rate scheduler.

Final Output
~~~~~~~~~~~~

The output of the model is an imputed tensor :math:`\hat{\mathbf{X}} \in \mathbb{R}^{n \times c}` that predicts the missing values in the input data.


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
        device="cpu", 
        n_jobs=8, # Number of CPUs to use with Optuna parameter tuning.
        verbose=1, 
        seed=42, # For reproducibility.
    )

Nonlinear Principal Component Analysis (NLPCA) Model
----------------------------------------------------

The NLPCA model is designed for imputing missing genotype values using a nonlinear transformation of the input data through a deep neural network architecture. The model refines both the input data and the weights via backpropagation, allowing it to capture nonlinear patterns in the data.

Model Overview
~~~~~~~~~~~~~~

The NLPCA model consists of a deep neural network with multiple hidden layers. The architecture is flexible and can be tuned to suit different types of input data. The key idea behind NLPCA is to learn a nonlinear mapping of the input space to a lower-dimensional latent space and then reconstruct the input from this latent representation.

### Network Architecture

The forward pass of the NLPCA model can be described mathematically as:

.. math::
    \mathbf{H}_{1} = \sigma(\mathbf{W}_{1} \mathbf{X} + \mathbf{b}_{1})

    \mathbf{H}_{2} = \sigma(\mathbf{W}_{2} \mathbf{H}_{1} + \mathbf{b}_{2})

    \vdots

    \mathbf{H}_{L} = \sigma(\mathbf{W}_{L} \mathbf{H}_{L-1} + \mathbf{b}_{L})

    \mathbf{Y} = \mathbf{W}_{\text{out}} \mathbf{H}_{L} + \mathbf{b}_{\text{out}}

where:
- :math:`\mathbf{X} \in \mathbb{R}^{n \times d}` is the input data with :math:`n` samples and :math:`d` features.
- :math:`\mathbf{H}_{i}` is the hidden representation at layer :math:`i`.
- :math:`\mathbf{W}_{i}` and :math:`\mathbf{b}_{i}` are the weights and biases at layer :math:`i`.
- :math:`\sigma` is the activation function, which can be ReLU, ELU, SELU, or Leaky ReLU.
- :math:`L` is the number of hidden layers.
- :math:`\mathbf{Y}` is the output representing the reconstructed input data.

Loss Function
^^^^^^^^^^^^^

The NLPCA model uses a masked focal loss to handle class imbalance and ignore missing values. The focal loss is defined as:

.. math::
    \mathcal{L}_{\text{focal}}(p_t) = - \alpha_t (1 - p_t)^{\gamma} \log(p_t)

where:
- :math:`p_t` is the probability of the correct class.
- :math:`\alpha_t` is a weighting factor for class balance.
- :math:`\gamma` controls the focus on difficult examples.

To ignore missing values, a masking operation is applied:

.. math::
    \mathcal{L} = \frac{1}{|M|} \sum_{(i,j) \in M} \mathcal{L}_{\text{focal}}(p_t)

where:
- :math:`M` is the set of valid (unmasked) samples.
- :math:`|M|` is the number of valid samples.

Training Procedure
~~~~~~~~~~~~~~~~~~

The NLPCA model is trained using backpropagation with the Adam optimizer and a cosine annealing learning rate scheduler. The key steps in training are as follows:

1. **Initialize model parameters**: Initialize the weights and biases of the network.
2. **Forward pass**: Compute the output of the network given the input data.
3. **Compute loss**: Calculate the masked focal loss to handle missing and imbalanced data.
4. **Backpropagation**: Compute gradients of the loss with respect to model parameters.
5. **Update parameters**: Use the Adam optimizer to update model weights.
6. **Refine inputs**: Manually update the input data to further improve imputation accuracy.

Refinement of Inputs
^^^^^^^^^^^^^^^^^^^^

During training, the model refines the input data to better represent the underlying patterns. This is achieved through a manual gradient update:

.. math::
    \mathbf{X} \leftarrow \mathbf{X} - \eta \frac{\partial \mathcal{L}}{\partial \mathbf{X}}

where:
- :math:`\eta` is the learning rate for input refinement.

Final Output
~~~~~~~~~~~~

The final output of the model is an imputed tensor :math:`\hat{\mathbf{X}}` that reconstructs the input data with missing values imputed using the learned nonlinear mappings.

Autoencoder Model for Genotype Data Imputation
----------------------------------------------

The Autoencoder model is designed to impute missing genotype data by encoding input data into a lower-dimensional latent representation and reconstructing the original input. This process helps capture complex patterns in the data and effectively handles missing values.

Model Overview
~~~~~~~~~~~~~~

An autoencoder consists of two main components:
1. **Encoder:** Maps the high-dimensional input data to a lower-dimensional latent space.
2. **Decoder:** Reconstructs the input data from the latent representation.

The model aims to minimize the reconstruction loss between the original and reconstructed inputs.

Encoder Network
^^^^^^^^^^^^^^^

The encoder network transforms the input data through several hidden layers:

.. math::
    \mathbf{H}_{1} = \sigma(\mathbf{W}_{1} \mathbf{X} + \mathbf{b}_{1})

    \mathbf{H}_{2} = \sigma(\mathbf{W}_{2} \mathbf{H}_{1} + \mathbf{b}_{2})

    \vdots

    \mathbf{Z} = \sigma(\mathbf{W}_{L} \mathbf{H}_{L-1} + \mathbf{b}_{L})

where:
- :math:`\mathbf{X} \in \mathbb{R}^{n \times d}` is the input data.
- :math:`\mathbf{H}_{i}` is the hidden representation at layer :math:`i`.
- :math:`\mathbf{W}_{i}` and :math:`\mathbf{b}_{i}` are the weights and biases of the encoder.
- :math:`\sigma` is the activation function.
- :math:`\mathbf{Z}` is the latent representation of dimension :math:`k`.

Decoder Network
^^^^^^^^^^^^^^^

The decoder reconstructs the original input data from the latent representation:

.. math::
    \mathbf{H}_{i} = \sigma(\mathbf{W}_{i} \mathbf{H}_{i-1} + \mathbf{b}_{i})

    \mathbf{\hat{X}} = \sigma(\mathbf{W}_{\text{out}} \mathbf{H}_{L} + \mathbf{b}_{\text{out}})

where:
- :math:`\mathbf{\hat{X}}` is the reconstructed input.

Loss Function
^^^^^^^^^^^^^

The model uses a masked focal loss to handle missing values and focus on difficult-to-predict data points. The masked focal loss is defined as:

.. math::
    \mathcal{L}_{\text{focal}}(p_t) = - \alpha_t (1 - p_t)^{\gamma} \log(p_t)

where:
- :math:`p_t` is the predicted probability of the correct class.
- :math:`\alpha_t` is a class balance weight.
- :math:`\gamma` is a parameter that controls the focus on hard-to-predict samples.

The overall loss is computed only over valid (unmasked) entries:

.. math::
    \mathcal{L} = \frac{1}{|M|} \sum_{(i,j) \in M} \mathcal{L}_{\text{focal}}(p_t)

where:
- :math:`M` is the set of valid (unmasked) samples.
- :math:`|M|` is the number of valid samples.

Training Procedure
~~~~~~~~~~~~~~~~~~

The autoencoder is trained using backpropagation with the Adam optimizer and a learning rate scheduler. The key steps in training are:
1. **Forward pass:** Compute the output of the model.
2. **Compute loss:** Calculate the masked focal loss between the original and reconstructed inputs.
3. **Backpropagation:** Compute the gradients of the loss.
4. **Update parameters:** Update the weights and biases of the encoder and decoder.

Final Output
~~~~~~~~~~~~

The output of the autoencoder is an imputed tensor :math:`\hat{\mathbf{X}}` that reconstructs the original input data while imputing the missing values.

Variational Autoencoder (VAE) Model for Genotype Data Imputation
-----------------------------------------------------------------

The Variational Autoencoder (VAE) model is designed to impute missing genotype data using a probabilistic approach. The model learns a distribution over the latent space and samples from this distribution to reconstruct the input data.

Model Overview
~~~~~~~~~~~~~~

A VAE consists of three key components:
1. **Encoder:** Maps the input data to a distribution in the latent space.
2. **Latent Space Sampling:** Samples latent variables from the distribution defined by the encoder.
3. **Decoder:** Reconstructs the input data from the sampled latent variables.

Encoder Network
^^^^^^^^^^^^^^^

The encoder maps the input :math:`\mathbf{X}` to the parameters of a Gaussian distribution over the latent space:

.. math::
    \mu = f_{\mu}(\mathbf{X})

    \log \sigma^{2} = f_{\sigma}(\mathbf{X})

where:
- :math:`\mu` is the mean of the distribution.
- :math:`\sigma^{2}` is the variance.
- :math:`f_{\mu}` and :math:`f_{\sigma}` are neural networks representing the encoder.

Latent Space Sampling
^^^^^^^^^^^^^^^^^^^^^

The model samples a latent variable :math:`\mathbf{z}` using the reparameterization trick:

.. math::
    \mathbf{z} = \mu + \epsilon \cdot \sigma, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})

This allows the model to backpropagate through the sampling step during training.

Decoder Network
^^^^^^^^^^^^^^^

The decoder reconstructs the input data from the sampled latent variables:

.. math::
    \mathbf{\hat{X}} = f_{\text{dec}}(\mathbf{z})

where :math:`f_{\text{dec}}` is a neural network representing the decoder.

Loss Function
^^^^^^^^^^^^^

The VAE loss consists of two components:
1. **Reconstruction Loss:** Measures the difference between the original and reconstructed inputs using a masked focal loss:

.. math::
    \mathcal{L}_{\text{recon}} = \frac{1}{|M|} \sum_{(i,j) \in M} \alpha_t (1 - p_t)^{\gamma} \log(p_t)

where:
- :math:`M` is the set of valid (unmasked) samples.
- :math:`\alpha_t` is a class weight.
- :math:`\gamma` controls the focus on hard-to-predict samples.

2. **KL Divergence:** Regularizes the learned latent distribution to be close to the prior distribution (a standard normal distribution):

.. math::
    \mathcal{L}_{\text{KL}} = D_{\text{KL}}(q(\mathbf{z} | \mathbf{X}) \| p(\mathbf{z}))

where:

- :math:`q(\mathbf{z} | \mathbf{X})` is the approximate posterior distribution.
- :math:`p(\mathbf{z})` is the prior distribution.

The total loss is given by:

.. math::
    \mathcal{L} = \mathcal{L}_{\text{recon}} + \beta \mathcal{L}_{\text{KL}}

where :math:`\beta` is a weighting factor that balances the reconstruction and KL divergence losses.

Training Procedure
~~~~~~~~~~~~~~~~~~

1. **Forward pass:** Compute the reconstruction, mean, and variance of the latent variables.
2. **Loss computation:** Calculate the total loss using the reconstruction and KL divergence components.
3. **Backpropagation:** Compute gradients and update model parameters.
4. **Latent sampling refinement:** Ensure robust learning of the latent variables through reparameterization.

Final Output
~~~~~~~~~~~~

The output of the VAE model is an imputed tensor :math:`\hat{\mathbf{X}}` that reconstructs the original input data while filling in the missing values probabilistically.


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

The `IterativeImputer` methods are in development and will be available in future versions of PG-SUI.

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
