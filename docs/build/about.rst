About PG-SUI
============

PG-SUI: Population Genomic Supervised and Unsupervised Imputation

PG-SUI Philosophy
-----------------

PG-SUI is a Python 3 API that uses machine learning to impute missing values from population genomic SNP data. There are several supervised and unsupervised machine learning algorithms available to impute missing data, as well as some non-machine learning imputers that are useful. 

### Supervised Imputation Methods

Supervised methods utilze the scikit-learn's IterativeImputer, which is based on the MICE (Multivariate Imputation by Chained Equations) algorithm ([1]_), and iterates over each SNP site (i.e., feature) while uses the N nearest neighbor features to inform the imputation. The number of nearest features can be adjusted by users. IterativeImputer currently works with any of the following scikit-learn classifiers: 

* K-Nearest Neighbors
* Random Forest
* Extra Trees
* XGBoost

See the `scikit-learn documentation <https://scikit-learn.org>`_ for more information on IterativeImputer and each of the classifiers.

### Unsupervised Imputation Methods

Unsupervised imputers include three custom neural network models:

    + Variational Autoencoder (VAE) ([2]_)
    + Standard Autoencoder (SAE) ([3]_)
    + Non-linear Principal Component Analysis (NLPCA) ([4]_)
    + Unsupervised Backpropagation (UBP) ([5]_)

To use the unsupervised neural networks for imputation, the real missing values are masked and missing values are simulated for training. The model gets trained to reconstruct only on known values. Once the model is trained, it is then used to predict the real missing values.

SAE models encode the input features (i.e., loci) into a reduced-dimensional layer (typically 2 or 3 dimensions). This reduced-dimensional layer is then input into the decoder and the model trains itself to reconstruct the  input (i.e., the genotypes). 

VAE models also have an encoder and a decoder and train themselves to reconstruct their input (i.e., the genotypes), but the reduced-dimensional layer in VAE (latent dimension) represent a sampling distribution with a mean and a variance (latent variables). This distribution gets sampled from during training, and the samples get input into the decoder.

NLPCA initializes random, reduced-dimensional input, then trains itself by using the known values (i.e., genotypes) as targets and refining the random input until it accurately predicts the genotype output. Essentially, NLPCA resembles the second half of a standard autoencoder.

UBP is an extension of NLPCA that runs over three phases. Phase 1 refines the randomly generated, reduced-dimensional input in a single layer perceptron neural network to obtain good initial input values. Phase 2 uses the refined reduced-dimensional input from phase 1 as input into a multi-layer perceptron (MLP), but in Phase 2 only the neural network weights are refined. Phase three uses an MLP to refine both the weights and the reduced-dimensional input.

### Non-Machine Learning Methods

We also include several non-machine learning options for imputing missing data, including:

    + Per-population mode per SNP site
    + Global mode per SNP site
    + Ref allele per SNP site
    + Using a phylogeny as input to inform the imputation
    + Matrix Factorization

These four non-machine learning imputation methods can be used as standalone imputers, as the initial imputation strategy for IterativeImputer (at least one method is required to be chosen), and to validate the accuracy of both IterativeImputer and the neural network models.

References
-----------

.. [1] Stef van Buuren, Karin Groothuis-Oudshoorn (2011). mice: Multivariate Imputation by Chained Equations in R. Journal of Statistical Software 45: 1-67.

.. [2] Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes. arXiv preprint arXiv:1312.6114.

.. [3] Hinton, G.E., & Salakhutdinov, R.R. (2006). Reducing the dimensionality of data with neural networks. Science, 313(5786), 504-507.

.. [4] Scholz, M., Kaplan, F., Guy, C. L., Kopka, J., & Selbig, J. (2005). Non-linear PCA: a missing data approach. Bioinformatics, 21(20), 3887-3895.

.. [5] Gashler, M. S., Smith, M. R., Morris, R., & Martinez, T. (2016). Missing value imputation with unsupervised backpropagation. Computational Intelligence, 32(2), 196-215.

