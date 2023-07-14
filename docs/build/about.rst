About PG-SUI
============

PG-SUI: Population Genomic Supervised and Unsupervised Imputation

PG-SUI Philosophy
-----------------

PG-SUI is aimed at scientific reasearchers, and is 


PG-SUI is a Python 3 API that uses machine learning to impute missing values from population genomic SNP data. There are several supervised and unsupervised machine learning algorithms available to impute missing data, as well as some non-machine learning imputers that are useful. 

### Supervised Imputation Methods

Supervised methods utilze the scikit-learn's IterativeImputer, which is based on the MICE (Multivariate Imputation by Chained Equations) algorithm [[1]](#1), and iterates over each SNP site (i.e., feature) while uses the N nearest neighbor features to inform the imputation. The number of nearest features can be adjusted by users. IterativeImputer currently works with any of the following scikit-learn classifiers: 

* K-Nearest Neighbors
* Random Forest
* Extra Trees
* Gradient Boosting
* XGBoost
* LightGBM

See the scikit-learn documentation (https://scikit-learn.org) for more information on IterativeImputer and each of the classifiers.

### Unsupervised Imputation Methods

Unsupervised imputers include three custom neural network models:

    + Variational Autoencoder (VAE) [[2]](#2)
    + Non-linear Principal Component Analysis (NLPCA) [[3]](#3)
    + Unsupervised Backpropagation (UBP) [[4]](#4)

VAE models train themselves to reconstruct their input (i.e., the genotypes. To use VAE for imputation, the missing values are masked and the VAE model gets trained to reconstruct only on known values. Once the model is trained, it is then used to predict the missing values.

NLPCA initializes random, reduced-dimensional input, then trains itself by using the known values (i.e., genotypes) as targets and refining the random input until it accurately predicts the genotype output. The trained model can then predict the missing values.

UBP is an extension of NLPCA that runs over three phases. Phase 1 refines the randomly generated, reduced-dimensional input in a single layer perceptron neural network to obtain good initial input values. Phase 2 uses the refined reduced-dimensional input from phase 1 as input into a multi-layer perceptron (MLP), but in Phase 2 only the neural network weights are refined. Phase three uses an MLP to refine both the weights and the reduced-dimensional input. Once the model is trained, it can be used to predict the missing values.

### Non-Machine Learning Methods

We also include several non-machine learning options for imputing missing data, including:

    + Per-population mode per SNP site
    + Global mode per SNP site
    + Using a phylogeny as input to inform the imputation
    + Matrix Factorization

These four "simple" imputation methods can be used as standalone imputers, as the initial imputation strategy for IterativeImputer (at least one method is required to be chosen), and to validate the accuracy of both IterativeImputer and the neural network models.