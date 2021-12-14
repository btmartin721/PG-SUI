To install PG-SUI, run:

python setup.py install

About PG-SUI:

PG-SUI is a Python 3 API that uses machine learning to impute missing values from population genomic SNP data. There are several supervised and unsupervised machine learning algorithms available to impute missing data, as well as some non-machine learning imputers that are useful. 

Supervised methods utilze the scikit-learn's IterativeImputer, which is based on the MICE (Multivariate Imputation by Chained Equations) algorithm [1], and iterates over each SNP site (i.e., feature) while uses the N nearest neighbor features to inform the imputation. The number of nearest features can be adjusted by users. IterativeImputer currently works with any of the following scikit-learn classifiers: 

    1. K-Nearest Neighbors
    2. Random Forest
    3. Extra Trees
    4. Gradient Boosting
    5. XGBoost
    6. LightGBM

See the scikit-learn documentation (https://scikit-learn.org) for more information on IterativeImputer and each of the classifiers.

Unsupervised imputers include three custom neural network models:

    1. Variational Autoencoder (VAE) [2]
    2. Non-linear Principal Component Analysis (NLPCA) [3]
    3. Unsupervised Backpropagation (UBP) [4]

VAE models train themselves to reconstruct their input (i.e., the genotypes. To use VAE for imputation, the missing values are masked and the VAE model gets trained to reconstruct only on known values. Once the model is trained, it is then used to predict the missing values.

NLPCA initializes random, reduced-dimensional input, then trains itself by using the known values (i.e., genotypes) as targets and refining the random input until it accurately predicts the genotype output. The trained model can then predict the missing values.

UBP is an extension of NLPCA that runs over three phases. Phase 1 refines the randomly generated, reduced-dimensional input in a single layer perceptron neural network to obtain good initial input values. Phase 2 uses the refined reduced-dimensional input from phase 1 as input into a multi-layer perceptron (MLP), but in Phase 2 only the neural network weights are refined. Phase three uses an MLP to refine both the weights and the reduced-dimensional input. Once the model is trained, it can be used to predict the missing values.

We also include several non-machine learning options for imputing missing data, including:

    1. Per-population mode per SNP site
    2. Global mode per SNP site
    3. Using a phylogeny as input to inform the imputation
    4. Matrix Factorization

These four "simple" imputation methods can be used as standalone imputers, as the initial imputation strategy for IterativeImputer (at least one method is required to be chosen), and to validate the accuracy of both IterativeImputer and the neural network models.

References:
   
    [1] Stef van Buuren, Karin Groothuis-Oudshoorn (2011). mice: Multivariate Imputation by Chained Equations in R. Journal of Statistical Software 45: 1-67.

    [2] Kingma, D.P. & Welling, M. (2013). Auto-encoding variational bayes. In: Proceedings  of  the  International Conference on Learning Representations (ICLR). arXiv:1312.6114 [stat.ML].

    [3] Scholz, M., Kaplan, F., Guy, C. L., Kopka, J., & Selbig, J. (2005). Non-linear PCA: a missing data approach. Bioinformatics, 21(20), 3887-3895.
    
    [4] Gashler, M. S., Smith, M. R., Morris, R., & Martinez, T. (2016). Missing value imputation with unsupervised backpropagation. Computational Intelligence, 32(2), 196-215.


