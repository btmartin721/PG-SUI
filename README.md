
<img src="https://github.com/btmartin721/PG-SUI/blob/master/img/pgsui-logo-faded.png" alt="PG-SUI Logo" width="50%" height="50%">


# PG-SUI

Population Genomic Supervised and Unsupervised Imputation.

## About PG-SUI

PG-SUI is a Python 3 API that uses machine learning to impute missing values from population genomic SNP data. There are several supervised and unsupervised machine learning algorithms available to impute missing data, as well as some non-machine learning imputers that are useful. 

Below is some general information and a basic tutorial. For more detailed information, see our [API Documentation](https://pg-sui.readthedocs.io/en/latest/).

### Supervised Imputation Methods

Supervised methods utilze the scikit-learn's IterativeImputer, which is based on the MICE (Multivariate Imputation by Chained Equations) algorithm ([1](#1)), and iterates over each SNP site (i.e., feature) while uses the N nearest neighbor features to inform the imputation. The number of nearest features can be adjusted by users. IterativeImputer currently works with any of the following scikit-learn classifiers: 

+ K-Nearest Neighbors
+ Random Forest
+ XGBoost

See the scikit-learn documentation (https://scikit-learn.org) for more information on IterativeImputer and each of the classifiers.

### Unsupervised Imputation Methods

Unsupervised imputers include three custom neural network models:

+ Variational Autoencoder (VAE) ([2](#2))
+ Standard Autoencoder (SAE) ([3](#3))
+ Non-linear Principal Component Analysis (NLPCA) ([4](#4))
+ Unsupervised Backpropagation (UBP) ([5](#5))

VAE models train themselves to reconstruct their input (i.e., the genotypes). To use VAE for imputation, the missing values are masked and the VAE model gets trained to reconstruct only on known values. Once the model is trained, it is then used to predict the missing values.

SAE is a standard autoencoder that trains the input to predict itself. As with VAE, missing values are masked and the model gets trained only on known values. Predictions are then made on the missing values.

NLPCA initializes random, reduced-dimensional input, then trains itself by using the known values (i.e., genotypes) as targets and refining the random input until it accurately predicts the genotype output. The trained model can then predict the missing values.

UBP is an extension of NLPCA that runs over three phases. Phase 1 refines the randomly generated, reduced-dimensional input in a single layer perceptron neural network to obtain good initial input values. Phase 2 uses the refined reduced-dimensional input from phase 1 as input into a multi-layer perceptron (MLP), but in Phase 2 only the neural network weights are refined. Phase three uses an MLP to refine both the weights and the reduced-dimensional input. Once the model is trained, it can be used to predict the missing values.

### Non-Machine Learning Methods

We also include several non-machine learning options for imputing missing data, including:

+ Per-population mode per SNP site
+ Global mode per SNP site
+ Using a phylogeny as input to inform the imputation
+ Matrix Factorization

These four "simple" imputation methods can be used as standalone imputers, as the initial imputation strategy for IterativeImputer (at least one method is required to be chosen), and to validate the accuracy of both IterativeImputer and the neural network models.

## Installing PG-SUI

The easiest way to install PG-SUI is to use pip:

```
pip install pgsui
```

If you have an Intel CPU and want to use the sklearn-genetic-intelex package to speed up scikit-learn computations, you can do:

```
pip install pgsui[intel]
```

## Manual Installation

### Dependencies

+ python >= 3.8
+ pandas
+ numpy==1.24.3
+ scipy
+ matplotlib
+ seaborn
+ plotly
+ kaleido
+ jupyterlab
+ tqdm
+ toytree
+ pyvolve
+ scikit-learn
+ tensorflow >= 2.7
+ keras >= 2.7
+ xgboost
+ scikeras >= 0.6.0
+ snpio


### Manual Install

If you want to install everything manually, the requirements can be installed with conda and pip. sklearn-genetic-opt and scikeras are only avaiable via pip, and scikeras requires tensorflow >= 2.7 and scikit-learn >= 1.0.

```
conda create -n pg-sui python
conda activate pg-sui

conda install matplotlib seaborn jupyterlab scikit-learn tqdm pandas numpy scipy xgboost lightgbm kaleido

# Only works if using Intel CPUs; speeds up processing
conda install scikit-learn-intelex

conda install -c conda-forge toytree kaleido

conda install -c bioconda pyvolve

conda install -c plotly plotly

pip install sklearn-genetic-opt[all]

pip install scikeras snpio

pip install tensorflow-cpu
```

#### Installation troubleshooting

##### "use_2to3 is invalid" error

Users running setuptools v58 may encounter this error during the last step of installation, using pip to install sklearn-genetic-opt:

```
ERROR: Command errored out with exit status 1:
   command: /Users/tyler/miniforge3/envs/pg-sui/bin/python3.8 -c 'import io, os, sys, setuptools, tokenize; sys.argv[0] = '"'"'/private/var/folders/6x/t6g4kn711z5cxmc2_tvq0mlw0000gn/T/pip-install-6y5g_mhs/deap_1d32f65d60a44056bd7031f3aad44571/setup.py'"'"'; __file__='"'"'/private/var/folders/6x/t6g4kn711z5cxmc2_tvq0mlw0000gn/T/pip-install-6y5g_mhs/deap_1d32f65d60a44056bd7031f3aad44571/setup.py'"'"';f = getattr(tokenize, '"'"'open'"'"', open)(__file__) if os.path.exists(__file__) else io.StringIO('"'"'from setuptools import setup; setup()'"'"');code = f.read().replace('"'"'\r\n'"'"', '"'"'\n'"'"');f.close();exec(compile(code, __file__, '"'"'exec'"'"'))' egg_info --egg-base /private/var/folders/6x/t6g4kn711z5cxmc2_tvq0mlw0000gn/T/pip-pip-egg-info-7hg3hcq2
       cwd: /private/var/folders/6x/t6g4kn711z5cxmc2_tvq0mlw0000gn/T/pip-install-6y5g_mhs/deap_1d32f65d60a44056bd7031f3aad44571/
  Complete output (1 lines):
  error in deap setup command: use_2to3 is invalid.
```

This occurs during the installation of DEAP, one of the dependencies for sklearn-genetic-opt. As a workaround, first downgrade setuptools, and then proceed with the installation as normal:
```
pip install setuptools==57
pip install sklearn-genetic-opt[all]

```

##### Mac ARM architecture

PG-SUI has been tested on the new Mac M1 chips and is working fine, but some changes to the installation process were necessary as of 9-December-21. Installation was successful using the following:

```
### Install Miniforge3 instead of Miniconda3
### Download: https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh
bash ~/Downloads/Miniforge3-MacOSX-arm64.sh

# Close and re-open terminal #

#create and activate conda environment
conda create -n pg-sui python

#activate environment
conda activate pg-sui

#install packages
conda install -c conda-forge matplotlib seaborn jupyterlab scikit-learn tqdm pandas numpy scipy xgboost lightgbm tensorflow keras sklearn-genetic-opt toytree
conda install -c bioconda pyvolve

#downgrade setuptools (may or may not be necessary)
pip install setuptools==57

#install sklearn-genetic-opt and mlflow
pip install sklearn-genetic-opt mlflow

```

Any other problems we run into testing on the Mac ARM architecture will be adjusted here. Note that the step installing scikit-learn-intelex was skipped here. PG-SUI will automatically detect the CPU architecture you are running, and forgo importing this package (which will only work on Intel processors)

## Input Data

You can read your input files as a GenotypeData object from the [SNPio](https://snpio.readthedocs.io/en/latest/) package:

```

# Import snpio. Automatically installed with pgsui when using pip.
from snpio import GenotypeData 

# Read in PHYLIP, VCF, or STRUCTURE-formatted alignments.
data = GenotypeData(
    filename="example_data/phylip_files/phylogen_nomx.u.snps.phy",
    popmapfile="example_data/popmaps/phylogen_nomx.popmap",
    force_popmap=True,
    filetype="auto",
    qmatrix_iqtree="example_data/trees/test.qmat",
    siterates_iqtree="example_data/trees/test.rate",
    guidetree="example_data/trees/test.tre",
    include_pops=["EA", "TT", "GU"], # Only include these populations. There's also an exclude_pops option that will exclude the provided populations.
)
```

## Supported Imputation Methods

There are numerous supported algorithms to impute missing data. Each one can be run by calling the corresponding class. You must provide a GenotypeData instance as the first positional argument.

You can import all the supported methods with:

```
from pgsui import *
```

Or you can import them one at a time.

```
from pgsui import ImputeVAE
```

### Supervised Imputers

Various supervised imputation options are supported:

```
# Supervised IterativeImputer classifiers
knn = ImputeKNN(data) # K-Nearest Neighbors
rf = ImputeRandomForest(data) # Random Forest or Extra Trees
xgb = ImputeXGBoost(data) # XGBoost
```

### Non-machine learning methods

Use phylogeny to inform imputation:

```
phylo = ImputePhylo(data)
```

Use by-population or global allele frequency to inform imputation

```
pop_af = ImputeAlleleFreq(data, by_populations=True)
global_af = ImputeAlleleFreq(data, by_populations=False)
ref_af = ImputeRefAllele(data)
```

Non-matrix factorization:

```
mf = ImputeMF(*args) # Matrix factorization
```

### Unsupervised Neural Networks

```
vae = ImputeVAE(data) # Variational autoencoder
nlpca = ImputeNLPCA(data) # Nonlinear PCA
ubp = ImputeUBP(data) # Unsupervised backpropagation
sae = ImputeStandardAutoEncoder(data) # standard autoencoder
```

## To-Dos

- simulations
- Documentation

## References:
   
<a name="1">1. </a>Stef van Buuren, Karin Groothuis-Oudshoorn (2011). mice: Multivariate Imputation by Chained Equations in R. Journal of Statistical Software 45: 1-67.

<a name="2">2. </a>Kingma, D.P. & Welling, M. (2013). Auto-encoding variational bayes. In: Proceedings  of  the  International Conference on Learning Representations (ICLR). arXiv:1312.6114 [stat.ML].

<a name="3">3. </a>Hinton, G.E., & Salakhutdinov, R.R. (2006). Reducing the dimensionality of data with neural networks. Science, 313(5786), 504-507.

<a name="4">4. </a>Scholz, M., Kaplan, F., Guy, C. L., Kopka, J., & Selbig, J. (2005). Non-linear PCA: a missing data approach. Bioinformatics, 21(20), 3887-3895.
    
<a name="5">5. </a>Gashler, M. S., Smith, M. R., Morris, R., & Martinez, T. (2016). Missing value imputation with unsupervised backpropagation. Computational Intelligence, 32(2), 196-215.
