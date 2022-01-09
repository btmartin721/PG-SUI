
<img src="url" alt="PG-SUI Logo" width="100%" height="100%">


# PG-SUI

Population Genomic Supervised and Unsupervised Imputation

## About PG-SUI

PG-SUI is a Python 3 API that uses machine learning to impute missing values from population genomic SNP data. There are several supervised and unsupervised machine learning algorithms available to impute missing data, as well as some non-machine learning imputers that are useful. 

### Supervised Imputation Methods

Supervised methods utilze the scikit-learn's IterativeImputer, which is based on the MICE (Multivariate Imputation by Chained Equations) algorithm [[1]](#1), and iterates over each SNP site (i.e., feature) while uses the N nearest neighbor features to inform the imputation. The number of nearest features can be adjusted by users. IterativeImputer currently works with any of the following scikit-learn classifiers: 

    + K-Nearest Neighbors
    + Random Forest
    + Extra Trees
    + Gradient Boosting
    + XGBoost
    + LightGBM

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

## Dependencies

+ python >= 3.7
+ pandas == 1.2.5
+ numpy == 1.20
+ scipy >= 1.6.2 and < 1.7.0
+ matplotlib
+ seaborn
+ jupyterlab
+ tqdm
+ toytree
+ scikit-learn >= 0.24
+ tensorflow >= 2.0
+ keras
+ xgboost
+ lightgbm

### Installation

The requirements can mostly be installed with conda. The only module that isn't available on conda is sklearn-genetic-opt, which can be installed via pip.

```
conda create -n pg-sui python=3.7
conda activate pg-sui

conda install matplotlib seaborn jupyterlab scikit-learn tqdm pandas=1.2.5 numpy=1.20.2 scipy=1.6.2 xgboost lightgbm tensorflow keras

# Only works if using Intel CPUs; speeds up processing
conda install scikit-learn-intelex

conda install -c conda-forge toytree

# For genetic algorithm plotting functions
pip install sklearn-genetic-opt[all]
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

#close and re-open terminal

#create and activate conda environment
conda create -n pg-sui python

#activate environment
conda activate pg-sui

#install packages
conda install -c conda-forge matplotlib seaborn jupyterlab scikit-learn tqdm pandas=1.2.5 numpy=1.20.2 scipy=1.6.2 xgboost lightgbm tensorflow keras sklearn-genetic toytree

#downgrade setuptools (may or may not be necessary)
pip install setuptools==57

#install sklearn-genetic-opt and mlflow
pip install sklearn-genetic-opt mlflow

```

Any other problems we run into testing on the Mac ARM architecture will be adjusted here. Note that the step installing scikit-learn-intelex was skipped here. PG-SUI will automatically detect the CPU architecture you are running, and forgo importing this package (which will only work on Intel processors)

## Input Data

Takes a STRUCTURE or PHYLIP file and a population map (popmap) file as input.  
There are a number of options for the structure file format. See the help menu:

```python pg_sui.py -h``` 

You can read your input files like this:

```
# Read in PHYLIP or STRUCTURE-formatted file
data = GenotypeData(...)
```

The data can be retrieved as a pandas DataFrame, a 2D numpy array, or a 2D list, each with shape (n_samples, n_SNPs):

```
df = data.genotypes012_df
arr = data.genotypes012_array
l = data.genotypes012_list
```

You can also retrieve the number of individuals and SNP sites:

```
num_inds = data.indcount
num_snps = data.snpcount
```

And to retrieve a list of sample IDs or population IDs:

```
inds = data.individuals
pops = data.populations
```

## Supported Imputation Methods

There are numerous supported algorithms to impute missing data. Each one can be run by calling the corresponding class.

```
# Various imputation options are supported

# Supervised IterativeImputer classifiers
knn = ImputeKNN(...) # K-Nearest Neighbors
rf = ImputeRandomForest(...) # Random Forest or Extra Trees
gb = ImputeGradientBoosting(...) # Gradient Boosting
xgb = ImputeXGBoost(...) # XGBoost
lgbm = ImputeLightGBM(...) # LightGBM

# Non-machine learning methods

# Use phylogeny to inform imputation
phylo = ImputePhylo(...)

# Use by-population or global allele frequency to inform imputation
pop_af = ImputeAlleleFreq(by_populations=True, ...)
global_af = ImputeAlleleFreq(by_populations=False, ...)

mf = ImputeMF(...) # Matrix factorization

# Unsupervised neural network models

vae = ImputeVAE(...) # Variational autoencoder
nlpca = ImputeNLPCA(...) # Nonlinear PCA
ubp = ImputeUBP(...) # Unsupervised backpropagation
```

## To-Dos

- read_vcf
- matrix factorization
- simulations
- Documentation

## References:
   
    <a id="1">[1]</a>Stef van Buuren, Karin Groothuis-Oudshoorn (2011). mice: Multivariate Imputation by Chained Equations in R. Journal of Statistical Software 45: 1-67.

     <a id="2">[2]</a>Kingma, D.P. & Welling, M. (2013). Auto-encoding variational bayes. In: Proceedings  of  the  International Conference on Learning Representations (ICLR). arXiv:1312.6114 [stat.ML].

    <a id="3">[3]</a>Scholz, M., Kaplan, F., Guy, C. L., Kopka, J., & Selbig, J. (2005). Non-linear PCA: a missing data approach. Bioinformatics, 21(20), 3887-3895.
    
    <a id="4">[4]</a>Gashler, M. S., Smith, M. R., Morris, R., & Martinez, T. (2016). Missing value imputation with unsupervised backpropagation. Computational Intelligence, 32(2), 196-215.
