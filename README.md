# PG-SUI

Population Genomic Supervised and Unsupervised Imputation

## Requirements

+ python == 3.7
+ pandas == 1.2.5
+ numpy == 1.20
+ matplotlib
+ seaborn
+ tqdm
+ jupyterlab
+ scikit-learn == 0.24
+ sklearn-genetic-opt >= 0.6.0
+ toytree
+ scipy >= 1.6.2 and < 1.7.0

Python versions other than 3.7 are not currently supported.  

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

## Input files

Takes a structure or phylip file and a popmap file as input.  
There are a number of options for the structure file format. See the help menu:

```python pg_sui.py -h```  

## API Imputation Options

```
# Read in PHYLIP or STRUCTURE-formatted file
data = GenotypeData(...)

# Various imputation options are supported

# Supervised IterativeImputer classifiers
knn = ImputeKNN(...)
rf = ImputeRandomForest(...)
gb = ImputeGradientBoosting(...)
xgb = ImputeXGBoost(...)
lgbm = ImputeLightGBM(...)

# Use phylogeny to inform imputation
phylo = ImputePhylo(...)

# Use by-population or global allele frequency to inform imputation
af = ImputeAlleleFreq(...)

# Unsupervised neural network models
vae = ImputeVAE(...)
ubp = ImputeUBP(...)


## To-Dos

- read_vcf
- matrix factorization
- simulations
