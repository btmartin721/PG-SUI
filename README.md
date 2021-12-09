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
