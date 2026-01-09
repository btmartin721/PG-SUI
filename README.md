# PG-SUI

![PG-SUI Logo: Stylized blue and purple gradient design with faded appearance representing PG-SUI - Population Genomic Supervised and Unsupervised Imputation](https://github.com/btmartin721/PG-SUI/blob/master/img/pgsui-logo-faded.png)

Population Genomic Supervised and Unsupervised Imputation.

## About PG-SUI

PG-SUI is a Python 3 API that uses machine learning to impute missing values from population genomic SNP data. There are several supervised and unsupervised machine learning algorithms available to impute missing data, as well as some non-machine learning imputers that are useful.

Below is some general information and a basic tutorial. For more detailed information, see our [API Documentation](https://pg-sui.readthedocs.io/en/latest/).

### Unsupervised Imputation Methods

Unsupervised imputers include three custom neural network models:

+ Variational Autoencoder (VAE) [1](#1)
  + VAE models train themselves to reconstruct their input (i.e., the genotypes) [1](#1). To use VAE for imputation, the missing values are masked and the VAE model gets trained to reconstruct only on known values. Once the model is trained, it is then used to predict the missing values.
+ Autoencoder [2](#2)
  + A standard autoencoder that trains the input to predict itself [2](#2). As with VAE, missing values are masked and the model gets trained only on known values. Predictions are then made on the missing values.

See the below diagram for an overview of implemented features for each model.

![Side-by-side comparison of two neural network architectures for genomic imputation. Left diagram with blue boxes shows ImputeAutoencoder workflow: input genotypes with missing data encoded as 0=REF, 1=HET, 2=ALT, -9 or -1=Missing flows through gamma Schedule, Encoder Network, Latent Space, Decoder Network, Reconstruction Loss, to produce Imputed Genotype Output. Right diagram with orange boxes shows ImputeVAE architecture: genotype input flows through Encoder Network to Mean and Log Variance outputs, then Sampling with Reparameterization, KL-beta Schedule, KL Divergence Loss, Decoder Network, Reconstruction Loss, producing Imputed Genotype Output. Both models output refilled missing values. The comparison illustrates how the autoencoder differs from VAE through additional scheduled parameters and loss components in the variational model.](./img/autoencoder_vae_model_diagrams.png)

### Supervised Imputation Methods

Supervised methods utilze the scikit-learn's ``IterativeImputer``, which is based on the MICE (Multivariate Imputation by Chained Equations) algorithm [3](#3), and iterates over each SNP site (i.e., feature) while uses the N nearest neighbor features to inform the imputation. The number of nearest features can be adjusted by users. IterativeImputer currently works with the following scikit-learn classifiers:

+ ImputeRandomForest
+ ImputeHistGradientBoosting

See the [scikit-learn documentation](https://scikit-learn.org) for more information on IterativeImputer and each of the classifiers.

### Non-Machine Learning (Deterministic) Methods

We also include several deterministic options for imputing missing data, including:

+ Per-population mode per SNP site
+ Overall mode per SNP site

## Installing PG-SUI

PG-SUI supports both pip and conda distributions. Both are kept current with up-to-date releases.

### Installation with Pip

To install PG-SUI with pip, do the following. It is strongly recommended to install pg-sui in a virtual environment.

``` shell
python3 -m venv .pgsui-venv
source .pgsui-venv/bin/activate
pip install pg-sui
```

### Installation with Anaconda

To install PG-SUI with Anaconda, do the following:

``` shell
conda create -n pgsui-env python=3.12
conda activate pgsui-env
conda install -c btmartin721 pg-sui
```

### Docker Container

We also maintains a Docker image that comes with PG-SUI preinstalled. This can be useful for automated worklows such as Nextflow or Snakemake.

``` shell
docker pull pg-sui:latest
```

### Optional MacOS GUI

PG-SUI ships an optional Electron GUI (Graphical User Interface) wrapper around the Python CLI. Currently for the GUI, only MacOS is supported.

1. Install the Python-side extras (FastAPI/ uvicorn helper) if you want to serve from Python:
   `pip install pg-sui[gui]`
2. Install [Node.js](https://nodejs.org) and fetch the app dependencies:
   `pgsui-gui-setup`
3. Launch the graphical interface:
   `pgsui-gui`

The GUI shells out to the same CLI underneath, so presets, overrides, and YAML configs behave identically.

## Input Data

You can read your input files as a GenotypeData object from the [SNPio](https://snpio.readthedocs.io/en/latest/) package. SNPio supports the VCF, PHYLIP, STRUCTURE, and GENEPOP input file formats.

``` python
# Import snpio. Automatically installed with pg-sui.
from snpio import VCFReader

# Read in VCF alignment.
# SNPio also supports PHYLIP, STRUCTURE, and GENEPOP input file formats.
data = VCFReader(
    filename="pgsui/example_data/phylogen_subset14K.vcf.gz",
    popmapfile="pgsui/example_data/popmaps/phylogen_nomx.popmap", # optional
    force_popmap=True, # optional
)
```

## Supported Imputation Methods

There are several supported algorithms PG-SUI uses to impute missing data. Each one can be run by calling the corresponding class. You must provide a GenotypeData instance as the first positional argument.

You can import all the supported methods with the following:

``` python
from pgsui import ImputeVAE, ImputeAutoencoder, ImputeRefAllele, ImputeMostFrequent, ImputeRandomForest, ImputeHistGradientBoosting
```

### Unsupervised Imputers

The four unsupervised imputers can be run by initializing them with the SNPio ``GenotypeData`` object and then calling ``fit()`` and ``transform()``.

``` python
# Initialize the models, then fit and impute
vae = ImputeVAE(data) # Variational autoencoder
vae.fit()
vae_imputed = vae.transform()

ae = ImputeAutoencoder(data) # standard autoencoder
ae.fit()
ae_imputed = ae.transform()
```

The ``*_imputed`` objects are NumPy arrays of IUPAC single-character codes that are compatible with SNPio's ``GenotypeData`` objects.

### Supervised Imputers

Various supervised imputation options are supported, and these use the same API design.

``` python
# Supervised IterativeImputer classifiers

# Random Forest
rf = ImputeRandomForest(data)
rf.fit()
imputed_rf = rf.transform()

# HistGradientBoosting
hgb = ImputeHistGradientBoosting(data)
hgb.fit()
imputed_hgb = hgb.transform()
```

### Non-machine learning methods

The following deterministic methods are supported. ``ImputeMostFrequent`` supports the mode-per-population or overall (global) mode options to inform imputation.

``` python
# Per-population, per-locus mode
pop_mode = ImputeMostFrequent(data, by_populations=True)
pop_mode.fit()
imputed_pop_mode = pop_mode.transform()

# Per-locus mode
mode = ImputeMostFrequent(data, by_populations=False)
mode.fit()
imputed_mode = mode.transform()
```

Or, always replace missing values with the reference allele.

``` python
ref = ImputeRefAllele(data)
ref.fit()
imputed_ref = ref.transform()
```

## Command-Line Interface

Run the PG-SUI CLI with ``pg-sui`` (installed alongside the library). The CLI follows the same precedence model as the Python API:

``code defaults  <  preset (--preset)  <  YAML (--config)  <  explicit CLI flags  <  --set key=value``.

Recent releases add explicit switches for the simulated-missingness workflow shared by the neural and supervised models:

+ ``--sim-strategy`` selects one of ``random``, ``random_weighted``, ``random_weighted_inv``, ``nonrandom``, ``nonrandom_weighted``.
+ ``--sim-prop`` sets the proportion of observed calls to temporarily mask when building the evaluation set.

Example:

``` shell
pg-sui \
  --input data.vcf.gz \
  --popmap pops.popmap \
  --models ImputeVAE ImputeAutoencoder \
  --preset balanced \
  --sim-strategy random_weighted_inv \
  --sim-prop 0.3 \
  --prefix ae_and_vae \
  --n-jobs 4 \
  --tune-n-trials 100 \
  --set tune.enabled=True
```

CLI overrides cascade into every selected model, so a single invocation can evaluate multiple imputers with a consistent simulation strategy and output prefix.

STRUCTURE inputs accept a few extra flags for parsing metadata:

``` shell
pg-sui \
  --input data.str \
  --format structure \
  --structure-has-popids \
  --structure-allele-start-col 2 \
  --structure-allele-encoding '{"1":"A","2":"C","3":"G","4":"T","-9":"N"}'
```

## References

1. Kingma, D.P. & Welling, M. (2013). Auto-encoding variational bayes. In: Proceedings  of  the  International Conference on Learning Representations (ICLR). arXiv:1312.6114 [stat.ML].

2. Hinton, G.E., & Salakhutdinov, R.R. (2006). Reducing the dimensionality of data with neural networks. Science, 313(5786), 504-507.

3. Stef van Buuren, Karin Groothuis-Oudshoorn (2011). mice: Multivariate Imputation by Chained Equations in R. Journal of Statistical Software 45: 1-67.
