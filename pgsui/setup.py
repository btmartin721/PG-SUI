from setuptools import setup, find_packages

VERSION = "0.1"
DESCRIPTION = "Python API to impute missing values frmo population genomic data"
LONG_DESCRIPTION = "PG-SUI is a python API that uses machine learning to impute missing values from SNP data. There are several supervised and unsupervised machine learning methods available to impute missing data, as well as some non-machine learning imputers that are useful."

AUTHORS = "Bradley T. Martin and Tyler K. Chafin"
AUTHOR_EMAIL = "evobio721@gmail.com"

# Setting up
setup(
    # the name must match the folder name 'verysimplemodule'
    name="pgsui",
    version=VERSION,
    author=AUTHORS,
    author_email=AUTHOR_EMAIL,
    maintainer="Bradley T. Martin",
    maintainer_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    url="https://github.com/btmartin721/PG-SUI",
    packages=find_packages(),
    install_requires=[
        "matplotlib",
        "seaborn",
        "jupyterlab",
        "scikit-learn>=0.24",
        "tqdm" "pandas>=1.2.5,<1.3.0",
        "numpy>=1.20.2,<1.21",
        "scipy>=1.6.2,<1.7",
        "xgboost",
        "lightgbm",
        "tensorflow",
        "keras",
        "toytree",
        "sklearn-genetic-opt[all]>=0.6.0",
    ],
    keywords=[
        "python",
        "impute",
        "imputation",
        "machine learning",
        "neural network",
        "api",
        "IterativeImputer",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8"
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: Unix",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
    ],
    license="GPL3",
)