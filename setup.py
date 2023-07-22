from setuptools import setup, find_packages
import os


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


NAME = "PG-SUI"
VERSION = "0.2"
AUTHORS = "Bradley T. Martin and Tyler K. Chafin"
AUTHOR_EMAIL = "evobio721@gmail.com"
MAINTAINER = "Bradley T. Martin"
DESCRIPTION = "Python machine and deep learning package to impute missing SNPs"
LONG_DESCRIPTION = open("README.md").read()

setup(
    name=NAME,
    version=VERSION,
    author=AUTHORS,
    author_email=AUTHOR_EMAIL,
    maintainer=MAINTAINER,
    maintainer_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/btmartin721/PG-SUI",
    project_urls={
        "Bug Tracker": "https://github.com/btmartin721/PG-SUI/issues"
    },
    keywords=[
        "python",
        "impute",
        "imputation",
        "imputer",
        "machine learning",
        "neural network",
        "api",
        "IterativeImputer",
        "vae",
        "ubp",
        "nlpca",
        "autoencoder",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Operating System :: OS Independent",
        "Natural Language :: English",
    ],
    license="GNU General Public License v3 (GPLv3)",
    packages=find_packages(),
    python_requires=">=3.8,<4",
    install_requires=[
        "matplotlib",
        "seaborn",
        "jupyterlab",
        "scikit-learn>=1.0",
        "tqdm",
        "pandas",
        "numpy==1.24.3",
        "scipy",
        "xgboost",
        "tensorflow",
        "keras",
        "toytree",
        "sklearn-genetic-opt[all]>=0.6.0",
        "importlib-resources>=1.1.0",
        "pyvolve",
        "scikeras",
        "snpio",
        "urllib3>=1.26.7,<2.0.0",
        "typing-extensions<4.6.0",
    ],
    extras_require={
        "intel": ["scikit-learn-intelex"],
        "docs": ["sphinx<7", "sphinx-rtd-theme", "sphinx_autodoc_typehints"],
    },
    package_data={
        "pgsui": [
            "example_data/structure_files/*.str",
            "example_data/phylip_files/*.phy",
            "example_data/vcf_files/*",
            "example_data/popmaps/test.popmap",
            "example_data/trees/test*",
        ]
    },
    include_package_data=True,
)
