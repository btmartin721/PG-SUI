#!/usr/bin/env python

# Standard library imports
import sys

import numpy as np
import pandas as pd
import scipy.stats as stats

from sklearn_genetic.space import Continuous, Categorical, Integer

from pgsui import GenotypeData
from pgsui.impute.estimators import *
from pgsui.impute.simple_imputers import *

if sys.version_info < (3, 9):
    # importlib.resources either doesn't exist or lacks the files()
    # function, so use the PyPI version:
    import importlib_resources
else:
    # importlib.resources has files(), so use that:
    import importlib.resources as importlib_resources


def main(phylip=True, onerow=False, popmap=True, iqtree=True):
    """Class instantiations and main package body"""

    pkg = importlib_resources.files("pgsui")

    if phylip:
        parent_dir = "phylip_files"
        aln = "test.phy"
        ft = "phylip"
    else:
        parent_dir = "structure_files"
        if onerow:
            aln = "test.oneline.10sites.str"
            ft = "structure1row"
        else:
            if popmap:
                aln = "test.nopops.10sites.str"
                ft = "structure2row"
            else:
                aln = "test.pops.10sites.str"
                ft = "structure2rowPopID"

    alnfile = pkg / "example_data" / parent_dir / aln
    popmapfile = pkg / "example_data" / "popmaps" / "test.popmap"
    treefile = pkg / "example_data" / "trees" / "test.tre"
    iqtreeqmatfile = pkg / "example_data" / "trees" / "test.iqtree"
    qmatfile = pkg / "example_data" / "trees" / "test.qmat"
    siteratefile = pkg / "example_data" / "trees" / "test_n10.rate"

    prefix = "setuptest"

    print("Testing GenotypeData...")

    data = GenotypeData(
        filename=alnfile,
        filetype=ft,
        popmapfile=popmapfile,
        guidetree=treefile,
        qmatrix_iqtree=iqtreeqmatfile,
        siterates_iqtree=siteratefile,
    )

    print("SUCCESS!\n")

    # For randomizedsearchcv
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=100, stop=1000, num=10)]

    # Number of features to consider at every split
    max_features = ["sqrt", "log2"]

    # Maximum number of levels in the tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)

    # Minimmum number of samples required to split a node
    min_samples_split = [int(x) for x in np.linspace(2, 10, num=5)]

    # Minimum number of samples required at each leaf node
    min_samples_leaf = [int(x) for x in np.linspace(1, 5, num=5)]

    # Proportion of dataset to use with bootstrapping
    # max_samples = [x for x in np.linspace(0.5, 1.0, num=6)]

    # Random Forest gridparams - RandomizedSearchCV
    grid_params_random = {
        "max_features": max_features,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
    }

    # Genetic Algorithm grid_params
    grid_params_ga = {
        "max_features": Categorical(["sqrt", "log2"]),
        "min_samples_split": Integer(2, 10),
        "min_samples_leaf": Integer(1, 10),
        "max_depth": Integer(2, 110),
    }

    print(
        "Testing ImputeRandomForest with randomized grid search and initial_strategy == 'populations'..."
    )

    # Random forest imputation with RandomizedSearchCV grid search
    rf_imp = ImputeRandomForest(
        data,
        prefix=prefix,
        n_estimators=50,
        n_nearest_features=2,
        gridparams=grid_params_random,
        cv=3,
        grid_iter=40,
        n_jobs=-1,
        max_iter=2,
        column_subset=1.0,
        ga=False,
        disable_progressbar=True,
        extratrees=False,
        mutation_probability=0.1,
        chunk_size=1.0,
        initial_strategy="populations",
    )

    print("SUCCESS!\n")

    print(
        "Testing ImputeRandomForest with GA grid search and initial_strategy == 'phylogeny'..."
    )

    # Genetic Algorithm grid search Test
    rf_imp2 = ImputeRandomForest(
        data,
        prefix=prefix,
        n_estimators=50,
        n_nearest_features=2,
        gridparams=grid_params_ga,
        cv=3,
        grid_iter=40,
        n_jobs=-1,
        max_iter=2,
        column_subset=1.0,
        ga=True,
        disable_progressbar=True,
        extratrees=False,
        chunk_size=1.0,
        initial_strategy="phylogeny",
    )
    print("SUCCESS!\n")

    print("Testing ImputeAlleleFreq by-population...")
    afpops = ImputeAlleleFreq(
        genotype_data=data,
        by_populations=True,
        prefix=prefix,
        write_output=False,
    )
    print("SUCCESS!\n")

    print("Testing ImputeAlleleFreq global...")
    afpops = ImputeAlleleFreq(
        genotype_data=data,
        by_populations=False,
        prefix=prefix,
        write_output=False,
    )
    print("SUCCESS!\n")

    print("Testing ImputePhylo...")
    phylo = ImputePhylo(
        genotype_data=data,
        prefix=prefix,
        disable_progressbar=True,
        write_output=False,
    )
    print("SUCCESS!\n")

    print("Testing ImputeNMF...")
    mf = ImputeNMF(
        genotype_data=data,
        prefix=prefix,
        write_output=False,
    )
    print("SUCCESS!\n")

    print(
        "Testing VAE with validation procedure with intial_strategy='populations'..."
    )
    vae = ImputeVAE(
        genotype_data=data,
        prefix=prefix,
        disable_progressbar=True,
        validation_only=1.0,
        initial_strategy="populations",
        cv=3,
    )
    print("SUCCESS!\n")

    print("Testing ImputeNLPCA with initial_strategy == 'phylogeny'...")
    nlpca = ImputeNLPCA(
        data,
        n_components=3,
        initial_strategy="phylogeny",
        disable_progressbar=True,
        cv=3,
        hidden_activation="elu",
        hidden_layer_sizes="midpoint",
        validation_only=None,
        num_hidden_layers=1,
        learning_rate=0.1,
    )
    print("SUCCESS!\n")

    print("Testing ImputeUBP with initial_strategy == 'nmf'...")
    ubp = ImputeUBP(
        genotype_data=data,
        initial_strategy="nmf",
        disable_progressbar=True,
        valildation_only=None,
        learning_rate=0.1,
        num_hidden_layers=1,
        hidden_layer_sizes=1,
        hidden_activation="elu",
        cv=3,
        n_components=3,
    )
    print("SUCCESS!\n")

    print("All tests passed successfully!")
