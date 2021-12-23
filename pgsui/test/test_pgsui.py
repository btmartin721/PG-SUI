#!/usr/bin/env python

# Standard library imports
import sys

try:
    from importlib.resources import files, as_file
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    from importlib_resources import files, as_file

import numpy as np
import pandas as pd
import scipy.stats as stats

from sklearn_genetic.space import Continuous, Categorical, Integer

from pgsui import GenotypeData
from pgsui.impute.estimators import *
from pgsui.impute.simple_imputers import *
from pgsui.example_data import structure_files, phylip_files, popmaps, trees


def main():
    """Class instantiations and main package body"""

    testaln = {
        "phylip": "test_n10.phy",
        "structure2row": "test.nopops.2row.10sites.str",
        "structure2rowPopID": "test.pops.2row.10sites.str",
        "structure1row": "test.nopops.1row.10sites.str",
        "structure1rowPopID": "test.pops.1row.10sites.str",
    }

    popmap = "test.popmap"
    tre = "test.tre"
    iqtre = "test.iqtree"
    qmat = "test.qmat"
    siterate = "test_n10.rate"
    prefix = "setuptest"

    strfile = files(structure_files).joinpath(testaln["structure2row"])
    popmapfile = files(popmaps).joinpath(popmap)
    treefile = files(trees).joinpath(tre)
    iqtreeqmatfile = files(trees).joinpath(iqtre)
    qmatfile = files(trees).joinpath(qmat)
    siteratefile = files(trees).joinpath(siterate)

    with as_file(popmapfile) as m, as_file(treefile) as t, as_file(
        iqtreeqmatfile
    ) as i, as_file(qmatfile) as q, as_file(siteratefile) as s:

        for ft, aln in testaln.items():
            if ft == "phylip":
                data_dir = phylip_files
            else:
                data_dir = structure_files

            alnfile = files(data_dir).joinpath(aln)

            print(f"Testing GenotypeData with {ft} filetype...")

            with as_file(alnfile) as a:

                data = GenotypeData(
                    filename=a,
                    filetype=ft,
                    popmapfile=m,
                    guidetree=t,
                    qmatrix_iqtree=i,
                    siterates_iqtree=s,
                )

        print("SUCCESS!\n")

        print("Testing GenotypeData with non-iqtree rate matrices..")

        with as_file(strfile) as a:
            data2 = GenotypeData(
                filename=a,
                filetype="structure2row",
                popmapfile=m,
                guidetree=t,
                qmatrix_iqtree=i,
                siterates_iqtree=s,
            )

            print("Testing simple imputers\n")

            print("Testing ImputeAlleleFreq by-population...")
            afpops = ImputeAlleleFreq(
                genotype_data=data2,
                by_populations=True,
                prefix=prefix,
                write_output=False,
            )
            print("SUCCESS!\n")

            print("Testing ImputeAlleleFreq global...")
            afpops = ImputeAlleleFreq(
                genotype_data=data2,
                by_populations=False,
                prefix=prefix,
                write_output=False,
            )
            print("SUCCESS!\n")

            print("Testing ImputePhylo...")
            phylo = ImputePhylo(
                genotype_data=data2,
                prefix=prefix,
                disable_progressbar=True,
                write_output=False,
            )
            print("SUCCESS!\n")

            print("Testing ImputeNMF...")
            mf = ImputeNMF(
                genotype_data=data2,
                prefix=prefix,
                write_output=False,
            )
            print("SUCCESS!\n")

            # For randomizedsearchcv
            # Number of trees in random forest
            n_estimators = [
                int(x) for x in np.linspace(start=100, stop=1000, num=10)
            ]

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

            print("Testing IterativeImputer\n")

            print(
                "Testing ImputeRandomForest with randomized grid search and initial_strategy == 'populations'..."
            )

            # Random forest imputation with RandomizedSearchCV grid search
            rf_imp = ImputeRandomForest(
                data2,
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
                data2,
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

            print("Testing neural networks\n")

            print(
                "Testing VAE with validation procedure with intial_strategy='populations'..."
            )
            vae = ImputeVAE(
                genotype_data=data2,
                prefix=prefix,
                disable_progressbar=True,
                validation_only=1.0,
                initial_strategy="populations",
                cv=3,
            )
            print("SUCCESS!\n")

            print("Testing ImputeNLPCA with initial_strategy == 'phylogeny'...")
            nlpca = ImputeNLPCA(
                data2,
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
                genotype_data=data2,
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
