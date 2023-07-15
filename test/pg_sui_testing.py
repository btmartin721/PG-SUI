#!/usr/bin/env python

# Standard library imports
import argparse
import sys

import numpy as np
import pandas as pd
import scipy.stats as stats

from sklearn_genetic.space import Continuous, Categorical, Integer

# from pgsui import GenotypeData
from snpio import GenotypeData
from impute.estimators import (
    ImputeNLPCA,
    ImputeUBP,
    ImputeRandomForest,
    ImputeVAE,
)
from impute.simple_imputers import ImputePhylo

# from snpio import GenotypeData
# from impute.estimators import *
# from impute.simple_imputers import ImputeAlleleFreq, ImputePhylo

# from read_input import GenotypeData
# from estimators import *


def main():
    """Class instantiations and main package body"""

    args = get_arguments()

    if args.str and args.phylip:
        sys.exit("Error: Only one file type can be specified")

        # If VCF file is specified.
    if args.str:
        if not args.pop_ids and args.popmap is None:
            raise TypeError("Either --pop_ids or --popmap must be specified\n")

        if args.pop_ids:
            print("\n--pop_ids was specified as column 2\n")
        else:
            print(
                "\n--pop_ids was not specified; "
                "using popmap file to get population IDs\n"
            )

        if args.onerow_perind:
            print("\nUsing one row per individual...\n")
        else:
            print("\nUsing two rows per individual...\n")

        if args.onerow_perind:
            data = GenotypeData(
                filename=args.str,
                filetype="structure1row",
                popmapfile=args.popmap,
                guidetree=args.treefile,
                qmatrix_iqtree=args.iqtree,
            )
        else:
            data = GenotypeData(
                filename=args.str,
                filetype="structure2row",
                popmapfile=args.popmap,
                guidetree=args.treefile,
                qmatrix_iqtree=args.iqtree,
            )

    if args.phylip:
        if args.pop_ids or args.onerow_perind:
            print(
                "\nPhylip file was used with structure arguments; ignoring "
                "structure file arguments\n"
            )

        if args.popmap is None:
            raise TypeError("No popmap file supplied with PHYLIP file\n")

        data = GenotypeData(
            filename=args.phylip,
            filetype="phylip",
            popmapfile=args.popmap,
            guidetree=args.treefile,
            qmatrix_iqtree=args.iqtree,
            siterates_iqtree="pgsui/example_data/trees/test_n10.rate",
        )

    if args.resume_imputed:
        pass
        # data.read_imputed(args.resume_imputed, impute_methods="rf")
        # data.write_imputed(data.imputed_rf_df, args.prefix)

    else:
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

        # # Random Forest gridparams - RandomizedSearchCV
        # grid_params = {
        #     "max_features": max_features,
        #     "max_depth": max_depth,
        #     "min_samples_split": min_samples_split,
        #     "min_samples_leaf": min_samples_leaf,
        # }

        # Random Forest gridparams - Genetic Algorithms
        # grid_params = {
        # 	"n_estimators": Integer(100, 500),
        # 	"max_features": max_features,
        # 	"max_depth": max_depth,
        # 	"min_samples_split": min_samples_split,
        # 	"min_samples_leaf": min_samples_leaf,
        # 	"max_samples": max_samples
        # }

        # # Genetic Algorithm grid_params
        # grid_params = {
        #     "max_features": Categorical(["sqrt", "log2"]),
        #     "min_samples_split": Integer(2, 10),
        #     "min_samples_leaf": Integer(1, 10),
        #     "max_depth": Integer(2, 110),
        # }

        # Bayesian Ridge gridparams - RandomizedSearchCV
        # grid_params = {
        # 	"alpha_1": stats.loguniform(1e-6, 0.01),
        # 	"alpha_2": stats.loguniform(1e-6, 0.01),
        # 	"lambda_1": stats.loguniform(1e-6, 0.01),
        # 	"lambda_2": stats.loguniform(1e-6, 0.01),
        # }

        # # Bayesian Ridge gridparams - Genetic algorithm
        # grid_params = {
        # 	"alpha_1": Continuous(1e-6, 1e-3, distribution="log-uniform"),
        # 	"alpha_2": Continuous(1e-6, 1e-3, distribution="log-uniform"),
        # 	"lambda_1": Continuous(1e-6, 1e-3, distribution="log-uniform"),
        # 	"lambda_2": Continuous(1e-6, 1e-3, distribution="log-uniform")
        # }

        # # Random forest imputation with genetic algorithm grid search
        # rf_imp = ImputeRandomForest(
        #     data,
        #     prefix=args.prefix,
        #     n_estimators=50,
        #     n_nearest_features=1,
        #     gridparams=grid_params,
        #     cv=3,
        #     grid_iter=40,
        #     n_jobs=4,
        #     max_iter=2,
        #     column_subset=1.0,
        #     ga=False,
        #     disable_progressbar=True,
        #     extratrees=False,
        #     mutation_probability=0.1,
        #     progress_update_percent=20,
        #     chunk_size=1.0,
        #     initial_strategy="phylogeny",
        # )

        # # Genetic Algorithm grid search Test
        # rf_imp2 = ImputeRandomForest(
        #     data,
        #     prefix=args.prefix,
        #     n_estimators=50,
        #     n_nearest_features=2,
        #     gridparams=grid_params,
        #     cv=3,
        #     grid_iter=40,
        #     n_jobs=-1,
        #     max_iter=2,
        #     column_subset=1.0,
        #     ga=True,
        #     disable_progressbar=True,
        #     extratrees=False,
        #     chunk_size=1.0,
        #     initial_strategy="phylogeny",
        # )

        # rfdata = rf_imp.imputed
        # print(rfdata.genotypes012_df)

        # rf_data = rf_imp.imputed
        # print(data.genotypes012_df)
        # print(rf_data.genotypes012_df)

        # imp_decoded = data.decode_imputed(rf_imp.imputed)
        # print(imp_decoded)

        # # RandomizedSearchCV Test
        # rf_imp = ImputeRandomForest(
        #     data,
        #     prefix=args.prefix,
        #     n_estimators=50,
        #     n_nearest_features=3,
        #     gridparams=grid_params,
        #     cv=3,
        #     grid_iter=40,
        #     n_jobs=4,
        #     max_iter=2,
        #     column_subset=5,
        #     ga=False,
        #     disable_progressbar=False,
        #     extratrees=False,
        #     progress_update_percent=20,
        #     chunk_size=0.2,
        #     initial_strategy="phylogeny",
        # )

        # lgbm = ImputeLightGBM(
        #     data,
        #     prefix=args.prefix,
        #     cv=3,
        #     n_jobs=4,
        #     n_estimators=50,
        #     disable_progressbar=True,
        #     chunk_size=0.2,
        #     validation_only=0.1,
        #     n_nearest_features=3,
        #     max_iter=2,
        #     initial_strategy="populations",
        # )

        # vae = ImputeVAE(
        #     genotype_data=data,
        #     prefix=args.prefix,
        #     disable_progressbar=True,
        #     validation_only=None,
        #     initial_strategy="populations",
        # )

        # vae_gtdata = vae.imputed
        # print(vae_gtdata.genotypes012_df)

        # complete_encoded = imputer.train(train_epochs=300, batch_size=256)
        # print(complete_encoded)

        # rf_imp = ImputeRandomForest(
        #     data,
        #     prefix=args.prefix,
        #     n_estimators=50,
        #     n_nearest_features=3,
        #     n_jobs=4,
        #     max_iter=2,
        #     disable_progressbar=True,
        #     extratrees=False,
        #     max_features="sqrt",
        #     min_samples_split=5,
        #     min_samples_leaf=2,
        #     max_depth=30,
        #     cv=3,
        #     validation_only=0.3,
        #     chunk_size=1.0,
        #     initial_strategy="populations",
        # )

        # afpops = ImputeAlleleFreq(
        #     genotype_data=data,
        #     by_populations=True,
        #     prefix=args.prefix,
        # )

        # print(data.genotypes012_df)
        # print(afpops.genotypes012_df)

        # br_imp = ImputeBayesianRidge(data, prefix=args.prefix, n_iter=100, gridparams=grid_params, grid_iter=3, cv=3, n_jobs=4, max_iter=5, n_nearest_features=3, column_subset=4, ga=False, disable_progressbar=True, progress_update_percent=20, chunk_size=1.0)

        # aftestpops = ImputeAlleleFreq(
        #     genotype_data=data, by_populations=True, prefix=args.prefix
        # )

        # aftestpops_data = aftestpops.imputed

        # print(data.genotypes012_df)
        # print(aftestpops_data.genotypes012_df)

    # vae = ImputeVAE(
    #     gt=np.array([[0, 1], [-9, 1], [2, -9]]),
    #     initial_strategy="most_frequent",
    #     cv=3,
    #     validation_only=None,
    # )

    # vae_data = vae.imputed

    # print(data.genotypes012_df)
    # print(vae_data.genotypes012_df)

    # For GridSearchCV. Generate parameters to sample from.
    learning_rate = [float(10) ** x for x in np.arange(-4, 0)]
    l1_penalty = [float(10) ** x for x in np.arange(-6, -1)]
    l1_penalty.append(0.0)
    l2_penalty = [float(10) ** x for x in np.arange(-6, -1)]
    l2_penalty.append(0.0)
    hidden_activation = ["elu", "relu"]
    num_hidden_layers = [1, 2, 3, 4, 5]
    hidden_layer_sizes = ["sqrt", "midpoint"]
    n_components = [2, 3]
    dropout_rate = [round(x, 1) for x in np.arange(0.0, 1.0, 0.1)]
    batch_size = [16, 32, 48, 64]
    optimizer = ["adam", "sgd", "adagrad"]

    # grid_params = {
    #     "learning_rate": Continuous(1e-6, 0.1, distribution="log-uniform"),
    #     "l2_penalty": Continuous(1e-6, 0.01, distribution="uniform"),
    #     "n_components": Integer(2, 3),
    #     # "hidden_activation": Categorical(["elu", "relu"]),
    # }

    grid_params = {
        # "learning_rate": learning_rate,
        # "l1_penalty": l1_penalty,
        "l2_penalty": l2_penalty,
        # "hidden_activation": hidden_activation,
        # "hidden_layer_sizes": hidden_layer_sizes,
        "n_components": n_components,
        # "dropout_rate": dropout_rate,
        # "batch_size": batch_size,
        # "optimizer": optimizer,
    }

    ubp = ImputeUBP(
        data,
        disable_progressbar=False,
        cv=3,
        column_subset=1.0,
        validation_split=0.0,
        learning_rate=0.1,
        num_hidden_layers=1,
        verbose=1,
        dropout_rate=0.2,
        hidden_activation="elu",
        batch_size=64,
        l1_penalty=1e-6,
        l2_penalty=1e-6,
        gridparams=grid_params,
        n_jobs=4,
        grid_iter=5,
        sim_strategy="nonrandom_weighted",
        sim_prop_missing=0.4,
        scoring_metric="precision_recall_macro",
        gridsearch_method="randomized_gridsearch",
        early_stop_gen=5,
        # sample_weights={0: 1.0, 1: 0.0, 2: 1.0},
        # sample_weights="auto",
    )

    # ubp = ImputeVAE(
    #     data,
    #     # gridparams=grid_params,
    #     # initial_strategy="populations",
    #     # disable_progressbar=True,
    #     # cv=3,
    #     # column_subset=1.0,
    #     # validation_size=0.3,
    #     # learning_rate=0.1,
    #     # num_hidden_layers=1,
    #     # verbose=1,
    #     # gridparams=grid_params,
    # )

    # nlpca_data = nlpca.imputed
    # print(nlpca_data.genotypes012_df)

    # print(data.genotypes012_df)
    # print(nlpca_data.genotypes012_df)

    # ubp = ImputeUBP(
    #     genotype_data=data,
    #     test_categorical=np.array([[0, 1], [-9, 1], [2, -9]]),
    # )

    # ubp = ImputeVAE(
    #     gt=np.array([[0, 1], [-9, 1], [2, -9]]),
    #     initial_strategy="most_frequent",
    # )

    # br_imp = ImputeBayesianRidge(
    #     data,
    #     prefix=args.prefix,
    #     alpha_1=0.0002689638465560243,
    #     alpha_2=0.0001473822173361299,
    #     lambda_1=0.0003281735206234651,
    #     lambda_2=0.00020767920087590963,
    #     n_iter=100,
    #     n_nearest_features=3,
    #     progress_update_percent=20,
    #     disable_progressbar=True,
    #     max_iter=2,
    #     cv=3,
    #     initial_strategy="group_mode",
    # )

    # phylo = ImputePhylo(
    #     genotype_data=data, save_plots=False, disable_progressbar=True
    # )

    # phylodata = phylo.imputed
    # print(phylodata.genotypes012_df)


def get_arguments():
    """[Parse command-line arguments. Imported with argparse]

    Returns:
        [argparse object]: [contains command-line arguments; accessed as method]
    """

    parser = argparse.ArgumentParser(
        description="Machine learning missing data imputation and species delimitation",
        add_help=False,
    )

    required_args = parser.add_argument_group("Required arguments")
    filetype_args = parser.add_argument_group(
        "File type arguments (choose only one)"
    )
    structure_args = parser.add_argument_group("Structure file arguments")
    optional_args = parser.add_argument_group("Optional arguments")

    # File Type arguments
    filetype_args.add_argument(
        "-s", "--str", type=str, required=False, help="Input structure file"
    )
    filetype_args.add_argument(
        "-p", "--phylip", type=str, required=False, help="Input phylip file"
    )

    filetype_args.add_argument(
        "-t",
        "--treefile",
        type=str,
        required=False,
        default=None,
        help="Newick-formatted treefile",
    )

    filetype_args.add_argument(
        "-i",
        "--iqtree",
        type=str,
        required=False,
        help=".iqtree output file containing Rate Matrix Q",
    )

    # Structure Arguments
    structure_args.add_argument(
        "--onerow_perind",
        default=False,
        action="store_true",
        help="Toggles on one row per individual option in structure file",
    )
    structure_args.add_argument(
        "--pop_ids",
        default=False,
        required=False,
        action="store_true",
        help="Toggles on population ID column (2nd col) in structure file",
    )

    ## Optional Arguments
    optional_args.add_argument(
        "-m",
        "--popmap",
        type=str,
        required=False,
        default=None,
        help="Two-column tab-separated population map file: inds\tpops. No header line",
    )
    optional_args.add_argument(
        "--prefix",
        type=str,
        required=False,
        default="output",
        help="Prefix for output files",
    )

    optional_args.add_argument(
        "--resume_imputed",
        type=str,
        required=False,
        help="Read in imputed data from a file instead of doing the imputation",
    )
    # Add help menu
    optional_args.add_argument(
        "-h", "--help", action="help", help="Displays this help menu"
    )

    # If no command-line arguments are called then exit and call help menu.
    if len(sys.argv) == 1:
        print("\nExiting because no command-line options were called.\n")
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
