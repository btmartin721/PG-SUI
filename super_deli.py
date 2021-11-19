#!/usr/bin/env python

# Standard library imports
import argparse
import sys

import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn_genetic.space import Continuous, Categorical, Integer

from utils.misc import get_processor_name
from utils.misc import generate_012_genotypes

# Custom module imports
from read_input.read_input import GenotypeData
from impute.estimators import *
from impute.neural_network_imputers import ImputeVAE
from impute.neural_network_imputers import ImputeUBP

from dim_reduction.dim_reduction import DimReduction
from dim_reduction.embed import *

from clustering.clustering import *


def main():
    """[Class instantiations and main package body]"""

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
        grid_params = {
            "max_features": max_features,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
        }

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
        #     n_nearest_features=3,
        #     gridparams=grid_params,
        #     cv=3,
        #     grid_iter=40,
        #     n_jobs=4,
        #     max_iter=2,
        #     column_subset=0.2,
        #     ga=True,
        #     disable_progressbar=True,
        #     extratrees=False,
        #     mutation_probability=0.1,
        #     progress_update_percent=20,
        #     chunk_size=0.5,
        #     initial_strategy="phylogeny",
        # )

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
        #     initial_strategy="populations",
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

        # vae_imp = ImputeVAE(
        #     # gt=np.array([[0.0, 2.0], [np.nan, 2.0], [1.0, np.nan]]),
        #     genotype_data=data,
        #     cv=3,
        #     prefix=args.prefix,
        #     disable_progressbar=True,
        #     validation_only=0.2,
        #     initial_strategy="group_mode",
        # )

        # complete_encoded = imputer.train(train_epochs=300, batch_size=256)
        # print(complete_encoded)

        # nnbp = ImputeBackPropogation(
        #     data,
        #     num_reduced_dims=3,
        #     hidden_layers=3,
        #     hidden_layer_sizes=[100, 100, 100],
        # )

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
        #     validation_only=0.5,
        #     chunk_size=1.0,
        #     initial_strategy="populations",
        # )

        # afpops = ImputeAlleleFreq(
        #     genotype_data=data,
        #     by_populations=True,
        #     prefix=args.prefix,
        #     write_output=False,
        # )

        # br_imp = ImputeBayesianRidge(data, prefix=args.prefix, n_iter=100, gridparams=grid_params, grid_iter=3, cv=3, n_jobs=4, max_iter=5, n_nearest_features=3, column_subset=4, ga=False, disable_progressbar=True, progress_update_percent=20, chunk_size=1.0)

        # vae = ImputeVAE(
        #     gt=np.array([[0, 1], [-9, 1], [2, -9]]),
        #     initial_strategy="most_frequent",
        #     cv=3,
        #     validation_only=None,
        # )

        ubp = ImputeUBP(
            genotype_data=data,
            n_components=3,
            initial_strategy="populations",
            disable_progressbar=True,
            cv=3,
            hidden_activation="elu",
            hidden_layer_sizes="sqrt",
            validation_only=0.3,
        )

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

        # ImputePhylo(genotype_data=data, save_plots=False)

    # colors = {
    # 	"GU": "#FF00FF",
    # 	"EA": "#FF8C00",
    # 	"TT": "#228B22",
    # 	"TC": "#6495ED",
    # 	"DS": "#00FFFF",
    # 	"ON": "#800080",
    # 	"CH": "#696969",
    # 	"FL": "#FFFF00",
    # 	"MX": "#FF0000"
    # }

    # dr = DimReduction(
    # 	data.imputed_rf_df,
    # 	data.populations,
    # 	data.individuals,
    # 	args.prefix,
    # 	colors=colors,
    # 	reps=2
    # )

    # pca = runPCA(dimreduction=dr, plot_cumvar=False, keep_pcs=10)

    # pca.plot(plot_3d=True)

    # rf = runRandomForestUML(dimreduction=dr, n_estimators=1000, n_jobs=4, min_samples_leaf=4)

    # rf_cmds = runMDS(
    # 	dimreduction=dr,
    # 	distances=rf.dissimilarity,
    # 	keep_dims=3,
    # 	n_jobs=4,
    # 	max_iter=1000,
    # 	n_init=25
    # )

    # rf_isomds = runMDS(dr, distances=rf.dissimilarity_matrix, metric=False, keep_dims=3, n_jobs=4, max_iter=1000, n_init=25)

    # rf_cmds.plot(plot_3d=True)
    # rf_isomds.plot(plot_3d=True)

    # tsne = runTSNE(dimreduction=dr, keep_dims=3, n_iter=20000, perplexity=15.0)
    # tsne.plot(plot_3d=True)

    # maxk = 9

    # pam_rf = PamClustering(rf_cmds, use_embedding=False, dimreduction=dr, distances=rf.dissimilarity, maxk=9, max_iter=2500)

    # pam.msw(plot_msw_clusters=True, plot_msw_line=True, axes=3)

    # pam_rf.gapstat(plot_gap=True, show_plot=False)

    # kmeans = KMeansClustering(rf_cmds, dimreduction=dr, sampleids=data.individuals, maxk=9)
    # kmeans.msw(plot_msw_clusters=True, plot_msw_line=True, axes=3)

    # tsne_pam = PamClustering(tsne, dimreduction=dr, sampleids=data.individuals)
    # tsne_pam.msw(plot_msw_clusters=True, plot_msw_line=True, axes=3)

    # cmds_dbscan = DBSCANClustering(
    # 	rf_cmds, dimreduction=dr, sampleids=data.individuals, plot_eps=True
    # )

    # cmds_dbscan.plot(plot_3d=True)

    # cmds_affprop = AffinityPropogationClustering(rf_cmds, dimreduction=dr, sampleids=data.individuals)

    # cmds_affprop.plot(plot_3d=True)

    # cmds_aggclust = AgglomHier(rf_cmds, dimreduction=dr, maxk=maxk, sampleids=data.individuals)
    # cmds.msw(plot_msw_clusters=True, plot_msw_line=True, axes=3)


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
