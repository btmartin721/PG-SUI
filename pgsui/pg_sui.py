#!/usr/bin/env python

# Standard library imports
import argparse
import sys

import numpy as np
import pandas as pd

from sklearn_genetic.space import Continuous, Categorical, Integer

from snpio import GenotypeData
from snpio import Plotting
from pgsui import *


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
                prefix=args.prefix,
            )
        else:
            data = GenotypeData(
                filename=args.str,
                filetype="structure2row",
                popmapfile=args.popmap,
                guidetree=args.treefile,
                qmatrix_iqtree=args.iqtree,
                prefix=args.prefix,
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
            siterates_iqtree=args.site_rate,
            prefix=args.prefix,
        )

    data.missingness_reports(prefix=args.prefix, plot_format="png")

    # For GridSearchCV. Generate parameters to sample from.
    learning_rate = [float(10) ** x for x in np.arange(-4, -1)]
    l1_penalty = [float(10) ** x for x in np.arange(-5, -1)]
    l1_penalty.append(0.0)
    l2_penalty = [float(10) ** x for x in np.arange(-5, -1)]
    l2_penalty.append(0.0)
    hidden_activation = ["elu", "relu"]
    num_hidden_layers = [1, 2, 3]
    hidden_layer_sizes = ["sqrt", "midpoint"]
    n_components = [2, 3, 5, 10]
    dropout_rate = [0.0, 0.2, 0.4]
    # batch_size = [16, 32, 48, 64]
    optimizer = ["adam", "sgd", "adagrad"]

    # Some are commented out for testing purposes.
    # grid_params = {
    #     "learning_rate": learning_rate,
    #     # "l1_penalty": l1_penalty,
    #     # "l2_penalty": l2_penalty,
    #     # "hidden_layer_sizes": hidden_layer_sizes,
    #     "n_components": n_components,
    #     # "dropout_rate": dropout_rate,
    #     # # "optimizer": optimizer,
    #     # "num_hidden_layers": num_hidden_layers,
    #     # "hidden_activation": hidden_activation,
    # }

    imp = ImputeXGBoost(
        data,
        max_iter=3,
        gridparams={"n_estimators": [100, 200]},
        n_nearest_features=5,
        # disable_progressbar=False,
        # epochs=100,
        # cv=3,
        # column_subset=1.0,
        # learning_rate=0.01,
        # num_hidden_layers=1,
        # hidden_layer_sizes="midpoint",
        # verbose=10,
        # dropout_rate=0.2,
        # hidden_activation="relu",
        # batch_size=32,
        # l1_penalty=1e-6,
        # l2_penalty=1e-6,
        # # gridparams=grid_params,
        # n_jobs=4,
        # grid_iter=5,
        # sim_strategy="nonrandom_weighted",
        # sim_prop_missing=0.5,
        # scoring_metric="precision_recall_macro",
        # gridsearch_method="gridsearch",
        # early_stop_gen=5,
        # n_components=3,
        # sample_weights={0: 1.0, 1: 0.0, 2: 1.0},
        # sample_weights="auto",
    )

    gd_imp = imp.imputed

    components, model = Plotting.run_pca(
        data,
        plot_format="png",
        center=True,
        scale=False,
        prefix=args.prefix,
        # n_axes=3,
    )

    components_imp, model_imp = Plotting.run_pca(
        gd_imp,
        plot_format="png",
        center=True,
        scale=False,
        prefix=args.prefix + "_imputed",
    )


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

    filetype_args.add_argument(
        "--site_rate",
        type=str,
        required=False,
        help="Specify site rate input file.",
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
        default="imputer",
        help="Prefix for output directory. Output directory will be '<prefix>_output'",
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
