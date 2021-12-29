#!/usr/bin/env python

# Standard library imports
import argparse
import sys

import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn_genetic.space import Continuous, Categorical, Integer

# Custom module imports
from pgsui import *
# from pgsui import GenotypeData
# from pgsui import SimGenotypeData
# from pgsui import ImputeNLPCA, ImputeRandomForest
# from pgsui import ImputePhylo, ImputeAlleleFreq

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
            siterates_iqtree=args.rates
        )

        prefix="c0.001_s0.009_gtrgamma_i0.0"
        sim = SimGenotypeData(data,
                prop_missing=0.1,
                strategy="random")

        # nmf = ImputeNMF(genotype_data=sim)
        #
        # accuracy = sim.accuracy(nmf)
        # print("Accuracy:",accuracy)
        #
        # phylo = ImputePhylo(genotype_data=sim, save_plots=False)

        # accuracy = sim.accuracy(phylo)
        # print("Accuracy:",accuracy)

        grid_params = {
            "dropout_probability": Continuous(1e-2, 1.0, distribution="log-uniform")
        }

        vae = ImputeVAE(
            genotype_data=sim,
            initial_strategy="populations",
            ga=False,
            #gridparams=grid_params
        )
        accuracy = sim.accuracy(vae)
        print("VAE Accuracy:",accuracy)



        sys.exit()

        #
        # nlpca = ImputeNLPCA(
        #     genotype_data=sim,
        #     initial_strategy="populations",
        #     cv=5,
        #     num_hidden_layers=3
        # )
        # accuracy = sim.accuracy(nlpca)
        # print("NLPCA Accuracy (num_hidden_layers=3):",accuracy)
        #
        # nlpca = ImputeNLPCA(
        #     genotype_data=sim,
        #     initial_strategy="populations",
        #     cv=5,
        #     num_hidden_layers=5
        # )
        # accuracy = sim.accuracy(nlpca)
        # print("NLPCA Accuracy (num_hidden_layers=5):",accuracy)
        #
        # nlpca = ImputeNLPCA(
        #     genotype_data=sim,
        #     initial_strategy="populations",
        #     cv=5,
        #     num_hidden_layers=8
        # )
        # accuracy = sim.accuracy(nlpca)
        # print("NLPCA Accuracy (num_hidden_layers=8):",accuracy)
        #
        # nlpca = ImputeNLPCA(
        #     genotype_data=sim,
        #     initial_strategy="populations",
        #     cv=5,
        #     num_hidden_layers=10
        # )
        # accuracy = sim.accuracy(nlpca)
        # print("NLPCA Accuracy (num_hidden_layers=10):",accuracy)
        #
        #
        #
        #
        # nlpca = ImputeNLPCA(
        #     genotype_data=sim,
        #     initial_strategy="populations",
        #     cv=5,
        #     n_components=2
        # )
        # accuracy = sim.accuracy(nlpca)
        # print("NLPCA Accuracy (n_components=2):",accuracy)
        #
        # nlpca = ImputeNLPCA(
        #     genotype_data=sim,
        #     initial_strategy="populations",
        #     cv=5,
        #     n_components=3
        # )
        # accuracy = sim.accuracy(nlpca)
        # print("NLPCA Accuracy (n_components=3):",accuracy)
        #
        # nlpca = ImputeNLPCA(
        #     genotype_data=sim,
        #     initial_strategy="populations",
        #     cv=5,
        #     n_components=4
        # )
        # accuracy = sim.accuracy(nlpca)
        # print("NLPCA Accuracy (n_components=4):",accuracy)


        # ubp = ImputeUBP(
        #     genotype_data=sim,
        #     initial_strategy="populations"
        # )
        # accuracy = sim.accuracy(ubp)
        # print("Accuracy:",accuracy)

        # vae = ImputeVAE(
        #     genotype_data=sim,
        #     initial_strategy="populations"
        # )
        # accuracy = sim.accuracy(vae)
        # print("Accuracy:",accuracy)




def get_arguments():
    """[Parse command-line arguments. Imported with argparse]

    Returns:
        [argparse object]: [contains command-line arguments; accessed as method]
    """

    parser = argparse.ArgumentParser(
        description="Simulate missing data on GenotypeData object",
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
        "-i",
        "--iqtree",
        type=str,
        required=False,
        help=".iqtree output file containing Rate Matrix Q",
    )

    optional_args.add_argument(
        "-r",
        "--rates",
        type=str,
        required=False,
        help="IQ-TREE site-rates output file",
    )

    optional_args.add_argument(
        "--prefix",
        type=str,
        required=False,
        default="output",
        help="Prefix for output files",
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
