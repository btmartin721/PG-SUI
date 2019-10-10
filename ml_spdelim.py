#!/usr/bin/env python

#*****************************************************************************
# Script created by Bradley T. Martin, University of Arkansas
# btm002@email.uark.edu
# https://github.com/btmartin721
# This script is intended to perform species delimitation using various machine learning methods
#*****************************************************************************

# Import necessary modules
import allel
import argparse
import sys
import vcf

def main():

    args = Get_Arguments()

    popmap = read_popmap(args.popmap)


def read_popmap(file):
    """
    Reads a popmap file to a dictionary
    Input: filename (string)
    Returns: dict[sampleID] = popID
    """
    my_dict = dict()
    with open(file, "r") as fin:
        for line in fin:
            line = line.strip()
            cols = line.split()
            my_dict[cols[0]] = cols[1]
    return my_dict

def Get_Arguments():
    """
    Parse command-line arguments. Imported with argparse.
    Returns: object of command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Do species delimitation analyses using machine learning", add_help=False)

    required_args = parser.add_argument_group("Required Arguments")
    optional_args = parser.add_argument_group("Optional Arguments")

    ## Required Arguments
    required_args.add_argument("--vcf",
                                type=str,
                                required=True,
                                help="Input VCF file")
    required_args.add_argument("-p", "--popmap",
                                type=str,
                                required=True,
                                help="population map file with two tab-separated columns (sampleID\tpopID)")
    optional_args.add_argument("-o", "--outfile",
                                type=str,
                                required=False,
                                default="dadi_output",
                                help="Specify output prefix for plots")
    optional_args.add_argument("-h", "--help",
                                action="help",
                                help="Displays this help menu")

    if len(sys.argv)==1:
        print("\nExiting because no command-line options were called.\n")
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()
