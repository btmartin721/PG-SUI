#!/usr/bin/env python

#*****************************************************************************
# Script created by Bradley T. Martin, University of Arkansas
# btm002@email.uark.edu
# https://github.com/btmartin721
# This package is intended to perform species delimitation using various machine learning methods
#*****************************************************************************

# Import necessary modules
import allel
import argparse
import matplotlib as mpl
mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sys

from collections import Counter
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


def main():

    args = Get_Arguments()

    print("\nUsing scikit-allel version {}\n".format(allel.__version__))

    # Read popmap file to dictionary
    print("Reading popmap file...")
    popmap = read_popmap(args.popmap)
    popcount = Counter(popmap.values()).most_common()

    # Print population info
    print("Done! Read {} individuals across {} populations\n\n".format(str(len(popmap)), str(len(popcount))))
    print("Populations as read from popmap")
    for pop in popcount:
        print("{}: {}".format(pop[0], pop[1]))
    print("\n")

    # Read VCF file
    print("Reading VCF file...")
    callset = allel.read_vcf(args.vcf)
    print("Done! Read {} loci across {} individuals\n".format(len(callset["calldata/GT"]), len(callset["samples"])))

    # Setup the genotype data
    genotypes = allel.GenotypeChunkedArray(callset["calldata/GT"])

    n_variants = len(callset["calldata/GT"])
    pc_missing = genotypes.count_missing(axis=0)[:] * 100 / n_variants
    pc_het = genotypes.count_het(axis=0)[:] * 100 / n_variants
    pops = [p[0] for p in popcount]

    # Plot missing data and heterozygosity per individual colored by pop
    plot_genotype_frequency(pc_missing, "Missing", popmap, len(pops), popcount)
    plot_genotype_frequency(pc_het, "Heterozygous", popmap, len(pops), popcount)

    # Get allele counts
    ac = genotypes.count_alleles()

    # Count multiallelic SNPs
    multi_ac = np.count_nonzero(ac.max_allele() > 1)

    # Count biallelic singletons
    bi_singletons = np.count_nonzero((ac.max_allele() == 1) & ac.is_singleton(1))

    print("\nNumber of multiallelic SNPs: {}".format(multi_ac))
    print("Number of singletons: {}\n".format(bi_singletons))

    # Filter multiallelic SNPs and bi_singletons
    flt = (ac.max_allele() == 1) & (ac[:, :2].min(axis=1) > 1)
    gf = genotypes.compress(flt, axis=0)

    # Transform filtered genotype data to 2D matrix where each cell has number of non-reference alleles.
    gn = gf.to_n_alt()


    print("Filtered out multiallelic SNPs and singletons. Final alignment has {} SNPs\n".format(len(gf)))

    # Make LD plot
    plot_ld(gn[:1000], "Pairwise Linkage Disequilibrium")

    # Prune correlated snps then plot it.
    gnu = ld_prune(gn, size=50, step=20, threshold=.1, n_iter=1)
    plot_ld(gnu[:1000], "Pairwise LD after LD pruning")

    ############################################################################
    # Do PCA
    ############################################################################
    coords1, model1 = allel.pca(gnu, n_components=10, scaler="patterson")
    # Convert popmap to pandas df.
    df_samples = pd.DataFrame.from_dict(popmap, orient="index", columns=["popid"])
    df_samples.reset_index(inplace=True) # reset index to column
    df_samples.columns = ["sampleid", "popid"] # rename the reset column

    # Get unique Populations
    populations = df_samples.popid.unique()
    #***************************************************************************
    # Set these and/or add/take away colors for your specific Populations
    #***************************************************************************
    pop_colours = {
        "CH": "#696969",
        "DS": "#00FFFF",
        "EA": "#FF8C00",
        "FL": "#CCCC00",
        "GUFL": "#FF00FF",
        "GUMS": "#0000FF",
        "MX": "#FF0000",
        "ON": "#4B0082",
        "TT": "#228B22"
    }

    # Plot the PCA
    fig_pca(coords1, model1, "Conventional PCA", df_samples, populations, pop_colours)

    ############################################################################
    # Support Vector Machine (SVM)
    ############################################################################

    #xcoords = list()
    #ycoords = list()
    #for ind in coords1:
        #xcoords.append(ind[0])
        #ycoords.append(ind[1])

    # Make list of pops
    pop_int_list = list(range(1, len(pops)+1))

    sample_ids = list()
    pop_list = list()
    for k, v in popmap.items():
        sample_ids.append(k)
        pop_list.append(v)

    # Assign integer population IDs to sample IDs in dict
    unique_ids = dict()
    popnum = 1
    pop_int_dict = dict()
    for i in range(len(pop_list)):
        print(pop_list[i])
        popnum = get_unique_identifiers(pop_list[i], unique_ids, popnum)
        popid = unique_ids[pop_list[i]]
        pop_int_dict[sample_ids[i]] = popid

    sys.exit()
    # Split the data into train and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(xcoords, ycoords, )

    pipe_steps = [("scaler", StandardScaler()), ("pca", PCA()), ("SupVM", SVC(kernal="rbf"))]

    check_params = {
        "pca__n_components": [2],
        "SupVM__C": [0.1, 0.5, 1, 10, 30, 40, 50, 75, 100, 500, 1000],
        "SupVM__gamma": [0.001, 0.005, 0.01, 0.05, 0.07, 0.1, 0.5, 1, 5, 10, 50]
    }

    pipeline = Pipeline(pipe_steps)


def get_unique_identifiers(pattern, hit, number):

    if not hit:
        hit[pattern] = number

    elif pattern not in hit:
        number += 1
        hit[pattern] = number

    return number

def plot_pca_coords(coords, model, pc1, pc2, ax, sample_population, populations, pop_colours):
    """
    Plots PCA
    """
    sns.despine(ax=ax, offset=5)
    x = coords[:, pc1]
    y = coords[:, pc2]
    for pop in populations:
        flt = (sample_population == pop)
        ax.plot(x[flt], y[flt], marker='o', linestyle=' ', color=pop_colours[pop],
                label=pop, markersize=6, mec='k', mew=.5)
    ax.set_xlabel('PC%s (%.1f%%)' % (pc1+1, model.explained_variance_ratio_[pc1]*100))
    ax.set_ylabel('PC%s (%.1f%%)' % (pc2+1, model.explained_variance_ratio_[pc2]*100))

def fig_pca(coords, model, title, df_samples, populations, pop_colours, sample_population=None):
    """
    Makes PCA figure
    """
    if sample_population is None:
        sample_population = df_samples.popid.values
    # plot coords for PCs 1 vs 2, 3 vs 4
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 2, 1)
    plot_pca_coords(coords, model, 0, 1, ax, sample_population, populations, pop_colours)
    ax = fig.add_subplot(1, 2, 2)
    plot_pca_coords(coords, model, 2, 3, ax, sample_population, populations, pop_colours)
    ax.legend(bbox_to_anchor=(1, 1), loc='upper left')
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig("{}.pdf".format(title), bbox_inches="tight")

def ld_prune(gn, size, step, threshold=.1, n_iter=1):
    """
    Prunes correlated SNPs from dataset.
    Input: genotypes, sliding window size, Nsteps, threshold, n_iter
    Returns: Pruned genotype dataset, saves plot to file.
    """
    for i in range(n_iter):
        loc_unlinked = allel.locate_unlinked(gn, size=size, step=step, threshold=threshold)
        n = np.count_nonzero(loc_unlinked)
        n_remove = gn.shape[0] - n
        print('iteration', i+1, 'retaining', n, 'removing', n_remove, 'variants')
        gn = gn.compress(loc_unlinked, axis=0)
    return gn

def plot_ld(gn, title):
    """
    Make linkage disequilibrium plot showing correlated features.
    Input: 2D matrix of non-reference alleles.
    Returns: Makes LD plot.
    """
    m = allel.rogers_huff_r(gn) ** 2
    ax = allel.plot_pairwise_ld(m)
    ax.set_title(title)
    plt.savefig("{}.pdf".format(title))

def plot_genotype_frequency(pc, title, popdict, pops, popcount):
    """
    Plot genotype frequency data, colored by population.
    Input: Percent, plot title, popmap dictionary, Npops, popcount dict
    Returns: Writes PDF file with plot.
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.despine(ax=ax, offset=10)
    left = np.arange(len(pc))
    palette = sns.hls_palette(pops, l=.3, s=.8)
    pop2color = dict()
    for pop in range(pops):
        pop2color[popcount[pop][0]] = palette[pop]
    names = ["id", "pop"]
    formats = ["U50", "U50"]
    dtype = dict(names = names, formats = formats)
    npa = np.array(list(popdict.items()), dtype = dtype)
    colors = [pop2color[p] for p in npa["pop"]]
    ax.bar(left, pc, color=colors)
    ax.set_xlim(0, len(pc))
    ax.set_xlabel('Sample index')
    ax.set_ylabel('Percent calls')
    ax.set_title(title)
    handles = list()
    for pal in pop2color.values():
        handles.append(mpl.patches.Patch(color=pal))
    poplist = list(pop2color.keys())
    ax.legend(handles=handles, labels=poplist, title='Population', bbox_to_anchor=(1, 1), loc='upper left')
    fig.savefig("{}.pdf".format(title))
    plt.close(fig)

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
