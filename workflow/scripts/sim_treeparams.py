#!/usr/bin/env python
import sys
import os
import errno

import argparse
import toytree
import toyplot.pdf
import pyvolve
import copy
import random
import re
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

def main():
    """
    Using pyvolve and toytree to simulate data for PG-SUI

    Two ways to run:
    - Simulate SNPs along a 'pseudo-chromosome' from which SNPs are sampled
    - Simulate genes/ loci, sampling SNPs separately for each

    Pseudo-chromosome(s):
    Set num_loci = 1 (or number of desired chromosomes)
        loc_length = chromosome length (e.g., 50000)
        snps_per_locus = # total snps you want (e.g., 1000)
        make_gene_trees = False

    Separate loci (e.g., you want gene trees)
    Set num_loci = number of genes/ loci (e.g., 1000)
        loc_length = locus length (e.g., 500 or 1000)
        snps_per_locus = 1 (usually, but could be higher)
        make_gene_trees = True (if you want gene trees)

    If you just want a SNP matrix for testing PG-SUI, option 1
    is faster, and functionally not very different given the simulation
    model has no explicit mechanism of linkage (i.e. so site independence
    is true regardless). The second option will create
    a greater amount of rate heterogeneity, since each locus will be
    initialized with its own rate matrix (GTR or GTR+Gamma). In either case,
    setting write_gene_alignments = True will create sub-directories called
    'full_alignments/' containing the full sequences from with snps_per_locus
    number of SNPs will be sampled (without replacement) and concatenated to
    create the final outputs.

    """

    params = parse_args()

    ###################

    if params.seed is not None:
        random.seed(params.seed)

    clade_height = np.around(params.tree_height*params.relative_clade_height, decimals=3)
    stem_height = np.around(params.tree_height*(1.0-params.relative_clade_height), decimals=3)

    print("Clade height:", clade_height)
    print("Stem height:", stem_height)
    print("Alpha:", params.alpha)

    clades=[]
    poplabels=[]
    indlabels=[]
    for i in range(params.num_clades):
        clades.append(("pop"+str(i)))
        for j in range(params.samples_per_clade):
            poplabels.append(("pop"+str(i)))
            indlabels.append(("pop"+str(i)+"_"+str(j)))
    outgroup = "pop"+str(params.num_clades-1)+"_"

    #### Simulate trees with toytree ####

    #skeleton tree as newick
    skeleton_tree = toytree.rtree.unittree(ntips=params.num_clades,
                                treeheight=stem_height,
                                random_names=False,
                                seed=random.randint(1, (sys.maxsize * 2 + 1))).write(tree_format=5)
    #print(skeleton_tree)
    #grab newick trees for each clade
    pop_idx=0
    guidetree = skeleton_tree
    for clade in clades:
        clade_tree = toytree.rtree.unittree(ntips=params.samples_per_clade,
                                    treeheight=clade_height,
                                    random_names=False,
                                    seed=random.randint(1, (sys.maxsize * 2 + 1))).write(tree_format=5)
        clade_tree = clade_tree.replace(";","")
        for i in range(params.samples_per_clade):
            #indlabels.append((clade+"_"+str(j)))
            clade_tree = re.sub("r", (clade+"_"), clade_tree)
        guidetree = guidetree.replace(("r"+str(pop_idx)), clade_tree)
        pop_idx+=1

    tobj=toytree.tree(guidetree, tree_format=0)

    #save guide trees
    basic_tree_plot(tobj, (params.prefix+"_guidetree.pdf"))
    tobj.write((params.prefix+"_guidetree.tre"), tree_format=5)


    #### pyvolve ####
    data = dict()
    for ind in indlabels:
        data[ind] = list()

    my_tree = pyvolve.read_tree(tree=guidetree)

# Define output path for alignment files
    if params.write_gene_alignments:
        fasta_outpath = params.prefix + "_alignments"
        try:
            if not os.path.exists(fasta_outpath):
                os.mkdir(fasta_outpath)
        except PermissionError:
            print("Error: Permission denied to create directory", fasta_outpath)
            return  # terminate the script

    for locus in range(params.num_loci):
        # Generate random parameters for mutation model
        f = np.random.random(4)
        f /= f.sum()
        parameters = {
            "mu": {
                "AC": np.random.uniform(low=0.0, high=1.0),
                "AG": np.random.uniform(low=0.0, high=1.0),
                "AT": np.random.uniform(low=0.0, high=1.0),
                "CG": np.random.uniform(low=0.0, high=1.0),
                "CT": np.random.uniform(low=0.0, high=1.0),
                "GT": np.random.uniform(low=0.0, high=1.0),
            },
            "state_freqs": [f[0], f[1], f[2], f[3]],
        }
        
        # Define mutation model
        if params.model == "gtr":
            # GTR model, without rate heterogeneity
            my_model = pyvolve.Model("nucleotide", parameters)
        else:
            my_model = pyvolve.Model(
                "nucleotide",
                parameters,
                rate_factors=[
                    np.random.uniform(low=0.1, high=0.7, size=1),
                    np.random.uniform(low=0.5, high=1.2, size=1),
                    np.random.uniform(low=1.0, high=1.8, size=1),
                    np.random.uniform(low=1.5, high=5.0, size=1),
                ],
                rate_probs=[0.4, 0.3, 0.2, 0.1],
            )
        
        # Define output path for gene alignments
        if params.write_gene_alignments:
            fastaout = os.path.join(fasta_outpath, "_loc{}.fasta".format(locus))
        else:
            fastaout = params.prefix + "_loc{}.fasta".format(locus)

        # Sample a gene alignment
        while True:
            loc = sample_locus(my_tree, my_model, params.loc_length, params.snps_per_locus, fastaout)
            if loc:
                # Sample SNP(s) from gene alignment
                sampled = sample_snp(read_fasta(fastaout), params.snps_per_locus)
                if sampled:
                    data = add_locus(data, sampled)
                    break

        # Clean up if not writing gene alignments
        if not params.write_gene_alignments:
            os.remove(fastaout)

    #Modeled as contemporary exchange from pop2 -> pop1
    snp_out=params.prefix+".phylip"

    if params.alpha > 0:
        source_pool = [indlabels[i] for i, pop in enumerate(poplabels) if pop == "pop2"]
        target_pool = [indlabels[i] for i, pop in enumerate(poplabels) if pop == "pop1"]
        introgressed_data = hybridization(data,
                                prob=params.alpha,
                                source=source_pool,
                                target=target_pool)
        write_phylip(introgressed_data, snp_out)
    else:
        write_phylip(data, snp_out)


def reroot_tree(tree, rooted="out.rooted.tre", outgroup_wildcard="out"):
    t=toytree.tree(tree)
    try:
        rt=t.root(wildcard=outgroup_wildcard)
        rt.write(rooted, tree_format=5)
        return(rt)
    except Exception:
        t.write(rooted, tree_format=5)
        return(None)


def hybridization(dat, prob=0.1, source=None, target=None):
    new_dat=dict()
    if source is None:
        source = [key for key in dat.keys()]
    if target is None:
        target = [key for key in dat.keys()]

    for individual in dat.keys():
        new_dat[individual] = dat[individual]
        aln_len=len(dat[individual])
    all_indices=list(range(aln_len))
    num=int(aln_len*prob)

    for target_individual in target:
        snp_indices = np.random.choice(all_indices, size=num, replace=False)
        for index in snp_indices:
            source_ind=np.random.choice(source, size=1)[0]
            new_dat[target_individual][index] = new_dat[source_ind][index]
    return(new_dat)

def add_locus(d, new):
    for sample in d.keys():
        for snp in new[sample]:
            d[sample].append(snp)
    return(d)

def write_fasta(seqs, fas):
    with open(fas, 'w') as fh:
        #Write seqs to FASTA first
        for a in seqs.keys():
            name = ">" + str(a) + "\n"
            seq = "".join(seqs[a]) + "\n"
            fh.write(name)
            fh.write(seq)
        fh.close()

def write_phylip(seqs, phy):
    #get header
    samps=0
    snps=None
    for key in seqs.keys():
        samps+=1
        if snps is None:
            snps = len(seqs[key])
        elif snps != len(seqs[key]):
            raise ValueError(("Error writing file"+phy+"- sequences not equal length\n"))
    with open(phy, 'w') as fh:
        header=str(samps)+"\t"+str(snps)+"\n"
        fh.write(header)
        #Write seqs to FASTA first
        for a in seqs.keys():
            line = str(a) + "\t" + "".join(seqs[a]) + "\n"
            fh.write(line)
        fh.close()

def read_phylip(phy):
    data = dict()
    header=True
    sample=None
    with open(phy, "r") as fin:
        for line in fin:
            line = line.strip()
            if not line:  # If blank line.
                continue
            else:
                if header==True:
                    header=False
                    continue
                else:
                    stuff = line.split()
                    data[stuff[0]] = stuff[1]
        fin.close()
        return(data)

def read_fasta(fasta):
    data = dict()
    header=False
    sample=None
    sequence=""
    with open(fasta, "r") as fin:
        for line in fin:
            line = line.strip()
            if not line:  # If blank line.
                continue
            if line[0] == ">":
                if sample:
                    data[sample] = sequence
                    sequence = ""
                sample=line[1:]
            else:
                sequence = sequence + line
        data[sample] = sequence
        fin.close()
        return(data)

def sample_snp(aln_dict, snps_per_locus=1):
    snp_indices = []
    snp_aln = dict()
    
    # Get a list of all sample names
    samples = list(aln_dict.keys())

    # Infer the length of the alignment from the first sample
    aln_len = len(aln_dict[samples[0]])

    # Initialize the dictionary to hold the SNP alignments
    for sample in samples:
        snp_aln[sample] = []

    # Loop over all positions in the alignment
    for i in range(aln_len):
        vars=[]
        for sample in samples:
            nuc=aln_dict[sample][i]
            if len(vars) == 0:
                vars.append(nuc)
            elif nuc not in vars:
                snp_indices.append(i)
                break

    # Check if any SNPs were found
    if len(snp_indices) == 0:
        return False  # No SNPs found

    # If the number of SNPs is less than or equal to `snps_per_locus`, return all SNPs
    elif len(snp_indices) <= snps_per_locus:
        for sample in samples:
            for i in snp_indices:
                snp_aln[sample].append(aln_dict[sample][i])
    # If there's more SNPs than `snps_per_locus`, randomly choose `snps_per_locus` of them
    else:
        sampled_indices = np.random.choice(snp_indices, size=snps_per_locus, replace=False)
        for sample in samples:
            for i in sampled_indices:
                snp_aln[sample].append(aln_dict[sample][i])
    return snp_aln


def sample_locus(tree, model, gene_len=1000, num_snps=1, out="out.fasta"):
    try:
        my_partition = pyvolve.Partition(models=model, size=gene_len)
        my_evolver = pyvolve.Evolver(partitions=my_partition, tree=tree)
        my_evolver(seqfile=out, seqfmt="fasta", ratefile=False, infofile=False)
        return True
    except Exception as e:
        print("Error in sample_locus:", str(e))
        return False


def silentremove(filename):
    try:
        os.remove(filename)
    except OSError as e: # this would be "except OSError, e:" before Python 2.6
        if e.errno != errno.ENOENT: # errno.ENOENT = no such file or directory
            raise # re-raise exception if a different error occurred

def get_tree_tips(tree):
    tips = re.split('[ ,\(\);]', tree)
    return([i for i in tips if i])

def basic_tree_plot(tree, out="out.pdf"):
    mystyle = {
        "edge_type": "p",
        "edge_style": {
            "stroke": toytree.colors[0],
            "stroke-width": 1,
        },
        "tip_labels_align": True,
        "tip_labels_style": {"font-size": "5px"},
        "node_labels": False,
        "tip_labels": True
    }

    canvas, axes, mark = tree.draw(
        width=400,
        height=600,
        **mystyle,
    )

    toyplot.pdf.render(canvas, out)

def mutation_model_type(value):
    allowed_values = ["gtr", "gtrgamma"]
    if value not in allowed_values:
        raise argparse.ArgumentTypeError(f"Invalid model. Allowed values are {allowed_values}")
    return value


def parse_args():
    parser = argparse.ArgumentParser(
        description="""Tree-based sequence simulation with toytree and pyvolve"""
    )

    parser.add_argument("-p", "--prefix", type=str, default="", help="Output file prefix.")
    parser.add_argument("-r", "--seed", type=int, default=None, help="Seed for random number generator.")
    parser.add_argument("-n", "--num_clades", type=int, default=4, help="Number of clades.")
    parser.add_argument("-S", "--samples_per_clade", type=int, default=20, help="Number of samples per clade.")
    parser.add_argument("-l", "--num_loci", type=int, default=1000, help="Number of loci.")
    parser.add_argument("-L", "--loc_length", type=int, default=250, help="Length of each locus.")
    parser.add_argument("-w", "--write_gene_alignments", action='store_true', help="If set, write gene alignments.")
    parser.add_argument("-s", "--snps_per_locus", type=int, default=1, help="Number of SNPs per locus.")
    parser.add_argument("-t", "--tree_height", type=float, default=0.01, help="Height of the tree.")
    parser.add_argument("-c", "--relative_clade_height", type=float, default=0.1, help="Relative height of the clade within the total tree height.")
    parser.add_argument("-m", "--model", type=mutation_model_type, default="gtrgamma", help="Mutation model [gtr or gtrgamma]")
    parser.add_argument("-a", "--alpha", type=float, default=0.0, help="Proportion of alleles to introgress between pop1 and pop2.")

    args = parser.parse_args()

    # Warn if alpha is set but num_clades < 2
    if args.alpha > 0 and args.num_clades < 2:
        print("Warning: Alpha is set but num_clades is less than 2. No introgression will occur.")

    # Construct the file output name prefix
    args.prefix = "{}t{}_c{}_a{}_{}".format(args.prefix, args.tree_height, args.relative_clade_height, args.alpha, args.model)
    args.base = os.path.basename(args.prefix)

    return args

if __name__ == "__main__":
    main()
