#!/usr/bin/env python
import sys
import os
import subprocess

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

    NOTE: This script is not a part of the PG-SUI API, and is written
    for a single purpose, i.e., is not generalized beyond some options
    which can be manually set below. It is intended to provide transparency
    for the simulation process used in the PG-SUI manuscript, *not* as
    a flexible/ portable tool -- meaning a lot of things are hard-coded.

    """
    seed=1234
    random.seed(seed)

    num_clades=3
    samples_per_clade=20
    num_loci=1
    loc_length=10000
    write_gene_alignments=False
    make_gene_trees=False
    snps_per_locus=100
    iqtree_bin="iqtree"

    ###################

    clades=[]
    poplabels=[]
    indlabels=[]
    for i in range(num_clades):
        clades.append(("pop"+str(i)))
        poplabels.append(("pop"+str(i)))
        for j in range(samples_per_clade):
            indlabels.append(("pop"+str(i)+"_"+str(j)))

    ####### Varying clade vs. stem heights
    for clade_height in [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009]:
        print("clade heights: ", clade_height)
        stem_height = np.around(0.01-clade_height, decimals=3)
        print("stem height: ", stem_height)

        #skeleton tree as newick
        skeleton_tree = toytree.rtree.unittree(ntips=num_clades,
                                    treeheight=stem_height,
                                    random_names=False,
                                    seed=random.randint(1, (sys.maxsize * 2 + 1))).write(tree_format=5)
        #grab newick trees for each clade
        pop_idx=0
        guidetree = skeleton_tree
        for clade in clades:
            clade_tree = toytree.rtree.unittree(ntips=samples_per_clade,
                                        treeheight=clade_height,
                                        random_names=False,
                                        seed=random.randint(1, (sys.maxsize * 2 + 1))).write(tree_format=5)
            clade_tree = clade_tree.replace(";","")
            for i in range(samples_per_clade):
                #indlabels.append((clade+"_"+str(j)))
                clade_tree = re.sub("r", (clade+"_"), clade_tree)
            guidetree = guidetree.replace(("r"+str(pop_idx)), clade_tree)
            pop_idx+=1

        base="c"+str(clade_height)+"_s"+str(stem_height)
        tobj=toytree.tree(guidetree, tree_format=0)


        #Set up directory structure for this set of tree params
        treeset_path = "sim_"+base
        if not os.path.exists(treeset_path):
            os.mkdir(treeset_path)

        #save guide trees
        basic_tree_plot(tobj, (treeset_path+"/"+base+"_guidetree.pdf"))
        tobj.write((treeset_path+"/"+base+"_guidetree.tre"), tree_format=5)


        ######## With and without rate heterogeneity
        #NOTE: Run alignments through IQ-TREE to get optional
        #Rate matrix and site-specific mutation rates
        data = dict()
        for ind in indlabels:
            data[ind] = list()

        my_tree = pyvolve.read_tree(tree=guidetree)

        for model in ["gtr","gtrgamma"]:
            model_outpath=treeset_path+"/"+base+"_"+model
            if not os.path.exists(model_outpath):
                os.mkdir(model_outpath)

            if model == "gtr":
                #GTR model, without rate heterogeneity
                my_model = pyvolve.Model("nucleotide")
            else:
                my_model = pyvolve.Model("nucleotide", alpha = 0.4, num_categories = 4)

            for locus in range(num_loci):
                if write_gene_alignments:
                    fasta_outpath=model_outpath + "/full_alignments"
                    if not os.path.exists(fasta_outpath):
                        os.mkdir(fasta_outpath)
                else:
                    fasta_outpath=model_outpath
                fastaout=fasta_outpath +"/"+ base+"_"+model+"_loc"+str(locus) + "_gene-alignment.fasta"
                #sample a gene alignment
                sample_locus(my_tree, my_model, loc_length, snps_per_locus, fastaout)

                #sample SNP(s) from gene alignment
                sampled = sample_snp(read_fasta(fastaout), loc_length, snps_per_locus)
                data = add_locus(data,sampled)

                if not write_gene_alignments:
                    os.remove(fastaout)
                    if make_gene_trees:
                        print("ERROR: Can't make gene trees when write_gene_alignments = False")
                elif make_gene_trees:
                    run_iqtree(fastaout, iqtree_path=iqtree_bin, keep_all=False, keep_report=False)

            #write full SNP alignment & generate tree
            all_snp_out=model_outpath+"/"+base+"_"+model+"_base-snps-concat.fasta"
            write_fasta(data, all_snp_out)
            run_iqtree(all_snp_out, iqtree_path=iqtree_bin, keep_all=False, keep_report=False)

            ######## Varying introgression weight
            num_sampled_loci = len(data[indlabels[0]])
            #Modeled as contemporary exchange from pop2 -> pop1
            model_base=model_outpath+"/"+base+"_"+model

            for alpha in [0.1, 0.0, 0.2, 0.3, 0.4, 0.5]:
                alpha_base=model_base + "_i" + str(alpha)

                #write a logfile recording which loci are introgressed

                #for introgression, choose a SNP index, and then sample new
                #genotypes for recipient population from genotypes in the source
                #population.
                source_pool = [indlabels[i] for i, pop in enumerate(poplabels) if pop == "pop2"]
                target_pool = [indlabels[i] for i, pop in enumerate(poplabels) if pop == "pop1"]

                if alpha == 0.0:
                    introgressed_data = copy.copy(data)
                else:
                    introgressed_data = hybridization(data,
                                            prob=alpha,
                                            source=source_pool,
                                            target=target_pool)

                sys.exit()
                #at this step, infer site-specific rates and substitution models in IQ-TREE
                #using a concatenated SNP alignment

                #Now, our datasets are complete, so simulate different types of missing
                #data on the guidetree(s)

                ######## Varying type of missing data (systematic vs. random)
                #for missing_type in ["systematic", "random"]:


                    ####### Varying proportion of missing data
                    #for missing_prop in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:

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
            new_dat[target_individual][index] = new_dat[np.random.choice(source, size=1)[0]][index]
    return(new_dat)

def run_iqtree(aln, iqtree_path="iqtree", keep_all=False, keep_report=False):
    #run
    result = subprocess.run([iqtree_path,
                    "-s",
                    str(aln),
                    "-m",
                    "GTR+G",
                    "-redo",
                    "-nt",
                    "4"
                    ], capture_output=True, text=True)
    #print(result.stdout)
    #print(result.stderr)

    if not keep_all:
        #delete everything except treefile
        os.remove((aln + ".bionj"))
        os.remove((aln + ".ckp.gz"))
        os.remove((aln + ".log"))
        os.remove((aln + ".mldist"))
        os.remove((aln + ".uniqueseq.phy"))
        if not keep_report:
            os.remove((aln + ".iqtree"))
    return((aln + ".treefile"))

def add_locus(d, new):
    for sample in d.keys():
        for snp in new[sample]:
            d[sample].append(snp)
    return(d)

def write_fasta(seqs, fas):
	with open(fas, 'w') as fh:
		#Write seqs to FASTA first
		for a in seqs.keys():
			name = ">id_" + str(a) + "\n"
			seq = "".join(seqs[a]) + "\n"
			fh.write(name)
			fh.write(seq)
		fh.close()

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
        return(data)

def sample_snp(aln_dict, aln_len, snps_per_locus=1):
    snp_indices = []
    snp_aln = dict()
    if aln_len == 1:
        for sample in aln_dict.keys():
            snp_aln[sample] = aln_dict[sample][0]
        return(snp_aln)
    else:
        for sample in aln_dict.keys():
            snp_aln[sample] = []
    for i in range(aln_len):
        vars=[]
        for sample in aln_dict.keys():
            nuc=aln_dict[sample][i]
            if len(vars) == 0:
                vars.append(nuc)
            elif nuc not in vars:
                snp_indices.append(i)
                break
    if len(snp_indices) == 0:
        return(snp_indices)
    elif len(snp_indices) == 1:
        #sample them all
        for sample in aln_dict.keys():
            snp_aln[sample] = aln_dict[sample][snp_indices[0]]
    else:
        sampled_indices = np.random.choice(snp_indices, size=snps_per_locus, replace=False)
        for sample in aln_dict.keys():
            for i in sampled_indices:
                snp_aln[sample].append(aln_dict[sample][i])
    return(snp_aln)

def sample_locus(tree, model, gene_len=1000, num_snps=1, out="out.fasta"):
        my_partition = pyvolve.Partition(models = model, size=gene_len)
        # Evolve!
        my_evolver = pyvolve.Evolver(partitions = my_partition, tree = tree)
        my_evolver(seqfile = out,
            seqfmt = "fasta",
            ratefile=False,
            infofile=False)
        #read fasta file
        aln=read_fasta(out)

        #delete fasta file
        #os.remove(".temp.fasta")
        #return SNP

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

if __name__ == "__main__":
    main()
