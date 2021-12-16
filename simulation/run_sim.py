#!/usr/bin/env python
import sys
import os
import subprocess

import toytree
import toyplot.pdf
import pyvolve
import random
import re
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

def main():
    seed=1234
    random.seed(seed)

    num_clades=3
    samples_per_clade=20
    num_loci=100
    loc_length=100
    write_gene_alignments=False
    make_gene_trees=False
    snps_per_locus=1
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
                for snp in range(snps_per_locus):
                    sampled = sample_snp(read_fasta(fastaout), loc_length)
                    data = add_locus(data,sampled)

                if not write_gene_alignments:
                    os.remove(fastaout)

                if make_gene_trees:
                    pass

            #write full SNP alignment
            all_snp_out=model_outpath+"/"+base+"_"+model+"_base-snps-concat.fasta"
            #print(data)
            write_fasta(data, all_snp_out)
            sys.exit()

            ######## Varying introgression weight
            #Modeled as contemporary exchange from pop2 -> pop1
            #for lambda in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:

                #write a logfile recording which loci are introgressed
                #for introgression, choose a SNP index, and then sample new
                #genotypes for recipient population from genotypes in the source
                #population.

                #at this step, infer site-specific rates and substitution models in IQ-TREE
                #using a concatenated SNP alignment


                #Now, our datasets are complete, so simulate different types of missing
                #data on the guidetree(s)

                ######## Varying type of missing data (systematic vs. random)
                #for missing_type in ["systematic", "random"]:


                    ####### Varying proportion of missing data
                    #for missing_prop in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:

def add_locus(d, new):
    for sample in d.keys():
        d[sample].append(new[sample])
    return(d)

def write_fasta(seqs, fas):
	with open(fas, 'w') as fh:
		try:
			#Write seqs to FASTA first
			for a in seqs.keys():
				name = ">id_" + str(a) + "\n"
				seq = "".join(seqs[a]) + "\n"
				fh.write(name)
				fh.write(seq)
		except IOError as e:
			print("Could not read file:",e)
			sys.exit(1)
		except Exception as e:
			print("Unexpected error:",e)
			sys.exit(1)
		finally:
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

def sample_snp(aln_dict, aln_len):
    snp_indices = []
    snp_aln = dict()
    if aln_len == 1:
        for sample in aln_dict.keys():
            snp_aln[sample] = aln_dict[sample][0]
        return(snp_aln)
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
        sampled_indices = np.random.choice(snp_indices, size=1, replace=False)
        for sample in aln_dict.keys():
            snp_aln[sample] = aln_dict[sample][sampled_indices[0]]
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
