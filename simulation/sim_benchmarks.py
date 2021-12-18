#!/usr/bin/env python
import sys
import os
import subprocess
import errno

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

    Generates random data matrices w/ varying dimensions
    for the purposes of runtime benchmarking

    """
    seed=553423
    random.seed(seed)

    num_samples_range=[10, 100, 1000]
    num_loci_range=[100, 1000, 10000]
    loc_length=100
    write_gene_alignments=False
    make_gene_trees=False
    make_guidetrees=True #set to true to run IQTREE on simulated SNP matrices
    keep_all=False #set to true to keep ALL iqtree outputs
    keep_report=True #set to true to keep .iqtree files
    get_siterates=True #set to true to infer site-specific rates in IQTREE
    snps_per_locus=1
    iqtree_bin="iqtree2"
    get_rates=True
    iq_procs=2

    ###################

    if get_siterates and not make_guidetrees:
        print("ERROR: can't set get_siterates=True and make_guidetrees=False")
        print("Setting make_guidetrees=True and proceeding...")
        make_guidetrees=True

    for num_samples in num_samples_range:
        outgroup="r0"
        tobj = toytree.rtree.unittree(ntips=num_samples,
                                    treeheight=1.0,
                                    random_names=False,
                                    seed=random.randint(1, (sys.maxsize * 2 + 1)))
        guidetree=tobj.write(tree_format=5)
        for num_loci in num_loci_range:
            print("num_samples:",num_samples)
            print("num_loci:",num_loci)
            pass

            base="benchmark_"+"i"+str(num_samples)+"_l"+str(num_loci)
            tobj.write((base+"_guidetree.tre"))

            data=dict()
            for ind in tobj.get_tip_labels():
                data[ind] = list()

            my_tree = pyvolve.read_tree(tree=guidetree)


            #for model in ["gtr","gtrgamma"]:
            for model in ["gtrgamma"]:
                model_outpath=base+"_"+model

                for locus in range(num_loci):
                    print(locus)
                    f = np.random.random(4)
                    f /= f.sum()
                    parameters = {
                    "mu":
                        {"AC": np.random.uniform(low=0.0, high=1.0),
                        "AG": np.random.uniform(low=0.0, high=1.0),
                        "AT": np.random.uniform(low=0.0, high=1.0),
                        "CG": np.random.uniform(low=0.0, high=1.0),
                        "CT": np.random.uniform(low=0.0, high=1.0),
                        "GT": np.random.uniform(low=0.0, high=1.0)},
                    "state_freqs":
                        [f[0], f[1], f[2], f[3]]
                    }
                    if model == "gtr":
                        #GTR model, without rate heterogeneity
                        my_model = pyvolve.Model("nucleotide",
                            parameters)
                    else:
                        my_model = pyvolve.Model("nucleotide",
                            parameters,
                            rate_factors = [
                                        np.random.uniform(low=0.1, high=0.7, size=1),
                                        np.random.uniform(low=0.5, high=1.2, size=1),
                                        np.random.uniform(low=1.0, high=1.8, size=1),
                                        np.random.uniform(low=1.5, high=5.0, size=1)
                                        ],
                            rate_probs = [0.4, 0.3, 0.2, 0.1] )
                    if write_gene_alignments:
                        fasta_outpath="full_alignments/"
                        if not os.path.exists(fasta_outpath):
                            os.mkdir(fasta_outpath)
                    else:
                        fasta_outpath=model_outpath
                    fastaout=fasta_outpath +model_outpath+"_loc"+str(locus) + "_gene-alignment.fasta"
                    #sample a gene alignment
                    loc = sample_locus(my_tree, my_model, loc_length, snps_per_locus, fastaout)

                    if loc:
                        #sample SNP(s) from gene alignment
                        sampled = sample_snp(read_fasta(fastaout), loc_length, snps_per_locus)
                        if sampled is not None:
                            data = add_locus(data,sampled)

                        if not write_gene_alignments:
                            os.remove(fastaout)
                            if make_gene_trees:
                                print("ERROR: Can't make gene trees when write_gene_alignments = False")
                        elif make_gene_trees:
                            run_iqtree(fastaout,
                                iqtree_path=iqtree_bin,
                                keep_all=keep_all,
                                keep_report=keep_report,
                                rates=get_siterates,
                                procs=iq_procs)
                            reroot_tree(tree=(fastaout+".treefile"),
                                rooted=(fastaout+".rooted.tre"),
                                outgroup_wildcard=outgroup)

                #write full SNP alignment & generate tree
                all_snp_out=model_outpath+"_base-snps-concat.fasta"
                write_fasta(data, all_snp_out)
                if make_guidetrees:
                    run_iqtree(all_snp_out,
                        iqtree_path=iqtree_bin,
                        keep_all=keep_all,
                        keep_report=keep_report,
                        rates=get_siterates,
                        procs=iq_procs)
                    reroot_tree(tree=(all_snp_out+".treefile"),
                        rooted=(all_snp_out+".rooted.tre"),
                        outgroup_wildcard=outgroup)

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

def run_iqtree(aln,
    iqtree_path="iqtree",
    keep_all=False,
    keep_report=False,
    outgroup=None,
    rates=False,
    procs=4):
    #run
    cmd = [iqtree_path,
                    "-s",
                    str(aln),
                    "-m",
                    "GTR+I*G4",
                    "-redo",
                    "-T",
                    str(procs)
                    ]
    if outgroup is not None:
        cmd.append("-o")
        cmd.append(str(outgroup))
    if rates:
        #use -wst (NOT -mlrate) if using iq-tree 1.6xx
        #cmd.append("-wsr")
        cmd.append("--mlrate")
    result = subprocess.run(cmd, capture_output=True, text=True)
    #print(result.stdout)
    #print(result.stderr)

    if not keep_all:
        #delete everything except treefile
        silentremove((aln + ".bionj"))
        silentremove((aln + ".ckp.gz"))
        silentremove((aln + ".log"))
        silentremove((aln + ".mldist"))
        silentremove((aln + ".uniqueseq.phy"))
        if not keep_report:
            silentremove((aln + ".iqtree"))
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
			name = ">" + str(a) + "\n"
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
        return(None)
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
    try:
        my_partition = pyvolve.Partition(models = model, size=gene_len)
        my_evolver = pyvolve.Evolver(partitions = my_partition, tree = tree)
        my_evolver(seqfile = out,
            seqfmt = "fasta",
            ratefile=False,
            infofile=False)
        return(True)
    except Exception:
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

if __name__ == "__main__":
    main()
