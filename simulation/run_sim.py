#!/usr/bin/env python
import sys
import toytree
import random
import re
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

def main():
    seed=1234
    random.seed(seed)

    num_clades=3
    samples_per_clade=20
    clades=[]
    poplabels=[]
    indlabels=[]
    for i in range(num_clades):
        clades.append(("pop"+str(i)))
        poplabels.append(("pop"+str(i)))
        for j in range(samples_per_clade):
            indlabels.append(("pop"+str(i)+"_"+str(j)))

    ####### Varying clade vs. stem heights
    for clade_height in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        #print("clade heights: ", clade_height)
        stem_height = np.around(1.0-clade_height, decimals=1)
        #print(stem_height)

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

        oname="c"+str(clade_height)+"_s"+str(stem_height)
        tobj=toytree.tree(guidetree, tree_format=0)
        tree_tools.basic_tree_plot(tobj, (oname+".pdf"))

        ######## With and without rate heterogeneity
        #NOTE: Run alignments through IQ-TREE to get optional
        #Rate matrix and site-specific mutation rates
        #for model in ["gtr", "gtrgamma"]:


        ######## Varying introgression weight
        #Modeled as contemporary exchange from pop2 -> pop1
        #for lambda in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:


        ######## Varying type of missing data (systematic vs. random)
        #for missing_type in ["systematic", "random"]:



        ####### Varying proportion of missing data
        #for missing_prop in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:


def get_tree_tips(tree):
    tips = re.split('[ ,\(\);]', tree)
    return([i for i in tips if i])

def basic_tree_plot(tree, out="out.pdf"):
    mystyle = {
        "edge_type": "p",
        "edge_style": {
            "stroke": tt.colors[0],
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
