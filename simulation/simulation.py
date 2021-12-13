import sys

import toytree
import pyvolve
import re

from functools import partial

import pandas as pd
import numpy as np
import scipy as sp

from read_input import read_input
from utils import tree_tools

class SimGenotypeData(read_input.GenotypeData):
    """Simulate SNP matrices for use with ImputeXXX methods.

    Simulates SNP data given a provided population-level guide tree

    Args:
            genotype_data (GenotypeDate): Optional GenotypeData object (with genotype_data.guidetree defined) on which to simulate missing data. Defaults to None.

            poptree (str): Newick string representing a population-level guide tree [e.g., "(((A, B), C), D);")] Defaults to None.

            n_to_sample (int or Dict[str][int]): Integer or dict (mapping tips in guide tree to integer values) given the number of individuals to sample per population. Defaults to 10.

            bldist_ind (functools.partial or Dict[str][functools.partial]): Function or dict of functions defining a distribution from which terminal/ tip/ leaf branch lengths will be sampled.

            bldist_amongInd

            bldist_amongGroup


    Attributes:
            samples (List[str]): List containing sample IDs of shape (n_samples,).



    """

    def __init__(
        self,
        poptree = None,
        genotype_data = None,
        n_to_sample = 10,
    ) -> None:
        self.poptree = poptree
        self.genotype_data = genotype_data
        self.n_to_sample = n_to_sample

        super().__init__()

        if genotype_data is None and poptree is None:
            raise TypeError("genotype_data and poptree cannot both be NoneType")

        if genotype_data is not None and poptree is not None:
            raise TypeError("genotype_data and poptree cannot both be used")

        if poptree is not None:
            self.init_poptree()

        elif genotype_data is not None:
            self.filename = genotype_data.filename
            self.filetype = genotype_data.filetype
            self.popmapfile = genotype_data.popmapfile
            self.guidetree = genotype_data.guidetree
            self.qmatrix_iqtree = genotype_data.qmatrix_iqtree
            self.qmatrix = genotype_data.qmatrix

            self.snpsdict = genotype_data.snpsdict
            self.samples= genotype_data.samples
            self.snps= genotype_data.snps
            self.pops= genotype_data.pops
            self.onehot= genotype_data.onehot
            self.num_snps = genotype_data.num_snps
            self.num_inds = genotype_data.num_inds

            if self.guidetree is not None:
                self.tree = self.read_tree(self.guidetree)
            elif self.guidetree is None:
                self.tree = None

            if self.qmatrix_iqtree is not None and self.qmatrix is not None:
                raise TypeError("qmatrix_iqtree and qmatrix cannot both be defined")

            if self.qmatrix_iqtree is not None:
                self.q = self.q_from_iqtree(self.qmatrix_iqtree)
            elif self.qmatrix_iqtree is None and self.qmatrix is not None:
                self.q = self.q_from_file(self.qmatrix)
            elif self.qmatrix is None and self.qmatrix_iqtree is None:
                self.q = None

    def init_poptree(self):
        #get tiplabels from newick string
        tiplabels = tree_tools.get_tree_tips(self.poptree)

        #get individual and pop labels
        for pop in tiplabels:
            clade="("
            if type(self.n_to_sample) is dict:
                if pop not in self.n_to_sample:
                    raise ValueError(
                        f"Population  {pop} was not found in "
                        f"input list ``n_to_sample``\n"
                    )
                else:
                    end=tiplabels[pop]+1
            else:
                end=n_to_sample+1
            for i in range(1, self.n_to_sample+1):
                indlabel=str(pop)+"_"+str(i)
                self.samples.append(indlabel)
                self.pops.append(str(pop))
                self.num_inds+=1
                clade = clade + indlabel
                if i == self.n_to_sample:
                    clade = clade + ")"
                else:
                    clade = clade + ", "

            #NOTE: This assumes that no population label both unique and
            #not contained within another. For example, population labels like
            #[1, 12, 3, 5] will not work, as `1` is contained within `12`,
            #but pop labels like [pop-1, pop-12, ...] will work.
            self.poptree=re.sub(rf"{pop}", clade, self.poptree)
