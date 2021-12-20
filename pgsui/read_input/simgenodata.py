import sys

import toytree
import toyplot
import pyvolve
import re
import copy

from functools import partial

import pandas as pd
import numpy as np
import scipy as sp
from pgsui.read_input.read_input import GenotypeData

class SimGenotypeData(GenotypeData):
    """Simulate missing data on genotypes read/ encoded in a GenotypeData object.

    Copies metadata from a GenotypeData object and simulates user-specified proportion of missing data

    Args:
            genotype_data (GenotypeData): GenotypeData object. Assumes no missing data already present. Defaults to None.

            prop_missing (float, optional): Proportion of missing data desired in output. Defaults to 0.10

            strategy (str, optional): Strategy for simulating missing data. May be one of: \"nonrandom\", \"nonrandom_weighted\", or \"random\". When set to \"nonrandom\", branches from GenotypeData.guidetree will be randomly sampled to generate missing data on descendant nodes. For \"nonrandom_weighted\", missing data will be placed on nodes proportionally to their branch lengths (e.g., to generate data distributed as might be the case with mutation-disruption of RAD sites). Defaults to \"random\"

            verbose (bool, optional): Verbosity level. Defaults to True.

    Attributes:
            samples (List[str]): List containing sample IDs of shape (n_samples,).

            snps (List[List[str]]): 2D list of shape (n_samples, n_sites) containing genotypes.

            pops (List[str]): List of population IDs of shape (n_samples,).

            onehot (List[List[List[float]]]): One-hot encoded genotypes as a 3D list of shape (n_samples, n_sites, 4). The inner-most list represents the four nucleotide bases in the order of "A", "T", "G", "C". If position 0 contains a 1.0, then the site is an "A". If position 1 contains a 1.0, then the site is a "T"...etc. Two values of 0.5 indicates a heterozygote. Missing data is encoded as four values of 0.0.

            guidetree (toytree object): Input guide tree as a toytree object.

            num_snps (int): Number of SNPs (features) present in the dataset.

            num_inds: (int): Number of individuals (samples) present in the dataset.

    Properties:
            snpcount (int): Number of SNPs (features) in the dataset.

            indcount (int): Number of individuals (samples) in the dataset.

            populations (List[str]): List of population IDs of shape (n_samples,).

            individuals (List[str]): List of sample IDs of shape (n_samples,).

            genotypes_list (List[List[str]]): List of 012-encoded genotypes of shape (n_samples, n_sites), after inserting missing data.

            genotypes_nparray (numpy.ndarray): 012-encoded genotypes of shape (n_samples, n_sites), after inserting missing data

            genotypes_df (pandas.DataFrame): 012-encoded genotypes of shape (n_samples, n_sites), after inserting missing data. Missing values are encoded as -9.

            genotypes_onehot (numpy.ndarray of shape (n_samples, n_SNPs, 4)): One-hot encoded numpy array, after inserting missing data. The inner-most array consists of one-hot encoded values for the four nucleotides in the order of "A", "T", "G", "C". Values of 0.5 indicate heterozygotes, and missing values contain 0.0 for all four nucleotides.

            mask (numpy.ndarray): 2-dimensional array tracking the indices of sampled missing data sites (n_samples, n_sites)

"""
    def __init__(
        self,
        genotype_data = None,
        prop_missing = None,
        strategy = "random"
    ) -> None:
        self.genotype_data = genotype_data
        self.prop_missing = prop_missing
        self.strategy = strategy

        super().__init__()

        #Copy genotype_data attributes into local attributes
        #keep original genotype_data as a reference for calculating
        #accuracy after imputing masked sites
        if self.genotype_data is None:
            raise TypeError("genotype_data cannot be NoneType")
        else:
            self.filename = genotype_data.filename
            self.filetype = genotype_data.filetype
            self.popmapfile = genotype_data.popmapfile
            self.guidetree = copy.deepcopy(genotype_data.guidetree)
            self.qmatrix_iqtree = genotype_data.qmatrix_iqtree
            self.qmatrix = genotype_data.qmatrix
            self.siterates = genotype_data.siterates
            self.siterates_iqtree = genotype_data.siterates_iqtree

            self.snpsdict = copy.deepcopy(genotype_data.snpsdict)
            self.samples= copy.deepcopy(genotype_data.samples)
            self.snps= copy.deepcopy(genotype_data.snps)
            self.pops= copy.deepcopy(genotype_data.pops)
            self.onehot= copy.deepcopy(genotype_data.onehot)
            self.num_snps = copy.deepcopy(genotype_data.num_snps)
            self.num_inds = copy.deepcopy(genotype_data.num_inds)
            self.q = copy.deepcopy(genotype_data.q)
            self.site_rates = copy.deepcopy(genotype_data.site_rates)

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

            if self.prop_missing is None:
                raise TypeError("prop_missing cannot be NoneType")

            #add in missing data
            self.add_missing()

    def add_missing(self):
        print("\nAdding",self.prop_missing,"missing data using strategy:",self.strategy)

        if self.strategy == "random":
            self.mask = np.random.choice([0, 1],
                size=self.genotypes_nparray.shape,
                p=((1 - self.prop_missing), self.prop_missing)).astype(np.bool)

            #mask 012-encoded (self.snps) and one-hot encoded genotypes (self.onehot)
            self.mask_snps()

        elif self.strategy == "nonrandom" or self.strategy== "nonrandom_weighted":
            if self.tree is None:
                raise TypeError("SimGenotypeData.tree cannot be NoneType when strategy=\"systematic\"")
            mask = np.full_like(self.genotypes_nparray, 0.0, dtype=bool)

            while(True):
                samples = self.sample_tree()
                sys.exit()
        else:
            raise ValueError("Invalid SimGenotypeData.strategy value:",self.strategy)

    def accuracy(self, imputed):
        pass

    def sample_tree(self,
        internal_only=False,
        tips_only=False,
        skip_root=True,
        weighted=False):

        if tips_only and internal_only:
            raise ValueError("internal_only and tips_only cannot both be true")

        #to only sample internal nodes add  if not i.is_leaf()
        node_dict = dict()

        for i in self.tree.treenode.traverse("preorder"):
            if skip_root:
                if i.idx == self.tree.nnodes-1:
                    continue
            if tips_only:
                if not i.is_leaf():
                    continue
            elif internal_only:
                if i.is_leaf():
                    continue
            node_dict[i.idx] = i.dist
        print(node_dict)
        sys.exit()
        node_idx = np.random.choice(nodes, size=1)[0]
        print(self.tree.get_tip_labels(idx=node_idx))

    def mask_snps(self):
        i=0
        for row in self.mask:
            for j in row.nonzero()[0]:
                self.snps[i][j] = -9
                self.onehot[i][j] = [0.0,0.0,0.0,0.0]
            i=i+1
