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

            strategy (str, optional): Strategy for simulating missing data. May be one of: \"systematic\" or \"random\". When set to \"systematic\", internal branches from GenotypeData.guidetree will be used to generate non-random missing data. Defaults to \"random\"

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

            genotypes_reference (pandas.DataFrame): 012-encoded genotypes of shape (n_samples, n_sites) prior to inserting missing data. Will be used to assess accuracy of a given imputed matrix.

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

    def add_missing(self):
        pass

    def accuracy(self, imputed):
        pass
