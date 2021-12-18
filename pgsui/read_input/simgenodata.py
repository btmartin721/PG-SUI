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
from pgsui.read_input import GenotypeData

from pgsui.utils import tree_tools

class SimGenotypeData(GenotypeData):
    """Simulate SNP matrices for use with ImputeXXX methods.

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
            raise TypeError("genotype_data cannot be NoneType"
        elif genotype_data is not None:
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
                raise TypeError("prop_missing cannot be NoneType"
