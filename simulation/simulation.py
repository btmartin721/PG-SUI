import sys

import toytree
import toyplot
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

            guidetree (str): File containing newick-formatted guidetree. Defaults to None.

            n_to_sample (int or Dict[str][int]): Integer or dict (mapping tips in guide tree to integer values) given the number of individuals to sample per population. Defaults to 10.

            bldist_ind (functools.partial or Dict[str: functools.partial]): Function or dict of functions defining a distribution from which terminal/ tip/ leaf branch lengths will be sampled.

            bldist_amongInd

            bldist_amongGroup

    Attributes:
            samples (List[str]): List containing sample IDs of shape (n_samples,).

    # Raises:
    #     TypeError: Check whether the ``gridparams`` values are of the correct format if ``ga=True`` or ``ga=False``.
    #
    #
    # Examples:
    #     # Don't use parentheses after estimator object.
    #     >>> imputer = Impute(
    #             sklearn.ensemble.RandomForestClassifier,
    #             "classifier",
    #             {
    #                 "n_jobs": 4,
    #                 "initial_strategy": "populations",
    #                 "max_iter": 25,
    #                 "n_estimators": 100,
    #                 "ga": True
    #             }
    #         )
    #
    #     >>> self.imputed, self.best_params = imputer.fit_predict(df)
    #
    #     >>> imputer.write_imputed(self.imputed)


    """

    def __init__(
        self,
        genotype_data = None,
        filename: Optional[str] = None,
        filetype: Optional[str] = None,
        popmapfile: Optional[str] = None,
        guidetree: Optional[str] = None,
        qmatrix_iqtree: Optional[str] = None,
        qmatrix: Optional[str] = None,
        siterates: Optional[str] = None,
        siterates_iqtree: Optional[str] = None,
        missing_prop = 0.1,
        missing_type = "random",
    ) -> None:
        self.genotype_data = genotype_data
        self.guidetree = guidetree

        super().__init__()

        # if self.genotype_data is None:
        #     if guidetree is not None:
        #         raise TypeError("genotype_data and poptree cannot both be NoneType")
        #     else:
        #         #simulate data on guide tree
        #         self.guidetree = self.read_tree(guidetree)

        #otherwise, simulate missing data on input GenotypeData object
        elif genotype_data is not None:
            self.filename = genotype_data.filename
            self.filetype = genotype_data.filetype
            self.popmapfile = genotype_data.popmapfile
            self.guidetree = genotype_data.guidetree
            self.qmatrix_iqtree = genotype_data.qmatrix_iqtree
            self.qmatrix = genotype_data.qmatrix
            self.siterates = genotype_data.siterates
            self.siterates_iqtree = genotype_data.siterates_iqtree

            self.snpsdict = genotype_data.snpsdict
            self.samples= genotype_data.samples
            self.snps= genotype_data.snps
            self.pops= genotype_data.pops
            self.onehot= genotype_data.onehot
            self.num_snps = genotype_data.num_snps
            self.num_inds = genotype_data.num_inds
            self.q = genotype_data.q
            self.site_rates = genotype_data.site_rates

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
