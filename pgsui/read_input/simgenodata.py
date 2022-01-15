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

from sklearn.base import BaseEstimator, TransformerMixin

try:
    from .read_input import GenotypeData
except (ModuleNotFoundError, ValueError):
    from read_input.read_input import GenotypeData


class SimGenotypeData(GenotypeData):
    """Simulate missing data on genotypes read/ encoded in a GenotypeData object.

    Copies metadata from a GenotypeData object and simulates user-specified proportion of missing data

    Args:
            genotype_data (GenotypeData): GenotypeData object. Assumes no missing data already present. Defaults to None.

            prop_missing (float, optional): Proportion of missing data desired in output. Defaults to 0.10

            strategy (str, optional): Strategy for simulating missing data. May be one of: \"nonrandom\", \"nonrandom_weighted\", or \"random\". When set to \"nonrandom\", branches from GenotypeData.guidetree will be randomly sampled to generate missing data on descendant nodes. For \"nonrandom_weighted\", missing data will be placed on nodes proportionally to their branch lengths (e.g., to generate data distributed as might be the case with mutation-disruption of RAD sites). Defaults to \"random\"

            subset (float): Proportion of sites to randomly subset from the input data. Defaults to 1.0 (i.e., all data retained)

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

            genotypes012_list (List[List[str]]): List of 012-encoded genotypes of shape (n_samples, n_sites), after inserting missing data.

            genotypes012_array (numpy.ndarray): 012-encoded genotypes of shape (n_samples, n_sites), after inserting missing data

            genotypes012_df (pandas.DataFrame): 012-encoded genotypes of shape (n_samples, n_sites), after inserting missing data. Missing values are encoded as -9.

            genotypes_onehot (numpy.ndarray of shape (n_samples, n_SNPs, 4)): One-hot encoded numpy array, after inserting missing data. The inner-most array consists of one-hot encoded values for the four nucleotides in the order of "A", "T", "G", "C". Values of 0.5 indicate heterozygotes, and missing values contain 0.0 for all four nucleotides.

            missing_count (int): Number of genotypes masked by chosen missing data strategy

            prop_missing_real (float): True proportion of missing data generated using chosen strategy

            mask (numpy.ndarray): 2-dimensional array tracking the indices of sampled missing data sites (n_samples, n_sites)
    """

    def __init__(
        self,
        genotype_data=None,
        prop_missing=None,
        strategy="random",
        subset=1.0,
    ) -> None:
        self.prop_missing = prop_missing
        self.strategy = strategy
        self.subset = subset

        # Copy genotype_data attributes into local attributes
        # keep original genotype_data as a reference for calculating
        # accuracy after imputing masked sites
        if genotype_data is None:
            raise TypeError("genotype_data cannot be NoneType")
        else:
            self.genotype_data = copy.deepcopy(genotype_data)
            super().__init__(
                filename=self.genotype_data.filename,
                filetype=self.genotype_data.filetype,
                popmapfile=self.genotype_data.popmapfile,
                guidetree=self.genotype_data.guidetree,
                qmatrix_iqtree=self.genotype_data.qmatrix_iqtree,
                qmatrix=self.genotype_data.qmatrix,
                siterates=self.genotype_data.siterates,
                siterates_iqtree=self.genotype_data.siterates_iqtree,
                verbose=False,
            )

            if self.prop_missing is None:
                raise TypeError("prop_missing cannot be NoneType")

            # add in missing data
            self.add_missing()

    def add_missing(self, tol=None, max_tries=None):
        """Function to generate masked sites in a SimGenotypeData object

        Args:
            tol (float): Tolerance to reach proportion specified in self.prop_missing. Defaults to 1/num_snps*num_inds

            max_tries (int): Maximum number of tries to reach targeted missing data proportion within specified tol. Defaults to num_inds.
        """
        print(
            "\nAdding",
            self.prop_missing,
            "missing data using strategy:",
            self.strategy,
        )

        if self.strategy == "random":
            self.mask = np.random.choice(
                [0, 1],
                size=self.genotypes012_array.shape,
                p=((1 - self.prop_missing), self.prop_missing),
            ).astype(np.bool)

            self.validate_mask()

            # mask 012-encoded (self.snps) and one-hot encoded genotypes (self.onehot)
            self.mask_snps()

        elif (
            self.strategy == "nonrandom"
            or self.strategy == "nonrandom_weighted"
        ):
            if self.tree is None:
                raise TypeError(
                    'SimGenotypeData.tree cannot be NoneType when strategy="systematic"'
                )
            mask = np.full_like(self.genotypes012_array, 0.0, dtype=bool)

            if self.strategy == "nonrandom_weighted":
                weighted = True
            else:
                weighted = False

            sample_map = dict()
            for i, sample in enumerate(self.samples):
                sample_map[sample] = i

            # if no tolerance provided, set to 1 snp position
            if tol is None:
                tol = 1.0 / mask.size

            # if no max_tries provided, set to # inds
            if max_tries is None:
                max_tries = mask.shape[0]

            filled = False
            while not filled:
                # Get list of samples from tree
                samples = self.sample_tree(
                    internal_only=False, skip_root=True, weighted=weighted
                )

                # convert to row indices
                rows = [sample_map[i] for i in samples]

                # randomly sample a column
                col_idx = np.random.randint(0, mask.shape[1])
                sampled_col = copy.copy(mask[:, col_idx])

                # mask column
                sampled_col[rows] = True

                # check that column is not 100% missing now
                # if yes, sample again
                if np.sum(sampled_col) == sampled_col.size:
                    continue

                # if not, set values in mask matrix
                else:
                    mask[:, col_idx] = sampled_col
                    # if this addition pushes missing % > self.prop_missing,
                    # check previous prop_missing, remove masked samples from this
                    # column until closest to target prop_missing
                    current_prop = np.sum(mask) / mask.size
                    if abs(current_prop - self.prop_missing) <= tol:
                        filled = True
                        break
                    elif current_prop > self.prop_missing:
                        tries = 0
                        while (
                            abs(current_prop - self.prop_missing) > tol
                            and tries < max_tries
                        ):
                            r = np.random.randint(0, mask.shape[0])
                            c = np.random.randint(0, mask.shape[1])
                            mask[r, c] = False
                            tries = tries + 1
                            current_prop = np.sum(mask) / mask.size
                            # print("After removal:",(np.sum(mask)/mask.size))
                        filled = True
                    else:
                        continue
            # finish
            self.mask = mask
            self.validate_mask()
            self.mask_snps()
        else:
            raise ValueError(
                "Invalid SimGenotypeData.strategy value:", self.strategy
            )

    def validate_mask(self):
        """
        Internal function to make sure no entirely missing columns are simulated.
        """
        i = 0
        for column in self.mask.T:
            if np.sum(column) == column.size:
                self.mask[np.random.randint(0, mask.shape[0]), i] = False
            i = i + 1

    def accuracy(self, imputed):
        masked_sites = np.sum(self.mask)
        num_correct = np.sum(
            self.genotype_data.genotypes012_array[self.mask]
            == imputed.imputed.genotypes012_array[self.mask]
        )
        return num_correct / masked_sites

    def sample_tree(
        self,
        internal_only=False,
        tips_only=False,
        skip_root=True,
        weighted=False,
    ):
        """Function for randomly sampling clades from SimGenotypeData.tree

        Args:
            internal_only (bool): Only sample from NON-TIPS. Defaults to False.

            tips_only (bool): Only sample from tips. Defaults to False.

            skip_root (bool): Exclude sampling of root node. Defaults to True.

            weighted (bool): Weight sampling by branch length. Defaults to False.

        Returns:
            List[str]: List of descendant tips from the sampled node.
        """

        if tips_only and internal_only:
            raise ValueError("internal_only and tips_only cannot both be true")

        # to only sample internal nodes add  if not i.is_leaf()
        node_dict = dict()

        for i in self.tree.treenode.traverse("preorder"):
            if skip_root:
                if i.idx == self.tree.nnodes - 1:
                    continue
            if tips_only:
                if not i.is_leaf():
                    continue
            elif internal_only:
                if i.is_leaf():
                    continue
            node_dict[i.idx] = i.dist
        if weighted:
            s = sum(list(node_dict.values()))
            p = [i / s for i in list(node_dict.values())]
            node_idx = np.random.choice(list(node_dict.keys()), size=1, p=p)[0]
        else:
            node_idx = np.random.choice(list(node_dict.keys()), size=1)[0]
        return self.tree.get_tip_labels(idx=node_idx)

    def mask_snps(self):
        """Mask positions in SimGenotypeData.snps and SimGenotypeData.onehot"""
        i = 0
        for row in self.mask:
            for j in row.nonzero()[0]:
                self.snps[i][j] = -9
                self.onehot[i][j] = [0.0, 0.0, 0.0, 0.0]
            i = i + 1

    @property
    def missing_count(self) -> int:
        """Count of masked genotypes in SimGenotypeData.mask

        Returns:
            int: Integer count of masked alleles.
        """
        return np.sum(np.mask)

    @property
    def prop_missing_real(self) -> float:
        """Proportion of genotypes masked in SimGenotypeData.mask

        Returns:
            float: Total number of masked alleles divided by SNP matrix size.
        """
        return np.sum(np.mask) / mask.size


class SimGenotypeDataTransformer(BaseEstimator, TransformerMixin):
    """Simulate missing data on genotypes read/ encoded in a GenotypeData object.

    Copies metadata from a GenotypeData object and simulates user-specified proportion of missing data

    Args:
            genotype_data (GenotypeData): GenotypeData object. Assumes no missing data already present. Defaults to None.

            prop_missing (float, optional): Proportion of missing data desired in output. Defaults to 0.10

            strategy (str, optional): Strategy for simulating missing data. May be one of: \"nonrandom\", \"nonrandom_weighted\", or \"random\". When set to \"nonrandom\", branches from GenotypeData.guidetree will be randomly sampled to generate missing data on descendant nodes. For \"nonrandom_weighted\", missing data will be placed on nodes proportionally to their branch lengths (e.g., to generate data distributed as might be the case with mutation-disruption of RAD sites). Defaults to \"random\"

            subset (float): Proportion of sites to randomly subset from the input data. Defaults to 1.0 (i.e., all data retained)

            verbose (bool, optional): Verbosity level. Defaults to 0.

            tol (float): Tolerance to reach proportion specified in self.prop_missing. Defaults to 1/num_snps*num_inds

            max_tries (int): Maximum number of tries to reach targeted missing data proportion within specified tol. Defaults to num_inds.

    Attributes:
        mask_ (numpy.ndarray): Array with boolean mask for simulated missing locations.

        genotype_data_ (GenotypeData): Initialized GenotypeData object. The data should have already been imputed with one of the non-machine learning simple imputers.

    Properties:
        missing_count (int): Number of genotypes masked by chosen missing data strategy

        prop_missing_real (float): True proportion of missing data generated using chosen strategy

        mask (numpy.ndarray): 2-dimensional array tracking the indices of sampled missing data sites (n_samples, n_sites)
    """

    def __init__(
        self,
        prop_missing=None,
        strategy="random",
        subset=1.0,
        verbose=0,
        tol=None,
        max_tries=None,
    ) -> None:
        self.prop_missing = prop_missing
        self.strategy = strategy
        self.subset = subset
        self.verbose = verbose
        self.tol = tol
        self.max_tries = max_tries

    def fit(self, X):
        """Fit to input data X.

        Args:
            X (GenotypeData): Initialized GenotypeData object. No missing data should be present. It should have already been imputed with one of the non-machine learning simple imputers.
        """
        self._validate_input(X)
        self.genotype_data_ = X

        # Copy genotype_data attributes into local attributes
        # keep original genotype_data as a reference for calculating
        # accuracy after imputing masked sites
        if self.genotype_data_ is None:
            raise TypeError("genotype_data cannot be NoneType")

        if self.prop_missing is None:
            raise TypeError("prop_missing cannot be NoneType")

        if self.verbose > 0:
            print(
                "\nAdding",
                self.prop_missing,
                "missing data using strategy:",
                self.strategy,
            )

        if self.strategy == "random":
            # Generate mask of 0's and 1's.
            self.mask_ = np.random.choice(
                [0, 1],
                size=X.genotypes012_array.shape,
                p=((1 - self.prop_missing), self.prop_missing),
            ).astype(np.bool)

            # Make sure no entirely missing columns were simulated.
            self._validate_mask()

        elif (
            self.strategy == "nonrandom"
            or self.strategy == "nonrandom_weighted"
        ):
            if self.tree is None:
                raise TypeError(
                    'SimGenotypeData.tree cannot be NoneType when strategy="systematic"'
                )
            mask = np.full_like(X.genotypes012_array, 0.0, dtype=bool)

            if self.strategy == "nonrandom_weighted":
                weighted = True
            else:
                weighted = False

            sample_map = dict()
            for i, sample in enumerate(X.samples):
                sample_map[sample] = i

            # if no tolerance provided, set to 1 snp position
            if self.tol is None:
                self.tol = 1.0 / mask.size

            # if no max_tries provided, set to # inds
            if self.max_tries is None:
                self.max_tries = mask.shape[0]

            filled = False
            while not filled:
                # Get list of samples from tree
                samples = self._sample_tree(
                    internal_only=False, skip_root=True, weighted=weighted
                )

                # convert to row indices
                rows = [sample_map[i] for i in samples]

                # randomly sample a column
                col_idx = np.random.randint(0, mask.shape[1])
                sampled_col = copy.copy(mask[:, col_idx])

                # mask column
                sampled_col[rows] = True

                # check that column is not 100% missing now
                # if yes, sample again
                if np.sum(sampled_col) == sampled_col.size:
                    continue

                # if not, set values in mask matrix
                else:
                    mask[:, col_idx] = sampled_col
                    # if this addition pushes missing % > self.prop_missing,
                    # check previous prop_missing, remove masked samples from this
                    # column until closest to target prop_missing
                    current_prop = np.sum(mask) / mask.size
                    if abs(current_prop - self.prop_missing) <= self.tol:
                        filled = True
                        break
                    elif current_prop > self.prop_missing:
                        tries = 0
                        while (
                            abs(current_prop - self.prop_missing) > self.tol
                            and tries < self.max_tries
                        ):
                            r = np.random.randint(0, mask.shape[0])
                            c = np.random.randint(0, mask.shape[1])
                            mask[r, c] = False
                            tries = tries + 1
                            current_prop = np.sum(mask) / mask.size
                            # print("After removal:",(np.sum(mask)/mask.size))
                        filled = True
                    else:
                        continue
            # finish
            self.mask_ = mask
            self._validate_mask()

        else:
            raise ValueError(
                "Invalid SimGenotypeData.strategy value:", self.strategy
            )

        return self

    def transform(self, X):
        """Function to generate masked sites in a SimGenotypeData object

        Args:
            X (GenotypeData): Initialized GenotypeData object. No missing data should be present. It should have already been imputed with one of the non-machine learning simple imputers.

        Returns:
            numpy.ndarray: Transformed data with missing data added.
        """
        self._validate_input(X)

        # mask 012-encoded and one-hot encoded genotypes.
        return self._mask_snps(X.genotypes012_array)

    def accuracy(self, X_true, X_pred):
        masked_sites = np.sum(self.mask_)
        num_correct = np.sum(X_true[self.mask_] == X_pred[self.mask_])
        return num_correct / masked_sites

    def _sample_tree(
        self,
        internal_only=False,
        tips_only=False,
        skip_root=True,
        weighted=False,
    ):
        """Function for randomly sampling clades from SimGenotypeData.tree

        Args:
            internal_only (bool): Only sample from NON-TIPS. Defaults to False.

            tips_only (bool): Only sample from tips. Defaults to False.

            skip_root (bool): Exclude sampling of root node. Defaults to True.

            weighted (bool): Weight sampling by branch length. Defaults to False.

        Returns:
            List[str]: List of descendant tips from the sampled node.
        """

        if tips_only and internal_only:
            raise ValueError("internal_only and tips_only cannot both be true")

        # to only sample internal nodes add  if not i.is_leaf()
        node_dict = dict()

        for i in self.genotype_data_.tree.treenode.traverse("preorder"):
            if skip_root:
                if i.idx == self.genotype_data_.tree.nnodes - 1:
                    continue
            if tips_only:
                if not i.is_leaf():
                    continue
            elif internal_only:
                if i.is_leaf():
                    continue
            node_dict[i.idx] = i.dist
        if weighted:
            s = sum(list(node_dict.values()))
            p = [i / s for i in list(node_dict.values())]
            node_idx = np.random.choice(list(node_dict.keys()), size=1, p=p)[0]
        else:
            node_idx = np.random.choice(list(node_dict.keys()), size=1)[0]
        return self.genotype_data_.tree.get_tip_labels(idx=node_idx)

    def _validate_input(self, X):
        array_sum = np.sum(X.genotypes012_array)
        if np.isnan(array_sum):
            raise ValueError(
                "Found missing values in input. Use a simple imputer first."
            )

    def _validate_mask(self):
        """Make sure no entirely missing columns are simulated."""
        for i, column in enumerate(self.mask_.T):
            if np.sum(column) == column.size:
                self.mask_[np.random.randint(0, mask.shape[0]), i] = False

    def _mask_snps(self, X):
        """Mask positions in SimGenotypeData.snps and SimGenotypeData.onehot"""
        if len(X.shape) == 3:
            # One-hot encoded.
            mask_val = [0.0, 0.0, 0.0, 0.0]
        elif len(X.shape) == 2:
            # 012-encoded.
            mask_val = -9
        else:
            raise ValueError(f"Invalid shape of input X: {X.shape}")

        Xt = X.copy()
        for i, row in enumerate(self.mask_):
            for j in row.nonzero()[0]:
                Xt[i][j] = mask_val
        return Xt

    @property
    def missing_count(self) -> int:
        """Count of masked genotypes in SimGenotypeData.mask

        Returns:
            int: Integer count of masked alleles.
        """
        return np.sum(self.mask_)

    @property
    def prop_missing_real(self) -> float:
        """Proportion of genotypes masked in SimGenotypeData.mask

        Returns:
            float: Total number of masked alleles divided by SNP matrix size.
        """
        return np.sum(self.mask_) / self.mask_.size
