# Standard library imports
import gc
import math
import os
import shutil
import sys
import warnings
from collections import Counter

## For stats and numeric operations
import numpy as np
import numpy.ma as ma
import pandas as pd
from scipy import stats as st
from scipy import sparse as sp

from sklearn.impute import SimpleImputer
from sklearn.utils import is_scalar_nan
from sklearn.utils._mask import _get_mask
from sklearn.utils.validation import _deprecate_positional_args
from sklearn.utils.validation import check_is_fitted

from utils.misc import get_processor_name

# Uses scikit-learn-intellex package if CPU is Intel
if get_processor_name().strip().startswith("Intel"):
	try:
		from sklearnex import patch_sklearn

		patch_sklearn(verbose=False)
	except ImportError:
		print(
			"Processor not compatible with scikit-learn-intelex; using "
			"default configuration"
		)


class SimpleImputerCustom(SimpleImputer):
	"""[SimpleImputer overload to add imputation by groups as an initial strategy]"""

	@_deprecate_positional_args
	def __init__(
		self,
		*,
		missing_values=np.nan,
		strategy="mean",
		fill_value=None,
		verbose=0,
		copy=True,
		add_indicator=False,
		pops=None,
	):

		super(SimpleImputer, self).__init__(
			missing_values=missing_values, add_indicator=add_indicator
		)

		self.strategy = strategy
		self.fill_value = fill_value
		self.verbose = verbose
		self.copy = copy
		self.pops = pops

	def _dense_fit(self, X, strategy, missing_values, fill_value):
		"""Fit the transformer on dense data."""
		missing_mask = _get_mask(X, missing_values)
		masked_X = ma.masked_array(X, mask=missing_mask)

		super(SimpleImputer, self)._fit_indicator(missing_mask)

		# Mean
		if strategy == "mean":
			mean_masked = np.ma.mean(masked_X, axis=0)
			# Avoid the warning "Warning: converting a masked element to nan."
			mean = np.ma.getdata(mean_masked)
			mean[np.ma.getmask(mean_masked)] = np.nan

			return mean

		# Median
		elif strategy == "median":
			median_masked = np.ma.median(masked_X, axis=0)
			# Avoid the warning "Warning: converting a masked element to nan."
			median = np.ma.getdata(median_masked)
			median[np.ma.getmaskarray(median_masked)] = np.nan

			return median

		# Most frequent
		elif strategy == "most_frequent":
			# Avoid use of scipy.stats.mstats.mode due to the required
			# additional overhead and slow benchmarking performance.
			# See Issue 14325 and PR 14399 for full discussion.

			# To be able access the elements by columns
			X = X.transpose()
			mask = missing_mask.transpose()

			if X.dtype.kind == "O":
				most_frequent = np.empty(X.shape[0], dtype=object)
			else:
				most_frequent = np.empty(X.shape[0])

			for i, (row, row_mask) in enumerate(zip(X[:], mask[:])):
				row_mask = np.logical_not(row_mask).astype(bool)
				row = row[row_mask]
				most_frequent[i] = self._most_frequent(row, np.nan, 0)
			
			return most_frequent

		# Constant
		elif strategy == "constant":
			# for constant strategy, self.statistcs_ is used to store
			# fill_value in each column
			return np.full(X.shape[1], fill_value, dtype=X.dtype)

		elif strategy == "groups":
			if self.pops is None:
				raise TypeError("pops argument cannot be NoneType if initial_strategy='groups'")

			# Avoid use of scipy.stats.mstats.mode due to the required
			# additional overhead and slow benchmarking performance.
			# See Issue 14325 and PR 14399 for full discussion.

			# To be able access the elements by columns
			X = X.transpose()
			mask = missing_mask.transpose()
			groups = np.array(self.pops, dtype=object)
			uniq_groups = np.unique(groups)

			if X.dtype.kind == "O":
				most_frequent = np.empty(X.shape[0], dtype=object)
			else:
				most_frequent = np.empty(X.shape[0])

			for i, (row, row_mask) in enumerate(zip(X[:], mask[:])):
				grps = groups.copy()
				row_mask = np.logical_not(row_mask).astype(bool)
				row = row[row_mask]
				grps = grps[row_mask]

				most_frequent[i] = self._most_frequent(row, np.nan, 0, grps)
			sys.exit()


			return most_frequent

	def _most_frequent(self, array, extra_value, n_repeat, grps=None):
		"""[Compute the most frequent value in a 1d array extended with
		``extra_value * n_repeat``, where ``extra_value`` is assumed to be not part of the array]"""

		if grps is not None:
			if array.size > 0:
				if array.dtype == object:
					df = pd.DataFrame({"gt": array, "pops": grps})
					modes = df.groupby("pops").gt.apply(pd.Series.mode).reset_index().drop("level_1", axis=1)

					print(df)
					sys.exit(0)
					
					
		# group_modes = [
		# 	(j, st.mode(row[grps == j])[0]) for j in uniq_groups
		# ]
				

		# else:
		# Compute the most frequent value in array only
		if array.size > 0:
			if array.dtype == object:
				# scipy.stats.mode is slow with object dtype array.
				# Python Counter is more efficient
				counter = Counter(array)
				most_frequent_count = counter.most_common(1)[0][1]

				# tie breaking similarly to scipy.stats.mode
				most_frequent_value = min(
					value
					for value, count in counter.items()
					if count == most_frequent_count
				)
			else:
				mode = stats.mode(array)
				most_frequent_value = mode[0][0]
				most_frequent_count = mode[1][0]
		else:
			most_frequent_value = 0
			most_frequent_count = 0

		# Compare to array + [extra_value] * n_repeat
		if most_frequent_count == 0 and n_repeat == 0:
			return np.nan
		elif most_frequent_count < n_repeat:
			return extra_value
		elif most_frequent_count > n_repeat:
			return most_frequent_value
		elif most_frequent_count == n_repeat:
			# tie breaking similarly to scipy.stats.mode
			return min(most_frequent_value, extra_value)

	def _validate_input(self, X, in_fit):
		allowed_strategies = [
			"mean",
			"median",
			"most_frequent",
			"constant",
			"groups",
		]

		if self.strategy not in allowed_strategies:
			raise ValueError(
				f"Can only use these strategies: "
				f"{allowed_strategies} "
				f" got strategy={self.strategy}"
			)

		if self.strategy in ("most_frequent", "groups", "constant"):
			# If input is a list of strings, dtype = object.
			# Otherwise ValueError is raised in SimpleImputer
			# with strategy='most_frequent' or 'constant'
			# because the list is converted to Unicode numpy array
			if isinstance(X, list) and any(
				isinstance(elem, str) for row in X for elem in row
			):
				dtype = object
			else:
				dtype = None
		else:
			dtype = FLOAT_DTYPES

		if not is_scalar_nan(self.missing_values):
			force_all_finite = True
		else:
			force_all_finite = "allow-nan"

		try:
			X = self._validate_data(
				X,
				reset=in_fit,
				accept_sparse="csc",
				dtype=dtype,
				force_all_finite=force_all_finite,
				copy=self.copy,
			)
		except ValueError as ve:
			if "could not convert" in str(ve):
				new_ve = ValueError(
					f"Cannot use {self.strategy} strategy with non-numeric data:\n{ve}"
				)

				raise new_ve from None
			else:
				raise ve

		return X

	def transform(self, X):
		"""Impute all missing values in X.
		Parameters
		----------
		X : {array-like, sparse matrix}, shape (n_samples, n_features)
			The input data to complete.
		"""
		check_is_fitted(self)

		X = self._validate_input(X, in_fit=False)
		statistics = self.statistics_

		if X.shape[1] != statistics.shape[0]:
			raise ValueError("X has %d features per sample, expected %d"
							% (X.shape[1], self.statistics_.shape[0]))

		# compute mask before eliminating invalid features
		missing_mask = _get_mask(X, self.missing_values)

		# Delete the invalid columns if strategy is not constant
		if self.strategy == "constant":
			valid_statistics = statistics
			valid_statistics_indexes = None
		else:
			# same as np.isnan but also works for object dtypes
			invalid_mask = _get_mask(statistics, np.nan)
			valid_mask = np.logical_not(invalid_mask)
			valid_statistics = statistics[valid_mask]
			valid_statistics_indexes = np.flatnonzero(valid_mask)

			if invalid_mask.any():
				missing = np.arange(X.shape[1])[invalid_mask]
				if self.verbose:
					warnings.warn("Deleting features without "
								"observed values: %s" % missing)
				X = X[:, valid_statistics_indexes]

		# Do actual imputation
		if sp.issparse(X):
			if self.missing_values == 0:
				raise ValueError("Imputation not possible when missing_values "
								"== 0 and input is sparse. Provide a dense "
								"array instead.")
			else:
				# if no invalid statistics are found, use the mask computed
				# before, else recompute mask
				if valid_statistics_indexes is None:
					mask = missing_mask.data
				else:
					mask = _get_mask(X.data, self.missing_values)
				indexes = np.repeat(
					np.arange(len(X.indptr) - 1, dtype=int),
					np.diff(X.indptr))[mask]

				X.data[mask] = valid_statistics[indexes].astype(X.dtype,
																copy=False)
		else:
			# use mask computed before eliminating invalid mask
			if valid_statistics_indexes is None:
				mask_valid_features = missing_mask
			else:
				mask_valid_features = missing_mask[:, valid_statistics_indexes]
			n_missing = np.sum(mask_valid_features, axis=0)
			values = np.repeat(valid_statistics, n_missing)
			coordinates = np.where(mask_valid_features.transpose())[::-1]

			X[coordinates] = values

		X_indicator = super()._transform_indicator(missing_mask)

		return super()._concatenate_indicator(X, X_indicator)
