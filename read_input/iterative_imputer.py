from time import time
from collections import namedtuple
import warnings
import sys
import os
import shutil


from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings
from sklearn.preprocessing import normalize

from sklearn.utils import (
	check_array, check_random_state, _safe_indexing, is_scalar_nan
)

from sklearn.utils.validation import FLOAT_DTYPES, check_is_fitted
from sklearn.utils._mask import _get_mask

from sklearn.impute import SimpleImputer
from sklearn.impute._base import _BaseImputer
from sklearn.impute._base import _check_inputs_dtype

from sklearn.metrics import accuracy_score, mean_squared_error, make_scorer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold

from sklearn_genetic import GASearchCV
from sklearn_genetic.plots import plot_fitness_evolution
from sklearn_genetic.utils import logbook_to_pandas
from sklearn_genetic.space import Continuous, Categorical, Integer
from sklearn_genetic.callbacks import ConsecutiveStopping, DeltaThreshold, ThresholdStopping

from utils.misc import isnotebook

is_notebook = isnotebook()

if is_notebook:
	from tqdm.notebook import tqdm as progressbar
else:
	if sys.platform == "linux" or sys.platform == "linux2":
		from tqdm.auto import tqdm as progressbar
	else:
		from tqdm import tqdm as progressbar

_ImputerTriplet = namedtuple('_ImputerTriplet', ['feat_idx',
												'neighbor_feat_idx',
												'estimator'])

class IterativeImputer(_BaseImputer):
	"""Multivariate imputer that estimates each feature from all the others.
	A strategy for imputing missing values by modeling each feature with
	missing values as a function of other features in a round-robin fashion.
	Read more in the :ref:`User Guide <iterative_imputer>`.
	.. versionadded:: 0.21
	.. note::
		This estimator is still **experimental** for now: the predictions
		and the API might change without any deprecation cycle. To use it,
		you need to explicitly import ``enable_iterative_imputer``::
		>>> # explicitly require this experimental feature
		>>> from sklearn.experimental import enable_iterative_imputer  # noqa
		>>> # now you can import normally from sklearn.impute
		>>> from sklearn.impute import IterativeImputer
	.. note::
		IterativeImputer has been modified from the original source code by Bradley T. Martin.

	Parameters
	----------
	estimator : estimator object, default=BayesianRidge()
		The estimator to use at each step of the round-robin imputation.
		If ``sample_posterior`` is True, the estimator must support
		``return_std`` in its ``predict`` method.
	missing_values : int, np.nan, default=np.nan
		The placeholder for the missing values. All occurrences of
		`missing_values` will be imputed. For pandas' dataframes with
		nullable integer dtypes with missing values, `missing_values`
		should be set to `np.nan`, since `pd.NA` will be converted to `np.nan`.
	sample_posterior : boolean, default=False
		Whether to sample from the (Gaussian) predictive posterior of the
		fitted estimator for each imputation. Estimator must support
		``return_std`` in its ``predict`` method if set to ``True``. Set to
		``True`` if using ``IterativeImputer`` for multiple imputations.
	max_iter : int, default=10
		Maximum number of imputation rounds to perform before returning the
		imputations computed during the final round. A round is a single
		imputation of each feature with missing values. The stopping criterion
		is met once `max(abs(X_t - X_{t-1}))/max(abs(X[known_vals])) < tol`,
		where `X_t` is `X` at iteration `t`. Note that early stopping is only
		applied if ``sample_posterior=False``.
	tol : float, default=1e-3
		Tolerance of the stopping condition.
	n_nearest_features : int, default=None
		Number of other features to use to estimate the missing values of
		each feature column. Nearness between features is measured using
		the absolute correlation coefficient between each feature pair (after
		initial imputation). To ensure coverage of features throughout the
		imputation process, the neighbor features are not necessarily nearest,
		but are drawn with probability proportional to correlation for each
		imputed target feature. Can provide significant speed-up when the
		number of features is huge. If ``None``, all features will be used.
	initial_strategy : str, default='mean'
		Which strategy to use to initialize the missing values. Same as the
		``strategy`` parameter in :class:`~sklearn.impute.SimpleImputer`
		Valid values: {"mean", "median", "most_frequent", or "constant"}.
	imputation_order : str, default='ascending'
		The order in which the features will be imputed. Possible values:
		"ascending"
			From features with fewest missing values to most.
		"descending"
			From features with most missing values to fewest.
		"roman"
			Left to right.
		"arabic"
			Right to left.
		"random"
			A random order for each round.
	skip_complete : boolean, default=False
		If ``True`` then features with missing values during ``transform``
		which did not have any missing values during ``fit`` will be imputed
		with the initial imputation method only. Set to ``True`` if you have
		many features with no missing values at both ``fit`` and ``transform``
		time to save compute.
	min_value : float or array-like of shape (n_features,), default=-np.inf
		Minimum possible imputed value. Broadcast to shape (n_features,) if
		scalar. If array-like, expects shape (n_features,), one min value for
		each feature. The default is `-np.inf`.
		.. versionchanged:: 0.23
			Added support for array-like.
	max_value : float or array-like of shape (n_features,), default=np.inf
		Maximum possible imputed value. Broadcast to shape (n_features,) if
		scalar. If array-like, expects shape (n_features,), one max value for
		each feature. The default is `np.inf`.
		.. versionchanged:: 0.23
			Added support for array-like.
	verbose : int, default=0
		Verbosity flag, controls the debug messages that are issued
		as functions are evaluated. The higher, the more verbose. Can be 0, 1,
		or 2.
	random_state : int, RandomState instance or None, default=None
		The seed of the pseudo random number generator to use. Randomizes
		selection of estimator features if n_nearest_features is not None, the
		``imputation_order`` if ``random``, and the sampling from posterior if
		``sample_posterior`` is True. Use an integer for determinism.
		See :term:`the Glossary <random_state>`.
	add_indicator : boolean, default=False
		If True, a :class:`MissingIndicator` transform will stack onto output
		of the imputer's transform. This allows a predictive estimator
		to account for missingness despite imputation. If a feature has no
		missing values at fit/train time, the feature won't appear on
		the missing indicator even if there are missing values at
		transform/test time.
	Attributes
	----------
	initial_imputer_ : object of type :class:`~sklearn.impute.SimpleImputer`
		Imputer used to initialize the missing values.
	imputation_sequence_ : list of tuples
		Each tuple has ``(feat_idx, neighbor_feat_idx, estimator)``, where
		``feat_idx`` is the current feature to be imputed,
		``neighbor_feat_idx`` is the array of other features used to impute the
		current feature, and ``estimator`` is the trained estimator used for
		the imputation. Length is ``self.n_features_with_missing_ *
		self.n_iter_``.
	n_iter_ : int
		Number of iteration rounds that occurred. Will be less than
		``self.max_iter`` if early stopping criterion was reached.
	n_features_with_missing_ : int
		Number of features with missing values.
	indicator_ : :class:`~sklearn.impute.MissingIndicator`
		Indicator used to add binary indicators for missing values.
		``None`` if add_indicator is False.
	random_state_ : RandomState instance
		RandomState instance that is generated either from a seed, the random
		number generator or by `np.random`.
	See Also
	--------
	SimpleImputer : Univariate imputation of missing values.
	Examples
	--------
	>>> import numpy as np
	>>> from sklearn.experimental import enable_iterative_imputer
	>>> from sklearn.impute import IterativeImputer
	>>> imp_mean = IterativeImputer(random_state=0)
	>>> imp_mean.fit([[7, 2, 3], [4, np.nan, 6], [10, 5, 9]])
	IterativeImputer(random_state=0)
	>>> X = [[np.nan, 2, 3], [4, np.nan, 6], [10, np.nan, 9]]
	>>> imp_mean.transform(X)
	array([[ 6.9584...,  2.       ,  3.        ],
			[ 4.       ,  2.6000...,  6.        ],
			[10.       ,  4.9999...,  9.        ]])
	Notes
	-----
	To support imputation in inductive mode we store each feature's estimator
	during the ``fit`` phase, and predict without refitting (in order) during
	the ``transform`` phase.
	Features which contain all missing values at ``fit`` are discarded upon
	``transform``.
	References
	----------
	.. [1] `Stef van Buuren, Karin Groothuis-Oudshoorn (2011). "mice:
		Multivariate Imputation by Chained Equations in R". Journal of
		Statistical Software 45: 1-67.
		<https://www.jstatsoft.org/article/view/v045i03>`_
	.. [2] `S. F. Buck, (1960). "A Method of Estimation of Missing Values in
		Multivariate Data Suitable for use with an Electronic Computer".
		Journal of the Royal Statistical Society 22(2): 302-306.
		<https://www.jstor.org/stable/2984099>`_
	"""
	def __init__(self,
				search_space,
				clf_kwargs,
				ga_kwargs,
				prefix,
				estimator=None, 
				*,
				missing_values=np.nan,
				sample_posterior=False,
				max_iter=10,
				tol=1e-3,
				n_nearest_features=None,
				initial_strategy="mean",
				imputation_order='ascending',
				skip_complete=False,
				min_value=-np.inf,
				max_value=np.inf,
				verbose=0,
				random_state=None,
				add_indicator=False,
				grid_cv=5,
				grid_n_jobs=1,
				grid_n_iter=10,
				clf_type="classifier",
				ga=False,
				disable_progressbar=False
	):
				
		super().__init__(
			missing_values=missing_values,
			add_indicator=add_indicator
		)

		self.search_space = search_space
		self.clf_kwargs = clf_kwargs
		self.ga_kwargs = ga_kwargs
		self.prefix = prefix
		self.estimator = estimator
		self.sample_posterior = sample_posterior
		self.max_iter = max_iter
		self.tol = tol
		self.n_nearest_features = n_nearest_features
		self.initial_strategy = initial_strategy
		self.imputation_order = imputation_order
		self.skip_complete = skip_complete
		self.min_value = min_value
		self.max_value = max_value
		self.verbose = verbose
		self.random_state = random_state
		self.grid_cv = grid_cv
		self.grid_n_jobs = grid_n_jobs
		self.grid_n_iter = grid_n_iter
		self.clf_type = clf_type
		self.ga = ga
		self.disable_progressbar = disable_progressbar

	@ignore_warnings(category=UserWarning)
	def _impute_one_feature(self,
							X_filled,
							mask_missing_values,
							feat_idx,
							neighbor_feat_idx,
							estimator=None,
							fit_mode=True,
	):
		"""Impute a single feature from the others provided.
		This function predicts the missing values of one of the features using
		the current estimates of all the other features. The ``estimator`` must
		support ``return_std=True`` in its ``predict`` method for this function
		to work.
		Parameters
		----------
		X_filled : ndarray
			Input data with the most recent imputations.
		mask_missing_values : ndarray
			Input data's missing indicator matrix.
		feat_idx : int
			Index of the feature currently being imputed.
		neighbor_feat_idx : ndarray
			Indices of the features to be used in imputing ``feat_idx``.
		estimator : object
			The estimator to use at this step of the round-robin imputation.
			If ``sample_posterior`` is True, the estimator must support
			``return_std`` in its ``predict`` method.
			If None, it will be cloned from self._estimator.
		fit_mode : boolean, default=True
			Whether to fit and predict with the estimator or just predict.

		Returns
		-------
		X_filled : ndarray
			Input data with ``X_filled[missing_row_mask, feat_idx]`` updated.
		estimator : estimator with sklearn API
			The fitted estimator used to impute
			``X_filled[missing_row_mask, feat_idx]``.
		"""
		if estimator is None and fit_mode is False:
			raise ValueError("If fit_mode is False, then an already-fitted "
							"estimator should be passed in.")

		if estimator is None:
			estimator = clone(self._estimator)

		# Modified code
		acc_scorer = make_scorer(accuracy_score)

		rmse_scorer = make_scorer(
			mean_squared_error, 
			greater_is_better=False, 
			squared=False
		)

		cross_val = StratifiedKFold(n_splits=self.grid_cv, shuffle=False)

		# Modified code
		# If regressor
		if self.clf_type == "regressor":
			metric = rmse_scorer

			if self.ga:
				callback = DeltaThreshold(threshold=1e-3, metric="fitness")

		else:
			metric = acc_scorer

			if self.ga:
				callback = ConsecutiveStopping(generations=5, metric="fitness")			
		# Do randomized grid search
		if not self.ga:
			search = RandomizedSearchCV(
				estimator, 
				param_distributions=self.search_space, 
				n_iter=self.grid_n_iter, 
				scoring=metric, 
				n_jobs=self.grid_n_jobs, 
				cv=cross_val
			)

		# Do genetic algorithm
		else:
			search = GASearchCV(
				estimator=estimator,
				cv=cross_val,
				scoring=metric,
				generations=self.grid_n_iter,
				param_grid=self.search_space,
				n_jobs=self.grid_n_jobs,
				verbose=False,
				**self.ga_kwargs
			)
				
		missing_row_mask = mask_missing_values[:, feat_idx]
		if fit_mode:
			X_train = _safe_indexing(X_filled[:, neighbor_feat_idx],
									~missing_row_mask)

			y_train = _safe_indexing(X_filled[:, feat_idx],
									~missing_row_mask)

			if self.ga:
				search.fit(X_train, y_train, callbacks=callback)

			else:
				search.fit(X_train, y_train)
		
		# if no missing values, don't predict
		if np.sum(missing_row_mask) == 0:
			return X_filled, estimator, None

		# get posterior samples if there is at least one missing value
		X_test = _safe_indexing(
			X_filled[:, neighbor_feat_idx],	missing_row_mask
		)

		if self.sample_posterior:
			mus, sigmas = search.predict(X_test, return_std=True)
			imputed_values = np.zeros(mus.shape, dtype=X_filled.dtype)
			# two types of problems: (1) non-positive sigmas
			# (2) mus outside legal range of min_value and max_value
			# (results in inf sample)
			positive_sigmas = sigmas > 0
			imputed_values[~positive_sigmas] = mus[~positive_sigmas]
			mus_too_low = mus < self._min_value[feat_idx]
			imputed_values[mus_too_low] = self._min_value[feat_idx]
			mus_too_high = mus > self._max_value[feat_idx]
			imputed_values[mus_too_high] = self._max_value[feat_idx]
			# the rest can be sampled without statistical issues
			inrange_mask = positive_sigmas & ~mus_too_low & ~mus_too_high
			mus = mus[inrange_mask]
			sigmas = sigmas[inrange_mask]
			a = (self._min_value[feat_idx] - mus) / sigmas
			b = (self._max_value[feat_idx] - mus) / sigmas

			truncated_normal = stats.truncnorm(a=a, b=b,
											loc=mus, scale=sigmas)
			imputed_values[inrange_mask] = truncated_normal.rvs(
				random_state=self.random_state_)

		else:
			imputed_values = search.predict(X_test)
			imputed_values = np.clip(imputed_values,
									self._min_value[feat_idx],
									self._max_value[feat_idx])

		# update the feature
		X_filled[missing_row_mask, feat_idx] = imputed_values

		return X_filled, estimator, search

	def _get_neighbor_feat_idx(self,
							n_features,
							feat_idx,
							abs_corr_mat):
		"""Get a list of other features to predict ``feat_idx``.
		If self.n_nearest_features is less than or equal to the total
		number of features, then use a probability proportional to the absolute
		correlation between ``feat_idx`` and each other feature to randomly
		choose a subsample of the other features (without replacement).
		Parameters
		----------
		n_features : int
			Number of features in ``X``.
		feat_idx : int
			Index of the feature currently being imputed.
		abs_corr_mat : ndarray, shape (n_features, n_features)
			Absolute correlation matrix of ``X``. The diagonal has been zeroed
			out and each feature has been normalized to sum to 1. Can be None.
		Returns
		-------
		neighbor_feat_idx : array-like
			The features to use to impute ``feat_idx``.
		"""
		if (self.n_nearest_features is not None and
				self.n_nearest_features < n_features):
			p = abs_corr_mat[:, feat_idx]
			neighbor_feat_idx = self.random_state_.choice(
				np.arange(n_features), self.n_nearest_features, replace=False,
				p=p)
		else:
			inds_left = np.arange(feat_idx)
			inds_right = np.arange(feat_idx + 1, n_features)
			neighbor_feat_idx = np.concatenate((inds_left, inds_right))
		return neighbor_feat_idx

	def _get_ordered_idx(self, mask_missing_values):
		"""Decide in what order we will update the features.
		As a homage to the MICE R package, we will have 4 main options of
		how to order the updates, and use a random order if anything else
		is specified.
		Also, this function skips features which have no missing values.
		Parameters
		----------
		mask_missing_values : array-like, shape (n_samples, n_features)
			Input data's missing indicator matrix, where "n_samples" is the
			number of samples and "n_features" is the number of features.
		Returns
		-------
		ordered_idx : ndarray, shape (n_features,)
			The order in which to impute the features.
		"""
		frac_of_missing_values = mask_missing_values.mean(axis=0)
		if self.skip_complete:
			missing_values_idx = np.flatnonzero(frac_of_missing_values)
		else:
			missing_values_idx = np.arange(np.shape(frac_of_missing_values)[0])
		if self.imputation_order == 'roman':
			ordered_idx = missing_values_idx
		elif self.imputation_order == 'arabic':
			ordered_idx = missing_values_idx[::-1]
		elif self.imputation_order == 'ascending':
			n = len(frac_of_missing_values) - len(missing_values_idx)
			ordered_idx = np.argsort(frac_of_missing_values,
									kind='mergesort')[n:]
		elif self.imputation_order == 'descending':
			n = len(frac_of_missing_values) - len(missing_values_idx)
			ordered_idx = np.argsort(frac_of_missing_values,
									kind='mergesort')[n:][::-1]
		elif self.imputation_order == 'random':
			ordered_idx = missing_values_idx
			self.random_state_.shuffle(ordered_idx)
		else:
			raise ValueError("Got an invalid imputation order: '{0}'. It must "
							"be one of the following: 'roman', 'arabic', "
							"'ascending', 'descending', or "
							"'random'.".format(self.imputation_order))
		return ordered_idx

	def _num_features(self, X):
		"""Return the number of features in an array-like X.
		This helper function tries hard to avoid to materialize an array version
		of X unless necessary. For instance, if X is a list of lists,
		this function will return the length of the first element, assuming
		that subsequent elements are all lists of the same length without
		checking.
		Parameters
		----------
		X : array-like
			array-like to get the number of features.
		Returns
		-------
		features : int
			Number of features
		"""
		type_ = type(X)
		if type_.__module__ == "builtins":
			type_name = type_.__qualname__
		else:
			type_name = f"{type_.__module__}.{type_.__qualname__}"
		message = f"Unable to find the number of features from X of type {type_name}"
		if not hasattr(X, "__len__") and not hasattr(X, "shape"):
			if not hasattr(X, "__array__"):
				raise TypeError(message)
			# Only convert X to a numpy array if there is no cheaper, heuristic
			# option.
			X = np.asarray(X)

		if hasattr(X, "shape"):
			if not hasattr(X.shape, "__len__") or len(X.shape) <= 1:
				message += f" with shape {X.shape}"
				raise TypeError(message)
			return X.shape[1]

		first_sample = X[0]

		# Do not consider an array-like of strings or dicts to be a 2D array
		if isinstance(first_sample, (str, bytes, dict)):
			message += f" where the samples are of type {type(first_sample).__qualname__}"
			raise TypeError(message)

		try:
			# If X is a list of lists, for instance, we assume that all nested
			# lists have the same length without checking or converting to
			# a numpy array to keep this function call as cheap as possible.
			return len(first_sample)
		except Exception as err:
			raise TypeError(message) from err

	def _check_n_features(self, X, reset):
		"""Set the `n_features_in_` attribute, or check against it.
		Parameters
		----------
		X : {ndarray, sparse matrix} of shape (n_samples, n_features)
			The input samples.
		reset : bool
			If True, the `n_features_in_` attribute is set to `X.shape[1]`.
			If False and the attribute exists, then check that it is equal to
			`X.shape[1]`. If False and the attribute does *not* exist, then
			the check is skipped.
			.. note::
			It is recommended to call reset=True in `fit` and in the first
			call to `partial_fit`. All other methods that validate `X`
			should set `reset=False`.
		"""
		try:
			n_features = self._num_features(X)
		except TypeError as e:
			if not reset and hasattr(self, "n_features_in_"):
				raise ValueError(
					"X does not contain any features, but "
					f"{self.__class__.__name__} is expecting "
					f"{self.n_features_in_} features"
				) from e
			# If the number of features is not defined and reset=True,
			# then we skip this check
			return

		if reset:
			self.n_features_in_ = n_features
			return

		if not hasattr(self, "n_features_in_"):
			# Skip this check if the expected number of expected input features
			# was not recorded by calling fit first. This is typically the case
			# for stateless transformers.
			return

		if n_features != self.n_features_in_:
			raise ValueError(
				f"X has {n_features} features, but {self.__class__.__name__} "
				f"is expecting {self.n_features_in_} features as input."
			)

	def _validate_data(self, X, y='no_validation', reset=True,
				   validate_separately=False, **check_params):
		"""Validate input data and set or check the `n_features_in_` attribute.
		Parameters
		----------
		X : {array-like, sparse matrix, dataframe} of shape \
				(n_samples, n_features)
			The input samples.
		y : array-like of shape (n_samples,), default='no_validation'
			The targets.
			- If `None`, `check_array` is called on `X`. If the estimator's
			requires_y tag is True, then an error will be raised.
			- If `'no_validation'`, `check_array` is called on `X` and the
			estimator's requires_y tag is ignored. This is a default
			placeholder and is never meant to be explicitly set.
			- Otherwise, both `X` and `y` are checked with either `check_array`
			or `check_X_y` depending on `validate_separately`.
		reset : bool, default=True
			Whether to reset the `n_features_in_` attribute.
			If False, the input will be checked for consistency with data
			provided when reset was last True.
			.. note::
			It is recommended to call reset=True in `fit` and in the first
			call to `partial_fit`. All other methods that validate `X`
			should set `reset=False`.
		validate_separately : False or tuple of dicts, default=False
			Only used if y is not None.
			If False, call validate_X_y(). Else, it must be a tuple of kwargs
			to be used for calling check_array() on X and y respectively.
		**check_params : kwargs
			Parameters passed to :func:`sklearn.utils.check_array` or
			:func:`sklearn.utils.check_X_y`. Ignored if validate_separately
			is not False.
		Returns
		-------
		out : {ndarray, sparse matrix} or tuple of these
			The validated input. A tuple is returned if `y` is not None.
		"""

		if y is None:
			if self._get_tags()['requires_y']:
				raise ValueError(
					f"This {self.__class__.__name__} estimator "
					f"requires y to be passed, but the target y is None."
				)
			X = check_array(X, **check_params)
			out = X
		elif isinstance(y, str) and y == 'no_validation':
			X = check_array(X, **check_params)
			out = X
		else:
			if validate_separately:
				# We need this because some estimators validate X and y
				# separately, and in general, separately calling check_array()
				# on X and y isn't equivalent to just calling check_X_y()
				# :(
				check_X_params, check_y_params = validate_separately
				X = check_array(X, **check_X_params)
				y = check_array(y, **check_y_params)
			else:
				X, y = check_X_y(X, y, **check_params)
			out = X, y

		if check_params.get('ensure_2d', True):
			self._check_n_features(X, reset=reset)

		return out

	def _get_abs_corr_mat(self, X_filled, tolerance=1e-6):
		"""Get absolute correlation matrix between features.
		Parameters
		----------
		X_filled : ndarray, shape (n_samples, n_features)
			Input data with the most recent imputations.
		tolerance : float, default=1e-6
			``abs_corr_mat`` can have nans, which will be replaced
			with ``tolerance``.
		Returns
		-------
		abs_corr_mat : ndarray, shape (n_features, n_features)
			Absolute correlation matrix of ``X`` at the beginning of the
			current round. The diagonal has been zeroed out and each feature's
			absolute correlations with all others have been normalized to sum
			to 1.
		"""
		n_features = X_filled.shape[1]
		if (self.n_nearest_features is None or
				self.n_nearest_features >= n_features):
			return None
		with np.errstate(invalid='ignore'):
			# if a feature in the neighboorhood has only a single value
			# (e.g., categorical feature), the std. dev. will be null and
			# np.corrcoef will raise a warning due to a division by zero
			abs_corr_mat = np.abs(np.corrcoef(X_filled.T))
		# np.corrcoef is not defined for features with zero std
		abs_corr_mat[np.isnan(abs_corr_mat)] = tolerance
		# ensures exploration, i.e. at least some probability of sampling
		np.clip(abs_corr_mat, tolerance, None, out=abs_corr_mat)
		# features are not their own neighbors
		np.fill_diagonal(abs_corr_mat, 0)
		# needs to sum to 1 for np.random.choice sampling
		abs_corr_mat = normalize(abs_corr_mat, norm='l1', axis=0, copy=False)
		return abs_corr_mat

	def _initial_imputation(self, X, in_fit=False):
		"""Perform initial imputation for input X.
		Parameters
		----------
		X : ndarray, shape (n_samples, n_features)
			Input data, where "n_samples" is the number of samples and
			"n_features" is the number of features.
		in_fit : bool, default=False
			Whether function is called in fit.
		Returns
		-------
		Xt : ndarray, shape (n_samples, n_features)
			Input data, where "n_samples" is the number of samples and
			"n_features" is the number of features.
		X_filled : ndarray, shape (n_samples, n_features)
			Input data with the most recent imputations.
		mask_missing_values : ndarray, shape (n_samples, n_features)
			Input data's missing indicator matrix, where "n_samples" is the
			number of samples and "n_features" is the number of features.
		X_missing_mask : ndarray, shape (n_samples, n_features)
			Input data's mask matrix indicating missing datapoints, where
			"n_samples" is the number of samples and "n_features" is the
			number of features.
		"""
		if is_scalar_nan(self.missing_values):
			force_all_finite = "allow-nan"
		else:
			force_all_finite = True

		X = self._validate_data(X, dtype=FLOAT_DTYPES, order="F", reset=in_fit,
								force_all_finite=force_all_finite)
		_check_inputs_dtype(X, self.missing_values)

		X_missing_mask = _get_mask(X, self.missing_values)
		mask_missing_values = X_missing_mask.copy()
		if self.initial_imputer_ is None:
			self.initial_imputer_ = SimpleImputer(
				missing_values=self.missing_values,
				strategy=self.initial_strategy
			)
			X_filled = self.initial_imputer_.fit_transform(X)
		else:
			X_filled = self.initial_imputer_.transform(X)

		valid_mask = np.flatnonzero(np.logical_not(
			np.isnan(self.initial_imputer_.statistics_)))
		Xt = X[:, valid_mask]
		mask_missing_values = mask_missing_values[:, valid_mask]

		return Xt, X_filled, mask_missing_values, X_missing_mask

	@staticmethod
	def _validate_limit(limit, limit_type, n_features):
		"""Validate the limits (min/max) of the feature values
		Converts scalar min/max limits to vectors of shape (n_features,)
		Parameters
		----------
		limit: scalar or array-like
			The user-specified limit (i.e, min_value or max_value)
		limit_type: string, "max" or "min"
			n_features: Number of features in the dataset
		Returns
		-------
		limit: ndarray, shape(n_features,)
			Array of limits, one for each feature
		"""
		limit_bound = np.inf if limit_type == "max" else -np.inf
		limit = limit_bound if limit is None else limit
		if np.isscalar(limit):
			limit = np.full(n_features, limit)
		limit = check_array(
			limit, force_all_finite=False, copy=False, ensure_2d=False
		)
		if not limit.shape[0] == n_features:
			raise ValueError(
				f"'{limit_type}_value' should be of "
				f"shape ({n_features},) when an array-like "
				f"is provided. Got {limit.shape}, instead."
			)
		return limit

	@ignore_warnings(category=UserWarning)
	def fit_transform(self, X, y=None):
		"""Fits the imputer on X and return the transformed X.
		Parameters
		----------
		X : array-like, shape (n_samples, n_features)
			Input data, where "n_samples" is the number of samples and
			"n_features" is the number of features.
		y : ignored.
		Returns
		-------
		Xt : array-like, shape (n_samples, n_features)
			The imputed input data.
		"""
		self.random_state_ = getattr(self, "random_state_",
									check_random_state(self.random_state))

		if self.max_iter < 0:
			raise ValueError(
				"'max_iter' should be a positive integer. Got {} instead."
				.format(self.max_iter))

		if self.tol < 0:
			raise ValueError(
				"'tol' should be a non-negative float. Got {} instead."
				.format(self.tol)
			)

		if self.estimator is None:
			from ..linear_model import BayesianRidge
			self._estimator = BayesianRidge()
		else:
			self._estimator = clone(self.estimator)

		self.imputation_sequence_ = []

		self.initial_imputer_ = None

		X, Xt, mask_missing_values, complete_mask = (
			self._initial_imputation(X, in_fit=True))

		super()._fit_indicator(complete_mask)
		X_indicator = super()._transform_indicator(complete_mask)

		if self.max_iter == 0 or np.all(mask_missing_values):
			self.n_iter_ = 0
			return super()._concatenate_indicator(Xt, X_indicator)

		# Edge case: a single feature. We return the initial ...
		if Xt.shape[1] == 1:
			self.n_iter_ = 0
			return super()._concatenate_indicator(Xt, X_indicator)

		self._min_value = self._validate_limit(
			self.min_value, "min", X.shape[1])
		self._max_value = self._validate_limit(
			self.max_value, "max", X.shape[1])

		if not np.all(np.greater(self._max_value, self._min_value)):
			raise ValueError(
				"One (or more) features have min_value >= max_value.")

		# order in which to impute
		# note this is probably too slow for large feature data (d > 100000)
		# and a better way would be good.
		# see: https://goo.gl/KyCNwj and subsequent comments
		ordered_idx = self._get_ordered_idx(mask_missing_values)
		self.n_features_with_missing_ = len(ordered_idx)

		abs_corr_mat = self._get_abs_corr_mat(Xt)

		n_samples, n_features = Xt.shape

		if self.verbose > 0:
			print("[IterativeImputer] Completing matrix with shape %s"
				% (X.shape,))
		start_t = time()

		if not self.sample_posterior:
			Xt_previous = Xt.copy()
			normalized_tol = self.tol * np.max(
				np.abs(X[~mask_missing_values])
		)

		params_list = list()
		score_list = list()
		iter_list = list()
		if self.ga:
			sns.set_style("white")

		total_iter = self.max_iter

		for self.n_iter_ in progressbar(
			range(1, total_iter), 
			desc="Iteration: ", disable=self.disable_progressbar
		):

			if self.ga:
				iter_list.append(self.n_iter_)

				pp_oneline = PdfPages(".score_traces_separate_{}.pdf".format(self.n_iter_))

				pp_lines = PdfPages(".score_traces_combined_{}.pdf".format(self.n_iter_))

				pp_space = PdfPages(".search_space_{}.pdf".format(self.n_iter_))

			if self.imputation_order == 'random':
				ordered_idx = self._get_ordered_idx(mask_missing_values)

			# Reset lists for current iteration
			params_list.clear()
			score_list.clear()
			searches = list()

			if self.disable_progressbar:
				feat_counter = 0
				total_features = len(ordered_idx)

			for feat_idx in progressbar(
				ordered_idx, desc="Feature: ", leave=False, position=1, 
				disable=self.disable_progressbar
			):

				if self.disable_progressbar:
					feat_counter += 1
					print(f"\nIteration: {self.n_iter_}/{self.max_iter} ({int((self.n_iter_ / total_iter) * 100)}%)\nFeature: {feat_counter}/{total_features} ({int((feat_counter / total_features) * 100)}%)\n")

				neighbor_feat_idx = self._get_neighbor_feat_idx(n_features,
																feat_idx,
																abs_corr_mat)

				Xt, estimator, search = self._impute_one_feature(
					Xt, 
					mask_missing_values, 
					feat_idx, 
					neighbor_feat_idx,
					estimator=None, 
					fit_mode=True
				)

				searches.append(search)

				estimator_triplet = _ImputerTriplet(feat_idx,
													neighbor_feat_idx,
													estimator)

				self.imputation_sequence_.append(estimator_triplet)

				if search is not None:

					params_list.append(search.best_params_)
					score_list.append(search.best_score_)

					if self.ga:
						plt.cla()
						plt.clf()
						plt.close()

						plot_fitness_evolution(search)
						pp_oneline.savefig(bbox_inches="tight")
						plt.cla()
						plt.clf()
						plt.close()

						self.plot_search_space(search)
						pp_space.savefig(bbox_inches="tight")
						plt.cla()
						plt.clf()
						plt.close()

				else:
					# Search is None
					# Thus, there was no missing data in the given feature
					tmp_dict = dict()
					for k in self.search_space.keys():
						tmp_dict[k] = -9
					params_list.append(tmp_dict)

					score_list.append(-9)

			if self.verbose > 1:
				print('[IterativeImputer] Ending imputation round '
					'%d/%d, elapsed time %0.2f'
					% (self.n_iter_, self.max_iter, time() - start_t))

			if not self.sample_posterior:
				inf_norm = np.linalg.norm(Xt - Xt_previous, ord=np.inf,
										axis=None)
				if self.verbose > 0:
					print('[IterativeImputer] '
						'Change: {}, scaled tolerance: {} '.format(
							inf_norm, normalized_tol))

				if inf_norm < normalized_tol:
					if self.verbose > 0:
						print('[IterativeImputer] Early stopping criterion '
							'reached.')

					if self.ga:
						pp_oneline.close()
						pp_space.close()

						plt.cla()
						plt.clf()
						plt.close()
						for iter_search in searches:
							if iter_search is not None:
								plot_fitness_evolution(iter_search)

						pp_lines.savefig(bbox_inches="tight")

						plt.cla()
						plt.clf()
						plt.close()
						pp_lines.close()

					break
				Xt_previous = Xt.copy()
			
			if self.ga:
				pp_oneline.close()
				pp_space.close()

				plt.cla()
				plt.clf()
				plt.close()
				for iter_search in searches:
					if iter_search is not None:
						plot_fitness_evolution(iter_search)
				
				pp_lines.savefig(bbox_inches="tight")

				plt.cla()
				plt.clf()
				plt.close()
				pp_lines.close()

		else:
			if not self.sample_posterior:
				warnings.warn("[IterativeImputer] Early stopping criterion not"
							" reached.", ConvergenceWarning)

		Xt[~mask_missing_values] = X[~mask_missing_values]

		if self.ga:
			# Remove all files except last iteration
			final_iter = iter_list.pop()

			[os.remove(".score_traces_separate_{}.pdf".format(x)) for x in iter_list]

			[os.remove(".score_traces_combined_{}.pdf".format(x)) for x in iter_list]

			[os.remove(".search_space_{}.pdf".format(x)) for x in iter_list]

			shutil.move(".score_traces_separate_{}.pdf".format(final_iter), "{}_score_traces_separate.pdf".format(self.prefix))

			shutil.move(".score_traces_combined_{}.pdf".format(final_iter), "{}_score_traces_combined.pdf".format(self.prefix))

			shutil.move(".search_space_{}.pdf".format(final_iter), "{}_search_space.pdf".format(self.prefix))

		return super()._concatenate_indicator(Xt, X_indicator), params_list, score_list

	def transform(self, X):
		"""Imputes all missing values in X.
		Note that this is stochastic, and that if random_state is not fixed,
		repeated calls, or permuted input, will yield different results.
		Parameters
		----------
		X : array-like of shape (n_samples, n_features)
			The input data to complete.
		Returns
		-------
		Xt : array-like, shape (n_samples, n_features)
			The imputed input data.
		"""
		check_is_fitted(self)

		X, Xt, mask_missing_values, complete_mask = self._initial_imputation(X)

		X_indicator = super()._transform_indicator(complete_mask)

		if self.n_iter_ == 0 or np.all(mask_missing_values):
			return super()._concatenate_indicator(Xt, X_indicator)

		imputations_per_round = len(self.imputation_sequence_) // self.n_iter_
		i_rnd = 0
		if self.verbose > 0:
			print("[IterativeImputer] Completing matrix with shape %s"
				% (X.shape,))
		start_t = time()

		params_list = list()
		score_list = list()
		for it, estimator_triplet in enumerate(self.imputation_sequence_):
			Xt, _, search = self._impute_one_feature(
				Xt,
				mask_missing_values,
				estimator_triplet.feat_idx,
				estimator_triplet.neighbor_feat_idx,
				estimator=estimator_triplet.estimator,
				fit_mode=False
			)

			if search is not None:
				params_list.append(search.best_params_)
				score_list.append(search.best_score_)
			else:
				tmp_dict = dict()
				for k in search_space.keys():
					tmp_dict[k] = -9
				params_list.append(tmp_dict)

				score_list.append(-9)

			if not (it + 1) % imputations_per_round:
				if self.verbose > 1:
					print('[IterativeImputer] Ending imputation round '
							'%d/%d, elapsed time %0.2f'
							% (i_rnd + 1, self.n_iter_, time() - start_t))
				i_rnd += 1

		Xt[~mask_missing_values] = X[~mask_missing_values]

		return super()._concatenate_indicator(Xt, X_indicator), params_list, score_list

	def fit(self, X, y=None):
		"""Fits the imputer on X and return self.
		Parameters
		----------
		X : array-like, shape (n_samples, n_features)
			Input data, where "n_samples" is the number of samples and
			"n_features" is the number of features.
		y : ignored
		Returns
		-------
		self : object
			Returns self.
		"""
		self.fit_transform(X)
		return self

	def plot_search_space(self, estimator, height=2, s=25, features: list = None):
		"""
		Parameters
		----------
		estimator: estimator object
			A fitted estimator from :class:`~sklearn_genetic.GASearchCV`
		height: float, default=2
			Height of each facet
		s: float, default=5
			Size of the markers in scatter plot
		features: list, default=None
			Subset of features to plot, if ``None`` it plots all the features by default

		Returns
		-------
		Pair plot of the used hyperparameters during the search

		"""
		sns.set_style("white")

		df = logbook_to_pandas(estimator.logbook)
		if features:
			stats = df[features]
		else:
			variables = [*estimator.space.parameters, "score"]
			stats = df[variables]

		g = sns.PairGrid(stats, diag_sharey=False, height=height)

		g = g.map_upper(sns.scatterplot, s=s, color="r", alpha=0.2)

		try:
			g = g.map_lower(
				sns.kdeplot,
				shade=True,
				cmap=sns.color_palette("ch:s=.25,rot=-.25", as_cmap=True),
			)
		except np.linalg.LinAlgError as err:
			if "singular matrix" in str(err).lower():
				g = g.map_lower(
					sns.scatterplot, s=s, color="b", alpha=1.0
				)
			else:
				raise

		try:
			g = g.map_diag(
				sns.kdeplot, 
				shade=True, 
				palette="crest", 
				alpha=0.2, 
				color="red"
			)
		except np.linalg.LinAlgError as err:
			if "singular matrix" in str(err).lower():
				g = g.map_diag(
					sns.histplot,
					color="red",
					alpha=1.0,
					kde=False
				)
				
		return g

