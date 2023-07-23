# Standard library imports
import gc
import math
import os
import sys
import warnings

# from collections import namedtuple
from contextlib import redirect_stdout
from time import time
from typing import Optional, Union, List, Dict, Tuple, Any, Callable

# Third-party imports
## For stats and numeric operations
import numpy as np
import pandas as pd
from scipy import stats

# scikit-learn imports
from sklearn.base import clone
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
from sklearn.impute._base import _check_inputs_dtype

## For warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings

## Required for IterativeImputer.fit_transform()
from sklearn.utils import check_random_state, _safe_indexing, is_scalar_nan
from sklearn.utils._mask import _get_mask
from sklearn.utils.validation import FLOAT_DTYPES
from sklearn.preprocessing import LabelEncoder

# Custom function imports
try:
    from .. import simple_imputers
    from ...utils.misc import get_processor_name
    from ...utils.misc import HiddenPrints
    from ...utils.misc import isnotebook
except (ModuleNotFoundError, ValueError, ImportError):
    from pgsui.impute import simple_imputers
    from pgsui.utils.misc import get_processor_name
    from pgsui.utils.misc import HiddenPrints
    from pgsui.utils.misc import isnotebook

# Uses scikit-learn-intellex package if CPU is Intel
if get_processor_name().strip().startswith("Intel"):
    try:
        from sklearnex import patch_sklearn

        patch_sklearn(verbose=False)
    except (ImportError, TypeError):
        print(
            "Processor not compatible with scikit-learn-intelex; using "
            "default configuration"
        )

is_notebook = isnotebook()

if is_notebook:
    from tqdm.notebook import tqdm as progressbar
else:
    if sys.platform == "linux" or sys.platform == "linux2":
        from tqdm.auto import tqdm as progressbar
    else:
        from tqdm import tqdm as progressbar

# NOTE: Removed ImputeTriplets to save memory.
# ImputerTriplet is there so that the missing values
# can be predicted on an already-fit model using just the
# transform method. I didn't need it, so I removed it
# because it was saving thousands of fit estimator models into the object

# _ImputerTripletAll = namedtuple(
# 	'_ImputerTripletAll', ['feat_idx', 'neighbor_feat_idx', 'estimator'])


class IterativeImputerFixedParams(IterativeImputer):
    """Overridden IterativeImputer methods.

    Herein, progress status updates, optimizations to save RAM, and several other improvements have been added. IterativeImputer is a multivariate imputer that estimates each feature from all the others. A strategy for imputing missing values by modeling each feature with missing values as a function of other features in a round-robin fashion.Read more in the scikit-learn User Guide for IterativeImputer. scikit-learn versionadded: 0.21. NOTE: This estimator is still **experimental** for now: the predictions and the API might change without any deprecation cycle. To use it, you need to explicitly import ``enable_iterative_imputer``\.

        IterativeImputer is based on the R MICE (Multivariate Imputation by Chained Equationspackage) [3]_. See [4]_ for more information about multiple versus single imputations.

        >>> # explicitly require this experimental feature
        >>> from sklearn.experimental import enable_iterative_imputer
        >>>
        >>> # now you can import normally from sklearn.impute
        >>> from sklearn.impute import IterativeImputer

    Args:
        logfilepath (str): Path to the progress log file.

        clf_kwargs (Dict[str, Any]): A dictionary with the classifier keyword arguments.

        prefix (str): Prefix for output files.

        estimator (callable estimator object, optional): The estimator to use at each step of the round-robin imputation. If ``sample_posterior`` is True, the estimator must support ``return_std`` in its ``predict`` method. Defaults to BayesianRidge().

        clf_type (str, optional): Whether to run ```'classifier'``` or ``'regression'`` based imputation. Defaults to 'classifier'

        disable_progressbar (bool, optional): Whether or not to disable the tqdm progress bar. If True, disables the progress bar. If False, tqdm is used for the progress bar. This can be useful if you are running the imputation on an HPC cluster or are saving the standard output to a file. If True, progress updates will be printed to the screen every ``progress_update_percent`` iterations. Defaults to False.

        progress_update_percent (int, optional): How often to display progress updates (as a percentage) if ``disable_progressbar`` is True. If ``progress_update_frequency=10``\, then it displays progress updates every 10%. Defaults to 10.

        pops (List[Union[str, int]] or None): List of population IDs to be used with ImputeAlleleFreq if ``initial_strategy="populations"``\.

        missing_values (int or np.nan, optional): The placeholder for the missing values. All occurrences of ``missing_values`` will be imputed. For pandas dataframes with	nullable integer dtypes with missing values, ``missing_values`` should be set to ``np.nan``\, since ``pd.NA`` will be converted to ``np.nan``\. Defaults to np.nan.

        Sample_posterior (bool, optional): Whether to sample from the (Gaussian) predictive posterior of the fitted estimator for each imputation. Estimator must support ``return_std`` in its ``predict`` method if set to ``True``\. Set to ``True`` if using ``IterativeImputer`` for multiple imputations. Defaults to False.

        max_iter (int, optional): Maximum number of imputation rounds to perform before returning the imputations computed during the final round. A round is a single	imputation of each feature with missing values. The stopping criterion is met once ``max(abs(X_t - X_{t-1}))/max(abs(X[known_vals])) < tol``\,	where ``X_t`` is ``X`` at iteration ``t``\. Note that early stopping is only applied if ``sample_posterior=False``\. Defaults to 10.

        tol (float, optional): Tolerance of the stopping condition. Defaults to 1e-3.

        n_nearest_features (int, optional): Number of other features to use to estimate the missing values of each feature column. Nearness between features is measured using the absolute correlation coefficient between each feature pair (after initial imputation). To ensure coverage of features throughout the imputation process, the neighbor features are not necessarily nearest,	but are drawn with probability proportional to correlation for each	imputed target feature. Can provide significant speed-up when the number of features is huge. If ``None``\, all features will be used. Defaults to None.

        initial_strategy (str, optional): Which strategy to use to initialize the missing values. Same as the ``strategy`` parameter in :class:`~sklearn.impute.SimpleImputer`	Valid values: "mean", "median", "most_frequent", "populations", "phylogeny", "mf", or "constant". Defaults to "mean".

        imputation_order (str, optional): The order in which the features will be imputed. Possible values: "ascending" (From features with fewest missing values to most), "descending" (From features with most missing values to fewest, "roman" (Left to right), "arabic" (Right to left),  random" (A random order for each round). Defaults to "ascending".

        skip_complete (bool, optional): If ``True`` then features with missing values during ``transform`` that did not have any missing values during ``fit`` will be imputed with the initial imputation method only. Set to ``True`` if you have	many features with no missing values at both ``fit`` and ``transform`` time to save compute. Defaults to False.

        min_value (float or array-like of shape (n_features,), optional): Minimum possible imputed value. Broadcast to shape (n_features,) if scalar. If array-like, expects shape (n_features,), one min value for each feature. The default is `-np.inf`...sklearn versionchanged:: 0.23 (Added support for array-like). Defaults to -np.inf.

        max_value (float or array-like of shape (n_features,), optional): Maximum possible imputed value. Broadcast to shape (n_features,) if scalar. If array-like, expects shape (n_features,), one max value for each feature..sklearn versionchanged:: 0.23 (Added support for array-like). Defaults to np.inf.

        verbose (int, optional): Verbosity flag, controls the debug messages that are issued as functions are evaluated. The higher, the more verbose. Can be 0, 1, or 2. Defaults to 0.

        random_state (int or RandomState instance, optional): The seed of the pseudo random number generator to use. Randomizes selection of estimator features if n_nearest_features is not None, the ``imputation_order`` if ``random``\, and the sampling from posterior if ``sample_posterior`` is True. Use an integer for determinism. Defaults to None.

        add_indicator (bool, optional): If True, a :class:`MissingIndicator` transform will stack onto output of the imputer's transform. This allows a predictive estimator to account for missingness despite imputation. If a feature has no missing values at fit/train time, the feature won't appear on the missing indicator even if there are missing values at transform/test time. Defaults to False.

        genotype_data (GenotypeData object, optional): GenotypeData object containing dictionary with keys=sampleIds and values=list of genotypes for the corresponding key. If using ``initial_strategy="phylogeny``\, then this object also needs contain the treefile and qmatrix objects. Defaults to None.

        str_encodings (dict(str: int), optional): Integer encodings used in STRUCTURE-formatted file. Should be a dictionary with keys=nucleotides and values=integer encodings. The missing data encoding should also be included. Argument is ignored if using a PHYLIP-formatted file. Defaults to {"A": 1, "C": 2, "G": 3, "T": 4, "N": -9}

        kwargs (Dict[str, Any]): For compatibility with grid search IterativeImputer.

    Attributes:
        initial_imputer_ (sklearn.impute.SimpleImputer):  Imputer used to initialize the missing values.

        n_iter_ (int): Number of iteration rounds that occurred. Will be less than ``self.max_iter`` if early stopping criterion was reached.

        n_features_with_missing_ (int): Number of features with missing values.

        indicator_ (sklearn.impute.MissingIndicator): Indicator used to add binary indicators for missing values ``None`` if add_indicator is False.

        random_state_ (RandomState instance): RandomState instance that is generated either from a seed, the random number generator or by ``np.random``\.

        logfilepath (str): Path to status logfile.

        clf_kwargs (Dict[str, Any]): Keyword arguments for estimator.

        prefix (str): Prefix for output files.

        clf_type (str): Type of estimator, either "classifier" or "regressor".

        disable_progressbar (bool): Whether to disable the tqdm progress bar. If True, writes status updates to file instead of tqdm progress bar.

        progress_update_percent (float or None): Print feature progress update every ``progress_update_percent`` percent.

        pops (List[Union[str, int]]): List of population IDs of shape (n_samples,).

        estimator (estimator object): Estimator to impute data with.

        sample_posterior (bool): Whether to use the sample_posterior option. This overridden class does not currently support sample_posterior.

        max_iter (int): The maximum number of iterations to run.

        tol (float): Convergence criteria.

        n_nearest_features (int): Number of nearest features to impute target with.

        initial_strategy (str): Strategy to use with SimpleImputer for training data.

        imputation_order (str): Order to impute.

        skip_complete (bool): Whether to skip features with no missing data.

        min_value (int or float): Minimum value of imputed data.

        max_value (int or float): Maximum value of imputed data.

        verbose (int): Verbosity level.

        genotype_data (GenotypeData object): GenotypeData object.

        str_encodings (Dict[str, int]): Dictionary with integer encodings for converting from STRUCTURE-formatted file to IUPAC nucleotides.

    See Also:
            SimpleImputer : Univariate imputation of missing values.

    Examples:
        >>> import numpy as np
        >>> from sklearn.experimental import enable_iterative_imputer
        >>> from sklearn.impute import IterativeImputer
        >>> imp_mean = IterativeImputer(random_state=0)
        >>> imp_mean.fit([[7, 2, 3], [4, np.nan, 6], [10, 5, 9]])
        >>> X = [[np.nan, 2, 3], [4, np.nan, 6], [10, np.nan, 9]]
        >>> imp_mean.transform(X)
        array([[ 6.9584...,  2.       ,  3.        ],
                        [ 4.       ,  2.6000...,  6.        ],
                        [10.       ,  4.9999...,  9.        ]])

    Notes:
        To support imputation in inductive mode we store each feature's estimator during the ``fit`` phase, and predict without refitting (in order) during	the ``transform`` phase. Features which contain all missing values at ``fit`` are discarded upon ``transform``\.

        NOTE: Inductive mode support was removed herein.

    References:
        .. [3] Stef van Buuren, Karin Groothuis-Oudshoorn (2011). mice: Multivariate Imputation by Chained Equations in R. Journal of Statistical Software 45: 1-67.

        .. [4] S. F. Buck, (1960). A Method of Estimation of Missing Values in	Multivariate Data Suitable for use with an Electronic Computer. Journal of the Royal Statistical Society 22(2): 302-306.
    """

    def __init__(
        self,
        logfilepath: str,
        clf_kwargs: Dict[str, Any],
        *,
        estimator: Callable = None,
        clf_type: str = "classifier",
        disable_progressbar: bool = False,
        progress_update_percent: Optional[int] = None,
        pops: Optional[List[Union[str, int]]] = None,
        missing_values: Union[float, int] = np.nan,
        sample_posterior: bool = False,
        max_iter: int = 10,
        tol: float = 1e-3,
        n_nearest_features: Optional[int] = None,
        initial_strategy: str = "mean",
        imputation_order: str = "ascending",
        skip_complete: bool = False,
        min_value: Union[int, float] = -np.inf,
        max_value: Union[int, float] = np.inf,
        verbose: int = 0,
        random_state: Optional[int] = None,
        add_indicator: bool = False,
        genotype_data: Optional[Any] = None,
        str_encodings: Optional[Dict[str, int]] = None,
        prefix="imputer",
        **kwargs,
    ) -> None:
        super().__init__(
            estimator=estimator,
            missing_values=missing_values,
            sample_posterior=sample_posterior,
            max_iter=max_iter,
            tol=tol,
            n_nearest_features=n_nearest_features,
            initial_strategy=initial_strategy,
            imputation_order=imputation_order,
            skip_complete=skip_complete,
            min_value=min_value,
            max_value=max_value,
            verbose=verbose,
            random_state=random_state,
            add_indicator=add_indicator,
        )

        self.logfilepath = logfilepath
        self.clf_kwargs = clf_kwargs
        self.prefix = prefix
        self.clf_type = clf_type
        self.disable_progressbar = disable_progressbar
        self.progress_update_percent = progress_update_percent
        self.pops = pops
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
        self.genotype_data = genotype_data
        self.str_encodings = str_encodings
        self.missing_values = missing_values

    def _initial_imputation(
        self, X: np.ndarray, cols_to_keep: np.ndarray, in_fit: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Perform initial imputation for input X.

        Performs initial imputation on training data (neighbors).

        Args:
            X (ndarray): Input data of shape (n_samples, n_features), where n_samples is the number of samples and n_features is the number of features.

            cols_to_keep (numpy.ndarray): Column indices of shape (n_features,) to keep. Only used if ``initial_strategy=="phylogeny"``\.

            in_fit (bool, optional): True if function is called in fit, otherwise False. Defaults to False.

        Returns:
            Xt (numpy.ndarray): Input data of shape (n_samples, n_features), where n_samples is the number of samples and "n_features" is the number of features.

            X_filled (numpy.ndarray): Input data of shape (n_samples, features) with the most recent imputations.

            mask_missing_values (numpy.ndarray): Input data missing indicator matrix of of shape (n_samples, n_features), where n_samples is the number of samples and "n_features" is the number of features.

            X_missing_mask (numpy.ndarray): Input data mask matrix of shape (n_samples, n_features) indicating missing datapoints, where
                n_samples is the number of samples and n_features is the
                number of features.

        Raises:
            AttributeError: GenotypeData object must be initialized with guidetree and qmatrix if using ``initial_strategy=phylogeny``\.
        """
        if is_scalar_nan(self.missing_values):
            force_all_finite = "allow-nan"
        else:
            force_all_finite = True

        X = self._validate_data(
            X,
            dtype=FLOAT_DTYPES,
            order="F",
            reset=in_fit,
            force_all_finite=force_all_finite,
        )

        X[X < 0] = np.nan

        _check_inputs_dtype(X, self.missing_values)

        X_missing_mask = _get_mask(X, self.missing_values)
        mask_missing_values = X_missing_mask.copy()

        if self.initial_strategy == "populations":
            self.initial_imputer_ = simple_imputers.ImputeAlleleFreq(
                self.genotype_data,
                gt=np.nan_to_num(X, nan=-9).tolist(),
                pops=self.pops,
                by_populations=True,
                missing=-9,
                verbose=False,
                iterative_mode=True,
                validation_mode=True,
            )

            X_filled = np.array(self.initial_imputer_.imputed)
            Xt = X.copy()

        elif self.initial_strategy == "phylogeny":
            if (
                self.genotype_data.qmatrix is None
                and self.genotype_data.qmatrix_iqtree is None
            ) or self.genotype_data.guidetree is None:
                raise AttributeError(
                    "GenotypeData object was not initialized with "
                    "qmatrix/ qmatrix_iqtree or guidetree arguments, "
                    "but initial_strategy == 'phylogeny'"
                )

            else:
                self.initial_imputer_ = simple_imputers.ImputePhylo(
                    genotype_data=self.genotype_data,
                    str_encodings=self.str_encodings,
                    write_output=False,
                    disable_progressbar=True,
                    column_subset=cols_to_keep,
                    validation_mode=True,
                )

                X_filled = self.initial_imputer_.imputed.to_numpy()
                valid_sites = self.initial_imputer_.valid_sites

                valid_mask = np.flatnonzero(
                    np.logical_not(np.isnan(valid_sites))
                )

                Xt = X[:, valid_mask]
                mask_missing_values = mask_missing_values[:, valid_mask]

        elif self.initial_strategy == "mf":
            self.initial_imputer_ = simple_imputers.ImputeMF(
                self.genotype_data,
                gt=np.nan_to_num(X, nan=-9),
                missing=-9,
                verbose=False,
                validation_mode=True,
            )

            X_filled = np.array(self.initial_imputer_.imputed)
            Xt = X.copy()

        elif self.initial_strategy == "phylogeny":
            if (
                self.genotype_data.qmatrix is None
                and self.genotype_data.qmatrix_iqtree is None
            ) or self.genotype_data.guidetree is None:
                raise AttributeError(
                    "GenotypeData object was not initialized with "
                    "qmatrix/ qmatrix_iqtree or guidetree arguments, "
                    "but initial_strategy == 'phylogeny'"
                )

            else:
                self.initial_imputer_ = simple_imputers.ImputePhylo(
                    genotype_data=self.genotype_data,
                    str_encodings=self.str_encodings,
                    write_output=False,
                    disable_progressbar=True,
                    column_subset=cols_to_keep,
                    validation_mode=True,
                )

            X_filled = self.initial_imputer_.imputed.to_numpy()
            Xt = X.copy()

        else:
            if self.initial_imputer_ is None:
                self.initial_imputer_ = SimpleImputer(
                    missing_values=self.missing_values,
                    strategy=self.initial_strategy,
                )
                X_filled = self.initial_imputer_.fit_transform(X)

            else:
                X_filled = self.initial_imputer_.transform(X)

            valid_mask = np.flatnonzero(
                np.logical_not(np.isnan(self.initial_imputer_.statistics_))
            )

            Xt = X[:, valid_mask]
            mask_missing_values = mask_missing_values[:, valid_mask]

        return Xt, X_filled, mask_missing_values, X_missing_mask

    @ignore_warnings(category=UserWarning)
    def _impute_one_feature(
        self,
        X_filled: np.ndarray,
        mask_missing_values: np.ndarray,
        feat_idx: int,
        neighbor_feat_idx: np.ndarray,
        estimator: Optional[Any] = None,
        fit_mode: bool = True,
    ) -> np.ndarray:
        """Impute a single feature from the others provided.

        This function predicts the missing values of one of the features using the current estimates of all the other features. The ``estimator`` must support ``return_std=True`` in its ``predict`` method for this function to work.

        Args:
            X_filled (numpy.ndarray): Input data with the most recent imputations.

            mask_missing_values (numpy.ndarray): Input data's missing indicator matrix.

            feat_idx (int): Index of the feature currently being imputed.

            neighbor_feat_idx (numpy.ndarray): Indices of the features to be used in imputing ``feat_idx``\.

            estimator (object): The estimator to use at this step of the round-robin imputation If ``sample_posterior`` is True, the estimator must support ``return_std`` in its ``predict`` method.If None, it will be cloned from self._estimator.

            fit_mode (bool, optional): Whether to fit and predict with the estimator or just predict. Defaults to True.

        Returns:
            X_filled (numpy.ndarray)]: Input data with ``X_filled[missing_row_mask, feat_idx]`` updated.
        """
        if estimator is None and fit_mode is False:
            raise ValueError(
                "If fit_mode is False, then an already-fitted "
                "estimator should be passed in."
            )

        if estimator is None:
            estimator = clone(self._estimator)

        missing_row_mask = mask_missing_values[:, feat_idx]

        if fit_mode:
            X_train = _safe_indexing(
                X_filled[:, neighbor_feat_idx], ~missing_row_mask
            )

            y_train = _safe_indexing(X_filled[:, feat_idx], ~missing_row_mask)

            try:
                estimator.fit(X_train, y_train)
                le = None
            except ValueError as e:
                # Happens in newer versions of XGBClassifier.
                if str(e).startswith(
                    "Invalid classes inferred from unique values of `y`"
                ):
                    le = LabelEncoder()
                    y_train = le.fit_transform(y_train)
                    estimator.fit(X_train, y_train)

        # if no missing values, don't predict
        if np.sum(missing_row_mask) == 0:
            return X_filled

        X_test = _safe_indexing(
            X_filled[:, neighbor_feat_idx], missing_row_mask
        )

        if self.sample_posterior:
            raise NotImplementedError(
                "sample_posterior is not currently supported. "
                "Please set sample_posterior to False"
            )

        else:
            imputed_values = estimator.predict(X_test)

            if le is not None:
                imputed_values = le.inverse_transform(imputed_values)

            imputed_values = np.clip(
                imputed_values,
                self._min_value[feat_idx],
                self._max_value[feat_idx],
            )

        # update the feature
        X_filled[missing_row_mask, feat_idx] = imputed_values

        del estimator
        del X_train
        del y_train
        del X_test
        del imputed_values
        gc.collect()

        return X_filled

    @ignore_warnings(category=UserWarning)
    def fit_transform(
        self,
        X: np.ndarray,
        valid_cols: Optional[np.ndarray] = None,
        y: None = None,
    ) -> np.ndarray:
        """Fits the imputer on X and return the transformed X.

        Args:
            X (array-like, shape (n_samples, n_features)): Input data, where "n_samples" is the number of samples and "n_features" is the number of features.

            valid_cols (numpy.ndarray, optional): Array with column indices to keep. Defaults to None.

            y (None): Ignored. Here for compatibility with other sklearn classes.

        Returns:
            Xt (array-like, shape (n_samples, n_features)): The imputed input data.

        Raises:
            ValueError: "max_iter" must be a positive integer.
            ValueError: "tol" should be non-negative float.
            ValueError: One or more features has min_value >= max_value.
            ConvergenceWarning: Early stopping criterion not reached.
        """
        self.random_state_ = getattr(
            self, "random_state_", check_random_state(self.random_state)
        )

        if self.max_iter < 0:
            raise ValueError(
                f"'max_iter' should be a positive integer. Got {self.max_iter} instead."
            )

        if self.tol < 0:
            raise ValueError(
                f"'tol' should be a non-negative float. Got {self.tol} instead"
            )

        self._estimator = clone(self.estimator)

        self.initial_imputer_ = None

        # X is the input data subset to only valid features (not nan)
        # Xt is the data imputed with SimpleImputer
        # mask_missing_values is the missing indicator matrix
        # complete_mask is the input data's mask matrix
        X, Xt, mask_missing_values, complete_mask = self._initial_imputation(
            X, valid_cols, in_fit=True
        )

        super(IterativeImputer, self)._fit_indicator(complete_mask)
        X_indicator = super(IterativeImputer, self)._transform_indicator(
            complete_mask
        )

        if self.max_iter == 0 or np.all(mask_missing_values):
            self.n_iter_ = 0
            return super(IterativeImputer, self)._concatenate_indicator(
                Xt, X_indicator
            )

        # Edge case: a single feature. We return the initial ...
        if Xt.shape[1] == 1:
            self.n_iter_ = 0
            return super(IterativeImputer, self)._concatenate_indicator(
                Xt, X_indicator
            )

        self._min_value = self._validate_limit(
            self.min_value, "min", X.shape[1]
        )
        self._max_value = self._validate_limit(
            self.max_value, "max", X.shape[1]
        )

        if not np.all(np.greater(self._max_value, self._min_value)):
            raise ValueError(
                "One (or more) features have min_value >= max_value."
            )

        # order in which to impute
        # note this is probably too slow for large feature data (d > 100000)
        # and a better way would be good.
        # see: https://goo.gl/KyCNwj and subsequent comments
        ordered_idx = self._get_ordered_idx(mask_missing_values)
        total_features = len(ordered_idx)

        self.n_features_with_missing_ = len(ordered_idx)

        abs_corr_mat = self._get_abs_corr_mat(Xt)

        n_samples, n_features = Xt.shape

        if self.verbose > 0:
            print(
                f"[IterativeImputer] Completing matrix with shape ({X.shape},)"
            )
        start_t = time()

        if not self.sample_posterior:
            Xt_previous = Xt.copy()
            normalized_tol = self.tol * np.max(np.abs(X[~mask_missing_values]))

        total_iter = self.max_iter

        ###########################################
        ### Iteration Start
        ###########################################
        for self.n_iter_ in progressbar(
            range(1, total_iter + 1),
            desc="Iteration: ",
            disable=self.disable_progressbar,
        ):
            if self.imputation_order == "random":
                ordered_idx = self._get_ordered_idx(mask_missing_values)

            if self.disable_progressbar:
                with open(self.logfilepath, "a") as fout:
                    # Redirect to progress logfile
                    with redirect_stdout(fout):
                        print(
                            f"Iteration Progress: "
                            f"{self.n_iter_}/{self.max_iter} "
                            f"({int((self.n_iter_ / total_iter) * 100)}%)"
                        )

                if self.progress_update_percent is not None:
                    print_perc_interval = self.progress_update_percent

            ##########################
            ### Feature Start
            ##########################
            for i, feat_idx in enumerate(
                progressbar(
                    ordered_idx,
                    desc="Feature: ",
                    leave=False,
                    position=1,
                    disable=self.disable_progressbar,
                ),
                start=1,
            ):
                neighbor_feat_idx = self._get_neighbor_feat_idx(
                    n_features, feat_idx, abs_corr_mat
                )

                Xt = self._impute_one_feature(
                    Xt,
                    mask_missing_values,
                    feat_idx,
                    neighbor_feat_idx,
                    estimator=None,
                    fit_mode=True,
                )

                # NOTE: The below source code has been commented out to save
                # RAM. estimator_triplet object contains numerous fit estimators
                # that demand a lot of resources. It is primarily used for the
                # transform function, which is not needed in this application.

                # estimator_triplet = _ImputerTripletAll(
                # 	feat_idx,
                # 	neighbor_feat_idx,
                # 	estimator)

                # self.imputation_sequence_.append(estimator_triplet)

                # Only print feature updates at each progress_update_percent
                # interval
                if (
                    self.progress_update_percent is not None
                    and self.disable_progressbar
                ):
                    current_perc = math.ceil((i / total_features) * 100)

                    if current_perc >= print_perc_interval:
                        with open(self.logfilepath, "a") as fout:
                            # Redirect progress to file
                            with redirect_stdout(fout):
                                print(
                                    f"Feature Progress (Iteration "
                                    f"{self.n_iter_}/{self.max_iter}): "
                                    f"{i}/{total_features} ({current_perc}"
                                    f"%)"
                                )

                                if i == len(ordered_idx):
                                    print("\n", end="")

                        while print_perc_interval <= current_perc:
                            print_perc_interval += self.progress_update_percent

            if self.verbose > 1:
                print(
                    f"[IterativeImputer] Ending imputation round "
                    f"{self.n_iter_}/{self.max_iter}, "
                    f"elapsed time {(time() - start_t):0.2f}"
                )

            if not self.sample_posterior:
                inf_norm = np.linalg.norm(
                    Xt - Xt_previous, ord=np.inf, axis=None
                )

                if self.verbose > 0:
                    print(
                        f"[IterativeImputer] Change: {inf_norm}, "
                        f"scaled tolerance: {normalized_tol} "
                    )

                if inf_norm < normalized_tol:
                    if self.disable_progressbar:
                        # Early stopping criteria has been reached
                        with open(self.logfilepath, "a") as fout:
                            # Redirect to progress logfile
                            with redirect_stdout(fout):
                                print(
                                    "[IterativeImputer] Early stopping "
                                    "criterion reached."
                                )
                    else:
                        print(
                            "[IterativeImputer] Early stopping criterion "
                            "reached."
                        )

                    break
                Xt_previous = Xt.copy()

        else:
            if not self.sample_posterior:
                warnings.warn(
                    "[IterativeImputer] Early stopping criterion not"
                    " reached.",
                    ConvergenceWarning,
                )

        Xt[~mask_missing_values] = X[~mask_missing_values]
        Xt = Xt.astype(int)

        return (
            super(IterativeImputer, self)._concatenate_indicator(
                Xt, X_indicator
            ),
            None,
            None,
        )
