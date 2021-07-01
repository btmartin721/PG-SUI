# Standard library imports
import sys
from collections import Counter
from operator import itemgetter
from statistics import mean

# Third party imports
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.experimental import enable_iterative_imputer
#from sklearn.impute import IterativeImputer
from read_input.iterative_imputer import IterativeImputer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import BayesianRidge
from sklearn.neighbors import KNeighborsClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

# Custom module imports
from read_input.read_input import GenotypeData
from utils import misc
from utils.misc import timer
from utils.misc import bayes_search_CV_init
from utils import sequence_tools

from utils.misc import isnotebook

is_notebook = isnotebook()

if is_notebook:
	from tqdm.notebook import tqdm as progressbar
else:
	from tqdm import tqdm as progressbar

class Impute:

	def __init__(
		self,
		clf,
		clf_type,
		kwargs
	):
		self.clf = clf
		self.clf_type = clf_type

		# Separate local variables into separate settings objects		
		self.gridparams, \
		self.imp_kwargs, \
		self.clf_kwargs, \
		self.grid_iter, \
		self.cv, \
		self.n_jobs, \
		self.prefix = self._gather_impute_settings(kwargs)

	@timer
	def fit_predict(self, X):

		#df = self._format_features(X)
		
		# Don't do a grid search
		if self.gridparams is None:
			imputed_df, best_acc, best_params = \
				self._impute_single(X)

		# Do a grid search and get the transformed data with the best parameters
		else:
			imputed_df, best_score, best_params = \
				self._impute_gridsearch(X)

			print("Grid Search Results:")
			if self.clf_type == "regressor":
				print("RMSE (best parameters): {:0.2f}".format(best_score))
			else:
				print("Accuracy (best parameters): {:0.2f}".format(best_score))
				
			print("Best Parameters: {}\n".format(best_params))

		print("\nDone!\n")
		return imputed_df, best_score, best_params

	def _impute_single(self, df):
		"""[Run IterativeImputer without a grid search]

		Args:
			df ([pandas.DataFrame]): [DataFrame of 012-encoded genotypes]

		Returns:
			[pandas.DataFrame]: [Imputed 012-encoded genotypes]
		"""
		print(
			"Doing {} imputation without grid search...".format(str(self.clf)))

		clf = self.clf(**self.clf_kwargs)
		imputer = self._define_iterative_imputer(clf)

		df_imp = self._impute_fit_transform(df, imputer)

		print("Done with {} imputation!\n".format(str(self.clf)))
		return df_imp, None, None
		
	def _impute_gridsearch(self, df):
		"""[Do IterativeImputer with RandomizedGridCV]

		Args:
			df ([pandas.DataFrame]): [DataFrame with 012-encoded genotypes]

		Returns:
			[pandas.DataFrame]: [DataFrame with 012-encoded genotypes imputed using the best parameters found from the grid search]

			[float]: [Absolute value of the best score found during the grid search]

			[dict]: [Best parameters found during the grid search]
		"""
		df_subset = self.subset_data_for_testing(df, 0.05)
		df_subset = self._remove_nonbiallelic(df_subset)

		print("Test dataset size: {}\n".format(len(df_subset.columns)))
		print("Doing grid search...")

		#search_space = self._validate_gridparams(self.gridparams)
		search_space = self.gridparams

		clf = self.clf()

		imputer = self._define_iterative_imputer(
			clf, search_space, self.n_jobs, self.grid_iter, self.cv, self.clf_type)

		imp_arr, params_list, score_list = imputer.fit_transform(df_subset)
		df_imp = pd.DataFrame(imp_arr)
		df_imp = df_imp.round(0).astype("Int8")

		best_params = self._get_best_params(params_list)
		avg_score = mean(x for x in score_list if x != -9)

		print("Done with {} imputation!\n".format(str(self.clf)))
		return df_imp, abs(avg_score), best_params
		
	def _get_best_params(self, params_list):
		best_params = dict()
		keys = list(params_list[0].keys())
		for k in keys:
			if all(isinstance(x[k], (int, float)) for x in params_list):
				if all(isinstance(y[k], int) for y in params_list):
					best_params[k] = \
						self._average_list_of_dicts(params_list, k, is_int=True)

				elif all(isinstance(z[k], float) for z in params_list):
					best_params[k] = self._average_list_of_dicts(params_list, k)

			elif all(isinstance(x[k], (str, bool)) for x in params_list):
				best_params[k] = self._mode_list_of_dicts(params_list, k)
			
			else:
				best_params[k] = self._mode_list_of_dicts(params_list, k)

		return best_params

	def write_imputed(self, data, best_score, best_params):
		"""[Save imputed data to a CSV file]

		Args:
			data ([pandas.DataFrame, numpy.array, or list(list)]): [Object returned from impute_missing()]

			best_score ([float]): [Best RMSE or accuracy score for the regressor or classifier, respectively]

			best_params ([dict]): [Best parameters from grid search]

		Raises:
			TypeError: [Must be of type pandas.DataFrame, numpy.array, or list]
		"""
		outfile = "{}_imputed_012.csv".format(self.prefix)
		if isinstance(data, pd.DataFrame):
			data.to_csv(outfile, header=False, index=False)

		elif isinstance(data, np.ndarray):
			np.savetxt(outfile, data, delimiter=",")

		elif isinstance(data, list):
			with open(outfile, "w") as fout:
				fout.writelines(
					",".join(str(j) for j in i) + "\n" for i in data
				)
		else:
			raise TypeError("'write_imputed()' takes either a pandas.DataFrame,"
				" numpy.ndarray, or 2-dimensional list"
			)

		best_score_outfile = "{}_imputed_best_score.csv".format(self.prefix)
		best_params_outfile = "{}_imputed_best_params.csv".format(self.prefix)

		with open(best_score_outfile, "w") as fout:
			fout.write("{}".format(best_score))

		with open(best_params_outfile, "w") as fout:
			fout.write("parameter\tbest_value\n")
			for k, v in best_params.items():
				fout.write("{},{}\n".format(k, v))

	def read_imputed(self, filename):
		"""[Read in imputed CSV file as formatted by write_imputed]

		Args:
			filename ([str]): [Name of imputed CSV file to be read]

		Returns:
			[pandas.DataFrame]: [Imputed data as DataFrame of 8-bit integers]
		"""
		return pd.read_csv(filename, dtype="Int8", header=None)

	def _mode_list_of_dicts(self, l, k):
		"""[Get mode for key k in a list of dictionaries]

		Args:
			l ([list(dict)]): [List of dictionaries]
			k ([str]): [Key to find the mode across all dictionaries in l]

		Returns:
			[str]: [Most common value across list of dictionaries for one key]
		"""
		k_count = Counter(map(itemgetter(k), l))
		return k_count.most_common()[0][0]

	def _average_list_of_dicts(self, l, k, is_int=False):
		"""[Get average of a given key in a list of dictionaries]

		Args:
			l ([list(dict)]): [List of dictionaries]

			k ([str]): [Key to find average across list of dictionaries]

			is_int (bool, optional): [Whether or not the value for key k is an integer. If False, it is expected to be of type float]. Defaults to False.

		Returns:
			[int or float]: [average of given key across list of dictionaries]
		"""
		if is_int:
			return int(sum(d[k] for d in l) / len(l))
		else:
			return sum(d[k] for d in l) / len(l)

	def _remove_nonbiallelic(self, df):
		"""[Remove sites that do not have both 0 and 2 encoded values in a column]

		Args:
			df ([pandas.DataFrame]): [DataFrame with 012-encoded genotypes]

		Returns:
			[pandas.DataFrame]: [DataFrame with non-biallelic sites removed]
		"""
		df_cp = df.copy()
		bad_cols = list()
		for col in df_cp.columns:
			if not df_cp[col].isin([0]).any() or not df_cp[col].isin([2]).any():
				bad_cols.append(col)

		return df_cp.drop(bad_cols, axis=1)

	def subset_data_for_testing(self, df, column_percent):
		"""[Randomly subsets pandas.DataFrame with ```column_percent``` fraction of the data. Allows faster testing]

		Args:
			df ([pandas.DataFrame]): [DataFrame with 012-encoded genotypes]
			column_percent ([float]): [Fraction of DataFrame to randomly subset. Should be between 0 and 1]

		Returns:
			[pandas.DataFrame]: [Subset DataFrame]
		"""
		cols = sorted(np.random.choice(df.columns, 
			int(len(df.columns) * column_percent), replace=False), reverse=False)

		df2 = df.loc[:, cols]
		return df2

	def _gather_impute_settings(self, kwargs):
		"""[Gather impute settings from the various imputation classes and IterativeImputer. Gathers them for use with the Impute() class. Returns dictionary with keys as keyword arguments and the values as the settings. The imputation can then be run by specifying e.g. IterativeImputer(**imp_kwargs)]

		Args:
			kwargs ([dict]): [Dictionary with keys as the keyword arguments and their corresponding values]

		Returns:
			[dict]: [Parameters to run with grid search]
			[dict]: [IterativeImputer kwargs]
			[dict]: [Classifier kwargs]
			[grid_iter]: [Number of iterations to run with grid search]
			[cv]: [Number of cross-validation folds to use]
			[n_jobs]: [Number of processors to use with grid search]
			[prefix]: [Prefix for output files]
		"""
		gridparams = kwargs.pop("gridparams")
		cv = kwargs.pop("cv")
		n_jobs = kwargs.pop("n_jobs")
		prefix = kwargs.pop("prefix")
		grid_iter = kwargs.pop("grid_iter")

		if prefix is None:
			prefix = "output"

		imp_kwargs = kwargs.copy()
		clf_kwargs = kwargs.copy()

		imp_keys = ["n_nearest_features", "max_iter", "tol", "initial_strategy", "imputation_order", "skip_complete", "random_state", "verbose", "sample_posterior"]

		to_remove = ["genotype_data", "self", "__class__"]

		for k, v in clf_kwargs.copy().items():
			if k in to_remove:
				clf_kwargs.pop(k)
			if k in imp_keys:
				clf_kwargs.pop(k)
				
		if "clf_random_state" in clf_kwargs:
			clf_kwargs["random_state"] = clf_kwargs.pop("clf_random_state")

		if "clf_tol" in clf_kwargs:
			clf_kwargs["tol"] = clf_kwargs.pop("clf_tol")
					
		for k, v in imp_kwargs.copy().items():
			if k not in imp_keys:
				imp_kwargs.pop(k)

		return gridparams, imp_kwargs, clf_kwargs, grid_iter, cv, n_jobs, prefix

	def _format_features(self, df, missing_val=-9):
		"""[Format a 2D list for input into iterative imputer]

		Args:
			df ([pandas.DataFrame]): [DataFrame of features with shape(n_samples, n_features)]

			missing_val (int, optional): [Missing value to replace with numpy.nan]. Defaults to -9.

		Returns:
			[pandas.DataFrame]: [Formatted pandas.DataFrame for input into IterativeImputer]
		"""
		# Replace missing data with NaN
		X = df.replace(missing_val, np.nan)
		return X

	def _validate_gridparams(self, params):
		"""[Convert values in dictionary of lists to Bayesian distribution space for use with BayesSearchCV]

		Args:
			params ([dict(list)]): [Dictionary of lists with grid search parameters]

		Raises:
			TypeError: [Unrecognized gridparam value types]

		Returns:
			[dict(Int, Real, Categorical)]: [Dictionary with values converted to distribution space]
		"""
		search_space = dict()
		for k, v in params.items():
			if all(isinstance(item, int) for item in v):
				search_space[k] = Integer(v)
			elif all(isinstance(item, str) for item in v):
				search_space[k] = Categorical(v)
			elif all(isinstance(item, float) for item in v):
				search_space[k] = Real(v)
			else:
				raise TypeError("Unknown type in gridparams values!")

		return search_space

	def _defile_dataset(self, df, col_selection_rate=0.40):
		"""[Function to select 40% of the columns in a pandas DataFrame and change anywhere from 15% to 50% of the values in each of those columns to np.nan (missing data). Since we know the true values of the ones changed to np.nan, we can assess the accuracy of the classifier and do a grid search]

		Args:
			df ([pandas.DataFrame]): [012-encoded genotypes to extract columns from.]

			col_selection_rate (float, optional): [Proportion of the DataFrame to extract]. Defaults to 0.40.

		Returns:
			[pandas.DataFrame]: [DataFrame with newly missing values]
			[numpy.ndarray]: [Columns that were extracted via random sampling]
		"""
		# Code adapted from: 
		# https://medium.com/analytics-vidhya/using-scikit-learns-iterative-imputer-694c3cca34de	
		cols = np.random.choice(df.columns, 
								int(len(df.columns) * col_selection_rate))

		df_cp = df.copy()
		for col in progressbar(cols, desc="CV Random Sampling: "):
			data_drop_rate = np.random.choice(np.arange(0.15, 0.5, 0.02), 1)[0]

			drop_ind = np.random.choice(np.arange(len(df_cp[col])), 
				size=int(len(df_cp[col])*data_drop_rate), replace=False)

			current_col = df_cp[col].values
			df_cp[col].iloc[drop_ind] = np.nan
		return df_cp, cols

	def _reshuffle_missing(self, df_cp, col, good_cols, retries):
		data_drop_rate = np.random.choice(np.arange(0.15, 0.5, 0.02), 1)[0]

		drop_ind = np.random.choice(np.arange(len(df_cp[col])), 
			size=int(len(df_cp[col])*data_drop_rate), replace=False)

		current_col = df_cp[col].values
		df_cp[col].iloc[drop_ind] = np.nan

		cnt = 1
		while cnt <= retries:
			if (
				not df_cp[col].isin([0]).any() or
				not df_cp[col].isin([1]).any() or
				not df_cp[col].isin([2]).any()
			):
				cnt += 1
				df_cp = df_cp.assign(col=current_col)
				
				data_drop_rate = \
					np.random.choice(np.arange(0.15, 0.5, 0.02), 1)[0]

				drop_ind = np.random.choice(np.arange(len(df_cp[col])), 
					size=int(len(df_cp[col])*data_drop_rate), 
					replace=False)

				df_cp[col].iloc[drop_ind] = np.nan

			elif (
					df_cp[col].isin([0]).any() and
					df_cp[col].isin([1]).any() and
					df_cp[col].isin([2]).any()
			):
				good_cols.append(col)
				return df_cp, good_cols

		return df_cp, good_cols

	def _impute_eval(self, df_orig, clf):
		"""[Function to run IterativeImputer on a DataFrame. The dataframe will be randomly sampled and a fraction of the known, true values are converted to missing data to allow evalutation of the model]

		Args:
			df_orig ([pandas.DataFrame]): [Original DataFrame with 012-encoded genotypes]

			clf ([sklearn Classifier]): [Classifier instance to use with IterativeImputer]

		Returns:
			[pandas.DataFrame]: [Subsampled DataFrame with known, true values]
			[pandas.DataFrame]: [Subsampled DataFrame with true values converted to missing data]
			[pandas.DataFrame]: [Subsampled DataFrame with imputed missing values]
			[int]: [Number of iterations required to converge]
		"""
		df_miss, cols = self._defile_dataset(df_orig)
		df_orig_slice = df_orig[cols]
		imputer = self._define_iterative_imputer(clf)
		df_stg = df_miss.copy()

		print("\nDoing initial imputation...")
		imp_arr = imputer.fit_transform(df_stg)
		print("Done!\n")

		return df_orig_slice, df_miss[cols], pd.DataFrame(imp_arr[:,[df_orig.columns.get_loc(i) for i in cols]], columns=cols), imputer.n_iter_, imputer

	def _impute_fit_transform(self, df, imputer):
		"""[Do the fit_transform for IterativeImputer and format as a pandas.dataFrame object]

		Args:

			df ([pandas.DataFrame]): [DataFrame with missing data to impute]

			imputer ([sklearn.impute.IterativeImputer]): [IterativeImputer instance]

		Returns:
			[pandas.DataFrame]: [Imputed DataFrame object]

		Raises:
			AssertionError: [Ensure no missing data remains in imputed DataFrame]
		"""
		arr = imputer.fit_transform(df)
		new_arr = arr.astype(dtype=np.int)
		new_df = pd.DataFrame(new_arr)

		# if new_df.isnull().values.any():
		# 	raise AssertionError("Imputation failed: There is still missing data")
					
		return new_df

	def _define_iterative_imputer(self, clf, search_space, n_jobs, n_iter, cv, clf_type):
		"""[Define an IterativeImputer instance]

		Args:
			clf ([sklearn Classifier]): [Classifier to use with IterativeImputer]

			search_space ([dict]): [gridparams dictionary with search space to explore in random grid search]

		Returns:
			[sklearn.impute.IterativeImputer]: [IterativeImputer instance]
		"""
		# Create iterative imputer
		imp = IterativeImputer(search_space, estimator=clf, grid_n_jobs=n_jobs, grid_n_iter=n_iter, grid_cv=cv, clf_type=clf_type, **self.imp_kwargs)
		return imp
		

class ImputeKNN:
	"""[Class to impute missing data from 012-encoded genotypes using K-Nearest Neighbors iterative imputation]
	"""
	def __init__(
		self, 
		genotype_data,
		*, 
		prefix=None,
		gridparams=None,
		grid_iter=50,
		cv=5,
		n_jobs=1,
		n_neighbors=5, 
		weights="distance", 
		algorithm="auto", 
		leaf_size=30, 
		p=2, 
		metric="minkowski", 
		max_iter=10,
		tol=1e-3,
		n_nearest_features=10,
		initial_strategy="most_frequent",
		imputation_order="ascending",
		skip_complete=False,
		random_state=None,
		verbose=0
	):
		"""[Does K-Nearest Neighbors Iterative imputation of missing data. Iterative imputation uses the n_nearest_features to inform the imputation at each feature (i.e., SNP site), using the N most correlated features per site. The N most correlated features are drawn with probability proportional to correlation for each imputed target feature to ensure coverage of features throughout the imputation process]

		Args:
			genotype_data ([GenotypeData]): [GenotypeData instance that was used to read in the sequence data]

			prefix ([str]): [Prefix for imputed data's output filename]

			n_neighbors (int, optional): [Number of neighbors to use by default for K-Nearest Neighbors queries]. Defaults to 5.

			weights (str, optional): [Weight function used in prediction. Possible values: 'Uniform': Uniform weights with all points in each neighborhood weighted equally; 'distance': Weight points by the inverse of their distance, in this case closer neighbors of a query point will have  agreater influence than neighbors that are further away; 'callable': A user-defined function that accepts an array of distances and returns an array of the same shape containing the weights]. Defaults to "distance".

			algorithm (str, optional): [Algorithm used to compute the nearest neighbors. Possible values: 'ball_tree', 'kd_tree', 'brute', 'auto']. Defaults to "auto".

			leaf_size (int, optional): [Leaf size passed to BallTree or KDTree. This can affect the speed of the construction and query, as well as the memory required to store the tree. The optimal value depends on the nature of the problem]. Defaults to 30.

			p (int, optional): [Power parameter for the Minkowski metric. When p=1, this is equivalent to using manhattan_distance (l1), and if p=2 it is equivalent to using euclidean distance (l2). For arbitrary p, minkowski_distance (l_p) is used]. Defaults to 2.

			metric (str, optional): [The distance metric to use for the tree. The default metric is minkowski, and with p=2 this is equivalent to the standard Euclidean metric. See the documentation of sklearn.DistanceMetric for a list of available metrics. If metric is 'precomputed', X is assumed to be a distance matrix and must be square during fit]. Defaults to "minkowski".

			n_jobs (int, optional): [The number of parallel jobs to run for neighbors search. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors]. Defaults to 1.

			max_iter (int, optional): [Maximum number of imputation rounds to perform before returning the imputations computed during the final round. A round is a single imputation of each feature with missing values]. Defaults to 10.

			tol ([type], optional): [Tolerance of the stopping condition]. Defaults to 1e-3.

			n_nearest_features (int, optional): [Number of other features to use to estimate the missing values of eacah feature column. If None, then all features will be used, but this can consume an  intractable amount of computing resources. Nearness between features is measured using the absolute correlation coefficient between each feature pair (after initial imputation). To ensure coverage of features throughout the imputation process, the neighbor features are not necessarily nearest, but are drawn with probability proportional to correlation for each imputed target feature. Can provide significant speed-up when the number of features is huge]. Defaults to 10.

			initial_strategy (str, optional): [Which strategy to use to initialize the missing values. Same as the strategy parameter in sklearn.impute.SimpleImputer Valid values: {“mean”, “median”, “most_frequent”, or “constant”}]. Defaults to "most_frequent".

			imputation_order (str, optional): [The order in which the features will be imputed. Possible values: 'ascending' (from features with fewest missing values to most), 'descending' (from features with most missing values to fewest), 'roman' (left to right), 'arabic' (right to left), 'random' (a random order for each round). ]. Defaults to "ascending".

			skip_complete (bool, optional): [If True, then features with missing values during transform that did not have any missing values during fit will be imputed with the initial imputation method only. Set to True if you have many features with no missing values at both fit and transform time to save compute time]. Defaults to False.

			random_state (int, optional): [The seed of the pseudo random number generator to use for the iterative imputer. Randomizes selection of etimator features if n_nearest_features is not None or the imputation_order is 'random'. Use an integer for determinism. If None, then uses a different random seed each time]. Defaults to None.

			verbose (int, optional): [Verbosity flag, controls the debug messages that are issues as functions are evaluated. The higher, the more verbose. Possible values are 0, 1, or 2]. Defaults to 0.
		"""
		# Get local variables into dictionary object
		kwargs = locals()

		self.clf_type = "classifier"
		self.clf = KNeighborsClassifier

		imputer = Impute(self.clf, self.clf_type, kwargs)

		self.imputed, self.best_score, self.best_params = \
			imputer.fit_predict(genotype_data.genotypes_df)

		imputer.write_imputed(self.imputed, self.best_score, self.best_params)

class ImputeRandomForest:
	"""[Class to impute missing data from 012-encoded genotypes using Random Forest iterative imputation]
	"""
	def __init__(
		self, 
		genotype_data,
		*, 
		prefix=None,
		gridparams=None,
		grid_iter=50,
		cv=5,
		n_jobs=1,
		n_estimators=100,
		criterion="gini",
		max_depth=None, 
		min_samples_split=2,
		min_samples_leaf=1, 
		min_weight_fraction_leaf=0.0,
		max_features="auto",
		max_leaf_nodes=None,
		min_impurity_decrease=0.0,
		bootstrap=False,
		oob_score=False,
		max_samples=None,
		clf_random_state=None,
		max_iter=10,
		tol=1e-3,
		n_nearest_features=10,
		initial_strategy="most_frequent",
		imputation_order="ascending",
		skip_complete=False,
		random_state=None,
		verbose=0
	):
		"""[Does Random Forest Iterative imputation of missing data. Iterative imputation uses the n_nearest_features to inform the imputation at each feature (i.e., SNP site), using the N most correlated features per site. The N most correlated features are drawn with probability proportional to correlation for each imputed target feature to ensure coverage of features throughout the imputation process]

		Args:
			genotype_data ([GenotypeData]): [GenotypeData instance that was used to read in the sequence data]

			prefix ([str]): [Prefix for imputed data's output filename]

			n_estimators (int, optional): [The number of trees in the forest. Increasing this value can improves the fit, but at the cost of compute time]. Defaults to 100.

			criterion (str, optional): [The function to measure the quality of a split. Supported values are 'gini' for the Gini impurity and 'entropy' for the information gain]. Defaults to "gini".

			max_depth (int, optional): [The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples]. Defaults to None.

			min_samples_split (int or float, optional): [The minimum number of samples required to split an internal node. If value is an integer, then considers min_samples_split as the minimum number. If value is a floating point, then min_samples_split is a fraction and (min_samples_split * n_samples), rounded up to the nearest integer, are the minimum number of samples for each split]. Defaults to 2.

			min_samples_leaf (int or float, optional): [The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches. This may have the effect of smoothing the model, especially in regression. If value is an integer, then min_samples_leaf is the minimum number. If value is floating point, then min_samples_leaf is a fraction and (min_samples_leaf * n_samples) rounded up to the nearest integer is the minimum number of samples for each node]. Defaults to 1.

			min_weight_fraction_leaf (float, optional): [The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node. Samples have equal weight when sample_weight is not provided.]. Defaults to 0.0.

			max_features (int or float, optional): [The number of features to consider when looking for the best split. If int, then consider 'max_features' features at each split. If float, then 'max_features' is a fraction and (max_features * n_samples) features, rounded to the nearest integer, are considered at each split. If 'auto', then max_features=sqrt(n_features). If 'sqrt', then max_features=sqrt(n_features). If 'log2', then max_features=log2(n_features). If None, then max_features=n_features]. Defaults to "auto".

			max_leaf_nodes (int, optional): [Grow trees with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes]. Defaults to None.

			min_impurity_decrease (float, optional): [A node will be split if this split induces a decrease of the impurity greater than or equal to this value. See sklearn.ensemble.ExtraTreesClassifier documentation for more information]. Defaults to 0.0.

			bootstrap (bool, optional): [Whether bootstrap samples are used when building trees. If False, the whole dataset is used to build each tree]. Defaults to False.

			oob_score (bool, optional): [Whether to use out-of-bag samples to estimate the generalization score. Only available if bootstrap=True]. Defaults to False.

			max_samples (int or float, optional): [If bootstrap is True, the number of samples to draw from X to train each base estimator. If None (default), then draws X.shape[0] samples. if int, then draw 'max_samples' samples. If float, then draw (max_samples * X.shape[0] samples) with max_samples in the interval (0, 1)]. Defaults to None.

			clf_random_state (int, optional): [Controls three sources of randomness for sklearn.ensemble.ExtraTreesClassifier: The bootstrapping of the samples used when building trees (if bootstrap=True), the sampling of the features to consider when looking for the best split at each node (if max_features < n_features), and the draw of the splits for each of the max_features. If None, then uses a different random seed each time the imputation is run]. Defaults to None.

			n_jobs (int, optional): [The number of parallel jobs to run for the random forest trees. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors]. Defaults to 1.

			max_iter (int, optional): [Maximum number of imputation rounds to perform before returning the imputations computed during the final round. A round is a single imputation of each feature with missing values]. Defaults to 10.

			tol ([type], optional): [Tolerance of the stopping condition]. Defaults to 1e-3.

			n_nearest_features (int, optional): [Number of other features to use to estimate the missing values of eacah feature column. If None, then all features will be used, but this can consume an  intractable amount of computing resources. Nearness between features is measured using the absolute correlation coefficient between each feature pair (after initial imputation). To ensure coverage of features throughout the imputation process, the neighbor features are not necessarily nearest, but are drawn with probability proportional to correlation for each imputed target feature. Can provide significant speed-up when the number of features is huge]. Defaults to 10.

			initial_strategy (str, optional): [Which strategy to use to initialize the missing values. Same as the strategy parameter in sklearn.impute.SimpleImputer Valid values: {“mean”, “median”, “most_frequent”, or “constant”}]. Defaults to "most_frequent".

			imputation_order (str, optional): [The order in which the features will be imputed. Possible values: 'ascending' (from features with fewest missing values to most), 'descending' (from features with most missing values to fewest), 'roman' (left to right), 'arabic' (right to left), 'random' (a random order for each round). ]. Defaults to "ascending".

			skip_complete (bool, optional): [If True, then features with missing values during transform that did not have any missing values during fit will be imputed with the initial imputation method only. Set to True if you have many features with no missing values at both fit and transform time to save compute time]. Defaults to False.

			random_state (int, optional): [The seed of the pseudo random number generator to use for the iterative imputer. Randomizes selection of etimator features if n_nearest_features is not None or the imputation_order is 'random'. Use an integer for determinism. If None, then uses a different random seed each time]. Defaults to None.

			verbose (int, optional): [Verbosity flag, controls the debug messages that are issues as functions are evaluated. The higher, the more verbose. Possible values are 0, 1, or 2]. Defaults to 0.
		"""
		# Get local variables into dictionary object
		kwargs = locals()

		self.clf = ExtraTreesClassifier
		self.clf_type = "classifier"

		imputer = Impute(self.clf, self.clf_type, kwargs)

		self.imputed, self.best_score, self.best_params = \
			imputer.fit_predict(genotype_data.genotypes_df)

		imputer.write_imputed(self.imputed, self.best_score, self.best_params)

class ImputeGradientBoosting:
	"""[Class to impute missing data from 012-encoded genotypes using Random Forest iterative imputation]
	"""
	def __init__(
		self,
		genotype_data,
		*,
		prefix=None,
		gridparams=None,
		grid_iter=50,
		cv=5,
		n_jobs=1,
		n_estimators=100,
		loss="deviance",
		learning_rate=0.1,
		subsample=1.0,
		criterion="friedman_mse",
		min_samples_split=2,
		min_samples_leaf=1,
		min_weight_fraction_leaf=0.0,
		max_depth=3,
		min_impurity_decrease=0.0,
		max_features=None,
		max_leaf_nodes=None,
		clf_random_state=None,
		max_iter=10,
		tol=1e-3,
		n_nearest_features=10,
		initial_strategy="most_frequent",
		imputation_order="ascending",
		skip_complete=False,
		random_state=None,
		verbose=0
	):
		"""[Does Random Forest Iterative imputation of missing data. Iterative imputation uses the n_nearest_features to inform the imputation at each feature (i.e., SNP site), using the N most correlated features per site. The N most correlated features are drawn with probability proportional to correlation for each imputed target feature to ensure coverage of features throughout the imputation process]

		Args:
			genotype_data ([GenotypeData]): [GenotypeData instance that was used to read in the sequence data]

			prefix ([str]): [Prefix for imputed data's output filename]
			
			n_estimators (int, optional): [The number of boosting stages to perform. Gradient boosting is fairly robust to over-fitting so a large number usually results in better performance]. Defaults to 100.

			loss (str, optional): [The loss function to be optimized. ‘deviance’ refers to deviance (=logistic regression) for classification with probabilistic outputs. For loss ‘exponential’ gradient boosting recovers the AdaBoost algorithm]. Defaults to "deviance".

			learning_rate (float, optional): [Learning rate shrinks the contribution of each tree by learning_rate. There is a trade-off between learning_rate and n_estimators]. Defaults to 0.1.

			subsample (float, optional): [The fraction of samples to be used for fitting the individual base learners. If smaller than 1.0 this results in Stochastic Gradient Boosting. subsample interacts with the parameter n_estimators. Choosing subsample < 1.0 leads to a reduction of variance and an increase in bias]. Defaults to 1.0.

			criterion (str, optional): [The function to measure the quality of a split. Supported criteria are 'friedman_mse' for the mean squared error with improvement score by Friedman and 'mse' for mean squared error. The default value of 'friedman_mse' is generally the best as it can provide a better approximation in some cases]. Defaults to "friedman_mse".

			min_samples_split (int or float, optional): [The minimum number of samples required to split an internal node. If value is an integer, then consider min_samples_split as the minimum number. If value is a floating point, then min_samples_split is a fraction and (min_samples_split * n_samples) is rounded up to the nearest integer and used as the number of samples per split]. Defaults to 2.

			min_samples_leaf (int or float, optional): [The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches. This may have the effect of smoothing the model, especially in regression. If value is an integer, consider min_samples_leaf as the minimum number. If value is a floating point, then min_samples_leaf is a fraction and (min_samples_leaf * n_samples) rounded up to the nearest integer is the minimum number of samples per node]. Defaults to 1.

			min_weight_fraction_leaf (float, optional): [The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node. Samples have equal weight when sample_weight is not provided]. Defaults to 0.0.

			max_depth (int, optional): [The maximum depth of the individual regression estimators. The maximum depth limits the number of nodes in the tree. Tune this parameter for best performance; the best value depends on the interaction of the input variables.]. Defaults to 3.

			min_impurity_decrease (float, optional): [A node will be split if this split induces a decrease of the impurity greater than or equal to this value]. Defaults to 0.0. See sklearn.ensemble.GradientBoostingClassifier documentation for more information]. Defaults to 0.0.

			max_features (int, float, or str, optional): [The number of features to consider when looking for the best split. If value is an integer, then consider 'max_features' features at each split. If value is a floating point, then 'max_features' is a fraction and (max_features * n_features) is rounded to the nearest integer and considered as the number of features per split. If 'auto', then max_features=sqrt(n_features). If 'sqrt', then max_features=sqrt(n_features). If 'log2', then max_features=log2(n_features). If None, then max_features=n_features]. Defaults to None.

			max_leaf_nodes (int, optional): [Grow trees with 'max_leaf_nodes' in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then uses an unlimited number of leaf nodes]. Defaults to None.

			clf_random_state (int, optional): [Controls the random seed given to each Tree estimator at each boosting iteration. In addition, it controls the random permutation of the features at each split. Pass an int for reproducible output across multiple function calls. If None, then uses a different random seed for each function call]. Defaults to None.

			max_iter (int, optional): [Maximum number of imputation rounds to perform before returning the imputations computed during the final round. A round is a single imputation of each feature with missing values]. Defaults to 10.

			tol ([type], optional): [Tolerance of the stopping condition]. Defaults to 1e-3.

			n_nearest_features (int, optional): [Number of other features to use to estimate the missing values of eacah feature column. If None, then all features will be used, but this can consume an  intractable amount of computing resources. Nearness between features is measured using the absolute correlation coefficient between each feature pair (after initial imputation). To ensure coverage of features throughout the imputation process, the neighbor features are not necessarily nearest, but are drawn with probability proportional to correlation for each imputed target feature. Can provide significant speed-up when the number of features is huge]. Defaults to 10.

			initial_strategy (str, optional): [Which strategy to use to initialize the missing values. Same as the strategy parameter in sklearn.impute.SimpleImputer Valid values: {“mean”, “median”, “most_frequent”, or “constant”}]. Defaults to "most_frequent".

			imputation_order (str, optional): [The order in which the features will be imputed. Possible values: 'ascending' (from features with fewest missing values to most), 'descending' (from features with most missing values to fewest), 'roman' (left to right), 'arabic' (right to left), 'random' (a random order for each round). ]. Defaults to "ascending".

			skip_complete (bool, optional): [If True, then features with missing values during transform that did not have any missing values during fit will be imputed with the initial imputation method only. Set to True if you have many features with no missing values at both fit and transform time to save compute time]. Defaults to False.

			random_state (int, optional): [The seed of the pseudo random number generator to use for the iterative imputer. Randomizes selection of etimator features if n_nearest_features is not None or the imputation_order is 'random'. Use an integer for determinism. If None, then uses a different random seed each time]. Defaults to None.

			verbose (int, optional): [Verbosity flag, controls the debug messages that are issues as functions are evaluated. The higher, the more verbose. Possible values are 0, 1, or 2]. Defaults to 0.
		"""
		# Get local variables into dictionary object
		kwargs = locals()

		self.clf_type = "classifier"
		self.clf = GradientBoostingClassifier

		imputer = Impute(self.clf, self.clf_type, kwargs)

		self.imputed, self.best_score, self.best_params = \
			imputer.fit_predict(genotype_data.genotypes_df)

		imputer.write_imputed(self.imputed, self.best_score, self.best_params)

class ImputeBayesianRidge:
	"""[Class to impute missing data from 012-encoded genotypes using Bayesian ridge iterative imputation]
	"""
	def __init__(
		self,
		genotype_data,
		*,
		prefix=None,
		gridparams=None,
		grid_iter=50,
		cv=5,
		n_jobs=1,
		n_iter=300,
		clf_tol=1e-3,
		alpha_1=1e-6,
		alpha_2=1e-6,
		lambda_1=1e-6,
		lambda_2=1e-6,
		alpha_init=None,
		lambda_init=None,
		max_iter=10,
		tol=1e-3,
		n_nearest_features=10,
		initial_strategy="most_frequent",
		imputation_order="ascending",
		skip_complete=False,
		random_state=None,
		verbose=0
	):
		"""[Does Bayesian Ridge Iterative imputation of missing data. Iterative imputation uses the n_nearest_features to inform the imputation at each feature (i.e., SNP site), using the N most correlated features per site. The N most correlated features are drawn with probability proportional to correlation for each imputed target feature to ensure coverage of features throughout the imputation process]

		Args:
			genotype_data ([GenotypeData]): [GenotypeData instance that was used to read in the sequence data]	

			prefix ([str]): [Prefix for imputed data's output filename]		
				
			n_iter (int, optional): [Maximum number of iterations. Should be greater than or equal to 1]. Defaults to 300.

			clf_tol (float, optional): [Stop the algorithm if w has converged]. Defaults to 1e-3.

			alpha_1 (float, optional): [Hyper-parameter: shape parameter for the Gamma distribution prior over the alpha parameter]. Defaults to 1e-6.

			alpha_2 (float, optional): [Hyper-parameter: inverse scale parameter (rate parameter) for the Gamma distribution prior over the alpha parameter]. Defaults to 1e-6.

			lambda_1 (float, optional): [Hyper-parameter: shape parameter for the Gamma distribution prior over the lambda parameter]. Defaults to 1e-6.

			lambda_2 (float, optional): [Hyper-parameter: inverse scale parameter (rate parameter) for the Gamma distribution prior over the lambda parameter]. Defaults to 1e-6.

			alpha_init (float, optional): [Initial value for alpha (precision of the noise). If None, alpha_init is 1/Var(y).]. Defaults to None.

			lambda_init (float, optional): [Initial value for lambda (precision of the weights). If None, lambda_init is 1]. Defaults to None.

			max_iter (int, optional): [Maximum number of imputation rounds to perform before returning the imputations computed during the final round. A round is a single imputation of each feature with missing values]. Defaults to 10.

			tol ([type], optional): [Tolerance of the stopping condition]. Defaults to 1e-3.

			n_nearest_features (int, optional): [Number of other features to use to estimate the missing values of eacah feature column. If None, then all features will be used, but this can consume an  intractable amount of computing resources. Nearness between features is measured using the absolute correlation coefficient between each feature pair (after initial imputation). To ensure coverage of features throughout the imputation process, the neighbor features are not necessarily nearest, but are drawn with probability proportional to correlation for each imputed target feature. Can provide significant speed-up when the number of features is huge]. Defaults to 10.

			initial_strategy (str, optional): [Which strategy to use to initialize the missing values. Same as the strategy parameter in sklearn.impute.SimpleImputer Valid values: {“mean”, “median”, “most_frequent”, or “constant”}]. Defaults to "most_frequent".

			imputation_order (str, optional): [The order in which the features will be imputed. Possible values: 'ascending' (from features with fewest missing values to most), 'descending' (from features with most missing values to fewest), 'roman' (left to right), 'arabic' (right to left), 'random' (a random order for each round). ]. Defaults to "ascending".

			skip_complete (bool, optional): [If True, then features with missing values during transform that did not have any missing values during fit will be imputed with the initial imputation method only. Set to True if you have many features with no missing values at both fit and transform time to save compute time]. Defaults to False.

			random_state (int, optional): [The seed of the pseudo random number generator to use for the iterative imputer. Randomizes selection of etimator features if n_nearest_features is not None or the imputation_order is 'random'. Use an integer for determinism. If None, then uses a different random seed each time]. Defaults to None.

			verbose (int, optional): [Verbosity flag, controls the debug messages that are issues as functions are evaluated. The higher, the more verbose. Possible values are 0, 1, or 2]. Defaults to 0.
		"""
		# Get local variables into dictionary object
		kwargs = locals()
		kwargs["normalize"] = True
		kwargs["sample_posterior"] = False

		self.clf_type = "regressor"
		self.clf = BayesianRidge

		imputer = Impute(self.clf, self.clf_type, kwargs)

		self.imputed, self.best_score, self.best_params = \
			imputer.fit_predict(genotype_data.genotypes_df)

		imputer.write_imputed(self.imputed, self.best_score, self.best_params)

class ImputeXGBoost:

	def __init__(
		self, 
		genotype_data, 
		*, 
		prefix=None,
		gridparams=None,
		grid_iter=50,
		cv=5,
		n_jobs=1, 
		n_estimators=100, 
		max_depth=6, 
		learning_rate=0.1, 
		booster="gbtree", 
		tree_method="auto", 
		gamma=0, 
		min_child_weight=1, 
		max_delta_step=0, 
		subsample=1, 
		colsample_bytree=1, 
		reg_lambda=1, 
		reg_alpha=0, 
		objective="multi:softmax", 
		eval_metric="error",
		clf_random_state=None,
		n_nearest_features=10,
		max_iter=10,
		tol=1e-3,
		initial_strategy="most_frequent",
		imputation_order="ascending",
		skip_complete=False,
		random_state=None,
		verbose=0
	):

		# Get local variables into dictionary object
		kwargs = locals()
		kwargs["num_class"] = 3
		kwargs["use_label_encoder"] = True

		self.clf_type = "classifier"
		self.clf = XGBClassifier
		
		imputer = Impute(self.clf, self.clf_type, kwargs)

		self.imputed, self.best_score, self.best_params = \
			imputer.fit_predict(genotype_data.genotypes_df)

		imputer.write_imputed(self.imputed, self.best_score, self.best_params)

class ImputeAlleleFreq(GenotypeData):
	"""[Class to impute missing data by global or population-wisse allele frequency]

	Args:
		GenotypeData ([GenotypeData]): [Inherits from GenotypeData class that reads input data from a sequence file]
	"""
	def __init__(
		self, 
		genotype_data, 
		*, 
		pops=None, 
		diploid=True, 
		default=0, 
		missing=-9
	):
		"""[Impute missing data by global allele frequency. Population IDs can be sepcified with the pops argument. if pops is None, then imputation is by global allele frequency. If pops is not None, then imputation is by population-wise allele frequency. A list of population IDs in the appropriate format can be obtained from the GenotypeData object as GenotypeData.populations]

		Args:
			pops ([list(str)], optional): [If None, then imputes by global allele frequency. If not None, then imputes population-wise and pops should be a list of population assignments. The list of population assignments can be obtained from the GenotypeData object as GenotypeData.populations]. Defaults to None.

			diploid (bool, optional): [When diploid=True, function assumes 0=homozygous ref; 1=heterozygous; 2=homozygous alt. 0-1-2 genotypes are decomposed to compute p (=frequency of ref) and q (=frequency of alt). In this case, p and q alleles are sampled to generate either 0 (hom-p), 1 (het), or 2 (hom-q) genotypes. When diploid=FALSE, 0-1-2 are sampled according to their observed frequency]. Defaults to True.

			default (int, optional): [Value to set if no alleles sampled at a locus]. Defaults to 0.

			missing (int, optional): [Missing data value]. Defaults to -9.
		"""
		self.pops = pops
		self.diploid = diploid
		self.default = default
		self.missing = missing

		super().__init__()

		self.imputed = self.fit_predict(genotype_data.genotypes_list)

	@timer
	def fit_predict(self, X):
		"""[Impute missing genotypes using allele frequencies, with missing alleles coded as negative; usually -9]
		
		Args:
			X ([list(list(int))]): [012-encoded genotypes obtained from the GenotypeData object as GenotypeData.genotypes_list]

		Returns:
			[pandas.DataFrame]: [Imputed genotypes of same dimensions as data]
		"""
		if self.pops:
			print("\nImputing by population allele frequencies...")
		else:
			print("\nImputing by global allele frequency...")

		data = [item[:] for item in X]

		if self.pops is not None:
			pop_indices = misc.get_indices(self.pops)

		loc_index=0
		for locus in data:
			if self.pops is None:
				allele_probs = self._get_allele_probs(locus, self.diploid)
				#print(allele_probs)
				if misc.all_zero(list(allele_probs.values())) or \
					not allele_probs:
					print("\nWarning: No alleles sampled at locus", 
						str(loc_index), 
						"setting all values to:",str(self.default))
					gen_index=0
					for geno in locus:
						data[loc_index][gen_index] = self.default
						gen_index+=1

				else:
					gen_index=0
					for geno in locus:
						if geno == self.missing:
							data[loc_index][gen_index] = \
								self._sample_allele(allele_probs, diploid=True)
						gen_index += 1
						
			else:
				for pop in pop_indices.keys():
					allele_probs = self._get_allele_probs(
						locus, self.diploid, 
						missing=self.missing, 
						indices=pop_indices[pop]
					)

					if misc.all_zero(list(allele_probs.values())) or \
						not allele_probs:
						print("\nWarning: No alleles sampled at locus", 
							str(loc_index), 
							"setting all values to:", 
							str(self.default)
						)
						gen_index=0
						for geno in locus:
							data[loc_index][gen_index] = self.default
							gen_index += 1
					else:
						gen_index=0
						for geno in locus:
							if geno == self.missing:
								data[loc_index][gen_index] = \
									self._sample_allele(
										allele_probs, 
										diploid=True
									)
							gen_index += 1
					
			loc_index += 1

		df = pd.DataFrame(data)

		print("Done!")
		return df

	def _sample_allele(self, allele_probs, diploid=True):
		if diploid:
			alleles=misc.weighted_draw(allele_probs, 2)
			if alleles[0] == alleles[1]:
				return alleles[0]
			else:
				return 1
		else:
			return misc.weighted_draw(allele_probs, 1)[0]

	def _get_allele_probs(
		self, genotypes, diploid=True, missing=-9, indices=None
	):
		data=genotypes
		length=len(genotypes)
		
		if indices is not None:
			data = [genotypes[index] for index in indices]
			length = len(data)
		
		if len(set(data))==1:
			if data[0] == missing:
				ret=dict()
				return ret
			else:
				ret=dict()
				ret[data[0]] = 1.0
				return ret
		
		if diploid:
			length = length*2
			ret = {0:0.0, 2:0.0}
			for g in data:
				if g == 0:
					ret[0] += 2
				elif g == 2:
					ret[2] += 2
				elif g == 1:
					ret[0] += 1
					ret[2] += 1
				elif g == missing:
					length -= 2
				else:
					print("\nWarning: Ignoring unrecognized allele", 
						str(g), 
						"in get_allele_probs\n"
					)
			for allele in ret.keys():
				ret[allele] = ret[allele] / float(length)
			return ret
		else:
			ret=dict()
			for key in set(data):
				if key != missing:
					ret[key] = 0.0
			for g in data:
				if g == missing:
					length -= 1
				else:
					ret[g] += 1
			for allele in ret.keys():
				ret[allele] = ret[allele] / float(length)
			return ret
        
