# Standard library imports


# Third party imports
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import BayesianRidge
from sklearn.neighbors import KNeighborsClassifier

# Custom module imports
from utils import misc
from utils.misc import timer

class ImputeKNN(GenotypeData):
	"""[Class to impute missing data from 012-encoded genotypes using K-Nearest Neighbors iterative imputation]

	Args:
		GenotypeData ([GenotypeData]): [Inherits from GenotypeData class, which reads input sequence data into a useable object]
	"""
	def __init__(
		self, 
		genotype_data,
		*, 
		n_neighbors=5, 
		weights="distance", 
		algorithm="auto", 
		leaf_size=30, 
		p=2, 
		metric="minkowski", 
		n_jobs=1,
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
		self.n_neighbors = n_neighbors
		self.weights = weights
		self.algorithm = algorithm
		self.leaf_size = leaf_size
		self.p = p
		self.metric = metric
		self.n_jobs = n_jobs
		self.max_iter = max_iter
		self.tol = tol
		self.n_nearest_features = n_nearest_features
		self.initial_strategy = initial_strategy
		self.imputation_order = imputation_order
		self.skip_complete = skip_complete
		self.random_state = random_state
		self.verbose = verbose

		super().__init__()

		self.imputed = self.fit_predict(genotype_data.genotypes_df)

	@timer
	def fit_predict(self, X):
		"""[Do K-nearest neighbors imputation using an Iterative Imputer.
		Iterative imputer iterates over N other features (i.e. SNP sites)
		and uses each them to inform missing data predictions in the input column]

		Args:
			X ([pandas.DataFrame]): [012-encoded genotypes from GenotypeData]

		Returns:
			[numpy.ndarray]: [2-D numpy array with imputed genotypes]
		"""
		print("\nDoing K-nearest neighbor iterative imputation...\n")
		print(
			"\nK Neighbors Classifier Settings:\n"
			"\tn_neighbors: "+str(self.n_neighbors)+"\n"
			"\tweights: "+str(self.weights)+"\n"
			"\talgorithm: "+str(self.algorithm)+"\n"
			"\tleaf_size: "+str(self.leaf_size)+"\n"
			"\tpower: "+str(self.p)+"\n"
			"\tmetric: "+str(self.metric)+"\n"
			"\tn_jobs: "+str(self.n_jobs)+"\n"
			"\n"
			"Iterative Imputer Settings:\n"
			"\tn_nearest_features: "+str(self.n_nearest_features)+"\n"
			"\tmax_iter: "+str(self.max_iter)+"\n" 
			"\ttol: "+str(self.tol)+"\n"
			"\tinitial strategy: "+str(self.initial_strategy)+"\n"
			"\timputation_order: "+str(self.imputation_order)+"\n"
			"\tskip_complete: "+str(self.skip_complete)+"\n"
			"\trandom_state: "+str(self.random_state)+"\n" 
			"\tverbose: "+str(self.verbose)+"\n"
		)

		df = self._format_features(X)

		# Create iterative imputer
		imputed = IterativeImputer(
			estimator=KNeighborsClassifier(
								n_neighbors=self.n_neighbors,
								weights=self.weights,
								algorithm=self.algorithm,
								leaf_size=self.leaf_size,
								p=self.p,
								metric=self.metric,
								n_jobs=self.n_jobs,
						),
			n_nearest_features=self.n_neareast_features, 
			max_iter=self.max_iter, 
			tol=self.tol, 
			initial_strategy=self.initial_strategy,
			imputation_order=self.imputation_order,
			skip_complete=self.skip_complete,
			random_state=self.random_state, 
			verbose=self.verbose
		)

		arr = imputed.fit_transform(df)
		new_arr = arr.astype(dtype=np.int)
		new_df = pd.DataFrame(new_arr)

		print("\nDone!")

		return new_df

class ImputeRandomForest(GenotypeData):
	"""[Class to impute missing data from 012-encoded genotypes using Random Forest iterative imputation]

	Args:
		GenotypeData ([GenotypeData]): [Inherits from GenotypeData class used to read in input sequence data]
	"""
	def __init__(
		self, 
		genotype_data,
		*, 
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
		rf_random_state=None,
		n_jobs=1,
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

			rf_random_state (int, optional): [Controls three sources of randomness for sklearn.ensemble.ExtraTreesClassifier: The bootstrapping of the samples used when building trees (if bootstrap=True), the sampling of the features to consider when looking for the best split at each node (if max_features < n_features), and the draw of the splits for each of the max_features. If None, then uses a different random seed each time the imputation is run]. Defaults to None.

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
		self.n_estimators = n_estimators
		self.criterion = criterion
		self.max_depth = max_depth 
		self.min_samples_split = min_samples_split
		self.min_samples_leaf = min_samples_leaf 
		self.min_weight_fraction_leaf = min_weight_fraction_leaf
		self.max_features = max_features
		self.max_leaf_nodes = max_leaf_nodes
		self.min_impurity_decrease = min_impurity_decrease
		self.bootstrap = bootstrap
		self.oob_score = oob_score
		self.max_samples = self.max_samples
		self.rf_random_state = self.rf_random_state
		self.n_jobs = n_jobs
		self.max_iter = max_iter
		self.tol = tol
		self.n_nearest_features = n_neareast_features
		self.initial_strategy = initial_strategy
		self.imputation_order = imputation_order
		self.skip_complete = skip_complete
		self.random_state = random_state
		self.verbose = verbose

		super().__init__()

		self.imputed = self.fit_predict(genotype_data.genotypes_df)

	@timer
	def fit_predict(self, X):
		"""[Do random forest imputation using Iterative Imputer. Iterative imputer iterates over all the other features (columns) and uses each one as a target variable, thereby informing missingness in the input column]

		Args:
			X ([pandas.DataFrame]): [012-encoded genotypes from GenotypeData object]

		Returns:
			[pandas.DataFrame]: [DataFrame with imputed missing data]
		"""
		print("\nDoing random forest imputation...\n")

		print(
			"\nRandom Forest Classifier Settings:\n"
			"\tn_estimators: "+str(self.n_estimators)+"\n"
			"\tcriterion: "+str(self.criterion)+"\n"
			"\tmax_depth: "+str(self.max_depth)+"\n"
			"\tmin_samples_split: "+str(self.min_samples_split)+"\n"
			"\tmin_samples_leaf: "+str(self.min_samples_leaf)+"\n"
			"\tmin_weight_fraction_leaf: "+str(self.min_weight_fraction_leaf)+"\n"
			"\tmax_features: "+str(self.max_features)+"\n"
			"\tmax_leaf_nodes: "+str(self.max_leaf_nodes)+"\n"
			"\tmin_impurity_decrease: "+str(self.min_impurity_decrease)+"\n"
			"\tbootstrap: "+str(self.bootstrap)+"\n"
			"\toob_score: "+str(self.oob_score)+"\n"
			"\tmax_samples: "+str(self.max_samples)+"\n"
			"\trf_random_state: "+str(self.rf_random_state)+"\n"
			"\tn_jobs: "+str(self.n_jobs)+"\n"
			"\n"
			"\nIterative Imputer Settings:\n"
			"\tn_nearest_features: "+str(self.n_nearest_features)+"\n"
			"\tmax_iter: "+str(self.max_iter)+"\n" 
			"\ttol: "+str(self.tol)+"\n"
			"\tinitial strategy: "+str(self.initial_strategy)+"\n"
			"\timputation_order: "+str(self.imputation_order)+"\n"
			"\tskip_complete: "+str(self.skip_complete)+"\n"
			"\trandom_state: "+str(self.random_state)+"\n" 
			"\tverbose: "+str(self.verbose)+"\n"
		)

		df = self._format_features(X)

		# Create iterative imputer
		imputed = IterativeImputer(
			estimator=ExtraTreesClassifier(
				n_estimators=self.n_estimators,
				criterion=self.criterion,
				max_depth=self.max_depth,
				min_samples_split=self.min_samples_split,
				min_samples_leaf=self.min_samples_leaf, 
				min_weight_fraction_leaf=self.min_weight_fraction_leaf,
				max_features=self.max_features,
				max_leaf_nodes=self.max_leaf_nodes,
				min_impurity_decrease=self.min_impurity_decrease,
				bootstrap=self.bootstrap,
				oob_score=self.oob_score,
				max_samples=self.max_samples,
				random_state=self.rf_random_state,
				n_jobs=self.n_jobs),
			n_nearest_features=self.n_nearest_features, 
			max_iter=self.max_iter, 
			tol=self.tol, 
			initial_strategy=self.initial_strategy,
			imputation_order=self.imputation_order,
			skip_complete=self.skip_complete,
			random_state=self.random_state, 
			verbose=self.verbose
		)

		arr = imputed.fit_transform(df)
		new_arr = arr.astype(dtype=np.int)
		new_df = pd.DataFrame(new_arr)

		print("\nDone!")

		return new_df

class ImputeGradientBoosting(GenotypeData):
	"""[Class to impute missing data from 012-encoded genotypes using Random Forest iterative imputation]

	Args:
		GenotypeData ([GenotypeData]): [Inherits from GenotypeData class used to read in sequence data]
	"""
	def __init__(
		self,
		genotype_data,
		*,
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
		gb_random_state=None,
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

			gb_random_state (int, optional): [Controls the random seed given to each Tree estimator at each boosting iteration. In addition, it controls the random permutation of the features at each split. Pass an int for reproducible output across multiple function calls. If None, then uses a different random seed for each function call]. Defaults to None.

			max_iter (int, optional): [Maximum number of imputation rounds to perform before returning the imputations computed during the final round. A round is a single imputation of each feature with missing values]. Defaults to 10.

			tol ([type], optional): [Tolerance of the stopping condition]. Defaults to 1e-3.

			n_nearest_features (int, optional): [Number of other features to use to estimate the missing values of eacah feature column. If None, then all features will be used, but this can consume an  intractable amount of computing resources. Nearness between features is measured using the absolute correlation coefficient between each feature pair (after initial imputation). To ensure coverage of features throughout the imputation process, the neighbor features are not necessarily nearest, but are drawn with probability proportional to correlation for each imputed target feature. Can provide significant speed-up when the number of features is huge]. Defaults to 10.

			initial_strategy (str, optional): [Which strategy to use to initialize the missing values. Same as the strategy parameter in sklearn.impute.SimpleImputer Valid values: {“mean”, “median”, “most_frequent”, or “constant”}]. Defaults to "most_frequent".

			imputation_order (str, optional): [The order in which the features will be imputed. Possible values: 'ascending' (from features with fewest missing values to most), 'descending' (from features with most missing values to fewest), 'roman' (left to right), 'arabic' (right to left), 'random' (a random order for each round). ]. Defaults to "ascending".

			skip_complete (bool, optional): [If True, then features with missing values during transform that did not have any missing values during fit will be imputed with the initial imputation method only. Set to True if you have many features with no missing values at both fit and transform time to save compute time]. Defaults to False.

			random_state (int, optional): [The seed of the pseudo random number generator to use for the iterative imputer. Randomizes selection of etimator features if n_nearest_features is not None or the imputation_order is 'random'. Use an integer for determinism. If None, then uses a different random seed each time]. Defaults to None.

			verbose (int, optional): [Verbosity flag, controls the debug messages that are issues as functions are evaluated. The higher, the more verbose. Possible values are 0, 1, or 2]. Defaults to 0.
		"""
		self.n_estimators = n_estimators
		self.loss = loss
		self.learning_rate = learning_rate
		self.subsample = subsample
		self.criterion = criterion
		self.min_samples_split = min_samples_split
		self.min_samples_leaf = min_samples_leaf
		self.min_weight_fraction_leaf = min_weight_fraction_leaf
		self.max_depth = max_depth
		self.min_impurity_decrease = min_impurity_decrease
		self.max_features = max_features
		self.max_leaf_nodes = max_leaf_nodes
		self.gb_random_state = gb_random_state
		self.max_iter = max_iter
		self.tol = tol
		self.n_nearest_features = n_nearest_features
		self.initial_strategy = initial_strategy
		self.imputation_order = imputation_order
		self.skip_complete = skip_complete
		self.random_state = random_state
		self.verbose = verbose

		super().__init__()

		self.imputed = self.fit_predict(genotype_data.genotypes_df)

	@timer
	def fit_predict(self, X):
		"""[Do gradient boosting iterative imputation. Iterative imputer iterates over all the other features (columns)	and uses each one as a target variable, thereby informing missingness in the input column]

		Args:
			X (pandas.DataFrame): [DataFrame of 012-encoded genotypes from GenotypeData object]

		Returns:
			[pandas.DataFrame]: [DataFrame with imputed 012-encoded genotypes]
		"""
		print("\nDoing gradient boosting iterative imputation...\n")
		print(
			"\nGradient Boosting Classifier Settings:\n"
			"\tn_estimators: "+str(self.n_estimators)+"\n"
			"\tloss: "+str(self.loss)+"\n"
			"\tlearning_rate: "+str(self.learning_rate)+"\n"
			"\tsubsample: "+str(self.subsample)+"\n"
			"\tcriterion: "+str(self.criterion)+"\n"
			"\tmin_samples_split: "+str(self.min_samples_split)+"\n"
			"\tmin_samples_leaf: "+str(self.min_samples_leaf)+"\n"
			"\tmin_weight_fraction_leaf: "+str(self.min_weight_fraction_leaf)+"\n"
			"\tmax_depth: "+str(self.max_depth)+"\n"
			"\tmin_impurity_decrease: "+str(self.min_impurity_decrease)+"\n"
			"\tmax_features: "+str(self.max_features)+"\n"
			"\tmax_leaf_nodes: "+str(self.max_leaf_nodes)+"\n"
			"\tgb_random_state: "+str(self.gb_random_state)+"\n"
			"\n"
			"Iterative Imputer Settings\n:
			"\tn_nearest_features: "+str(self.n_nearest_features)+"\n"
			"\tmax_iter: "+str(self.max_iter)+"\n" 
			"\ttol: "+str(self.tol)+"\n"
			"\tinitial strategy: "+str(self.initial_strategy)+"\n"
			"\timputation_order: "+str(self.imputation_order)+"\n"
			"\tskip_complete: "+str(self.skip_complete)+"\n"
			"\trandom_state: "+str(self.random_state)+"\n" 
			"\tverbose: "+str(self.verbose)+"\n"
		)

		df = self._format_features(X)

		"\tn_estimators: "+str(self.n_estimators)+"\n"
		"\tloss: "+str(self.loss)+"\n"
		"\tlearning_rate: "+str(self.learning_rate)+"\n"
		"\tsubsample: "+str(self.subsample)+"\n"
		"\tcriterion: "+str(self.criterion)+"\n"
		"\tmin_samples_split: "+str(self.min_samples_split)+"\n"
		"\tmin_samples_leaf: "+str(self.min_samples_leaf)+"\n"
		"\tmin_weight_fraction_leaf: "+str(self.min_weight_fraction_leaf)+"\n"
		"\tmax_depth: "+str(self.max_depth)+"\n"
		"\tmin_impurity_decrease: "+str(self.min_impurity_decrease)+"\n"
		"\tmax_features: "+str(self.max_features)+"\n"
		"\tmax_leaf_nodes: "+str(self.max_leaf_nodes)+"\n"
		"\tgb_random_state: "+str(self.gb_random_state)+"\n"

		# Create iterative imputer
		imputed = IterativeImputer(
			estimator=GradientBoostingClassifier(
				n_estimators=self.n_estimators, 
				loss=self.loss,
				learning_rate=self.learning_rate,
				subsample=self.subsample,
				criterion=self.criterion,
				min_samples_split=self.min_samples_split,
				min_samples_leaf=self.min_samples_leaf,
				min_weight_fraction_leaf=self.min_weight_fraction_leaf,
				max_depth=self.max_depth,
				min_impurity_decrease=self.min_impurity_decrease,
				max_features=self.max_features,
				max_leaf_nodes=self.max_leaf_nodes,
				random_state=self.gb_random_state
			),
			n_nearest_features=self.n_nearest_features, 
			max_iter=self.max_iter, 
			tol=self.tol, 
			initial_strategy=self.initial_strategy,
			imputation_order=self.imputation_order,
			random_state=self.random_state, 
			verbose=self.verbose
		)

		arr = imputed.fit_transform(df)
		new_arr = arr.astype(dtype=np.int)
		new_df = pd.DataFrame(new_arr)

		print("\nDone!")

		return new_df


class ImputeBayesianRidge(GenotypeData):
	"""[Class to impute missing data from 012-encoded genotypes using Bayesian ridge iterative imputation]

	Args:
		GenotypeData ([GenotypeData]): [Inherits from GenotypeData class used to read in sequence data]
	"""
	def __init__(
		self,
		genotype_data,
		*,
		n_iter=300,
		br_tol=1e-3,
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
				
			n_iter (int, optional): [Maximum number of iterations. Should be greater than or equal to 1]. Defaults to 300.

			br_tol (float, optional): [Stop the algorithm if w has converged]. Defaults to 1e-3.

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
		self.n_iter = n_iter,
		self.br_tol = br_tol,
		self.alpha_1 = alpha_1,
		self.alpha_2 = alpha_2,
		self.lambda_1 = lambda_1,
		self.lambda_2 = lambda_2,
		self.alpha_init = alpha_init,
		self.lambda_init = lambda_init,
		self.max_iter = max_iter,
		self.tol = tol,
		self.n_nearest_features = n_nearest_features,
		self.initial_strategy = initial_strategy,
		self.imputation_order = imputation_order,
		self.skip_complete = skip_complete,
		self.random_state = random_state,
		self.verbose = verbose

		self.normalize = True
		self.sample_posterior = True

		super().__init__()

		self.imputed = self.fit_predict(genotype_data.genotypes_df)

	@timer
	def fit_predict(self, X):
		"""[Do bayesian ridge imputation using Iterative Imputer.
		Iterative imputer iterates over all the other features (columns)
		and uses each one as a target variable, thereby informing missingness
		in the input column]

		Args:
			X ([pandas.DataFrame]): [DataFrame 012-encoded genotypes from GenotypeData instance]

		Returns:
			[pandas.DataFrame]: [DataFrame with 012-encoded imputed genotypes of shape(n_samples, n_features)]
		"""
		print("\nDoing Bayesian ridge iterative imputation...\n")
		print(
			"\nBayesian Ridge Regression Settings:\n"
			"\tn_iter: "+str(self.n_iter)+"\n"
			"\tbr_tol: "+str(self.br_tol)+"\n"
			"\talpha_1: "+str(self.alpha_1)+"\n"
			"\talpha_2: "+str(self.alpha_2)+"\n"
			"\tlambda_1: "+str(self.lambda_1)+"\n"
			"\tlambda_2: "+str(self.lambda_2)+"\n"
			"\talpha_init: "+str(self.alpha_init)+"\n"
			"\tlambda_init: "+str(self.lambda_init)+"\n"
			"\n"
			"Iterative Imputer Settings:\n"
			"\tn_nearest_features: "+str(self.n_nearest_features)+"\n"
			"\tmax_iter: "+str(self.max_iter)+"\n" 
			"\ttol: "+str(self.tol)+"\n"
			"\tinitial strategy: "+str(self.initial_strategy)+"\n"
			"\timputation_order: "+str(self.imputation_order)+"\n"
			"\tskip_complete: "+str(self.skip_complete)+"\n"
			"\trandom_state: "+str(self.random_state)+"\n" 
			"\tverbose: "+str(self.verbose)+"\n"
		)

		df = self._format_features(X)

		# Create iterative imputer
		imputed = IterativeImputer(
			estimator=BayesianRidge(
				n_iter=self.n_iter,
				tol=self.br_tol,
				alpha_1=self.alpha_1,
				alpha_2=self.alpha_2,
				lambda_1=self.lambda_1,
				lambda_2=self.lambda_2,
				alpha_init=self.alpha_init,
				lambda_init=self.lambda_init,
				normalize=self.normalize
			),
			n_nearest_features=self.n_nearest_features, 
			max_iter=self.max_iter, 
			tol=self.tol, 
			initial_strategy=self.initial_strategy,
			imputation_order=self.imputation_order,
			skip_complete=self.skip_complete,
			random_state=self.random_state, 
			verbose=self.verbose,
			sample_posterior=self.sample_posterior
		)

		arr = imputed.fit_transform(df)
		new_arr = arr.astype(dtype=np.int)
		new_df = pd.DataFrame(new_arr)

		print("\nDone!")

		return new_df

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
					#print(pop,"--",allele_probs)
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
