# Standard library imports
import sys

# Make sure python version is >= 3.6
if sys.version_info < (3, 6):
	raise ImportError("Python < 3.6 is not supported!")

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from kneed import KneeLocator
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.manifold import MDS
from sklearn.manifold import TSNE

# Custom imports
from dim_reduction.dim_reduction import DimReduction
from utils.misc import timer
from utils.misc import isnotebook

is_notebook = isnotebook()

if is_notebook:
	from tqdm.notebook import tqdm as progressbar
else:
	from tqdm import tqdm as progressbar

#from utils.misc import progressbar

class runPCA(DimReduction):
	"""[Class to run a principal component analysis]

	Args:
		DimReduction ([DimReduction]): [Inherits from DimReduction]
	"""
	def __init__(
		self, 
		*,
		dimreduction=None, 
		gt=None, 
		pops=None,
		sampleids=None,
		prefix=None, 
		reps=None, 
		scaler=None,
		colors=None, 
		palette="Set1", 
		keep_pcs=10, 
		plot_cumvar=False, 
		show_cumvar=False,
		elbow=False, 
		pc_var=None, 
		**kwargs
	):
		"""[Run a principal component analysis. The PCA axes can be plotted by calling runPCA.plot(). The PCA plot can be 2D or 3D. If dimreduction=None, gt, pops, sampleids, prefix, reps, and [colors or palette] must be supplied, and vice versa]

		Args:
			dimreduction ([DimReduction], optional): [Previous DimReduction instance. Allows a single initialization of DimReduction to be used for multiple embedding methods]. Defaults to None.

			gt ([pandas.DataFrame, numpy.ndarray, or list(list(int))], optional): [One of the imputed objects returned from GenotypeData of shape (n_samples, n_sites)]. Defaults to None.

			pops ([list(str)], optional): [Population IDs of shape (n_samples,). Can be retrieved as GenotypeData.populations]. Defaults to None.

			prefix ([str], optional): [Prefix for output files and plots]. Defaults to None.

			reps ([int], optional): [Number of replicates to perform. If None, 1 replicate is performed]. Defaults to None.

			scaler (str, optional): [Scaler to use for genotype data. Valid options include "patterson", "standard", and "center". "patterson" follows a Patterson standardization (Patterson, Price & Reich (2006)) recommended for genetic data that scales the data to unit variance at each SNP. "standard" does regular standardization to unit variance and centers the data. "center" does not do a standardization and just centers the data]. Defaults to "patterson".

			colors ([dict(str: str)], optional): [Dictionary with unique population IDs as the keys and hex-code colors as the corresponding values. If None, then uses the matplotlib palette supplied to palette]. Defaults to None.

			palette (str, optional): [matplotlib palette to use if colors=None]. Defaults to "Set1".

			keep_pcs (int, optional): [Number of principal components to retain]. Defaults to 10.

			plot_cumvar (bool, optional): [If True, save the PCA cumulative variance plot to disk]. Defaults to False.

			show_cumvar (bool, optional): [If True, shows the PCA cumulative variance plot in an active window]. Defaults to False.

			elbow (bool, optional): [If True, uses the elbow of the PCA cumulative variance plot to determine the number of principal components to retain. Cannot be set at same time as 'pc_var' argument]. Defaults to False.

			pc_var ([float], optional): [If not None, should be a value between 0 and 1 to retain percentage of the principal components. Cannot be used at same time as 'elbow' argument]. Defaults to None.

			**kwargs (optional): [Keyword arguments used for plot settings. Options include: text_size (default=14), style (default="white"), figwidth (default=6), figheight (default=6), cumvar_figwidth (default=6), cumvar_figheight (default=6), cumvar_linecolor (defaults="blue"), cumvar_linewidth (default=3), cumvar_xintercept_width (default=3), cumvar_xintercept_color (default="r"), cumvar_xintercept_style (default="--"), cumvar_style (default="white"), cumvar_text_size (default=14)]

		Raises:
			TypeError: [Unexpected keyword argument provided to **kwargs]
		"""
		# Initialize parent class
		super().__init__(gt, pops, sampleids, prefix, reps, scaler, colors, palette)

		# Validates passed arguments and sets parent class attributes
		self._validate_args(dimreduction, gt, pops, sampleids, prefix, reps, scaler)

		self.scale_gt(self.gt)

		# Child class attributes
		self.keep_pcs = keep_pcs
		self.scaler = scaler
		self.coords = list()
		self._pca_model = list()
		self.method = "PCA"

		# PCA scatterplot settings
		text_size = kwargs.pop("text_size", 14)
		style = kwargs.pop("style", "white")
		figwidth = kwargs.pop("figwidth", 6)
		figheight = kwargs.pop("figheight", 6)

		# PCA Cumulative Variance plot settings
		cumvar_figwidth = kwargs.pop("cumvar_figwidth", 6)
		cumvar_figheight = kwargs.pop("cumvar_figheight", 6)
		cumvar_linecolor = kwargs.pop("cumvar_linecolor", "blue")
		cumvar_linewidth = kwargs.pop("cumvar_linewidth", 3)
		cumvar_xintercept_width = kwargs.pop("cumvar_xintercept_width", 3)
		cumvar_xintercept_color = kwargs.pop("cumvar_xintercept_color", "r")
		cumvar_xintercept_style = kwargs.pop("cumvar_xintercept_style", "--")
		cumvar_style = kwargs.pop("cumvar_style", "white")
		cumvar_text_size = kwargs.pop("cumvar_text_size", 14)

		# If still items left in kwargs, then unrecognized keyword argument
		if kwargs:
			raise TypeError(
				"Unexpected keyword arguments provided: {}".format(
					list(kwargs.keys())
				)
			)


		self.coords, self._pca_model = self.fit_transform(self.gt, self.keep_pcs)

		if plot_cumvar:
			best_pcs = self._plot_pca_cumvar(
				self.coords, 
				self._pca_model, 
				prefix, 
				self.reps,
				show_cumvar,
				elbow, 
				pc_var, 
				cumvar_figwidth, 
				cumvar_figheight, 
				cumvar_linecolor, 
				cumvar_linewidth, 
				cumvar_xintercept_width, 
				cumvar_xintercept_color, 
				cumvar_xintercept_style, 
				cumvar_style, 
				cumvar_text_size
			)

			self.coords, self._pca_model = self.fit_transform(self.gt, best_pcs)

	@timer
	def fit_transform(self, X, pcs):
		"""[Does principal component analysis on 012-encoded genotypes. By default uses a Patterson scaler to standardize, but you can use two other scalers: 'standard' and 'center'. 'standard' centers and standardizes the data, whereas 'center' just centers it and doesn't standardize it. PCA is performed reps times, with the pca_coords and pca_model objects being stored as a list of embedded coordinates and a list of PCA models]

		Args:
			X ([pandas.DataFrame or numpy.ndarray]): [012-encoded genotypes]

			pcs ([list(int) or int): [Number of components to retain]

		Returns:
			coords_list ([list(numpy.ndarray)]): [List of PCA coordinate objects, with each item in the list corresponding to one replicate]

			model_list ([list(sklearn.decomposition.PCA)]): [List of PCA model objects, with each item in the list corresponding to one replicate]
		"""
		print("\nDoing {}...\n".format(self.method))
		print(
			"\nPCA Settings:\n"
			"\tn_components: """+str(pcs)+"\n"
			"\tscaler: """+str(self.scaler)+"\n"
		)

		model_list = list()
		coords_list = list()
		for rep in progressbar(
			range(self.reps),
			desc="{} Replicates: ".format(self.method), 
			leave=True,
			position=0
		):

			if isinstance(pcs, list):
				pca = PCA(n_components=pcs[rep])
			else:
				pca = PCA(n_components=pcs)

			model = pca.fit(X)
			model_list.append(model)
			coords_list.append(model.transform(X))

		return coords_list, model_list

	def _plot_pca_cumvar(
		self, 
		coords, 
		model, 
		prefix, 
		reps,
		show_cumvar,
		elbow, 
		pc_var, 
		figwidth, 
		figheight, 
		linecolor, 
		linewidth, 
		xintercept_width, 
		xintercept_color, 
		xintercept_style, 
		style, 
		text_size
	):
		"""[Plot cumulative variance for principal components with xintercept line at the inflection point]

		Args:
			coords ([numpy.ndarray]): [PCA coordinates to plot of shape (n_features, n_components)]

			model ([sklearn.decomposition.PCA]): [PCA model with corresponding attributes]

			prefix ([str]): [Prefix for output plots]

			reps ([int]): [Number of replicates to perform]

			elbow ([bool]): [If True, uses inflection point (elbow) to determine number of principal components to retain]

			pc_var ([bool]): [If True, uses a proportion between 0 and 1 to determine the number of principal components to retain]

			figwidth ([int]): [Width of figure in inches] 
			figheight ([int]): [Height of figure in inches] 
			linecolor ([str]): [Color of line plot]
			linewidth ([int]): [Width of line]
			xintercept_width ([int]): [Width of x-intercept line] 
			xintercept_color ([int]): [Color of x-intercept line]
			xintercept_style ([str]): [Style of x-intercept line]
			style ([str]): [Plot style set with seaborn] 
			text_size ([int]): [Font size for plot text]

		Raises:
			AttributeError: [pca_model must be defined prior to running this function]
		"""
		knees = list()
		for rep in range(reps):

			# Raise error if PCA hasn't been run yet
			if coords[rep] is None:
				raise AttributeError("\nA PCA coordinates are not defined!")

			if elbow is True and pc_var is not None:
				raise ValueError("elbow and pc_var cannot both be specified!")

			if pc_var is not None:
				assert isinstance(pc_var, float), "pc_var must be of type float"
				assert pc_var <= 1.0, "pc_var must be a value between 0.0 and 1.0"
				assert pc_var > 0.0, "pc_var must be a value greater than 0.0"

			if not elbow and pc_var is None:
				raise TypeError("plot_cumvar was set to True but either elbow or " "pc_var were not set")

			# Get the cumulative variance of the principal components
			cumsum = np.cumsum(model[rep].explained_variance_ratio_)

			# From the kneed package
			# Gets the knee/ elbow of the curve
			kneedle = KneeLocator(
				range(1, coords[rep].shape[1]+1), 
				cumsum, 
				curve="concave", 
				direction="increasing"
			)

			# Sets plot background style
			# Uses seaborn
			sns.set(style=style)

			# Plot the results
			# Uses matplotlib.pyplot
			fig = plt.figure(figsize=(figwidth, figheight))
			ax = fig.add_subplot(1,1,1)


			if elbow:
				# Plot the explained variance ratio
				ax.plot(kneedle.y, color=linecolor, linewidth=linewidth)

				# Add text to show inflection point
				ax.text(
					0.95, 
					0.01, 
					"Knee={}".format(kneedle.knee), 
					verticalalignment="bottom", 
					horizontalalignment="right", 
					transform=ax.transAxes, 
					color="k", 
					fontsize=text_size
				)

				knee = kneedle.knee
				knees.append(knee)

			if pc_var is not None:
				for idx, item in enumerate(cumsum):
					if cumsum[idx] < pc_var:
						knee = idx + 1
					else:
						break

				ax.plot(cumsum, color=linecolor, linewidth=linewidth)

				ax.text(
					0.95, 
					0.01, 
					"# PCs: {}".format(knee), 
					verticalalignment="bottom", 
					horizontalalignment="right", 
					transform=ax.transAxes, 
					color="k", 
					fontsize=text_size
				)

			ax.axvline(
				linewidth=xintercept_width, 
				color=xintercept_color, 
				linestyle=xintercept_style, 
				x=knee, 
				ymin=0, 
				ymax=1
			)
			
			# Set axis labels
			ax.set_xlabel("Number of Components")
			ax.set_ylabel("Cumulative Explained Variance")

			# Add prefix to filename
			plot_fn = "{}_pca_cumvar_{}.pdf".format(prefix, rep+1)

			# Save as PDF file
			fig.savefig(plot_fn, bbox_inches="tight")

			if show_cumvar:
				plt.show()

			plt.clf()
			plt.close(fig)

		# Returns number of principal components at elbow
		return knees

	@property
	def	pca_coords(self):
		"""[Getter for PCA coordinates]

		Raises:
			AttributeError: [Must have executed runPCA() first]

		Returns:
			[list(numpy.ndarray)]: [list of PCA coordinates, with each item in the list corresponding to one replicate's coordinates of shape (n_samples, n_components)]
		"""
		if self.coords is not None:
			return self.coords
		else:
			raise AttributeError("pca_coodinates attribute is not yet defined")

	@property
	def pca_model(self):
		"""[PCA model with corresponding sklearn attributes]

		Raises:
			AttributeError: [PCA must have been executed with runPCA()]

		Returns:
			[list(sklearn.decomposition.PCA)]: [Fit PCA model, with each item in the list being the PCA model for one replicate]
		"""
		if self._pca_model is not None:
			return self._pca_model
		else:
			raise AttributeError("pca_model_object is not yet defined")


class runRandomForestUML(DimReduction):
	"""[Class to do unsupervised random forest embedding]

	Args:
		DimReduction ([DimReduction]): [Inherits from DimReduction]
	"""

	def __init__(
		self, 
		*,
		dimreduction=None, 
		gt=None, 
		pops=None,
		sampleids=None,
		prefix=None, 
		reps=None, 
		scaler=None,
		colors=None, 
		palette="Set1", 
		pca=None, 
		n_estimators=100, 
		max_depth=5, 
		min_samples_split=2, 
		min_samples_leaf=1, 
		min_weight_fraction_leaf=0.0, 
		max_leaf_nodes=None, 
		min_impurity_decrease=0.0, 
		min_impurity_split=None, 
		sparse_output=True, 
		n_jobs=1, 
		random_state=None, 
		verbose=0, 
		warm_start=False
	):
		"""[Run unsupervised random forest embedding on 012-encoded genotypes. Calculates RF embedded coordinates, a proximity matrix, and a dissimilarity matrix. Data must have been imputed using one of the impute methods. Either a DimReduction object must be supplied to the dimreduction argument or gt, pops, sampleids, prefix, reps, and [colors or palette] must all be supplied]

		Args:
			dimreduction ([DimReduction], optional): [Previous DimReduction instance. Allows a single initialization of DimReduction to be used for multiple embedding methods]. Defaults to None.

			gt ([pandas.DataFrame, numpy.ndarray, or list(list(int))], optional): [One of the imputed objects returned from GenotypeData of shape (n_samples, n_sites)]. Defaults to None.

			pops ([list(str)], optional): [Population IDs of shape (n_samples,). Can be retrieved as GenotypeData.populations]. Defaults to None.

			prefix ([str], optional): [Prefix for output files and plots]. Defaults to None.

			reps ([int], optional): [Number of replicates to perform. If None, 1 replicate is performed]. Defaults to None.

			scaler (str, optional): [Scaler to use for genotype data. Valid options include "patterson", "standard", and "center". "patterson" follows a Patterson standardization (Patterson, Price & Reich (2006)) recommended for genetic data that scales the data to unit variance at each SNP. "standard" does regular standardization to unit variance and centers the data. "center" does not do a standardization and just centers the data]. Defaults to "patterson".

			colors ([dict(str: str)], optional): [Dictionary with unique population IDs as the keys and hex-code colors as the corresponding values. If None, then uses the matplotlib palette supplied to palette]. Defaults to None.

			palette (str, optional): [matplotlib palette to use if colors=None]. Defaults to "Set1".

			reps ([int], optional): [Number of replicates to perform]. Defaults to None.

			pca ([numpy.ndarray], optional): [If not None, runs random forest on the supplied PCA coordinates of shape (n_samples, n_components). Can be retrieved as runPCA.pca_coords]. Defaults to None.

			n_estimators (int, optional): [Number of decision trees in the forest]. Defaults to 100.

			max_depth (int, optional): [The maximum depth of each tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples]. Defaults to 5.
			
			min_samples_split (int or float, optional): [The minimum number of samples required to split an internal node. If int, then consier min_samples_split as the minimum number. If float, then min_samples_split is a fraction and ceil(min_samples_split * n_samples) is the minimum number of samples for each split]. Defaults to 2.

			min_samples_leaf (int or float, optional): [The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches. If int, then consider min_samples_leaf as the minimum number. If float, then min_samples_leaf is a fraction and ceil(min_samples_leaf * n_samples) is the minimum number of samples for each node]. Defaults to 1.

			min_weight_fraction_leaf (float, optional): [The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node. Samples have equal weight when sample_weight is nor provided]. Defaults to 0.0.

			max_leaf_nodes ([int], optional): [Grow trees with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes]. Defaults to None.

			min_impurity_decrease (float, optional): [A node will be split if this split induces a decrease of the impurity greater than or equal to this value. See sklearn.ensemble.RandomTreesEmbedding for further documentation]. Defaults to 0.0.

			sparse_output (bool, optional): [Whether or not to return a sparse CSR matrix, as default behavior, or to return a dense array compatible with dense pipeline operators]. Defaults to True.

			n_jobs ([int], optional): [Number of jobs to run in parallel. None means 1 unless in a joblib.parallel_backend context. -1 means use all processors]. Defaults to 1.

			random_state ([int], optional): [Controls the generation of random y used to fit the trees and the draw of the splits for each features at the trees' nodes. See sklearn Glossary for details]. Defaults to None.

			verbose (int, optional): [Controls the verbosity when fitting and predicting 0=low verbosity]. Defaults to 0.
		"""

		# Initialize parent class
		super().__init__(gt, pops, sampleids, prefix, reps, scaler, colors, palette)

		# Validates passed arguments and sets parent class attributes
		self._validate_args(dimreduction, gt, pops, sampleids, prefix, reps, scaler)

		self.scale_gt(self.gt)

		# Set class attributes
		self.pca = pca
		self.n_estimators = n_estimators
		self.max_depth = max_depth
		self.min_samples_split = min_samples_split
		self.min_samples_leaf = min_samples_leaf
		self.min_weight_fraction_leaf = min_weight_fraction_leaf
		self.max_leaf_nodes = max_leaf_nodes
		self.min_impurity_decrease = min_impurity_decrease
		self.sparse_output = sparse_output
		self.n_jobs = n_jobs
		self.random_state = random_state
		self.verbose = verbose

		self.rf = list()
		self._prox_matrix = list()
		self._diss_matrix = list()

		if self.pca is None:
			self.rf, self._prox_matrix, self._diss_matrix = \
				self.fit_transform(self.gt, self.pca)
		else:
			self.rf, self._prox_matrix, self._diss_matrix = \
				self.fit_transform(self.pca.coords, self.pca)


	@timer
	def fit_transform(self, X, pca):
		"""[Do unsupervised random forest embedding. Saves an rf model, a proximity score matrix, and a dissimilarity score matrix for each replicate]

		Args:
			X ([numpy.ndarray or pandas.DataFrame]): [Data to model of shape (n_samples, n_features). Can be principal components or 012-encoded genotypes]

			pca ([bool]): [True if input is embedded principal component coordinates, False if input is 012-eoncded genotypes]

		Returns:
			rf_list ([list(sklearn.ensemble.RandomTreesEmbeedding)]): [List of random forest models, with each replicate corresponding to one item in the list]

			prox_mat_list ([list(numpy.ndarray)]): [List of proximity score matrices, with each matrix being corresponding to one item in the list]

			diss_mat_list ([list(numpy.ndarray)]): [List of dissimilarity score matrices, with each matrix corresponding to one item in the list]
		"""
		# Print settings to command-line
		print("\nDoing random forest unsupervised embedding...")
		print(
		"\nRandom Forest Embedding Settings:\n"
		"\tn_estimators: "+str(self.n_estimators)+"\n"
		"\tmax_depth: "+str(self.max_depth)+"\n"
		"\tmin_samples_split: "+str(self.min_samples_split)+"\n"
		"\tmin_samples_leaf: "+str(self.min_samples_leaf)+"\n"
		"\tmin_weight_fraction_leaf: "+str(self.min_weight_fraction_leaf)+"\n"
		"\tmax_leaf_nodes: "+str(self.max_leaf_nodes)+"\n"
		"\tmin_impurity_decrease: "+str(self.min_impurity_decrease)+"\n"
		"\tsparse_output: "+str(self.sparse_output)+"\n"
		"\tn_jobs: "+str(self.n_jobs)+"\n"
		"\trandom_state: "+str(self.random_state)+"\n"
		"\tverbose: "+str(self.verbose)+"\n"
		)

		prox_mat_list = list()
		diss_mat_list = list()
		rf_list = list()
		for rep in progressbar(
			range(self.reps), 
			desc="RF Embedding Replicates: ",
			leave=True,
			position=0):

			clf = None
			rf_model = None
			_rf = None

			# Initialize a random forest
			clf = RandomTreesEmbedding(
				n_estimators=self.n_estimators,
				max_depth=self.max_depth,
				min_samples_split=self.min_samples_split,
				min_samples_leaf=self.min_samples_leaf,
				min_weight_fraction_leaf=self.min_weight_fraction_leaf,
				max_leaf_nodes=self.max_leaf_nodes,
				min_impurity_decrease=self.min_impurity_decrease,
				sparse_output=self.sparse_output,
				n_jobs=self.n_jobs,
				random_state=self.random_state,
				verbose=self.verbose
			)

			if pca:
				_X = X[rep]
			else:
				_X = X.copy()

			# Fit the model
			clf.fit(_X)

			# transform and return the model
			rf_model = clf.transform(_X)

			# Cast it to a numpy.ndarray
			_rf = rf_model.toarray()
			rf_list.append(_rf)

			_prox_matrix, _diss_matrix = \
				self._calculate_rf_proximity_dissimilarity_mat(
					_X, 
					clf, 
					self.n_estimators)

			prox_mat_list.append(_prox_matrix)
			diss_mat_list.append(_diss_matrix)


		return rf_list, prox_mat_list, diss_mat_list

	def _calculate_rf_proximity_dissimilarity_mat(self, X, clf, n_trees):
		"""[Calculate random forest proximity scores and a dissimilarity matrix from a fit sklearn.ensemble.RandomTreesEmbeddings model. Should be applied after model fitting but not on the transformed data. X is the modeled data]

		Args:
			X ([numpy.ndarray or pandas.DataFrame]): [The features to modeled with RandomTreesEmbeddings]

			clf ([RandomTreesEmbeddings object]): [The classifier returned from initializing RandomForestEmbeddings()]

			n_trees ([int]): [Number of estimator trees used in the RandomTreesEmbedding]

		Returns:
			[numpy.ndarray]: [Proximity score matrix and dissimilarity score matrix]
		"""
		# Apply trees in the forest to X, return leaf indices
		# This allows calculation of the proximity scores
		leaves = clf.apply(X)

		# Initialize proximity matrix
		prox = np.zeros((len(X), len(X)))
				
		# adapted implementation found here: 
		# https://stackoverflow.com/questions/18703136/proximity-matrix-in-sklearn-ensemble-randomforestclassifier

		# Generates proximity scores
		for tree_idx in range(n_trees):
			prox += np.equal.outer(
				leaves[:,tree_idx],
				leaves[:,tree_idx]
			).astype(float)
				
		# Divide by the number of estimators to normalize
		prox /= n_trees

		# Calculate dissimilarity matrix
		# Subtracts 1 from each element in numpy.ndarray
		diss = 1 - prox

		# Convert them to pandas.DataFrame objects
		diss = pd.DataFrame(diss)
		prox = pd.DataFrame(prox)

		return prox, diss

	@property
	def rf_model(self):
		"""[Getter for the random forest fit model that can be input into runMDS()]

		Raises:
			AttributeError: [If runRandomForestUML() has not yet been executed]

		Returns:
			[list(sklearn.ensemble.RandomTreesEmbedding)]: [The fit random forest model, with each list item being the model for one replicate]
		"""
		if self.rf is not None:
			return self.rf
		else:
			raise AttributeError("rf_model object is not yet defined")

	@property
	def proximity(self):
		"""[Getter for the proximity matrix]

		Raises:
			AttributeError: [If runRandomForestUML has not yet been executed]

		Returns:
			[list(numpy.ndarray)]: [Pairwise proximity scores between the points, with each list item=1 replicate of shape (n_samples, n_samples)]
		"""
		if self._prox_matrix is not None:
			return self._prox_matrix
		else:
			raise AttributeError("proximity_matrix object is not yet defined")

	@property
	def dissimilarity(self):
		"""[Getter for the random forest dissimilarity matrix that can be input into runMDS()]

		Raises:
			AttributeError: [if runRandomForestUML() has not yet been executed]

		Returns:
			[list(numpy.ndarray)]: [Pairwise dissimilarity scores between the points, with each list item=1 replicate of shape (n_samples, n_samples)]
		"""
		if self._diss_matrix is not None:
			return self._diss_matrix
		else:
			raise AttributeError("proximity object is not yet " "defined")

class runMDS(DimReduction):
	"""[Class to perform multi-dimensional scaling on 012-encoded genotypes or runRandomForestUML() output]

	Args:
		DimReduction ([DimReduction]): [Inherits from DimReduction]
	"""

	def __init__(
		self, 
		*,
		dimreduction=None, 
		distances=None,
		gt=None, 
		pops=None,
		sampleids=None,
		prefix=None, 
		reps=None,
		scaler=None, 
		colors=None, 
		palette="Set1", 
		rf=None, 
		metric=True, 
		keep_dims=2, 
		n_init=4, 
		max_iter=300, 
		eps=1e-3, 
		n_jobs=1, 
		verbose=0, 
		random_state=None
	):
		"""[Run multi-dimensional scaling on output 012-encoded genotypes or runRandomForest() high-dimensional output. If using runRandomForest() output, either the random forest embeddeding or the random forest dissimilarity matrix can be accessed as runRandomForestUML.dissimilarity to get distances. If using the random forest output, euclidean distances are used. Use the metric argument to either perform classical MDS (metric=True) or isotonic MDS (metric=False). >2 dimensions can be retained, and output can be plotted up to 3 dimensions by calling runMDS.plot()]

		Args:
			dimreduction ([DimReduction], optional): [Previous DimReduction instance. Allows a single initialization of DimReduction to be used for multiple embedding methods]. Defaults to None.

			distances ([numpy.ndarray], optional): [Pairwise distance matrix of shape (n_samples, n_samples). Can be any kind of distance matrix or the dissimilarity scores between points from runRandomForestUML() that can be accessed as runRandomForestUML.dissimilarity. Either distances or rf must be supplied. If using rf, euclidean distances are used]. Defaults to None.

			gt ([pandas.DataFrame, numpy.ndarray, or list(list(int))], optional): [One of the imputed objects returned from GenotypeData of shape (n_samples, n_sites)]. Defaults to None.

			pops ([list(str)], optional): [Population IDs of shape (n_samples,). Can be retrieved as GenotypeData.populations]. Defaults to None.

			prefix ([str], optional): [Prefix for output files and plots]. Defaults to None.

			reps ([int], optional): [Number of replicates to perform. If None, 1 replicate is performed]. Defaults to None.

			scaler (str, optional): [Scaler to use for genotype data. Valid options include "patterson", "standard", and "center". "patterson" follows a Patterson standardization (Patterson, Price & Reich (2006)) recommended for genetic data that scales the data to unit variance at each SNP. "standard" does regular standardization to unit variance and centers the data. "center" does not do a standardization and just centers the data]. Defaults to "patterson".

			colors ([dict(str: str)], optional): [Dictionary with unique population IDs as the keys and hex-code colors as the corresponding values. If None, then uses the matplotlib palette supplied to palette]. Defaults to None.

			palette (str, optional): [matplotlib palette to use if colors=None]. Defaults to "Set1".

			reps ([int], optional): [Number of replicates to perform]. Defaults to None.

			rf ([runRandomForestUML], optional): [runRandomForestUML.rf_model object. If not supplied, you must supply the dissimilarity matrix. Either rf or dissimilarity matrix must be supplied]. Defaults to None.

			metric (bool, optional): [If True, does classical multi-dimensional scaling. If False, does isotonic (non-metric) MDS]. Defaults to True.

			keep_dims (int, optional): [Number of MDS dimensions to retain]. Defaults to 2.

			n_init (int, optional): [Number of times the SMACOF algorithm will be run with different initializations. The final results will be the best output of the runs, determined by the run with the smallest final stress. See sklearn.manifold.MDS documentation]. Defaults to 4.

			max_iter (int, optional): [Maximum number of iterations of the SMACOF algorithm for a single run. See sklearn.manifold.MDS documentation]. Defaults to 300.

			eps ([float], optional): [Relative tolerance with respect to stress at which to declare convergence]. Defaults to 1e-3.

			n_jobs ([int], optional): [The number of jobs to use for the computation. If multiple initializations are used (n_init), each run of the algorithm is computed in parallel. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors. See sklearn.manifold.MDS documentation]. Defaults to 1.

			verbose (int, optional): [Level of verbosity. 0=less verbose]. Defaults to 0.

			random_state ([int], optional): [Determines the random number generator used to initialize the centers. Pass an int for reproducible results across multiple function calls or set to None for random initializations each time]. Defaults to None.

		Raises:
			TypeError: [Either rf or distances must be defined]
			TypeError: [rf and distances cannot both be defined]
		"""

		# Initialize parent class
		super().__init__(gt, pops, sampleids, prefix, reps, scaler, colors, palette)

		# Validates passed arguments and sets parent class attributes
		self._validate_args(dimreduction, gt, pops, sampleids, prefix, reps, scaler)

		self.metric = metric
		self.rf = rf
		self.distances = distances
		self.n_dims = keep_dims
		self.n_init = n_init
		self.max_iter = max_iter
		self.eps = eps
		self.n_jobs = n_jobs
		self.verbose = verbose
		self.random_state = random_state
		self.method = None

		if self.rf is None and self.distances is None:
			raise TypeError("Either rf or dissimilarity matrix must be "
				"defined!")

		if self.rf is not None and self.distances is not None:
			raise TypeError("rf and dissimilarity matrix cannot both be "
				"defined!")

		if self.rf is not None:
			self.coords = self.fit_transform(self.rf)

		elif self.distances is not None:
			self.coords = self.fit_transform(self.distances)

	@timer
	def fit_transform(self, X):
		"""[Do MDS embedding for X over multiple replicates if reps > 1. Stores the coordinates as a list of numpy.ndarrays]

		Args:
			X ([numpy.ndarray or pandas.DataFrame]): [Either runRandomForestUML.rf_model, runRandomForestUML.dissimilarity, or another pairwise distance matrix of shape (n_samples, n_samples)]

		Returns:
			[list(numpy.ndarray)]: [List of coordinates with each item as the coordinates of shape (n_samples, n_features) for one replicate]
		"""
	
		if self.metric:
			print("\nDoing cMDS dimensionality reduction...\n")
			self.method = "cMDS"
		else:
			print("\nDoing isoMDS dimensionality reduction...\n")
			self.method = "isoMDS"
				

		if self.distances is None:
			self.dissimilarity = "euclidean"
		else:
			self.dissimilarity = "precomputed"

		print(
			"MDS Settings:\n"
				"\tmetric "+str(self.metric)+"\n"
				"\tn_dims: "+str(self.n_dims)+"\n"
				"\tn_init: "+str(self.n_init)+"\n"
				"\tmax_iter: "+str(self.max_iter)+"\n"
				"\teps: "+str(self.eps)+"\n"
				"\tn_jobs: "+str(self.n_jobs)+"\n"
				"\tdissimilarity: "+str(self.dissimilarity)+"\n"
				"\trandom_state: "+str(self.random_state)+"\n"
				"\tverbose: "+str(self.verbose)+"\n"
		)

		coords_list = list()
		for rep in progressbar(
			range(self.reps), 
			desc="{} Replicates".format(self.method),
			leave=True,
			position=0
		):
			mds = None
			_coords = None
			mds = MDS(
				n_components=self.n_dims, 
				random_state=self.random_state, 
				metric=self.metric,
				n_init=self.n_init,
				max_iter=self.max_iter,
				verbose=self.verbose,
				eps=self.eps,
				n_jobs=self.n_jobs,
				dissimilarity=self.dissimilarity
			)

			_coords = mds.fit_transform(X[rep])
			coords_list.append(_coords)

		return coords_list

class runTSNE(DimReduction):
	"""[Class to do t-SNE embedding on 012-encoded genotypes]

	Args:
		DimReduction ([DimReduction]): [Inherits from DimReduction]
	"""

	def __init__(
		self, 
		*,
		dimreduction=None, 
		distances=None,
		gt=None, 
		pops=None,
		sampleids=None,
		prefix=None, 
		reps=None, 
		scaler=None,
		colors=None, 
		palette="Set1", 
		keep_dims=2, 
		perplexity=30.0, 
		early_exaggeration=12.0, 
		learning_rate=200.0, 
		n_iter=1000, 
		n_iter_without_progress=300, 
		min_grad_norm=1e-7, 
		init="random", 
		verbose=0, 
		random_state=None, 
		method="barnes_hut", 
		angle=0.5, 
		n_jobs=1, 
		metric="euclidean"
	):
		"""[Run t-SNE embedding on output 012-encoded genotypes. >2 dimensions can be retained, and output can be plotted up to 3 dimensions by calling runTSNE.plot()]

		Args:
			dimreduction ([DimReduction], optional): [Previous DimReduction instance. Allows a single initialization of DimReduction to be used for multiple embedding methods]. Defaults to None.

			distances (numpy.ndarray): [Pairwise distance matrix of shape (n_samples, n_samples). Any of the distance metrics from scipy.spatial.distance.pdist or pairwise.PAIRWISE_DISTANCE_FUNCTIONS can be used. Can also be dissimilarity scores retrieved as runRandomForestUML.dissimilarity. If None, then uses euclidean distances for the distance matrix]. Defaults to None.

			gt ([pandas.DataFrame, numpy.ndarray, or list(list(int))], optional): [One of the imputed objects returned from GenotypeData of shape (n_samples, n_sites)]. Defaults to None.

			pops ([list(str)], optional): [Population IDs of shape (n_samples,). Can be retrieved as GenotypeData.populations]. Defaults to None.

			prefix ([str], optional): [Prefix for output files and plots]. Defaults to None.

			reps ([int], optional): [Number of replicates to perform. If None, 1 replicate is performed]. Defaults to None.

			scaler (str, optional): [Scaler to use for genotype data. Valid options include "patterson", "standard", and "center". "patterson" follows a Patterson standardization (Patterson, Price & Reich (2006)) recommended for genetic data that scales the data to unit variance at each SNP. "standard" does regular standardization to unit variance and centers the data. "center" does not do a standardization and just centers the data]. Defaults to "patterson".

			colors ([dict(str: str)], optional): [Dictionary with unique population IDs as the keys and hex-code colors as the corresponding values. If None, then uses the matplotlib palette supplied to palette]. Defaults to None.

			palette (str, optional): [matplotlib palette to use if colors=None]. Defaults to "Set1".

			reps ([int], optional): [Number of replicates to perform]. Defaults to None.

			keep_dims (int, optional): [Number of dimensions to retain]. Defaults to 2.

			perplexity (float, optional): [THe perplexity is related to the number of nearest neighbors. Larger datasets usually require a larger perplexity. Consider selecting a value between 5 and 50. Different values can result in significantly different results. See sklearn.manifold.TSNE documentation]. Defaults to 30.0.

			early_exaggeration (float, optional): [Controls how tight natural clusters in the original space are in the embedded space and how much space will be between them. For larger values, the space between natural clusters will be larger in the embedded space. The choice of this parameter is not very critical. If the cost funtion increases during initial optimization, the early exaggeration factor or the learning rate might be too high. See sklearn.manifold.TSNE documentation]. Defaults to 12.0.

			learning_rate (float, optional): [The learning rate for t-SNE is usually in the range [10.0, 1000.0]. If the larning rate is too high, the data may look like a "ball" with any point approximately equidistant from its nearest neighbors. If the learning rate is too low, most points may look compressed in a dense cloud with few outliers. If the cost function gets stuck in a bad local minimum increasing the learning rate may help]. Defaults to 200.0.

			n_iter (int, optional): [Maximum number of iterations for the optimization. Should be at least 250]. Defaults to 1000.

			n_iter_without_progress (int, optional): [Maximum number of iterations without progress before we abort the optimization, used after 250 initial iterations with early exaggeration. Note that progress is only checked every 50 iterations so this value is rounded to the next multiple of 50]. Defaults to 300.

			min_grad_norm ([float], optional): [If the gradient norm is below this threshold, the optimization will be stopped]. Defaults to 1e-7.

			init (str, optional): [Initialization of embedding. Possible options are "random", "pca", and a numpy array of shape (n_samples, n_components). PCA initialization cannot be used with precomputed disatnces and is usually more globally stable than random initialization]. Defaults to "random".

			verbose (int, optional): [Verbosity level. 0=less verbose]. Defaults to 0.

			random_state ([int], optional): [Determines the random number generator. Pass an int for reproducible results across multiple function call, or pass None for random initialization seeds]. Defaults to None.

			method (str, optional): [By default the gradient calculation algorithm uses Barnes-Hut approximation running in O(NlogN) time. method=’exact’ will run on the slower, but exact, algorithm in O(N^2) time. The exact algorithm should be used when nearest-neighbor errors need to be better than 3%. However, the exact method cannot scale to millions of examples]. Defaults to "barnes_hut".

			angle (float, optional): [Only used if method=’barnes_hut’ This is the trade-off between speed and accuracy for Barnes-Hut T-SNE. ‘angle’ is the angular size (referred to as theta in [3]) of a distant node as measured from a point. If this size is below ‘angle’ then it is used as a summary node of all points contained within it. This method is not very sensitive to changes in this parameter in the range of 0.2 - 0.8. Angle less than 0.2 has quickly increasing computation time and angle greater 0.8 has quickly increasing error.]. Defaults to 0.5.

			n_jobs (int, optional): [The number of parallel jobs to run for neighbors search. This parameter has no impact when metric="precomputed" or (metric="euclidean" and method="exact"). None means 1 unless in a joblib.parallel_backend context. -1 means using all processors]. Defaults to 1.

			metric (str, optional): [The metric to use when calculating distance between instances in a feature array. Ignored if distances parameter is not None. If metric is a string, it must be one of the options allowed by scipy.spatial.distance.pdist for its metric parameter, or a metric listed in pairwise.PAIRWISE_DISTANCE_FUNCTIONS. If metric is “precomputed”, X is assumed to be a distance matrix. Alternatively, if metric is a callable function, it is called on each pair of instances (rows) and the resulting value recorded. The callable should take two arrays from X as input and return a value indicating the distance between them. The default is “euclidean” which is interpreted as squared euclidean distance]. Defaults to "euclidean".
		"""

		# Initialize parent class
		super().__init__(gt, pops, sampleids, prefix, reps, scaler, colors, palette)

		# Validates passed arguments and sets parent class attributes
		self._validate_args(dimreduction, gt, pops, sampleids, prefix, reps, scaler)

		self.distances = distances
		self.n_components = keep_dims
		self.perplexity = perplexity
		self.early_exaggeration = early_exaggeration
		self.learning_rate = learning_rate
		self.n_iter = n_iter
		self.n_iter_without_progress = n_iter_without_progress
		self.min_grad_norm = min_grad_norm
		self.metric = metric
		self.init = init
		self.verbose = verbose
		self.random_state = random_state
		self.tsne_method = method
		self.angle = angle
		self.n_jobs = n_jobs
		self.method = "t-SNE"

		if distances is not None:
			self.metric = "precomputed"
			self.coords = self.fit_transform(self.distances)
		else:
			self.coords = self.fit_transform(self.gt)

	@timer
	def fit_transform(self, X):
		"""[Do t-SNE embedding for X over multiple replicates if reps > 1. Stores the coordinates as a list of numpy.ndarrays]

		Args:
			X ([numpy.ndarray or pandas.DataFrame]): [012-encoded genotypes retrived from GenotypeData]

		Returns:
			[list(numpy.ndarray)]: [List of coordinates with each item as the coordinates of shape (n_samples, n_features) for one replicate]
		"""
		print("\nDoing T-SNE embedding...\n")

		print(
			"T-SNE Settings:\n"
				"\tn_components: "+str(self.n_components)+"\n"
				"\tperplexity: "+str(self.perplexity)+"\n"
				"\tearly exaggeration: "+str(self.early_exaggeration)+"\n"
				"\tlearning_rate: "+str(self.learning_rate)+"\n"
				"\tn_iter: "+str(self.n_iter)+"\n"
				"\tmin_grad_norm: "+str(self.min_grad_norm)+"\n"
				"\tmetric: "+str(self.metric)+"\n"
				"\tinit: "+str(self.init)+"\n"
				"\tverbose: "+str(self.verbose)+"\n"
				"\trandom_state: "+str(self.random_state)+"\n"
				"\tmethod: "+str(self.tsne_method)+"\n"
				"\tangle: "+str(self.angle)+"\n"
				"\tn_jobs: "+str(self.n_jobs)+"\n"
		)

		coords_list = list()
		for rep in progressbar(
			range(self.reps), 
			desc="t-SNE Replicates: ",
			leave=True,
			position=0):
			t = None
			_coords = None

			t = TSNE(
				n_components=self.n_components,
				perplexity=self.perplexity,
				early_exaggeration=self.early_exaggeration,
				learning_rate=self.learning_rate,
				n_iter=self.n_iter,
				n_iter_without_progress=self.n_iter_without_progress,
				min_grad_norm=self.min_grad_norm,
				metric=self.metric,
				init=self.init,
				verbose=self.verbose,
				random_state=self.random_state,
				method=self.tsne_method,
				angle=self.angle,
				n_jobs=self.n_jobs,
				square_distances=True
			)

			if isinstance(X, list):
				_coords = t.fit_transform(X[rep].values)
				coords_list.append(_coords)
			else:
				_coords = t.fit_transform(X.values)
				coords_list.append(_coords)

		return coords_list