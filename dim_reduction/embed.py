import sys

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from kneed import KneeLocator
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.manifold import MDS
from sklearn.manifold import TSNE

# Custom imports
from dim_reduction.dim_reduction2 import DimReduction
from utils.misc import timer

class runPCA(DimReduction):

	def __init__(self, dimreduction=None, gt=None, pops=None, prefix=None, colors=None, palette="Set1", keep_pcs=10, scaler="patterson", plot_cumvar=False, elbow=False, pc_var=None, **kwargs):

		# Initialize parent class
		super().__init__(gt, pops, prefix, colors, palette)

		# Validates passed arguments and sets parent class attributes
		self._validate_args(dimreduction, gt, pops, prefix)

		# Child class attributes
		self.keep_pcs = keep_pcs
		self.scaler = scaler
		self.coords = None
		self._pca_model = None
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
			raise TypeError("Unexpected keyword arguments provided: {}".format(list(kwargs.keys())))


		self.coords, self._pca_model = self.fit_transform()

		if plot_cumvar:
			self.keep_pcs = self._plot_pca_cumvar(self.coords, self._pca_model, prefix, elbow, pc_var, cumvar_figwidth, cumvar_figheight, cumvar_linecolor, cumvar_linewidth, cumvar_xintercept_width, cumvar_xintercept_color, cumvar_xintercept_style, cumvar_style, cumvar_text_size)

			self.coords, self._pca_model = self.fit_transform()

	@timer
	def fit_transform(self):
		"""[Does principal component analysis on 012-encoded genotypes. By default uses a Patterson scaler to standardize, but you can use two other scalers: 'standard' and 'center'. 'standard' centers and standardizes the data, whereas 'center' just centers it and doesn't standardize it]

		Args:
			data ([numpy.ndarray]): [012-encoded genotypes of shape (n_samples, n_features)]

			pca_arguments ([dict]): [Dictionary with option names as keys and the settings as values]
		"""
		print("\nDoing PCA...\n")
		print(
				"\nPCA Settings:\n"
				"\tn_components: """+str(self.keep_pcs)+"\n"
				"\tscaler: """+str(self.scaler)+"\n"
		)

		# Scale and center the data
		if self.scaler == "patterson":
			std = FunctionTransformer(self.patterson_scaler)
			X = std.fit_transform(self.gt)

		elif self.scaler == "standard":
			std = FunctionTransformer(self.standard_scaler)
			X = std.fit_transform(self.gt)

		elif self.scaler == "center":
			std = FunctionTransformer(self.center_scaler)
			X = std.fit_transform(self.gt)

		else:
			raise ValueError("Unsupported scaler argument provided: {}".format(self.scaler))

		pca = PCA(n_components=self.keep_pcs)

		model = pca.fit(X)
		coords = model.transform(X)

		print("\nDone!")

		return coords, model

	def patterson_scaler(self, data):
		"""[Patterson scaler for PCA. Basically the formula for calculating the unit variance per SNP site is: std = np.sqrt((mean / ploidy) * (1 - (mean / ploidy))). Then center the data by subtracting the mean and scale it by dividing by the unit variance per SNP site.]

		Args:
			data ([numpy.ndarray]): [012-encoded genotypes to transform of shape (n_samples, n_features)]

		Returns:
			[numpy.ndarray]: [Transformed data, centered and scaled with Patterson scaler]
		"""
		# Make sure type is np.ndarray
		if not isinstance(data, np.ndarray):
			try:
				data = np.asarray(data)
			except:
				raise TypeError("Input genotype data cannot be converted to a numpy.ndarray")
		
		# Find mean of each site
		mean = np.mean(data, axis=0, keepdims=True)

		# Super Deli only supports ploidy of 2 currently
		ploidy = 2

		# Do Patterson scaling
		# Ploidy is 2
		p = mean / ploidy
		std = np.sqrt(p * (1 - p))

		# Make sure dtype is np.float64
		data = data.astype(np.float64)

		# Center the data
		data -= mean

		# Scale the data using the Patterson scaler
		data /= std

		return data

	def standard_scaler(self, data):
		"""[Center and standardize data]

		Args:
			data ([numpy.ndarray]): [012-encoded genotypes to transform of shape (n_samples, n_features)]

		Returns:
			[numpy.ndarray]: [Transformed data, centered and standardized]
		"""
		# Make sure data is a numpy.ndarray
		if not isinstance(data, np.ndarray):
			data = np.asarray(data)
		
		# Make sure dtype is np.float64
		data = data.astype(np.float64)

		# Center and standardize the data
		mean = np.mean(data, axis=0, keepdims=True)
		std = np.std(data, axis=0, keepdims=True)
		data -= mean
		data /= std

		return data

	def center_scaler(self, data):
		"""[Just center the data; don't standardize]

		Args:
			data ([numpy.ndarray]): [012-encoded genoytpes to transform of shape (n_samples, n_features)]

		Returns:
			[numpy.ndarray]: [Centered 012-encoded genotypes; not standardized]
		"""
		if not isinstance(data, np.ndarray):
			data = np.asarray(data)

		# Make sure type is np.float64
		data = data.astype(np.float64)

		# Center the data
		mean = np.mean(data, axis=0, keepdims=True)
		data -= mean
		return data

	def _plot_pca_cumvar(self, coords, model, prefix, elbow, pc_var, figwidth, figheight, linecolor, linewidth, xintercept_width, xintercept_color, xintercept_style, style, text_size):
		"""[Plot cumulative variance for principal components with xintercept line at the inflection point]

		Args:
			coords ([numpy.ndarray]): [PCA coordinates as 2D array of shape (n_samples, n_components)]

			model ([sklearn.decomposision.PCA model object]): [PCA model returned from scikit-allel. See scikit-allel documentation]

			prefix ([str]): [Prefix for output PDF filename]

			user_settings ([dict]): [Dictionary with matplotlib arguments as keys and their corresponding values. Only some or all of the settings can be changed]

		Raises:
			AttributeError: [pca_model must be defined prior to running this function]
		"""
		# Raise error if PCA hasn't been run yet
		if coords is None:
			raise AttributeError("\nA PCA coordinates are not defined!")

		if elbow is True and pc_var is not None:
			raise ValueError("elbow and pc_var cannot both be specified!")

		if pc_var is not None:
			assert isinstance(pc_var, float), "pc_var must be of type float"
			assert pc_var <= 1.0, "pc_var must be a value between 0.0 and 1.0"
			assert pc_var > 0.0, "pc_var must be a value greater than 0.0"

		if not elbow and pc_var is None:
			raise TypeError("plot_cumvar was set to True but either elbow or pc_var were not set")

		# Get the cumulative variance of the principal components
		cumsum = np.cumsum(model.explained_variance_ratio_)

		# From the kneed package
		# Gets the knee/ elbow of the curve
		kneedle = KneeLocator(range(1, coords.shape[1]+1), cumsum, curve="concave", direction="increasing")

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
			ax.text(0.95, 0.01, "Knee={}".format(kneedle.knee), verticalalignment="bottom", horizontalalignment="right", transform=ax.transAxes, color="k", fontsize=text_size)

			knee = kneedle.knee

		if pc_var is not None:
			for idx, item in enumerate(cumsum):
				if cumsum[idx] < pc_var:
					knee = idx + 1
				else:
					break

			ax.plot(cumsum, color=linecolor, linewidth=linewidth)

			ax.text(0.95, 0.01, "# PCs: {}".format(knee), verticalalignment="bottom", horizontalalignment="right", transform=ax.transAxes, color="k", fontsize=text_size)

		ax.axvline(linewidth=xintercept_width, color=xintercept_color, linestyle=xintercept_style, x=knee, ymin=0, ymax=1)
		
		# Set axis labels
		ax.set_xlabel("Number of Components")
		ax.set_ylabel("Cumulative Explained Variance")

		# Add prefix to filename
		plot_fn = "{}_pca_cumvar.pdf".format(prefix)

		# Save as PDF file
		fig.savefig(plot_fn, bbox_inches="tight")

		# Returns number of principal components at elbow
		return knee

	@property
	def	pca_coords(self):
		if self.coords is not None:
			return self.coords
		else:
			raise AttributeError("pca_coodinates attribute is not yet defined")

	@property
	def pca_model(self):
		if self._pca_model is not None:
			return self._pca_model
		else:
			raise AttributeError("pca_model_object is not yet defined")


class runRandomForestUML(DimReduction):

	def __init__(self, dimreduction=None, gt=None, pops=None, prefix=None, colors=None, palette="Set1", pca=None, n_estimators=100, max_depth=5, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, sparse_output=True, n_jobs=None, random_state=None, verbose=0, warm_start=False):

		# Initialize parent class
		super().__init__(gt, pops, prefix, colors, palette)

		# Validates passed arguments and sets parent class attributes
		self._validate_args(dimreduction, gt, pops, prefix)

		# Set class attributes
		self.pca = pca
		self.n_estimators = n_estimators
		self.max_depth = max_depth
		self.min_samples_split = min_samples_split
		self.min_samples_leaf = min_samples_leaf
		self.min_weight_fraction_leaf = min_weight_fraction_leaf
		self.max_leaf_nodes = max_leaf_nodes
		self.min_impurity_decrease = min_impurity_decrease
		self.min_impurity_split = min_impurity_split
		self.sparse_output = sparse_output
		self.n_jobs = n_jobs
		self.random_state = random_state
		self.verbose = verbose
		self.warm_start = warm_start

		self.rf = None
		self._prox_matrix = None
		self._diss_matrix = None

		if pca is None:
			self.rf, self._prox_matrix, self._diss_matrix = self.fit_transform(self.gt)
		else:
			self.rf, self._prox_matrix, self._diss_matrix = self.fit_transform(self.pca)


	@timer
	def fit_transform(self, X):
		"""[Do unsupervised random forest embedding. Saves an rf model and a proximity score matrix]

		Args:
			X ([numpy.ndarray or pandas.DataFrame]): [Data to model. Can be principal components or 012-encoded genotypes]

			rf_settings_default ([dict]): [RandomTreesEmbedding settings with argument names as keys and their corresponding values]
		"""
		# Print settings to command-line
		print("\nDoing random forest unsupervised...")
		print(
		"\nRandom Forest Unsupervised Settings:\n"
			"\tn_estimators: "+str(self.n_estimators)+"\n"
			"\tmax_depth: "+str(self.max_depth)+"\n"
			"\tmin_samples_split: "+str(self.min_samples_split)+"\n"
			"\tmin_samples_leaf: "+str(self.min_samples_leaf)+"\n"
			"\tmin_weight_fraction_leaf: "+str(self.min_weight_fraction_leaf)+"\n"
			"\tmax_leaf_nodes: "+str(self.max_leaf_nodes)+"\n"
			"\tmin_impurity_decrease: "+str(self.min_impurity_decrease)+"\n"
			"\tmin_impurity_split: "+str(self.min_impurity_split)+"\n"
			"\tsparse_output: "+str(self.sparse_output)+"\n"
			"\tn_jobs: "+str(self.n_jobs)+"\n"
			"\trandom_state: "+str(self.random_state)+"\n"
			"\tverbose: "+str(self.verbose)+"\n"
			"\twarm_start: "+str(self.warm_start)+"\n"
		)

		# Initialize a random forest
		clf = RandomTreesEmbedding(
			n_estimators=self.n_estimators,
			max_depth=self.max_depth,
			min_samples_split=self.min_samples_split,
			min_samples_leaf=self.min_samples_leaf,
			min_weight_fraction_leaf=self.min_weight_fraction_leaf,
			max_leaf_nodes=self.max_leaf_nodes,
			min_impurity_decrease=self.min_impurity_decrease,
			min_impurity_split=self.min_impurity_split,
			sparse_output=self.sparse_output,
			n_jobs=self.n_jobs,
			random_state=self.random_state,
			verbose=self.verbose,
			warm_start=self.warm_start
		)

		# Fit the model
		clf.fit(X)

		# transform and return the model
		rf_model = clf.transform(X)

		# Cast it to a numpy.ndarray
		_rf = rf_model.toarray()

		_prox_matrix, _diss_matrix = self.calculate_rf_proximity_dissimilarity_mat(X, clf, self.n_estimators)

		return _rf, _prox_matrix, _diss_matrix

	def calculate_rf_proximity_dissimilarity_mat(self, X, clf, n_trees):
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
			prox += np.equal.outer(leaves[:,tree_idx],
												leaves[:,tree_idx]).astype(float)
				
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
		if self.rf is not None:
			return self.rf
		else:
			raise AttributeError("rf_model object is not yet defined")

	@property
	def proximity_matrix(self):
		if self._prox_matrix is not None:
			return self._prox_matrix
		else:
			raise AttributeError("proximity_matrix object is not yet defined")

	@property
	def dissimilarity_matrix(self):
		if self._diss_matrix is not None:
			return self._diss_matrix
		else:
			raise AttributeError("dissimilarity_matrix object is not yet defined")

class runMDS(DimReduction):

	def __init__(self, dimreduction=None, gt=None, pops=None, prefix=None, colors=None, palette="Set1", rf=None, dissimilarity_matrix=None, metric=True, keep_dims=2, n_init=4, max_iter=300, eps=1e-3, n_jobs=None, verbose=0, random_state=None):

		# Initialize parent class
		super().__init__(gt, pops, prefix, colors, palette)

		# Validates passed arguments and sets parent class attributes
		self._validate_args(dimreduction, gt, pops, prefix)

		self.metric = metric
		self.rf = rf
		self.dissimilarity_matrix = dissimilarity_matrix
		self.n_dims = keep_dims
		self.n_init = n_init
		self.max_iter = max_iter
		self.eps = eps
		self.n_jobs = n_jobs
		self.verbose = verbose
		self.random_state = random_state
		self.method = None

		if self.rf is None and self.dissimilarity_matrix is None:
			raise TypeError("Either rf or dissimilarity matrix must be defined!")

		if self.rf is not None and self.dissimilarity_matrix is not None:
			raise TypeError("rf and dissimilarity matrix cannot both be defined!")

		if self.rf is not None:
			self.coords = self.fit_transform(self.rf)

		elif self.dissimilarity_matrix is not None:
			self.coords = self.fit_transform(self.dissimilarity_matrix)

	@timer
	def fit_transform(self, X):
	
		if self.metric:
			print("\nDoing cMDS dimensionality reduction...\n")
			self.method = "cMDS"
		else:
			print("\nDoing isoMDS dimensionality reduction...\n")
			self.method = "isoMDS"
				

		if self.dissimilarity_matrix is None:
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

		_coords = mds.fit_transform(X)
		print("\nDone!")

		return _coords

class runTSNE(DimReduction):

	def __init__(self, dimreduction=None, gt=None, pops=None, prefix=None, colors=None, palette="Set1", keep_dims=2, perplexity=30.0, early_exaggeration=12.0, learning_rate=200.0, n_iter=1000, n_iter_without_progress=300, min_grad_norm=1e-7, metric="euclidean", init="random", verbose=0, random_state=None, method="barnes_hut", angle=0.5, n_jobs=None, square_distances="legacy"):

		# Initialize parent class
		super().__init__(gt, pops, prefix, colors, palette)

		# Validates passed arguments and sets parent class attributes
		self._validate_args(dimreduction, gt, pops, prefix)

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
		self.square_distances = square_distances
		self.method = "t-SNE"

		self.coords = self.fit_transform(self.gt)

	@timer
	def fit_transform(self, X):

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
				"\tsquare_distances: "+str(self.square_distances)+"\n"
		)

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
			square_distances=self.square_distances
		)

		_coords = t.fit_transform(X.values)

		print("Done!")

		return _coords