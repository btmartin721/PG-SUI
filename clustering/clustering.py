# Standard library imports
import sys

# Make sure python version is >= 3.6
if sys.version_info < (3, 6):
	raise ImportError("Python < 3.6 is not supported!")

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from kneed import KneeLocator

# Requires scikit-learn-intellex package
if get_processor_name().strip().startswith("Intel"):
	try:
		from sklearnex import patch_sklearn
		patch_sklearn()
	except ImportError:
		print("Warning: Intel CPU detected but scikit-learn-intellex is not installed. We recommend installing it to speed up computation.")

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering
from sklearn_extra.cluster import KMedoids

# Custom imports
from dim_reduction.dim_reduction import DimReduction
#from utils.misc import progressbar
from utils.misc import timer
from utils.misc import isnotebook

is_notebook = isnotebook()

if is_notebook:
	from tqdm.notebook import tqdm as progressbar
else:
	from tqdm import tqdm as progressbar

class PamClustering(DimReduction):
	"""[Class to perform unsupervised PAM clustering on embedded data]

	Args
		([DimReduction]): [Inherits from DimReduction]
	"""

	def __init__(
		self, 
		embedding, 
		*,
		use_embedding=True,
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
		maxk=8, 
		metric="euclidean", 
		init="heuristic", 
		max_iter=300, 
		random_state=None
	):
		"""[Do PAM clustering. The embedding object is required. Either a DimReduction object or each of the gt, pops, sampleids, prefix, reps, and [colors or palette] objects are also required. Clustering for K=2 to K=maxk is performed by calling PamClustering(). Optimal K can then be assessed using any of the various supported methods.]

		Args:
			embedding ([runPCA, runMDS, or runTSNE object]): [Embedded data object created by runPCA(), runMDS(), or runTSNE()]

			use_embedding (bool, optional): [Whether to use embedding for cluster analysis. If False, uses distances if not None or 012-encoded genotypes]. Defaults to True.

			dimreduction (DimReduction object, optional): [Initialized DimReduction object. If not supplied, gt, pops, sampleids, prefix, reps, colors, and palette must be supplied instead]. Defaults to None.

			distances (numpy.ndarray, optional): [Pairwise distance matrix of shape (n_samples, n_samples). If supplied, metric parameter is ignored. Can be genetic distances or the dissimilarity scores between points from runRandomForestUML() that can be accessed as runRandomForestUML.dissimilarity. If distances is not supplied, uses euclidean distances. Cannot be used alongside embedding parameter]. Defaults to None]

			gt (pandas.DataFrame, numpy.ndarray, or 2D-list, optional): [One of the retrievable objects created by GenotypeData() of shape (n_samples, n_sites)]. Defaults to None.

			pops (list(str), optional): [List of populations created by GenotypeData of shape (n_samples,). Can be retrieved as GenotypeData.populations]. Defaults to None.

			prefix ([str], optional): [Prefix for outptut files and plots]. Defaults to None.

			reps ([int], optional): [Number of replicates to perform]. Defaults to None.

			scaler (str, optional): [Scaler to use for genotype data. Valid options include "patterson", "standard", and "center". "patterson" follows a Patterson standardization (Patterson, Price & Reich (2006)) recommended for genetic data that scales the data to unit variance at each SNP. "standard" does regular standardization to unit variance and centers the data. "center" does not do a standardization and just centers the data]. Defaults to "patterson".

			colors ([dict(str)], optional): [Dictionary with unique populations IDs as the keys and hex-code colors as the values. If None, then uses the matplotlib palette supplied to palette]. Defaults to None.

			palette (str, optional): [matplotlib color palette to use if colors=None]. Defaults to "Set1".

			maxk (int, optional): [Highest K value to test]. Defaults to 8.

			sampleids ([list(str)], optional): [Sample IDs to write to labels files of shape (n_samples,). Can be retrieved as GenotypeData.individuals]. Defaults to None.

			metric (str, optional): [What distance metric to use for clustering. See sklearn.metrics.pairwise_distances documentation. If distances is not None, metric is ignored]. Defaults to "euclidean".

			init (str, optional): [Specify medoids initialization method. Supported options include "heuristic", "random", "k-medoids++", and "build". See sklearn_extra.cluster.KMedoids documentation]. Defaults to "heuristic".

			max_iter (int, optional): [Specify the maximum number of iterations when fitting. See sklearn_extra.cluster.KMedoids documentation]. Defaults to 300.

			random_state ([int], optional): [Specify random state for the random number generator. Used to initialize medoids when init="random". See sklearn_extra.cluster.KMedoids documentation]. Defaults to None.
		"""
		# Initialize parent class
		super().__init__(gt, pops, sampleids, prefix, reps, scaler, colors, palette)

		# Validates passed arguments and sets parent class attributes
		self._validate_args(dimreduction, gt, pops, sampleids, prefix, reps, scaler)

		self.scale_gt(self.gt)

		# Set child class attributes
		self.distances = distances
		self.use_embedding = use_embedding
		self.maxk = maxk
		
		self.metric = metric
		self.init = init
		self.max_iter = max_iter
		self.random_state = random_state
		self.clust_method = "PAM"

		# To store results
		self.labels = list()
		self.models = list()
		self.pred_labels = list()
		self.X = None

		if self.use_embedding and self.distances is not None:
			raise TypeError(
				"'use_embedding' cannot be True if 'distances' is supplied"
			)

		# Get attributes from already instantiated embedding
		self.coords = embedding.coords
		self.method = embedding.method

		# Do cluster analysis from embedding
		if self.use_embedding:
			self.fit_predict(self.coords)

		# Or from distance matrix
		elif self.distances is not None:
			self.metric = "precomputed"
			self.fit_predict(self.distances)
		
		# Or use 012-encoded genotypes
		else:
			self.fit_predict(self.gt)

	@timer
	def fit_predict(self, X):
		"""[Fit model and predict cluster labels. Sets self.labels (predicted labels) and self.models (fit model). self.labels and self.models are lists of dictionaries. Each list item corresponds to one replicate, and each dictionary has k values as the keys and the labels and models as values]

		Args:
			X ([pandas.DataFrame or numpy.ndarray]): [Coordinates following dimensionality reduction embedding of shape (n_samples, n_features) or distance matrix of shape (n_samples, n_samples). Accessed via self.coords]
		"""
		print("\nDoing {}...\n".format(self.clust_method))

		print(
			"PAM Clustering Settings:\n"
				"\tmetric: "+str(self.metric)+"\n"
				"\tinit: "+str(self.init)+"\n"
				"\tmax_iter: "+str(self.max_iter)+"\n"
				"\trandom_state: "+str(self.random_state)+"\n"
		)

		if not isinstance(X, list):
			_X = X.copy()
			self.X = _X
		else:
			self.X = list()

		for rep in progressbar(
			range(self.reps), 
			desc="{} Replicates: ".format(self.clust_method), 
			leave=True, 
			position=0):

			l = dict()
			m = dict()
			for k in progressbar(
				range(2, self.maxk+1), 
				desc="K-Values: ", 
				leave=False, 
				position=1):

				_X = None
				km = None
				km = KMedoids(
					n_clusters=k,
					metric=self.metric,
					method="pam",
					init=self.init,
					max_iter=self.max_iter,
					random_state=self.random_state
				)

				if isinstance(X, list):
					_X = X[rep]
					self.X.append(_X)

				self.X = _X

				km.fit(_X)

				l[k] = km.predict(_X)
				m[k] = km

			self.labels.append(l)
			self.models.append(m)

class KMeansClustering(DimReduction):
	"""[Class to perform K-Means clustering on embedded data]

	Args:
		DimReduction ([DimReduction]): [Inherits from DimReduction]
	"""

	def __init__(
		self, 
		embedding,
		*,
		use_embedding=False,
		dimreduction=None, 
		gt=None, 
		pops=None,
		sampleids=None,
		prefix=None, 
		reps=None, 
		scaler=None,
		colors=None, 
		palette="Set1", 
		maxk=8, 
		init="k-means++", 
		n_init=10, 
		max_iter=300, 
		tol=1e-4, 
		verbose=0, 
		random_state=None, 
		algorithm="auto"
	):
		"""[Do K-Means clustering. The embedding object is required. Either a DimReduction object or each of the gt, pops, sampleids, prefix, reps, colors, and palette objects are also required. Clustering for K=2 to K=maxk is performed by calling KMeansClustering(). Optimal K can then be assessed using any of the various supported methods.]

		Args:
			embedding (runPCA, runMDS, or runTSNE object, optional): [Embedded data created by runPCA(), runMDS(), or runTSNE() of shape (n_samples, n_features). If use_embedding is False, then uses the 012-encoded genotypes]

			dimreduction ([DimReduction object], optional): [Initialized DimReduction object. If not supplied, gt, pops, sampleids, prefix, reps, colors, and palette must be supplied instead]. Defaults to None.

			gt ([pandas.DataFrame, numpy.ndarray, or 2D-list], optional): [One of the retrievable objects created by GenotypeData() of shape (n_samples, n_sites)]. Defaults to None.

			pops ([list(str)], optional): [List of populations created by GenotypeData of shape (n_samples,). Can be retrieved as GenotypeData.populations]. Defaults to None.

			prefix ([str], optional): [Prefix for outptut files and plots]. Defaults to None.

			reps ([int], optional): [Number of replicates to perform]. Defaults to None.

			scaler (str, optional): [Scaler to use for genotype data. Valid options include "patterson", "standard", and "center". "patterson" follows a Patterson standardization (Patterson, Price & Reich (2006)) recommended for genetic data that scales the data to unit variance at each SNP. "standard" does regular standardization to unit variance and centers the data. "center" does not do a standardization and just centers the data]. Defaults to "patterson".

			colors ([dict(str)], optional): [Dictionary with unique populations IDs as the keys and hex-code colors as the values]. Defaults to None.

			palette (str, optional): [matplotlib color palette to use if colors=None]. Defaults to "Set1".

			maxk (int, optional): [Highest K value to test]. Defaults to 8.

			sampleids ([list(str)], optional): [Sample IDs to write to labels files of shape (n_samples,). Can be retrieved as GenotypeData.individuals]. Defaults to None.

			init (str, optional): [Method for initial cluster center initialization. Supported options include "k-means++" and "random". See sklearn.cluster.KMeans documentation]. Defaults to "k-means++".

			n_init (int, optional): [Number of times the K-Means algorithm will be run with different centroid seeds. The final results will be the best output of n_init consecutive runs in terms of inertia. See sklearn.cluster.KMeans documentation]. Defaults to 10.

			max_iter (int, optional): [Specify the maximum number of iterations of the K-Means algorithm for a single run. See sklearn.cluster.KMeans documentation]. Defaults to 300.

			tol ([float], optional): [Relative tolerance with regards to Frobenius norm of the difference in the cluster centers of two consecutive iterations to declare convergence. See sklearn.clsuter.KMeans documentation]. Defaults to 1e-4.

			verbose (int, optional): [Verbosity mode. 0=least verbose; 2=most verbose]. Defaults to 0.

			random_state ([int], optional): [Determines random number generation for centroid initialization. Use an int to make the randomness deterministic. See sklearn.cluster.KMeans documentation. If None, a different random number will be generated each time]. Defaults to None.

			algorithm (str, optional): [K-Means algorithm to use. Supported options include "auto", "full", and "elkan". The classical EM-style algorithm is "full". The "elkan" variation is more efficient on data with well-defined clusters, but is more memory intensive. For now, "auto" chooses "elkan". See sklearn.cluster.KMeans documentation]. Defaults to "auto".
		"""
		# Initialize parent class
		super().__init__(gt, pops, sampleids, prefix, reps, scaler, colors, palette)

		# Validates passed arguments and sets parent class attributes
		self._validate_args(dimreduction, gt, pops, sampleids, prefix, reps, scaler)

		self.scale_gt(self.gt)

		# Set child class attributes
		self.use_embedding = use_embedding
		self.maxk = maxk
		
		self.init = init
		self.n_init = n_init
		self.max_iter = max_iter
		self.tol = tol
		self.verbose = verbose
		self.random_state = random_state
		self.algorithm = algorithm
		self.clust_method = "K-Means"

		# To store results
		self.labels = list()
		self.models = list()
		self.pred_labels = list()
		self.X = None

		# Get attributes from already instantiated embedding
		self.coords = embedding.coords
		self.method = embedding.method

		if self.use_embedding:
			self.fit_predict(self.coords)

		else:
			self.fit_predict(self.gt)

	@timer
	def fit_predict(self, X):
		"""[Fit model and predict cluster labels. Sets self.labels (predicted labels) and self.models (fit model). self.labels and self.models are lists of dictionaries. Each list item corresponds to one replicate, and each dictionary has k values as the keys and the labels and models as values]

		Args:
			X ([pandas.DataFrame]): [Coordinates following dimensionality reduction embedding of shape (n_samples, n_features). Accessed via self.coords]
		"""
		print("\nDoing {} for K=2 "
			"to K={}...\n".format(self.clust_method, self.maxk+1))

		print(
			"K-Means Clustering Settings:\n"
				"\tinit "+str(self.init)+"\n"
				"\tn_init: "+str(self.n_init)+"\n"
				"\tmax_iter: "+str(self.max_iter)+"\n"
				"\ttol: "+str(self.tol)+"\n"
				"\tverbose: "+str(self.verbose)+"\n"
				"\random_state: "+str(self.random_state)+"\n"
				"\algorithm: "+str(self.algorithm)+"\n"
		)

		if not isinstance(X, list):
			_X = X.copy()
			self.X = _X
		else:
			self.X = list()

		for rep in progressbar(
			range(self.reps), 
			"K-Means Replicates: ", 
			leave=True, 
			position=0):

			l = dict()
			m = dict()
			for k in progressbar(
				range(2, self.maxk+1), 
				desc="K-Values: ", 
				leave=False, 
				position=1):

				_X = None
				km = None
				km = KMeans(
					n_clusters=k,
					init=self.init,
					n_init=self.n_init,
					max_iter=self.max_iter,
					tol=self.tol,
					verbose=self.verbose,
					random_state=self.random_state,
					algorithm=self.algorithm
					)

				if isinstance(X, list):
					_X = X[rep]
					self.X.append(_X)

				km.fit(_X)
				
				l[k] = km.predict(_X)
				m[k] = km

			self.labels.append(l)
			self.models.append(m)

class DBSCANClustering(DimReduction):
	"""[Class to perform DBSCAN density-based clustering on embedded data]

	Args:
		DimReduction ([DimReduction object]): [Inherits from DimReduction]
	"""

	def __init__(
		self, 
		embedding, 
		*,
		use_embedding=False,
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
		plot_eps=False, 
		show_eps=False,
		eps_curve="concave",
		min_samples=5, 
		metric="euclidean", 
		metric_params=None, 
		algorithm="auto", 
		leaf_size=30, 
		p=None
	):
		"""[Do DBSCAN density-based clustering. The embedding object is required. Either a DimReduction object or each of the gt, pops, sampleids, prefix, reps, colors, and palette objects are also required. Optimal K is assessed based on the density of the clusters. No additional methods are required to determine optimal K for DBSCAN. Noisy or outlier samples get labeled as -1]

		Args:
			embedding ([runPCA or runMDS object]): [Embedded data created by runPCA() or runMDS()]

			use_embedding (bool, optional): [Whether to use embedding for cluster analysis. If False, uses distances if not None or else 012-encoded genotypes]. Defaults to False.

			dimreduction ([DimReduction object], optional): [Initialized DimReduction object. If not supplied, gt, pops, sampleids, prefix, reps, colors, and palette must be supplied instead]. Defaults to None.

			distances ([numpy.ndarray], optional): [Pairwise distance matrix of shape (n_samples, n_samples). Can be genetic distances, any of the sklearn.metrics.pairwise_distances, or the dissimilarity scores between points from runRandomForestUML() that can be accessed as runRandomForestUML.dissimilarity. If distances is not supplied, uses euclidean distances. If distances is not None, metric parameter is ignored]. Defaults to None.

			gt ([pandas.DataFrame, numpy.ndarray, or 2D-list], optional): [One of the retrievable objects created by GenotypeData() of shape (n_samples, n_sites)]. Defaults to None.

			pops ([list(str)], optional): [List of populations created by GenotypeData of shape (n_samples,). Can be retrieved as GenotypeData.populations]. Defaults to None.

			prefix ([str], optional): [Prefix for outptut files and plots]. Defaults to None.

			reps ([int], optional): [Number of replicates to perform]. Defaults to None.

			scaler (str, optional): [Scaler to use for genotype data. Valid options include "patterson", "standard", and "center". "patterson" follows a Patterson standardization (Patterson, Price & Reich (2006)) recommended for genetic data that scales the data to unit variance at each SNP. "standard" does regular standardization to unit variance and centers the data. "center" does not do a standardization and just centers the data]. Defaults to "patterson".

			colors ([dict(str)], optional): [Dictionary with unique populations IDs as the keys and hex-code colors as the values]. Defaults to None.

			palette (str, optional): [matplotlib color palette to use if colors=None]. Defaults to "Set1".

			sampleids ([list(str)], optional): [Sample IDs to write to labels files of shape (n_samples,). Can be retrieved as GenotypeData.individuals]. Defaults to None.

			plot_eps (bool, optional): [If True, eps tuning plots are made and saved to disk]. Defaults to False.

			show_eps (bool, optional): [If True, eps tuning plots are shown in the window]. Defaults to False. 

			eps_curve (str, optional): [How to detect the knee of the eps curve. Valid options are "concave" (default) and "convex". You can change it depending on the curve shape and it will detect a different knee (inflection point) for optimal eps]. Defaults to "concave".

			min_samples (int, optional): [The number of samples in a neighborhood for a point to be considered as a core point. This includes the point itself. See sklearn.cluster.DBSCAN documentaation]. Defaults to 5.

			metric (str, optional): [The metric to use when calculating distance between instances in a feature array. See sklearn.metrics.pairwise_distances and sklearn.cluster.DBSCAN documentation]. Defaults to "euclidean".

			metric_params (dict, optional): [Additional keyword arguments for the metric function. See sklearn.cluster.DBSCAN documentation]. Defaults to None.

			algorithm (str, optional): [The algorithm to be used by the NearestNeighbors module to compute pointwise distances and find nearest neighbors. Supported options include "auto", "ball_tree", "kd_tree", and "brute". See sklearn.cluster.DBSCAN and sklearn Nearest Neighbor documentation]. Defaults to "auto".

			leaf_size (int, optional): [Leaf size passed to BallTree or cKDTree. This can affect the speed of the construction and query, as well as the memory required to store the tree. The optimal value depends on the nature of the problem. See sklearn.cluster.DBSCAN documentation]. Defaults to 30.

			p (float, optional): [The power of the Minkowski metric to be used to calculate distance between points. If None, then p=2 (equivalent to the Euclidean distance)]. Defaults to None.
		"""
		# Initialize parent class
		super().__init__(gt, pops, sampleids, prefix, reps, scaler, colors, palette)

		# Validates passed arguments and sets parent class attributes
		self._validate_args(dimreduction, gt, pops, sampleids, prefix, reps, scaler)

		self.scale_gt(self.gt)

		# Set child class attributes
		self.use_embedding = use_embedding
		self.distances = distances
		self.plot_eps = plot_eps
		self.show_eps = show_eps
		self.eps_curve = eps_curve
		
		self.min_samples = min_samples
		self.metric = metric
		self.metric_params = metric_params
		self.algorithm = algorithm
		self.leaf_size = leaf_size
		self.p = p
		self.clust_method = "DBSCAN"

		# To store results
		self.labels = list()
		self.models = list()
		self.pred_labels = list()

		# Get attributes from already instantiated embedding
		self.coords = embedding.coords
		self.method = embedding.method

		if self.use_embedding and self.distances is not None:
			raise TypeError(
				"'use_embedding' cannot be True if 'distances' is supplied"
			)

		if self.distances is not None:
			self.metric = "precomputed"
			self.fit_predict(self.distances)

		elif self.use_embedding:
			self.fit_predict(self.coords)

		else:
			self.fit_predict(self.gt)

	@timer
	def fit_predict(self, X):
		"""[Fit model and predict cluster labels. Sets self.labels (predicted labels) and self.models (fit model). self.labels and self.models are lists with each item corresponding to one replicate]

		Args:
			X ([pandas.DataFrame or numpy.ndarray]): [Coordinates following dimensionality reduction embedding of shape (n_samples, n_features), accessed via self.coords, or pairwise distance matrix of shape (n_samples, n_samples)]
		"""
		print("\nDoing {} Clustering...\n".format(self.clust_method))

		print(
			"DBSCAN Clustering Settings:\n"
				"\tmin_samples: "+str(self.min_samples)+"\n"
				"\tmetric: "+str(self.metric)+"\n"
				"\tmetric_params: "+str(self.metric_params)+"\n"
				"\talgorithm: "+str(self.algorithm)+"\n"
				"\tleaf_size: "+str(self.leaf_size)+"\n"
				"\tp: "+str(self.p)+"\n"
		)

		if not isinstance(X, list):
			_X = X.copy()

		eps = None
		for rep in progressbar(
			range(self.reps), 
			desc="{} Replicates: ".format(self.clust_method),
			leave=True,
			position=0):

			_X = None

			if isinstance(X, list):
				_X = X[rep]

			eps = self.tune_eps(_X, rep, self.eps_curve, plot_eps=self.plot_eps, show_eps=self.show_eps)

			db = DBSCAN(
				eps=eps,
				min_samples=self.min_samples,
				metric=self.metric,
				metric_params=self.metric_params,
				algorithm=self.algorithm,
				leaf_size=self.leaf_size,
				p=self.p
			)

			l = db.fit_predict(_X)
			m = db

			self.labels.append(l)
			self.models.append(m)

			self.save_labels(rep)
		
		print("Tuned eps setting: {}".format(eps))

	def tune_eps(self, X, rep, curve, plot_eps=False, show_eps=False):
		"""[Tune the eps parameter for DBSCAN by plotting distances (y-axis) versus the number of points. Selects the optimal distance (~eps) by calculating the knee (inflection point) of the curve via the kneed.KneeLocator module. Saves a plot with the curve and knee to disk if plot_eps=True]

		Args:
			X ([pandas.DataFrame]): [Coordinates following dimensionality reduction embedding of shape (n_samples, n_features). Accessed via self.coords]

			rep ([int]): [Current replicate number (0-based indexing)]

			plot_eps (bool, optional): [If True, saves the eps plot to disk]. Defaults to False.

			show_eps (bool, optional): [If True, shows the eps plot in the window]. Defaults to False.

		Returns:
			[float]: [The optimal eps DBSCAN setting calculated as the inflection point (knee) of the curve]
		"""
		neigh = NearestNeighbors(n_neighbors=2)
		nbrs = neigh.fit(X)
		dist, idx = nbrs.kneighbors(X)
		dist = dist[:,1]

		dist = np.sort(dist, axis=0)

		i = np.arange(len(dist))

		kneedle = KneeLocator(
			i, 
			dist, 
			S=1, 
			curve=curve, 
			direction="increasing", 
			interp_method="polynomial"
		)

		if plot_eps:
			fig = plt.figure(figsize=(5, 5))

			kneedle.plot_knee()
			plt.xlabel("Points")
			plt.ylabel("Distance")

			plt.savefig(
				"{}_output/{}/{}/plots/dbscan_eps_{}.pdf".format(
					self.prefix, self.method, self.clust_method, rep+1
				)
			)

			if show_eps:
				plt.show()

		return dist[kneedle.knee]

class AffinityPropogationClustering(DimReduction):
	"""[Class to perform Affinity Propogation clustering on embedded data]

	Args:
		DimReduction ([DimReduction object]): [Inherits from DimReduction]
	"""
	def __init__(
		self, 
		embedding,
		*,
		distances=None,
		dimreduction=None, 
		gt=None, 
		pops=None,
		sampleids=None,
		prefix=None, 
		reps=None, 
		scaler=None,
		colors=None, 
		palette="Set1", 
		initial_damping=0.5,
		max_iter=200,
		convergence_iter=15,
		verbose=0,
		random_state=None
	):
		"""[Do Affinity Propogation clustering. The embedding object is required. Can use the runRandomForestUML().dissimilarity matrix for the distances parameter. Either a DimReduction object or each of the gt, pops, sampleids, prefix, reps, and [colors or palette] objects are also required. Clusters are assessed as being nearby to examplar samples that best summarize the data. Clusters are determined as being similar to the examplar data points, and similarity is communicated among points using a propogation algorithm.]

		Args:
			embedding ([runPCA, runMDS, or runTSNE object]): [Embedded data created by runPCA(), runMDS(), or runTSNE()]

			distances (numpy.ndarray, optional). [Pairwise distance matrix of shape (n_samples, n_samples). If supplied, does affinity propagation clustering on the precomputed distance matrix. Can use the dissimilarity matrix from the random forest output retrieved as runRandomForestUML.dissimilarity, or can use any other distance matrix (e.g. pairwise genetic distances).Otherwise it computes euclidean distances from the 012-encoded genotypes]. Defaults to None.

			dimreduction ([DimReduction object], optional): [Initialized DimReduction object. If not supplied, gt, pops, sampleids, prefix, reps, colors, and palette must be supplied instead]. Defaults to None.

			gt ([pandas.DataFrame, numpy.ndarray, or 2D-list], optional): [One of the retrievable objects created by GenotypeData() of shape (n_samples, n_sites)]. Defaults to None.

			pops ([list(str)], optional): [List of populations created by GenotypeData of shape (n_samples,). Can be retrieved as GenotypeData.populations]. Defaults to None.

			prefix ([str], optional): [Prefix for outptut files and plots]. Defaults to None.

			reps ([int], optional): [Number of replicates to perform]. Defaults to None.

			scaler (str, optional): [Scaler to use for genotype data. Valid options include "patterson", "standard", and "center". "patterson" follows a Patterson standardization (Patterson, Price & Reich (2006)) recommended for genetic data that scales the data to unit variance at each SNP. "standard" does regular standardization to unit variance and centers the data. "center" does not do a standardization and just centers the data]. Defaults to "patterson".

			colors ([dict(str)], optional): [Dictionary with unique populations IDs as the keys and hex-code colors as the values]. Defaults to None.

			palette (str, optional): [matplotlib color palette to use if colors=None]. Defaults to "Set1".

			sampleids ([list(str)], optional): [Sample IDs to write to labels files of shape (n_samples,). Can be retrieved as GenotypeData.individuals]. Defaults to None.

			initial_damping (float, optional): [Initial damping factor (between 0.5 and 1). Damping is the extent to which the current value is maintained relative to incoming values (weighted 1 - damping). This is in order to avoid numerical oscilliations when updating the values (messages)]. Defaults to 0.5.

			max_iter (int, optional): [Maximum number of iterations]. Defaults to 200.

			convergence_iter (int, optional): [Number of iterations with no chnage in the number of estimated clusters that stops the convergence]. Defaults to 15.

			verbose (int, optional): [Verbosity level. 0=less verbose]. Defaults to 0. 

			random_state (int, optional): [Pseudo-random number generator to control the starting state. Use an int for reproducible results across function calls or leave as None for randomization between function calls]. Defaults to None.

		Raises:
			ValueError: ['verbose' argument must be an integer >= 0]
		"""
		# Initialize parent class
		super().__init__(gt, pops, sampleids, prefix, reps, scaler, colors, palette)

		# Validates passed arguments and sets parent class attributes
		self._validate_args(dimreduction, gt, pops, sampleids, prefix, reps, scaler)

		self.scale_gt(self.gt)

		# Child class attributes
		self.distances = distances
		
		self.initial_damping = initial_damping
		self.max_iter = max_iter
		self.convergence_iter = convergence_iter
		self.verbose = verbose
		self.random_state = random_state
		self.clust_method = "AffinityPropagation"

		# To store results
		self.labels = list()
		self.models = list()
		self.pred_labels = list()
		self.cluster_centers_indices = list()
		self.best_preference = list()
		self.best_damping = list()

		# Get attributes from already instantiated embedding
		self.coords = embedding.coords
		self.method = embedding.method

		if self.verbose == 0:
			self.verbose = False
		elif self.verbose >=1:
			self.verbose = True
		else:
			raise ValueError("'verbose' argument must be an integer >= 0")

		if self.distances is not None:
			self.affinity = "precomputed"
			self.tune_clusters(self.distances)
		else:
			self.affinity = "euclidean"
			self.tune_clusters(self.gt)

		print("\nOptimized preference: {}".format(str(
			self.best_preference)))

		print("Optimized damping: {}\n".format(str(
			self.best_damping)))
				
	def fit_predict(self, X, damping, preference, affinity="precomputed"):
		"""[Fit model and predict cluster labels. Sets self.labels (predicted labels) and self.models (fit model). self.labels and self.models are lists with each item corresponding to one replicate]

		Args:
			X ([pandas.DataFrame or numpy.ndarray]): [Coordinates following dimensionality reduction embedding of shape (n_samples, n_features), accessed via self.coords, or distance matrix of shape (n_samples, n_samples)]

			damping ([float]): [Damping parameter setting between 0.5 and 1]

			preference ([float]): [Preference parameter setting between 0 and 1]

			affinity (str, optional): [Distance metric used. Either "precomputed" or "euclidean" are supported]. Defaults to "precomputed".

		Returns:
			l ([list(int)]): [List of predicted labels, with each item being one replicate]

			([int]): [Count of unique labels]

			af.cluster_center_indices_ ([numpy.ndarray]): [Indices for each cluster center]

			m ([sklearn.cluster.AffinityPropagation]): [Fit model]
		"""

		af = AffinityPropagation(
			damping=damping,
			max_iter=self.max_iter,
			convergence_iter=self.convergence_iter,
			preference=preference,
			affinity=affinity,
			verbose=self.verbose,
			random_state=self.random_state
		)

		l = af.fit_predict(X)
		m = af

		return l, len(set(l)), af.cluster_centers_indices_, m

	def is_tuning_required(self, similarity_matrix, rows_of_cluster):
		df = similarity_matrix[rows_of_cluster]

		for val in df.values:
			if all(np.where(val > 0.5, True, False)):
				continue

			return True

		return False

	def get_damp_range(self, similarity):

		starting_point = 0.5
		step = 0.05

		damping_tuning_range = [starting_point]
		max_val = starting_point
		while max_val <= 1 - step:
			max_val += step
			damping_tuning_range.append(max_val)

		return damping_tuning_range

	def get_pref_range(self, similarity):
		starting_point = np.median(similarity)

		if starting_point == 0:
			starting_point = np.mean(similarity)

		# Let's try to accelerate the pace of values picking
		if starting_point >= 0.05:
			step = 1.25
		else:
			step = 2.0

		preference_tuning_range = [starting_point]
		max_val = starting_point
		while max_val < 1:
			max_val *= step
			preference_tuning_range.append(max_val)

		min_val = starting_point
		while min_val > 0.01:
			min_val /= step
			preference_tuning_range.append(min_val)

		return preference_tuning_range

	@timer
	def tune_clusters(self, X):
		"""[Run Affinity Propagation Clustering. This function tunes the preference parameter. The self.fit_predict() method tunes the damping parameter. Saves labels and models into lists for each replicate]

		Args:
			X ([pandas.DataFrame or numpy.ndarray]): [Input distance matrix or 012-encoded genotypes]
		"""
		print("\nDoing {}...\n".format(self.clust_method))

		print(
			"Affinity Propagation Settings:\n"
				"\tmax_iter: "+str(self.max_iter)+"\n"
				"\tconvergence_iter: "+str(self.convergence_iter)+"\n"
				"\taffinity: "+str(self.affinity)+"\n"
				"\tverbose: "+str(self.verbose)+"\n"
				"\trandom_state: "+str(self.random_state)+"\n"
		)

		if not isinstance(X, list):
			_X = X.copy()

		for rep in progressbar(
			range(self.reps), 
			desc="{} Replicates: ".format(self.clust_method),
			leave=True,
			position=0
		):

			if isinstance(X, list):
				_X = X[rep]

			similarity = None
			if self.affinity == "euclidean":				
				_, __, ___, tmp_model = \
					self.fit_predict(_X, self.initial_damping, None, affinity="euclidean")
				similarity = pd.DataFrame(euclidean_distances(_X, squared=True))
			else:
				similarity = _X

			preference_tuning_range = self.get_pref_range(similarity)

			damping_tuning_range = self.get_damp_range(similarity)

			grid = \
				[(d, p) for d in damping_tuning_range for p in preference_tuning_range]

			best_tested_preference = None
			best_tested_damping = None
			for damping, preference in grid:
				labels, labels_count, cluster_centers_indices, af_model = \
					self.fit_predict(similarity, damping, preference)

				needs_tuning = False
				wrong_clusters = 0
				for label_index in range(labels_count):
					cluster_elements_indexes = np.where(labels == label_index)[0]

					tuning_required = self.is_tuning_required(similarity, cluster_elements_indexes)

					if tuning_required:
						wrong_clusters += 1

						if not needs_tuning:
							needs_tuning = True

				if best_tested_preference is None or wrong_clusters < best_tested_preference[1]:
					best_tested_preference = (preference, wrong_clusters)

				if best_tested_damping is None or wrong_clusters < best_tested_damping[1]:
					best_tested_damping = (damping, wrong_clusters)

				if not needs_tuning:
					self.labels.append(labels), 
					self.models.append(af_model)
					self.save_labels(rep)
					self.best_preference.append(best_tested_preference[0])
					self.best_damping.append(best_tested_damping[0])
					break

			# The clustering has not been tuned enough during the iterations, we choose the less wrong clusters
			labs, lab_count, centers, mymodel = self.fit_predict(similarity, best_tested_damping[0], best_tested_preference[0])

			self.labels.append(labs)
			self.models.append(mymodel)
			self.save_labels(rep)
			self.best_preference.append(best_tested_preference[0])
			self.best_damping.append(best_tested_damping[0])

class AgglomHier(DimReduction):
	"""[Class to perform Agglomerative Hierarchical Clustering on embedded data]

	Args:
		DimReduction ([DimReduction object]): [Inherits from DimReduction]
	"""
	def __init__(
		self, 
		embedding,
		*,
		use_embedding=False,
		distances=None,
		dimreduction=None, 
		gt=None, 
		pops=None,
		sampleids=None,
		prefix=None, 
		reps=None, 
		scaler=None,
		colors=None, 
		palette="Set1",
		maxk=8, 
		affinity="euclidean",
		linkage="average"
	):
		"""[Do Agglomerative Hierarchical Clustering. The embedding object is required. Either a DimReduction object or each of the gt, pops, sampleids, prefix, reps, and [colors or palette] objects are also required. Clusters are assessed by recursively merging the pair of clusters that minimally increase a given linkage distance]

		Args:
			embedding ([runPCA or runMDS object]): [Embedded data created by runPCA() or runMDS()]

			use_embedding (bool, optional): [Whether to use embedding for cluster analysis. If False, uses distances if not None or else 012-encoded genotypes]. Defaults to False.

			dimreduction ([DimReduction object], optional): [Initialized DimReduction object. If not supplied, gt, pops, sampleids, prefix, reps, colors, and palette must be supplied instead]. Defaults to None.

			distances (numpy.ndarray, optional). [Pairwise distance matrix of shape (n_samples, n_samples). Can use dissimilarity matrix from runRandomForestUML.dissimilarity. If supplied, does affinity propagation clustering on the distance matrix instead of the embedded coordinates]. Defaults to None.

			gt ([pandas.DataFrame, numpy.ndarray, or 2D-list], optional): [One of the retrievable objects created by GenotypeData() of shape (n_samples, n_sites)]. Defaults to None.

			pops ([list(str)], optional): [List of populations created by GenotypeData of shape (n_samples,). Can be retrieved as GenotypeData.populations]. Defaults to None.

			prefix ([str], optional): [Prefix for outptut files and plots]. Defaults to None.

			reps ([int], optional): [Number of replicates to perform]. Defaults to None.

			scaler (str, optional): [Scaler to use for genotype data. Valid options include "patterson", "standard", and "center". "patterson" follows a Patterson standardization (Patterson, Price & Reich (2006)) recommended for genetic data that scales the data to unit variance at each SNP. "standard" does regular standardization to unit variance and centers the data. "center" does not do a standardization and just centers the data]. Defaults to "patterson".

			colors ([dict(str)], optional): [Dictionary with unique populations IDs as the keys and hex-code colors as the values]. Defaults to None.

			palette (str, optional): [matplotlib color palette to use if colors=None]. Defaults to "Set1".

			maxk (int, optional): [Maximum number of clusters to test]. Defaults to 8.

			sampleids (list(str), optional): [Sample IDs to write to labels files of shape (n_samples,). Can be retrieved as GenotypeData.individuals]. Defaults to None.

			affinity (str, optional): [Metric to compute the linkage. Can be "euclidean", "l1", "l2", "manhattan", "cosine", or "precomputed". If linkage is "ward", only "euclidean" is accepted. If "precomputed", a distance matrix (instead of a similarity matrix) is needed as input for the fit method]. Defaults to "euclidean".

			linkage (str, optional): [Which linkage criterion to use. The linkage criterion determines which distance to use between sets of observation. The algorithm will merge the pairs of clusters that minimize this criterion. "ward" minimizes the variance of the clusters being merged. "average" uses the average of the distances of each observation of the two sets. "complete" or "maximum" linkage uses the maximum distances betwee all observations of the two sets. "single" uses the minimum of the distances between all observations of the two sets]. Defaults to "ward".
		"""
		# Initialize parent class
		super().__init__(gt, pops, sampleids, prefix, reps, scaler, colors, palette)

		# Validates passed arguments and sets parent class attributes
		self._validate_args(dimreduction, gt, pops, sampleids, prefix, reps, scaler)

		self.scale_gt(self.gt)

		# Child class attributes
		self.use_embedding = use_embedding
		self.maxk = maxk
		self.distances = distances
		
		self.linkage = linkage
		self.clust_method = "Agglomerative"

		# To store results
		self.labels = list()
		self.models = list()
		self.pred_labels = list()
		self.X = None

		# Get attributes from already instantiated embedding
		self.coords = embedding.coords
		self.method = embedding.method

		if self.use_embedding and self.distances is not None:
			raise TypeError(
				"'use_embedding' cannot be True if 'distances' is supplied"
			)

		if self.distances is not None:
			self.affinity = "precomputed"
			self.fit_predict(self.distances)

		elif self.use_embedding:
			self.fit_predict(self.coords)

		else:
			self.fit_predict(self.gt)

	@timer
	def fit_predict(self, X):
		"""[Fit model and predict cluster labels. Sets self.labels (predicted labels) and self.models (fit model). self.labels and self.models are lists of dictionaries. Each list item corresponds to one replicate, and each dictionary has k values as the keys and the labels and models as values]

		Args:
			X ([pandas.DataFrame or numpy.ndarray]): [Coordinates following dimensionality reduction embedding of shape (n_samples, n_features), accessed via self.coords, or pairwise distance matrix of shape (n_samples, n_samples)]
		"""
		print("\nDoing {} for K=2 "
			"to K={}...\n".format(self.clust_method, self.maxk+1))

		print(
			"Agglomerative Clustering Settings:\n"
				"\taffinity: "+str(self.affinity)+"\n"
				"\tlinkage: "+str(self.linkage)+"\n"
		)

		self.X = X

		for rep in progressbar(
			range(self.reps),
			desc="{} Repilcates: ".format(self.clust_method),
			leave=True,
			position=0
		):
			l = dict()
			m = dict()
			for k in progressbar(
				range(2, self.maxk+1), 
				desc="K-Values: ",
				leave=False,
				position=1):

				clust = None
				clust = AgglomerativeClustering(
					n_clusters=k,
					affinity=self.affinity,
					linkage=self.linkage
				)
						
				l[k] = clust.fit_predict(X[rep])
				m[k] = clust

			self.labels.append(l)
			self.models.append(m)












