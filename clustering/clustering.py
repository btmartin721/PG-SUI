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
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn_extra.cluster import KMedoids

# Custom imports
from dim_reduction.dim_reduction import DimReduction
from utils.misc import progressbar

class PamClustering(DimReduction):
	"""[Class to perform unsupervised PAM clustering on embedded data]

	Args
		([DimReduction]): [Inherits from DimReduction]
	"""

	def __init__(
		self, 
		embedding, 
		dimreduction=None, 
		gt=None, 
		pops=None, 
		prefix=None, 
		reps=None, 
		colors=None, 
		palette="Set1", 
		maxk=8, 
		sampleids=None, 
		metric="euclidean", 
		clust_method="pam", 
		init="heuristic", 
		max_iter=300, 
		random_state=None
	):
		"""[Do PAM clustering. The embedding object is required. Either a DimReduction object or each of the gt, pops, prefix, reps, colors, and palette objects are also required. Clustering for K=2 to K=maxk is performed by calling PamClustering(). Optimal K can then be assessed using any of the various supported methods.]

		Args:
			embedding ([runPCA or runMDS object]): [Embedded data created by runPCA() or runMDS()]

			dimreduction ([DimReduction object], optional): [Initialized DimReduction object. If not supplied, gt, pops, prefix, reps, colors, and palette must be supplied instead]. Defaults to None.

			gt ([pandas.DataFrame, numpy.ndarray, or 2D-list], optional): [One of the retrievable objects created by GenotypeData() of shape (n_samples, n_sites)]. Defaults to None.

			pops ([list(str)], optional): [List of populations created by GenotypeData of shape (n_samples,). Can be retrieved as GenotypeData.populations]. Defaults to None.

			prefix ([str], optional): [Prefix for outptut files and plots]. Defaults to None.

			reps ([int], optional): [Number of replicates to perform]. Defaults to None.

			colors ([dict(str)], optional): [Dictionary with unique populations IDs as the keys and hex-code colors as the values. If None, then uses the matplotlib palette supplied to palette]. Defaults to None.

			palette (str, optional): [matplotlib color palette to use if colors=None]. Defaults to "Set1".

			maxk (int, optional): [Highest K value to test]. Defaults to 8.

			sampleids ([list(str)], optional): [Sample IDs to write to labels files of shape (n_samples,). Can be retrieved as GenotypeData.individuals]. Defaults to None.

			metric (str, optional): [What distance metric to use for clustering. See sklearn_extra.cluster.KMedoids documentation]. Defaults to "euclidean".

			clust_method (str, optional): [Clustering method to use. Supported options include "pam" and "alternate". See sklearn_extra.cluster.KMedoids documentation]. Defaults to "pam".

			init (str, optional): [Specify medoids initialization method. Supported options include "heuristic", "random", "k-medoids++", and "build". See sklearn_extra.cluster.KMedoids documentation]. Defaults to "heuristic".

			max_iter (int, optional): [Specify the maximum number of iterations when fitting. See sklearn_extra.cluster.KMedoids documentation]. Defaults to 300.

			random_state ([int], optional): [Specify random state for the random number generator. Used to initialize medoids when init="random". See sklearn_extra.cluster.KMedoids documentation]. Defaults to None.
		"""
		# Initialize parent class
		super().__init__(gt, pops, prefix, reps, colors, palette)

		# Validates passed arguments and sets parent class attributes
		self._validate_args(dimreduction, gt, pops, prefix, reps)

		# Set child class attributes
		self.maxk = maxk
		self.sampleids = sampleids
		self.metric = metric
		self.clust_method = clust_method
		self.init = init
		self.max_iter = max_iter
		self.random_state=random_state

		# To store results
		self.labels = list()
		self.models = list()
		self.pred_labels = list()


		# Get attributes from already instantiated embedding
		self.coords = embedding.coords
		self.method = embedding.method

		# self.kmedoids = dict()
		# self.dbscan = None
		# self.affinity_prop = None
		# self.hier = None
		# self.birch = None
		# self.kmeans = None
		# self.mean_shift = None
		# self.optics = None
		# self.spectral = None
		# self.gaussian_mix = None

		self.fit_predict(self.coords)

	def fit_predict(self, X):
		"""[Fit model and predict cluster labels. Sets self.labels (predicted labels) and self.models (fit model). self.labels and self.models are lists of dictionaries. Each list item corresponds to one replicate, and each dictionary has k values as the keys and the labels and models as values]

		Args:
			X ([pandas.DataFrame]): [Coordinates following dimensionality reduction embedding of shape (n_samples, n_features). Accessed via self.coords]
		"""

		print("\nDoing K-Medoids Clustering...\n")

		for rep in progressbar(range(self.reps), "K-Medoids: "):
			l = dict()
			m = dict()
			for k in range(2, self.maxk+1):

				km = None
				km = KMedoids(
					n_clusters=k,
					metric=self.metric,
					method=self.clust_method,
					init=self.init,
					max_iter=self.max_iter,
					random_state=self.random_state
				)

				km.fit(X[rep])

				l[k] = km.predict(X[rep])
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
		dimreduction=None, 
		gt=None, 
		pops=None, 
		prefix=None, 
		reps=None, 
		colors=None, 
		palette="Set1", 
		maxk=8, 
		sampleids=None, 
		init="k-means++", 
		n_init=10, 
		max_iter=300, 
		tol=1e-4, 
		verbose=0, 
		random_state=None, 
		algorithm="auto"
	):
		"""[Do K-Means clustering. The embedding object is required. Either a DimReduction object or each of the gt, pops, prefix, reps, colors, and palette objects are also required. Clustering for K=2 to K=maxk is performed by calling KMeansClustering(). Optimal K can then be assessed using any of the various supported methods.]

		Args:
			embedding ([runPCA or runMDS object]): [Embedded data created by runPCA() or runMDS() of shape (n_samples, n_features)]

			dimreduction ([DimReduction object], optional): [Initialized DimReduction object. If not supplied, gt, pops, prefix, reps, colors, and palette must be supplied instead]. Defaults to None.

			gt ([pandas.DataFrame, numpy.ndarray, or 2D-list], optional): [One of the retrievable objects created by GenotypeData() of shape (n_samples, n_sites)]. Defaults to None.

			pops ([list(str)], optional): [List of populations created by GenotypeData of shape (n_samples,). Can be retrieved as GenotypeData.populations]. Defaults to None.

			prefix ([str], optional): [Prefix for outptut files and plots]. Defaults to None.

			reps ([int], optional): [Number of replicates to perform]. Defaults to None.

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
		super().__init__(gt, pops, prefix, reps, colors, palette)

		# Validates passed arguments and sets parent class attributes
		self._validate_args(dimreduction, gt, pops, prefix, reps)

		# Set child class attributes
		self.maxk = maxk
		self.sampleids = sampleids
		self.init = init
		self.n_init = n_init
		self.max_iter = max_iter
		self.tol = tol
		self.verbose = verbose
		self.random_state = random_state
		self.algorithm = algorithm
		self.clust_method = "kmeans"

		# To store results
		self.labels = list()
		self.models = list()
		self.pred_labels = list()


		# Get attributes from already instantiated embedding
		self.coords = embedding.coords
		self.method = embedding.method

		self.fit_predict(self.coords)

	def fit_predict(self, X):
		"""[Fit model and predict cluster labels. Sets self.labels (predicted labels) and self.models (fit model). self.labels and self.models are lists of dictionaries. Each list item corresponds to one replicate, and each dictionary has k values as the keys and the labels and models as values]

		Args:
			X ([pandas.DataFrame]): [Coordinates following dimensionality reduction embedding of shape (n_samples, n_features). Accessed via self.coords]
		"""
		print("\nDoing K-Means Clustering for K=2 "
			"to K={}...\n".format(self.maxk+1))

		for rep in progressbar(range(self.reps), "K-Means: "):
			l = dict()
			m = dict()
			for k in range(2, self.maxk+1):

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

				km.fit(X[rep])
				
				l[k] = km.predict(X[rep])
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
		dimreduction=None, 
		gt=None, 
		pops=None, 
		prefix=None, 
		reps=None, 
		colors=None, 
		palette="Set1", 
		sampleids=None, 
		plot_eps=False, 
		min_samples=5, 
		metric="euclidean", 
		metric_params=None, 
		algorithm="auto", 
		leaf_size=30, 
		p=None
	):
		"""[Do DBSCAN density-based clustering. The embedding object is required. Either a DimReduction object or each of the gt, pops, prefix, reps, colors, and palette objects are also required. Optimal K is assessed based on the density of the clusters. No additional methods are required to determine optimal K for DBSCAN. Noisy or outlier samples get labeled as -1]

		Args:
			embedding ([runPCA or runMDS object]): [Embedded data created by runPCA() or runMDS()]

			dimreduction ([DimReduction object], optional): [Initialized DimReduction object. If not supplied, gt, pops, prefix, reps, colors, and palette must be supplied instead]. Defaults to None.

			gt ([pandas.DataFrame, numpy.ndarray, or 2D-list], optional): [One of the retrievable objects created by GenotypeData() of shape (n_samples, n_sites)]. Defaults to None.

			pops ([list(str)], optional): [List of populations created by GenotypeData of shape (n_samples,). Can be retrieved as GenotypeData.populations]. Defaults to None.

			prefix ([str], optional): [Prefix for outptut files and plots]. Defaults to None.

			reps ([int], optional): [Number of replicates to perform]. Defaults to None.

			colors ([dict(str)], optional): [Dictionary with unique populations IDs as the keys and hex-code colors as the values]. Defaults to None.

			palette (str, optional): [matplotlib color palette to use if colors=None]. Defaults to "Set1".

			sampleids ([list(str)], optional): [Sample IDs to write to labels files of shape (n_samples,). Can be retrieved as GenotypeData.individuals]. Defaults to None.

			plot_eps (bool, optional): [If True, eps tuning plots are made and saved to disk]. Defaults to False.

			min_samples (int, optional): [The number of samples in a neighborhood for a point to be considered as a core point. This includes the point itself. See sklearn.cluster.DBSCAN documentaation]. Defaults to 5.

			metric (str, optional): [The metric to use when calculating distance between instances in a feature array. See sklearn.cluster.DBSCAN documentation]. Defaults to "euclidean".

			metric_params (dict, optional): [Additional keyword arguments for the metric function. See sklearn.cluster.DBSCAN documentation]. Defaults to None.

			algorithm (str, optional): [The algorithm to be used by the NearestNeighbors module to compute pointwise distances and find nearest neighbors. Supported options include "auto", "ball_tree", "kd_tree", and "brute". See sklearn.cluster.DBSCAN and sklearn Nearest Neighbor documentation]. Defaults to "auto".

			leaf_size (int, optional): [Leaf size passed to BallTree or cKDTree. This can affect the speed of the construction and query, as well as the memory required to store the tree. The optimal value depends on the nature of the problem. See sklearn.cluster.DBSCAN documentation]. Defaults to 30.

			p (float, optional): [The power of the Minkowski metric to be used to calculate distance between points. If None, then p=2 (equivalent to the Euclidean distance)]. Defaults to None.
		"""

		# Initialize parent class
		super().__init__(gt, pops, prefix, reps, colors, palette)

		# Validates passed arguments and sets parent class attributes
		self._validate_args(dimreduction, gt, pops, prefix, reps)

		# Set child class attributes
		self.plot_eps = plot_eps
		self.sampleids = sampleids
		self.min_samples = min_samples
		self.metric = metric
		self.metric_params = metric_params
		self.algorithm = algorithm
		self.leaf_size = leaf_size
		self.p = p
		self.clust_method = "dbscan"

		# To store results
		self.labels = list()
		self.models = list()
		self.pred_labels = list()

		# Get attributes from already instantiated embedding
		self.coords = embedding.coords
		self.method = embedding.method

		self.fit_predict(self.coords)

	def fit_predict(self, X):
		"""[Fit model and predict cluster labels. Sets self.labels (predicted labels) and self.models (fit model). self.labels and self.models are lists with each item corresponding to one replicate]

		Args:
			X ([pandas.DataFrame]): [Coordinates following dimensionality reduction embedding of shape (n_samples, n_features). Accessed via self.coords]
		"""
		print("\nDoing DBSCAN Clustering...\n")

		for rep in progressbar(
			range(self.reps), 
			"{}: ".format(self.clust_method.upper())):

			eps = self.tune_eps(X, rep, plot_eps=self.plot_eps)

			db = DBSCAN(
				eps=eps,
				min_samples=self.min_samples,
				metric=self.metric,
				metric_params=self.metric_params,
				algorithm=self.algorithm,
				leaf_size=self.leaf_size,
				p=self.p
			)

			l = db.fit_predict(X[rep])
			m = db

			self.labels.append(l)
			self.models.append(m)

			self.save_labels(rep)

	def tune_eps(self, X, rep, plot_eps=False):
		"""[Tune the eps parameter for DBSCAN by plotting distances (y-axis) versus the number of points. Selects the optimal distance (~eps) by calculating the knee (inflection point) of the curve via the kneed.KneeLocator module. Saves a plot with the curve and knee to disk if plot_eps=True]

		Args:
			X ([pandas.DataFrame]): [Coordinates following dimensionality reduction embedding of shape (n_samples, n_features). Accessed via self.coords]

			rep ([int]): [Current replicate number (0-based indexing)]

			plot_eps (bool, optional): [If True, saves the eps plot to disk]. Defaults to False.

		Returns:
			[float]: [The optimal eps DBSCAN setting calculated as the inflection point (knee) of the curve]
		"""
		neigh = NearestNeighbors(n_neighbors=2)
		nbrs = neigh.fit(X[rep])
		dist, idx = nbrs.kneighbors(X[rep])
		dist = dist[:,1]

		dist = np.sort(dist, axis=0)
		i = np.arange(len(dist))

		kneedle = KneeLocator(
			i, 
			dist, 
			S=1, 
			curve="convex", 
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

		return dist[kneedle.knee]


