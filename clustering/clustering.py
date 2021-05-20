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
from sklearn.cluster import AffinityPropagation
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

class AffinityPropogationClustering(DimReduction):
	"""[Class to perform Affinity Propogation clustering on embedded data]

	Args:
		DimReduction ([DimReduction object]): [Inherits from DimReduction]
	"""
	def __init__(
		self, 
		embedding,
		proximity_matrix,
		dimreduction=None, 
		gt=None, 
		pops=None, 
		prefix=None, 
		reps=None, 
		colors=None, 
		palette="Set1", 
		sampleids=None, 
		damping=0.5,
		max_iter=200,
		convergence_iter=15,
		verbose=0,
		random_state=None
	):
		"""[Do Affinity Propogation clustering. The embedding object is required. Either a DimReduction object or each of the gt, pops, prefix, reps, and [colors or palette] objects are also required. Clusters are assessed as being nearby to examplar samples that best summarize the data. Clusters are determined as being similar to the examplar data points, and similarity is communicated among points using a propogation algorithm]

		Args:
			embedding ([runPCA or runMDS object]): [Embedded data created by runPCA() or runMDS()]

			proximity_matrix ([runRandomForestUML proximity matrix]). [Can be retrieved as runRandomForestUML.proximity_matrix]

			dimreduction ([DimReduction object], optional): [Initialized DimReduction object. If not supplied, gt, pops, prefix, reps, colors, and palette must be supplied instead]. Defaults to None.

			gt ([pandas.DataFrame, numpy.ndarray, or 2D-list], optional): [One of the retrievable objects created by GenotypeData() of shape (n_samples, n_sites)]. Defaults to None.

			pops ([list(str)], optional): [List of populations created by GenotypeData of shape (n_samples,). Can be retrieved as GenotypeData.populations]. Defaults to None.

			prefix ([str], optional): [Prefix for outptut files and plots]. Defaults to None.

			reps ([int], optional): [Number of replicates to perform]. Defaults to None.

			colors ([dict(str)], optional): [Dictionary with unique populations IDs as the keys and hex-code colors as the values]. Defaults to None.

			palette (str, optional): [matplotlib color palette to use if colors=None]. Defaults to "Set1".

			sampleids ([list(str)], optional): [Sample IDs to write to labels files of shape (n_samples,). Can be retrieved as GenotypeData.individuals]. Defaults to None.

			damping (float, optional): [Damping factor (between 0.5 and 1) is the extent to which the current value is maintained relative to incoming values (weighted 1 - damping). This is in order to avoid numerical oscilliations when updating the values (messages)]. Defaults to 0.5.

			max_iter (int, optional): [Maximum number of iterations]. Defaults to 200.

			convergence_iter (int, optional): [Number of iterations with no chnage in the number of estimated clusters that stops the convergence]. Defaults to 15.

			verbose (int, optional): [Verbosity level. 0=less verbose]. Defaults to 0. 

			random_state (int, optional): [Pseudo-random number generator to control the starting state. Use an int for reproducible results across function calls or leave as None for randomization between function calls]. Defaults to None.

		Raises:
			ValueError: ['verbose' argument must be an integer >= 0]
		"""
		# Initialize parent class
		super().__init__(gt, pops, prefix, reps, colors, palette)

		# Validates passed arguments and sets parent class attributes
		self._validate_args(dimreduction, gt, pops, prefix, reps)

		# Child class attributes
		self.proximity_matrix = proximity_matrix
		self.sampleids = sampleids
		self.damping = damping
		self.max_iter = max_iter
		self.convergence_iter = convergence_iter
		self.verbose = verbose
		self.random_state = random_state
		self.clust_method = "AffinityPropogation"
		self.affinity = "precomputed"

		# To store results
		self.labels = list()
		self.models = list()
		self.pred_labels = list()
		self.cluster_centers_indices = list()
		self.best_preference = None
		self.best_damping = None

		# Get attributes from already instantiated embedding
		self.coords = embedding.coords
		self.method = embedding.method

		if self.verbose == 0:
			self.verbose = False
		elif self.verbose >=1:
			self.verbose = True
		else:
			raise ValueError("'verbose' argument must be an integer >= 0")

		self.run_ideal_clustering(self.proximity_matrix)

		print("\nBest optimized preference setting: {}".format(
			self.best_preference)
		)
		print("Best optimized damping setting: {}\n".format(
			self.best_damping)
		)
				
	def fit_predict(self, X, preference):
		"""[Fit model and predict cluster labels. Sets self.labels (predicted labels) and self.models (fit model). self.labels and self.models are lists with each item corresponding to one replicate]

		Args:
			X ([pandas.DataFrame]): [Coordinates following dimensionality reduction embedding of shape (n_samples, n_features). Accessed via self.coords]

			damping (float, optional): [damping parameter setting between 0.5 and 1]
		"""
		print("\nDoing Affinity Propogation Clustering...\n")

		initial_damping = self.damping

		af = AffinityPropagation(
			damping=initial_damping,
			max_iter=self.max_iter,
			convergence_iter=self.convergence_iter,
			preference=preference,
			affinity=self.affinity,
			verbose=self.verbose,
			random_state=self.random_state
		)

		l = af.fit_predict(X)
		m = af

		l, m, final_damping = \
			self.tune_damping(X, l, m, initial_damping, preference)

		self.best_damping = final_damping

		return l, len(set(l)), af.cluster_centers_indices_, m

	def tune_damping(self, _X, _l, _m, _damping, _preference, increment=0.05):
		"""[Checks if predicted labels are all -1, and if so increases the damping parameter by increment]

		Args:
			_X ([pandas.DataFrame]): [Input coordinates as pandas.DataFrame of shape (n_samples, n_features)]

			_l ([list(int)]): [Predicted labels for affinity propogation]

			_damping ([type]): [Current damping parameter setting]

			increment (float, optional): [Value to increment by if damping setting predicts all -1 labels]

		Returns:
			_l: [list(int)]: [Predicted labels]
			_m: [sklearn.cluster.AffinityPropogation]: [Fit model]
			_damping: [float]: [Current damping value]
		"""
		if all(x == -1 for x in _l):
			if _damping <= 1.0 - increment:
				_damping += increment
			else:
				return _l, _m, _damping

			_af = AffinityPropogation(
			damping=_damping,
			max_iter=self.max_iter,
			convergence_iter=self.convergence_iter,
			preference=_preference,
			affinity=self.affinity,
			verbose=self.verbose,
			random_state=self.random_state
		)
			_l = None
			_m = None
			_l = _af.fit_predict(_X)
			_m = _af

			_l, _m, _damping = self.tune_damping(_X, _l, _damping)

			return _l, _m, _damping

		else:
			return _l, _m, _damping

	def is_tuning_required(self, similarity_matrix, rows_of_cluster):
		df = similarity_matrix[rows_of_cluster]

		for val in df.values:
			if all(np.where(val > 0.5, True, False)):
				continue

			return True

		return False

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

	def run_ideal_clustering(self, similarity):

		for rep in progressbar(
			range(self.reps), 
			"{}: ".format(self.clust_method.upper())
		):
			preference_tuning_range = self.get_pref_range(similarity[rep])

			best_tested_preference = None
			for preference in preference_tuning_range:
				labels, labels_count, cluster_centers_indices, af_model = self.fit_predict(similarity[rep], preference)

				needs_tuning = False
				wrong_clusters = 0
				for label_index in range(labels_count):
					cluster_elements_indexes = np.where(labels == label_index)[0]

					tuning_required = self.is_tuning_required(similarity[rep], cluster_elements_indexes)

					if tuning_required:
						wrong_clusters += 1

						if not needs_tuning:
							needs_tuning = True

				if best_tested_preference is None or wrong_clusters < best_tested_preference[1]:
					best_tested_preference = (preference, wrong_clusters)

				if not needs_tuning:
					self.labels.append(labels), 
					self.models.append(af_model)
					self.save_labels(rep)
					self.best_preference = best_tested_preference[0]
					break

			# The clustering has not been tuned enough during the iterations, we choose the less wrong clusters
			labs, lab_count, centers, mymodel = self.fit_predict(similarity[rep], best_tested_preference[0])

			self.labels.append(labs)
			self.models.append(mymodel)
			self.save_labels(rep)
			self.best_preference = best_tested_preference[0]







