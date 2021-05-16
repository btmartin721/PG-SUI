# Standard library imports

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
	"""[Class to perform unsupervised clustering on embedded data]
	"""

	def __init__(self, embedding, dimreduction=None, gt=None, pops=None, prefix=None, reps=None, colors=None, palette="Set1", maxk=8, sampleids=None, metric="euclidean", clust_method="pam", init="heuristic", max_iter=300, random_state=None):

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
		self.clusters = list()
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

		print("\nDoing K-Medoids Clustering...\n")

		for rep in progressbar(range(self.reps), "K-Medoids: "):
			c = dict()
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

				c[k] = km.transform(X[rep])
				l[k] = km.predict(X[rep])
				m[k] = km

			self.clusters.append(c)
			self.labels.append(l)
			self.models.append(m)

class KMeansClustering(DimReduction):

	def __init__(self, embedding, dimreduction=None, gt=None, pops=None, prefix=None, reps=None, colors=None, palette="Set1", maxk=8, sampleids=None, init="k-means++", n_init=10, max_iter=300, tol=1e-4, verbose=0, random_state=None, algorithm="auto"):

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
		self.clusters = list()
		self.labels = list()
		self.models = list()
		self.pred_labels = list()


		# Get attributes from already instantiated embedding
		self.coords = embedding.coords
		self.method = embedding.method

		self.fit_predict(self.coords)

	def fit_predict(self, X):

		print("\nDoing K-Means Clustering for K=2 "
			"to K={}...\n".format(self.maxk+1))

		for rep in progressbar(range(self.reps), "K-Means: "):
			c = dict()
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
				
				c[k] = km.transform(X[rep])
				l[k] = km.predict(X[rep])
				m[k] = km

			self.clusters.append(c)
			self.labels.append(l)
			self.models.append(m)

class DBSCANClustering(DimReduction):

	def __init__(self, embedding, dimreduction=None, gt=None, pops=None, prefix=None, reps=None, colors=None, palette="Set1", plot_eps=False, sampleids=None, min_samples=5, metric="euclidean", metric_params=None, algorithm="auto", leaf_size=30, p=None):

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
		self.clusters = list()
		self.labels = list()
		self.models = list()
		self.pred_labels = list()

		# Get attributes from already instantiated embedding
		self.coords = embedding.coords
		self.method = embedding.method

		self.fit_predict(self.coords)

	def fit_predict(self, X):

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
		neigh = NearestNeighbors(n_neighbors=2)
		nbrs = neigh.fit(X[rep])
		dist, idx = nbrs.kneighbors(X[rep])
		dist = dist[:,1]

		dist = np.sort(dist, axis=0)
		i = np.arange(len(dist))

		kneedle = KneeLocator(i, dist, S=1, curve="convex", direction="increasing", interp_method="polynomial")

		if plot_eps:
			fig = plt.figure(figsize=(5, 5))

			kneedle.plot_knee()
			plt.xlabel("Points")
			plt.ylabel("Distance")

			plt.savefig("{}_output/{}/{}/plots/dbscan_eps_{}.pdf".format(self.prefix, self.method, self.clust_method, rep+1))

		return dist[kneedle.knee]


