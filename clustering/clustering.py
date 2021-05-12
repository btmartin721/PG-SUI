# Standard library imports
import sys

# Third-party imports
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score

# Custom imports
from dim_reduction.dim_reduction import DimReduction

class PamClustering(DimReduction):
	"""[Class to perform unsupervised clustering on embedded data]
	"""

	def __init__(self, embedding, dimreduction=None, gt=None, pops=None, prefix=None, colors=None, palette="Set1", maxk=8, sampleids=None, plot=False, metric="euclidean", clust_method="pam", init="heuristic", max_iter=300, random_state=None):

		# Initialize parent class
		super().__init__(gt, pops, prefix, colors, palette)

		# Validates passed arguments and sets parent class attributes
		self._validate_args(dimreduction, gt, pops, prefix)

		# Set child class attributes
		self.maxk = maxk
		self.sampleids = sampleids
		self.plot = plot
		self.metric = metric
		self.clust_method = clust_method
		self.init = init
		self.max_iter = max_iter
		self.random_state=random_state

		# To store results
		self.clusters = dict()
		self.labels = dict()
		self.models = dict()

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

		self.fit(self.coords)

	def fit(self, X):

		print("\nDoing K-Medoids Clustering...\n")
		print("Searching for optimal K...\n")

		for k in range(2, self.maxk+1):

			km = KMedoids(
				n_clusters=k,
				metric=self.metric,
				method=self.clust_method,
				init=self.init,
				max_iter=self.max_iter,
				random_state=self.random_state
			)

			km.fit(X)

			self.clusters[k] = km.transform(X)
			self.labels[k] = km.predict(X)
			self.models[k] = km

