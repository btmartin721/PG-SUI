# Standard library imports
import sys

# Third-party imports
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score

class Clustering:
	"""[Class to perform unsupervised clustering on embedded data]
	"""

	def __init__(self):
		self.pam = None
		self.dbscan = None
		self.affinity_prop = None
		self.hier = None
		self.birch = None
		self.kmeans = None
		self.mean_shift = None
		self.optics = None
		self.spectral = None
		self.gaussian_mix = None

		self.coords = None

	def pam_clustering(self, embedding, maxk, pops, algorithm, method, prefix, pam_settings, is_3d=False, plot_settings=None, colors=None, palette="Set1"):

		df = self._validate_type(embedding)

		if is_3d:
			df.columns = ["axis1", "axis2", "axis3"]

		else:
			df.columns = ["axis1", "axis2"]

		sw = list()
		for i in range(2, maxk):
			medoids = KMedoids(
				n_clusters=i, 
				metric=pam_settings["metric"], 
				method=pam_settings["method"], 
				init=pam_settings["init"], 
				max_iter=pam_settings["max_iter"], 
				random_state=pam_settings["random_state"]
			)

			medoids.fit(embedding)
			y_kmed = medoids.fit_predict(embedding)
			silhouette_avg = silhouette_score(embedding, y_kmed)
			sw.append(silhouette_avg)

	def plot_sil(self, sw, maxk, plot_settings, algorithm, method, prefix):
		fig = plt.figure(figsize=(plot_settings["figwidth"], plot_settings["figheight"]))

		fig.plot(range(2, maxk), sw)

		fig.title("Silhouette Score")
		fig.xlabel("Number of Clusters")
		fig.ylabel("Silhouette Width")

		plot_fn = "{}_{}_{}.pdf".format(prefix, algorithm, method)

		fig.savefig(plot_fn, bbox_inches="tight")
	
	def plot_clusters(self, coords, pops, method, colors, palette, plot_settings, is_3d)

		if method == "cmds":
			method = "cMDS"

		elif method == "isomds": 
			method = "isoMDS"

		elif method == "tsne":
			method = "T-SNE"

		elif method == "pca":
			method = "PCA"

		fig = plt.figure(1, figsize=(plot_settings["figwidth"], plot_settings["figheight"]))

		ax = Axes3D(fig, elev=plot_settings["elev"], azim=plot_settings["azim"])

		x_idx = int(plot_settings["axis1"] - 1)
		y_idx = int(plot_settings["axis2"] - 1)

		x = coords[:, x_idx]
		y = coords[:, y_idx]

		if is_3d:
			z_idx = int(plot_settings["axis3"] - 1)
			z = coords[:, z_idx]

		unique_populations = list(set(pops))
		populations = pd.DataFrame(pops, columns=["population"])

		colors = self._get_pop_colors(targets, palette, colors)

		for pop in unique_populations:
			flt = (populations.population == pop)

			if is_3d:			
				ax.scatter(
					x[flt], 
					y[flt], 
					z[flt],
					marker=marker, 
					linestyle=' ', 
					color=pop_colors[pop], 
					label=pop, 
					markersize=markersize, 
					mec=markeredgecolor, 
					mew=markeredgewidth, 
					alpha=alpha
				)

			else:
				ax.scatter(
					x[flt], 
					y[flt], 
					marker=marker, 
					linestyle=' ', 
					color=pop_colors[pop], 
					label=pop, 
					markersize=markersize, 
					mec=markeredgecolor, 
					mew=markeredgewidth, 
					alpha=alpha
				)

		if method == "PCA":
			ax.set_xlabel('PC%s (%.1f%%)' % (axis1, model.explained_variance_ratio_[axis1_idx]*100))

			ax.set_ylabel('PC%s (%.1f%%)' % (axis2, model.explained_variance_ratio_[axis2_idx]*100))

			if plot_3d:
				ax.set_zlabel('PC%s (%.1f%%)' % (axis3, model.explained_variance_ratio_[axis3_idx]*100))

		else:
			ax.set_xlabel("{} Axis {}".format(method, axis1))
			ax.set_ylabel("{} Axis {}".format(method, axis2))

			if plot_3d:
				ax.set_zlabel("{} Axis {}".format(method, axis3))

	def _get_pop_colors(self, uniq_pops, palette, colors):
		"""[Get population color codes if colors=None]

		Args:
			uniq_pops ([list]): [Unique population IDs]

			palette ([str]): [Color palette to use. See matplotlib.colors documentation]

			colors ([dict]): [Dictionary with unique population IDs as keys and hex-code colors as values. If colors=None, the color palette is automatically set]

		Raises:
			ValueError: [colors must be equal to the number of unique populations]

		Returns:
			[dict]: [Dictionary with unique population IDs as the keys and hex-code colors as the values]
		"""

		if not colors: # Make a list of hex-coded colors
			colors = dict()
			cmap = plt.get_cmap(palette, len(uniq_pops))

			for i in range(cmap.N):
				rgba = cmap(i)
				colors[uniq_pops[i]] = mcolors.rgb2hex(rgba)

		else:
			if len(colors.keys()) != len(uniq_pops):
				raise ValueError("\nThe colors argument's list length must equal the number of unique populations!")

		return colors


	def _validate_type(self, obj):
	"""[Validate that the genotypes are of the correct type. Also converts numpy.ndarrays and list(list) into pandas.DataFrame for use with DelimModel]

	Args:
		obj ([numpy.ndarray, pandas.DataFrame, or list(list)]) : [GenotypeData object containing the genotype data]

	Returns:
		([pandas.DataFrame]): [Object converted to pandas DataFrame]

	Raises:
		TypeError: [Must be of type numpy.ndarray, list, or pandas.DataFrame]
	"""
	if isinstance(obj, np.ndarray):
		df = pd.DataFrame(obj)

	elif isinstance(obj, list):
		df = pd.DataFrame.from_records(obj)

	elif isinstance(obj, pd.DataFrame):
		df = obj.copy()

	else:
		raise TypeError("\nThe genotype data must be a numpy.ndarray, a pandas.DataFrame, or a 2-dimensional list! Any of these can be retrieved from the GenotypeData object")

	return df


