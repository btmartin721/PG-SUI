import sys
import allel

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
import seaborn as sns
sns.set_style("white")
sns.set_style("ticks")

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class DimReduction:

	def __init__(self, data, pops, algorithms=None, settings=None):
		self.data = data
		self.pops = pops
		self.settings = settings
		self.algorithms = algorithms
		self.pca_coords = None
		self.pca_model = None
		self.target = None

		self.all_settings = {"n_components": 10, 
									"copy": True, 
									"scaler": "patterson", 
									"ploidy": 2
								}

		if self.algorithms:
			self.algorithms = [self.algorithms]

		for arg in self.algorithms:
			if arg not in ["standard-pca"]:
				raise ValueError("\nThe argument {} is not supported. Supported options include: [standard-pca]".format(arg))

			if self.settings and arg == "standard-pca":
				self.all_settings.update(self.settings)
				self.standard_pca(self.all_settings)
			elif not self.settings and arg == "standard-pca":
				self.standard_pca(self.all_settings)

		

	def standard_pca(self, all_settings):

		pca_arguments = ["n_components", "copy", "scaler", "ploidy"]

		# Subset all_settings to only keys in pca_arguments
		pca_arguments = {key: all_settings[key] for key in pca_arguments}

		# Standardize the features (mean = 0, variance = 1)
		# Important for doing PCA
		# X = StandardScaler().fit_transform(self.gn.values)
		# self.target = pd.DataFrame(self.pops, columns=["target"])

		# # Do the PCA using scikit-learn
		# pca = PCA(n_components = pca_arguments["n_components"])
		# principal_components = pca.fit_transform(X)

		# # Get list of all the PC numbers
		# pc_max = pca_arguments["n_components"] + 1
		
		# # Make the list into PC names and numbers
		# pc_list = list()
		# pcs = list(range(1, pc_max))
		# for pc in pcs:
		# 	pc_list.append("Principal Component {}".format(str(pc)))

		# # Make a pandas DataFrame object
		# self.pca_coords = pd.DataFrame(principal_components, columns=pc_list)

		gn = np.array(self.data).transpose()

		self.pca_coords, self.pca_model = allel.pca(gn, n_components=pca_arguments["n_components"], copy=pca_arguments["copy"], scaler=pca_arguments["scaler"], ploidy=pca_arguments["ploidy"])


	def plot_pca(self, prefix, pc1=1, pc2=2, figwidth=6, figheight=6, alpha=1.0, colors=None, palette="Set1", legend_loc="upper left"):

		pc1_idx = pc1 - 1
		pc2_idx = pc2 - 1

		x = self.pca_coords[:, pc1_idx]
		y = self.pca_coords[:, pc2_idx]

		targets = list(set(self.pops))

		pca_df = pd.DataFrame({
			"pc1": x,
			"pc2": y,
			"pop": self.pops
		})
		pop_df = pd.DataFrame(self.pops, columns=["population"])
		
		fig = plt.figure(figsize=(figwidth, figheight))
		ax = fig.add_subplot(1,1,1)

		colors = self._get_pop_colors(targets, palette, colors)

		self._plot_pca_coords(self.pca_coords, self.pca_model, pc1, pc2, ax, pop_df, targets, colors)

		ax.legend(loc=legend_loc)
		plot_fn = "{}_pca.pdf".format(prefix)
		fig.savefig(plot_fn, bbox_inches="tight")
		
	def _plot_pca_coords(self, coords, model, pc1, pc2, ax, populations, unique_populations, pop_colors):

		sns.despine(ax=ax, offset=5)

		pc1_idx = pc1 - 1
		pc2_idx = pc2 - 1

		x = coords[:, pc1_idx]
		y = coords[:, pc2_idx]

		for pop in unique_populations:
			flt = (populations.population == pop)
			ax.plot(x[flt], y[flt], marker='o', linestyle=' ', color=pop_colors[pop], 
                label=pop, markersize=6, mec='k', mew=.5)

		ax.set_xlabel('PC%s (%.1f%%)' % (pc1, model.explained_variance_ratio_[pc1_idx]*100))
		
		ax.set_ylabel('PC%s (%.1f%%)' % (pc2, model.explained_variance_ratio_[pc2_idx]*100))

	def _get_pop_colors(self, uniq_pops, palette, colors):

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

	@property
	def get_pca_coords(self):
		return self.pca_coords

	@property
	def get_pca_model(self):
		return self.pca_model
