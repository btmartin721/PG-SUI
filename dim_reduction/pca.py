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

	def __init__(self, gn, pops, algorithms=None, settings=None):
		self.gn = gn
		self.pops = pops
		self.settings = settings
		self.algorithms = algorithms
		self.pca_coords = None
		self.pca_model = None

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
		X = StandardScaler().fit_transform(self.gn.values)
		y = np.asarray(self.pops)
		y_df = pd.DataFrame(y, columns=["target"])

		# Do the PCA using scikit-learn
		pca = PCA(n_components = pca_arguments["n_components"])
		principal_components = pca.fit_transform(X)

		# Get list of all the PC numbers
		pc_max = pca_arguments["n_components"] + 1
				
		pc_list = list()
		pcs = list(range(1, pc_max))
		for pc in pcs:
			pc_list.append("Principal Component {}".format(str(pc)))

		# Make a pandas DataFrame object
		principal_df = pd.DataFrame(principal_components, columns=pc_list)

		self.pca_coords = pd.concat([principal_df, y_df[["target"]]], axis=1)

		# gn = np.array(self.data).transpose()
		# print(gn.shape)

		# self.pca_coords, self.pca_model = allel.pca(gn, n_components=pca_arguments["n_components"], copy=pca_arguments["copy"], scaler=pca_arguments["scaler"], ploidy=pca_arguments["ploidy"])


	def plot_pca(self, prefix, pc1=1, pc2=2, figwidth=8, figheight=8, alpha=1.0, colors=None):

		fig = plt.figure(figsize = (8, 8))
		ax = fig.add_subplot(1, 1, 1)
		ax.set_xlabel("Principal Component {}".format(str(pc1)), fontsize = 15)
		ax.set_ylabel("Principal Component {}".format(str(pc2)), fontsize = 15)
		ax.set_title("PCA", fontsize = 20)

		targets = list(set(self.pops))

		if not colors: # Make a list of hex-coded colors
			colors = list()
			cmap = plt.get_cmap("Set1", len(targets))

			for i in range(cmap.N):
				rgba = cmap(i)
				colors.append(mcolors.rgb2hex(rgba))

		else:
			if len(colors) != len(targets):
				raise ValueError("\nThe colors argument's list length must equal the number of unique populations!")

		for target, color in zip(targets, colors):
			indices_to_keep = self.pca_coords["target"] == target
			ax.scatter(self.pca_coords.loc[indices_to_keep, "Principal Component {}".format(pc1)],
			self.pca_coords.loc[indices_to_keep, "Principal Component {}".format(pc2)],
			c = color,
			s = 50)

			

		ax.legend(targets)
		plot_fn = "{}_pca.pdf".format(prefix)
		fig.savefig(plot_fn, bbox_inches="tight")
		


			

		# fig, ax = plt.subplots(figsize=(figwidth, figheight))
		# sns.despine(ax=ax, offset=10)
		# x = self.pca_coords[:, pc1]
		# y = self.pca_coords[:, pc2]

		# pca_df = pd.DataFrame({
		# 	"pc1": x,
		# 	"pc2": y,
		# 	"pop": self.pops
		# })
		#pop_df = pd.DataFrame(self.pops, columns=["population"])
		
		#for pop in list(set(self.pops)):
		# if colors:
		# 	grouped = pca_df.groupby("population")
		# 	for key, group in grouped:
		# 		#group. 
		# 		#flt = (pop_df.population == pop).values
		# 	ax.plot(x[flt], y[flt], marker='o', linestyle=' ', label=pop, markersize=6, alpha=alpha)
		# ax.set_xlabel('PC%s (%.1f%%)' % (pc1+1, self.pca_model.explained_variance_ratio_[pc1]*100))
		# ax.set_ylabel('PC%s (%.1f%%)' % (pc2+1, self.pca_model.explained_variance_ratio_[pc2]*100))
		# ax.legend();
		# plot_fn = "{}_pca.pdf".format(prefix)
		# plt.savefig(plot_fn, bbox_inches="tight")



	@property
	def get_pca_coords(self):
		return self.pca_coords

	@property
	def get_pca_model(self):
		return self.pca_model
