import sys
import allel

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
import seaborn as sns
sns.set_style("white")
sns.set_style("ticks")

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import MDS

# Custom module imports
from utils import settings

class DimReduction:
	"""[Class to perform dimensionality reduction on genotype features]
	"""

	def __init__(self, data, pops):
		"""[Class constructor]

		Args:
			data ([pandas.DataFrame]): [012-encoded genotypes. There should not be any missing data or else the PCA will not run. If your dataset has missing data, use one of our imputation methods.]

			pops ([list(str)]): [Population IDs]

			algorithms ([list or str], optional): [Dimensionality reduction algorithms to use]. Defaults to None.

			settings ([dict], optional): [Dictionary with dimensionality reduction option names as keys and the settings as values]. Defaults to None.

		Raises:
			ValueError: [Must be a supported dimensionality reduction algorithm]
		"""
		self.data = data
		self.pops = pops
		self.pca_coords = None
		self.pca_model = None
		self.cmds_model = None
		self.isomds_model = None
		self.target = None
		self.inflection = None

	def standard_pca(self, pca_arguments):
		"""[Does standard PCA using scikit-allel. By default uses a Patterson scaler]

		Args:
			all_settings ([dict]): [Dictionary with option names as keys and the settings as values]
		"""
		print("\nDoing PCA...\n")
		print(
				"""
				PCA Settings:
					n_components: """+str(pca_arguments["n_components"])+"""
					scaler: """+str(pca_arguments["scaler"])+"""
					ploidy: """+str(pca_arguments["ploidy"])+"""
				"""
		)

		gn = np.array(self.data).transpose()

		self.pca_coords, self.pca_model = allel.pca(gn, n_components=pca_arguments["n_components"], scaler=pca_arguments["scaler"], ploidy=pca_arguments["ploidy"])

		print("\nDone!")

	def do_mds(self, data, mds_arguments, metric=True):
		
		if metric:
			print("\nDoing cMDS dimensionality reduction...\n")
		else:
			print("\nDoing isoMDS dimensionality reduction...\n")
			
		print(
				"""
				MDS Settings:
					n_dims: """+str(mds_arguments["n_dims"])+"""
					n_init: """+str(mds_arguments["n_init"])+"""
					max_iter: """+str(mds_arguments["max_iter"])+"""
					eps: """+str(mds_arguments["eps"])+"""
					n_jobs: """+str(mds_arguments["n_jobs"])+"""
					dissimilarity: """+str(mds_arguments["dissimilarity"])+"""
					random_state: """+str(mds_arguments["random_state"])+"""
					verbose: """+str(mds_arguments["verbose"])+"""
				"""
		)

		scaler = MinMaxScaler()
		df_X_scaled = scaler.fit_transform(self.data)

		mds = MDS(
					n_components=mds_arguments["n_dims"], 
					random_state=mds_arguments["random_state"], 
					metric=metric,
					n_init=mds_arguments["n_init"],
					max_iter=mds_arguments["max_iter"],
					verbose=mds_arguments["verbose"],
					eps=mds_arguments["eps"],
					n_jobs=mds_arguments["n_jobs"],
					dissimilarity=mds_arguments["dissimilarity"]
				)

		if metric:
			self.cmds_model = mds.fit_transform(df_X_scaled)

		else:
			self.isomds_model = mds.fit_transform(df_X_scaled)

		print("\nDone!")

	def plot_dimreduction(self, prefix, pca=False, cmds=False, isomds=False, user_settings=None, colors=None, palette="Set1"):
		"""[Plot PCA results as a scatterplot and save it as a PDF file]

		Args:
			prefix ([str]): [Prefix for output PDF filename]

			pca (bool, optional): [True if plotting PCA results. Cannot be set at same time as cmds and isomds]. Defaults to False.

			cmds (bool, optional): [True if plotting cmds results. Cannot be set at same time as pca and isomds]. Defaults to False.

			isomds (bool, optional): [True if plotting isomds results. Cannot be set at same time as pca and cmds]. Defaults to False.

			settings (dict, optional): [Dictionary with plot setting arguments as keys and their corresponding values. Some or all of the arguments can be set]. Defaults to None.

			colors (dict, optional): [Dictionary with unique population IDs as keys and hex-code colors as values]. Defaults to None.

			palette (str, optional): [matplotlib.colors palette to be used if colors=None]. Defaults to 'Set1'.

		Raises:
			ValueError: [Only one of pca, cmds, and isomds can be set to True]

		"""
		if not pca and not cmds and not isomds:
			raise ValueError("One of the pca, cmds, or isomds arguments must be set to True")

		if (pca and cmds) or (pca and isomds) or (cmds and isomds) or (cmds and pca and isomds):
			raise ValueError("Only one of the pca, cmds, or isomds arguments can be set to True")

		settings_default = settings.dimreduction_plot_settings()

		if user_settings:
			settings_default.update(user_settings)

		axis1_idx = settings_default["axis1"] - 1
		axis2_idx = settings_default["axis2"] - 1

		if pca:
			print("\nPlotting PCA results...")
			x = self.pca_coords[:, axis1_idx]
			y = self.pca_coords[:, axis2_idx]

		if cmds:
			print("\nPlotting cMDS results...")
			x = self.cmds_model[:, axis1_idx]
			y = self.cmds_model[:, axis2_idx]

		if isomds:
			print("\nPlotting isoMDS results...")
			x = self.isomds_model[:, axis1_idx]
			y = self.isomds_model[:, axis2_idx]

		targets = list(set(self.pops))

		pop_df = pd.DataFrame(self.pops, columns=["population"])
		
		fig = plt.figure(figsize=(settings_default["figwidth"], settings_default["figheight"]))
		ax = fig.add_subplot(1,1,1)

		colors = self._get_pop_colors(targets, palette, colors)

		if pca:
			self._plot_coords(self.pca_coords, settings_default["axis1"], settings_default["axis2"], ax, pop_df, targets, colors, settings_default["alpha"], settings_default["marker"], settings_default["markersize"], settings_default["markeredgecolor"], settings_default["markeredgewidth"], pca, cmds, isomds, model=self.pca_model)

		elif cmds:
			self._plot_coords(self.cmds_model, settings_default["axis1"], settings_default["axis2"], ax, pop_df, targets, colors, settings_default["alpha"], settings_default["marker"], settings_default["markersize"], settings_default["markeredgecolor"], settings_default["markeredgewidth"], pca, cmds, isomds)

		elif isomds:
			self._plot_coords(self.isomds_model, settings_default["axis1"], settings_default["axis2"], ax, pop_df, targets, colors, settings_default["alpha"], settings_default["marker"], settings_default["markersize"], settings_default["markeredgecolor"], settings_default["markeredgewidth"], pca, cmds, isomds)

		if settings_default["legend"]:
			if settings_default["legend_inside"]:
				if settings_default["bbox_to_anchor"][0] > 1 or \
					settings_default["bbox_to_anchor"] > 1:
					print("Warning: bbox_to_anchor was set grater than 1.0 (outside plot margins) but legend_inside was set to True. Setting bbox_to_anchor to (1.0, 1.0)")

				ax.legend(loc=settings_default["legend_loc"], labelspacing=settings_default["labelspacing"], columnspacing=settings_default["columnspacing"], title=settings_default["title"], title_fontsize=settings_default["title_fontsize"], markerfirst=settings_default["markerfirst"], markerscale=settings_default["markerscale"], ncol=settings_default["ncol"], bbox_to_anchor=settings_default["bbox_to_anchor"], borderaxespad=settings_default["borderaxespad"], edgecolor=settings_default["legend_edgecolor"], facecolor=settings_default["facecolor"], framealpha=settings_default["framealpha"], shadow=settings_default["shadow"])

			else:
				if settings_default["bbox_to_anchor"][0] < 1 and \
					settings_default["bbox_to_anchor"][1] < 1:
					print("Warning: bbox_to_anchor was set less than 1.0 (inside the plot margins) but legend_inside was set to False. Setting bbox_to_anchor to (1.05, 1.0)")

				ax.legend(loc=settings_default["legend_loc"], labelspacing=settings_default["labelspacing"], columnspacing=settings_default["columnspacing"], title=settings_default["title"], title_fontsize=settings_default["title_fontsize"], markerfirst=settings_default["markerfirst"], markerscale=settings_default["markerscale"], ncol=settings_default["ncol"], bbox_to_anchor=settings_default["bbox_to_anchor"], borderaxespad=settings_default["borderaxespad"], edgecolor=settings_default["legend_edgecolor"], facecolor=settings_default["facecolor"], framealpha=settings_default["framealpha"], shadow=settings_default["shadow"])

		if pca:
			plot_fn = "{}_pca.pdf".format(prefix)
		elif cmds:
			plot_fn = "{}_cmds.pdf".format(prefix)
		elif isomds:
			plot_fn = "{}_isomds.pdf".format(prefix)

		fig.savefig(plot_fn, bbox_inches="tight")

		print("Done!")

		if pca:
			print("\nSaved PCA scatterplot to {}".format(plot_fn))
		elif cmds:
			print("\nSaved cMDS scatterplot to {}".format(plot_fn))
		elif isomds:
			print("\nSaved isoMDS scatterplot to {}".format(plot_fn))

	def plot_pca_cumvar(self, prefix, user_settings):
		"""[Plot cumulative variance for principal components with xintercept line at the inflection point]

		Args:
			prefix ([str]): [Prefix for output PDF filename]

			user_settings ([dict]): [Dictionary with matplotlib arguments as keys and their corresponding values. Only some or all of the settings can be changed]

		Raises:
			AttributeError: [pca_model must be defined prior to running this function]
		"""
		# Raise error if PCA hasn't been run yet
		if not self.pca_model:
			raise AttributeError("\nA PCA object has not yet been created! Please do so with DelimModel([arguments]).dim_reduction(dim_red_algorithms['standard-pca'], [other_arguments])")

		# Retrieve default plot settings
		settings_default = settings.pca_cumvar_default_settings()

		# Update plot settings with user-specified settings
		if user_settings:
			settings_default.update(user_settings)

		# Get the cumulative sum of the explained variance ratio
		cumsum = np.cumsum(self.pca_model.explained_variance_ratio_)

		# Compute first derivative with gaussian filter
		# to get smoothed histogram
		# Uses scipy
		smooth = gaussian_filter1d(cumsum, 100)

		# Compute second derivative
		smooth_d2 = np.gradient(np.gradient(smooth))

		# Get the inflection point
		inflection = np.where(np.diff(np.sign(smooth_d2)))[0]

		# Sets plot background style
		# Uses seaborn
		sns.set(style=settings_default["style"])

		# Plot the results
		# Uses matplotlib.pyplot
		fig = plt.figure(figsize=(settings_default["figwidth"], settings_default["figheight"]))
		ax = fig.add_subplot(1,1,1)

		# Plot the explained variance ratio
		ax.plot(np.cumsum(self.pca_model.explained_variance_ratio_), color=settings_default["linecolor"], linewidth=settings_default["linewidth"])

		ax.set_xlabel("Number of Components")
		ax.set_ylabel("Cumulative Explained Variance")

		# Add text to show inflection point
		ax.text(0.95, 0.01, "Inflection Point={}".format(inflection[0]), verticalalignment="bottom", horizontalalignment="right", transform=ax.transAxes, color="k", fontsize=settings_default["text_size"])

		# Add inflection point
		ax.axvline(linewidth=settings_default["xintercept_width"], color=settings_default["xintercept_color"], linestyle=settings_default["xintercept_style"], x=inflection[0], ymin=0, ymax=1)

		# Add prefix to filename
		plot_fn = "{}_pca_cumvar.pdf".format(prefix)

		# Save as PDF file
		fig.savefig(plot_fn, bbox_inches="tight")

		self.inflection = inflection[0]
		
	def _plot_coords(self, coords, axis1, axis2, ax, populations, unique_populations, pop_colors, alpha, marker, markersize, markeredgecolor, markeredgewidth, pca, cmds, isomds, model=None):
		"""[Map colors to populations and make the scatterplot]

		Args:
			coords ([numpy.array]): [pca_coords object returned from scikit-allel PCA or cmds_model or isomds_model objects stored in do_mds()]

			axis1 ([int]): [First axis to plot. Starts at 1]

			axis2 ([int]): [Second axis to plot. Starts at 1]

			ax ([matplotlib object]): [ax object from matplotlib]

			populations ([pandas.DataFrame]): [pandas.DataFrame with population IDs]

			unique_populations ([list]): [Unique populations in the populations argument]

			pop_colors ([dict]): [Dictionary with unique population IDs as keys and hex-code colors as values]

			alpha ([float]): [Set transparency of points; lower = more transparent. Should be between 0 and 1]

			marker ([str]): [Set the marker shape. See matplotlib.markers documentation].

			markersize ([int]): [Set size of the markers]

			markeredgecolor ([str]): [Set the color of the marker edge]. See matplotlib.pyplot.plot documentation.

			markeredgewidth ([float]): [Set the width of the marker edge].

			pca ([bool]): [True if doing PCA. False if doing cmds or isomds]

			cmds ([bool]): [True if doing cmds. False if doing pca or isomds]

			isomds ([bool]): [True if doing isomds. False if doing pca or cmds]

			model (scikit-allel object, optional): [Second object returned from scikit-allel PCA. Set model=None if doing cmds or isomds]

		Raises:
			ValueError: [Make sure model argument is set if pca is True]
			ValueError: [Make sure one of pca, cmds, or isomds is True]
		"""
		if pca and not model:
			raise ValueError("The model argument must be set if pca is True")

		sns.despine(ax=ax, offset=5)

		axis1_idx = axis1 - 1
		axis2_idx = axis2 - 1

		x = coords[:, axis1_idx]
		y = coords[:, axis2_idx]

		for pop in unique_populations:
			flt = (populations.population == pop)
			ax.plot(
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

		if pca:
			ax.set_xlabel('PC%s (%.1f%%)' % (axis1, model.explained_variance_ratio_[axis1_idx]*100))

			ax.set_ylabel('PC%s (%.1f%%)' % (axis2, model.explained_variance_ratio_[axis2_idx]*100))

		elif cmds:
			ax.set_xlabel("cMDS Axis {}".format(axis1))
			ax.set_ylabel("cMDS Axis {}".format(axis2))

		elif isomds:
			ax.set_xlabel("isoMDS Axis {}".format(axis1))
			ax.set_ylabel("isoMDS Axis {}".format(axis2))

		else:
			raise ValueError("Either pca, cmds, or isomds must be set to True")

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

	@property
	def get_pca_coords(self):
		"""[Getter for PCA coordinates]

		Returns:
			[numpy.ndarray, float]: [PCA coordinates for each component of shape(n_samples, n_components)]
		"""
		return self.pca_coords

	@property
	def get_pca_model(self):
		"""[Getter for PCA model]

		Returns:
			[allel.stats.decomposition.GenotypePCA]: [Model info from PCA. See sklearn.decomposition.PCA documentation]
		"""
		return self.pca_model

	@property
	def explained_variance(self):
		"""[Getter for the explained variance in the PCA model]

		Returns:
			[numpy.ndarray]: [Explained variance with shape (n_components,)]
		"""
		return self.pca_model.explained_variance_ratio_

	@property
	def explained_variance_ratio(self):
		"""[Getter for the explained variance ratio in the PCA model]

		Returns:
			[numpy.ndarray]: [Explained variance ratio, shape(n_components,)]
		"""
		return self.pca_model.explained_variance_ratio_

	@property
	def cmds_dissimilarity_matrix(self):
		"""[Getter for cMDS dissimilarity matrix]

		Returns:
			[numpy.ndarray]: [cMDS dissimilarity matrix]
		"""
		return self.cmds_model.dissimilarity_matrix_

	@property
	def isomds_dissimilarity_matrix(self):
		"""[Getter for isoMDS dissimilarity matrix]

		Returns:
			[numpy.ndarray]: [isoMDS dissimilarity matrix]
		"""
		return self.isomds_model.dissimilarity_matrix_
	
	@property
	def pca_components_elbow(self):
		"""[Getter for the inflection point of PC cumulative variance]

		Returns:
			[int]: [Number of principal component axes at inflection point]
		"""
		return self.inflection

