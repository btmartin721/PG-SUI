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

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import MDS

class DimReduction:
	"""[Class to perform dimensionality reduction on genotype features]
	"""

	def __init__(self, data, pops, algorithms=None):
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
		self.algorithms = algorithms
		self.pca_coords = None
		self.pca_model = None
		self.cmds_model = None
		self.isomds_model = None
		self.target = None

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

	def do_mds(self, mds_arguments, metric=True):
		
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

	def plot_dimred(self, prefix, pca=False, cmds=False, isomds=False, axis1=1, axis2=2, figwidth=6, figheight=6, alpha=1.0, colors=None, palette="Set1", legend=True, legend_inside=False, legend_loc="upper left", marker='o', markersize=6, markeredgecolor='k', markeredgewidth=0.5, labelspacing=0.5, columnspacing=2.0, title=None, title_fontsize=None, markerfirst=True, markerscale=1.0, ncol=1, bbox_to_anchor=(1.0, 1.0), borderaxespad=0.5, legend_edgecolor="black", facecolor="white", framealpha=0.8, shadow=False):
		"""[Plot PCA results as a scatterplot and save it as a PDF file]

		Args:
			prefix ([str]): [Prefix for output PDF filename]

			pca (bool, optional): [True if plotting PCA results. Cannot be set at same time as cmds and isomds]

			cmds (bool, optional): [True if plotting cmds results. Cannot be set at same time as pca and isomds]

			isomds (bool, optional): [True if plotting isomds results. Cannot be set at same time as pca and cmds]

			axis1 (int, optional): [First principal component axis to plot; starts at 1]. Defaults to 1.

			axis2 (int, optional): [Second principal component axis to plot; starts at 1]. Defaults to 2.

			figwidth (int, optional): [Set width of the plot]. Defaults to 6.

			figheight (int, optional): [Set height of the plot]. Defaults to 6.

			alpha (float, optional): [Set transparency of sample points. Should be between 0 and 1. 0 = fully transparent, 1 = no transparency. Allows for better visualization if many points overlay one another]. Defaults to 1.0.

			colors ([dict], optional): [Dictionary with unique population IDs as keys and hex color codes as values. If colors=None, a diverging color palette automatically gets applied.]. Defaults to None.

			palette (str, optional): [Set the automatically generated color palette if colors=None. See matplotlib.colors documentation]. Defaults to "Set1".

			legend (boolean, optional): [If True, a legend is included]. Defaults to True.

			legend_inside (boolean, optional): [If True, the legend is located inside the plot]. Defaults to False.

			legend_loc (str, optional): [Set the location of the legend. If some of your points get covered with the legend you can change its location]. Defaults to "upper left".

			marker (str, optional): [Set the marker shape. See matplotlib.markers documentation]. Defaults to 'o' (a circle).

			markersize (int, optional): [Set size of the markers]. Defaults to 6.

			markeredgecolor (str, optional): [Set the color of the marker edge]. Defaults to 'k' (black). See matplotlib.pyplot.plot documentation.

			markeredgewidth (float, optional): [Set the width of the marker edge]. Defaults to 0.5.

			labelspacing (float, optional): [The vertical space between the legend entries, in font-size units]. Defaults to 0.5.

			columnspacing (float, optional): [The spacing between columns, in font-size units]. Defaults to 2.0.

			title (str or None, optional): [The legend's title]. Defaults to None.

			title_fontsize (int or {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}, optional): [The font size of the legend's title]. Defaults to None.

			markerfirst (boolean, optional): [If True, legend marker is placed to the left of the legend label. If False, legend marker is placed to the right of the legend label]. Defaults to True.

			markerscale (float, optional): [The relative size of legend markers compared with the originally drawn ones]. Defaults to 1.0.

			ncol (int, optional): [The number of columns that the legend has]. Defaults to 1.

			bbox_to_anchor (tuple(float, float) or tuple(float, float, float, float), optional): [Box that is used to position the legend in conjuction with legend_loc. If a 4-tuple bbox is given, then it specifies the bbox(x, y, width, height) that the legend is placed in]. Defaults to (1.0, 1.0).

			borderaxespad (float, optional): [The pad between the axes and legend border, in font-size units]. Defaults to 0.5.

			legend_edgecolor (float, optional): [The legend's background patch edge color. If "inherit", use take rcParams["axes.edgecolor"]]. Defaults to "black".

			facecolor (str, optional): [The legend's background color. If "inherit", use rcParams["axes.facecolor"]]. Defaults to "white".

			framealpha (float, optional): [The alpha transparency of the legend's background. If shadow is activated and framealpha is None, the default value is ignored]. Defaults to 0.8.

			shadow (boolean, optional): [Whether to draw a shadow behind the legend]. Defaults to False.

		Raises:
			ValueError: [Only one of pca, cmds, and isomds can be set to True]

		"""
		if not pca and not cmds and not isomds:
			raise ValueError("One of the pca, cmds, or isomds arguments must be set to True")

		if (pca and cmds) or (pca and isomds) or (cmds and isomds):
			raise ValueError("Only one of the pca, cmds, or isomds arguments can be set to True at a time")

		axis1_idx = axis1 - 1
		axis2_idx = axis2 - 1

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
		
		fig = plt.figure(figsize=(figwidth, figheight))
		ax = fig.add_subplot(1,1,1)

		colors = self._get_pop_colors(targets, palette, colors)

		if pca:
			self._plot_coords(self.pca_coords, axis1, axis2, ax, pop_df, targets, colors, alpha, marker, markersize, markeredgecolor, markeredgewidth, pca, cmds, isomds, model=self.pca_model)

		elif cmds:
			self._plot_coords(self.cmds_model, axis1, axis2, ax, pop_df, targets, colors, alpha, marker, markersize, markeredgecolor, markeredgewidth, pca, cmds, isomds)

		elif isomds:
			self._plot_coords(self.isomds_model, axis1, axis2, ax, pop_df, targets, colors, alpha, marker, markersize, markeredgecolor, markeredgewidth, pca, cmds, isomds)

		if legend:
			if legend_inside:
				if bbox_to_anchor[0] > 1 or bbox_to_anchor > 1:
					print("Warning: bbox_to_anchor was set grater than 1.0 (outside plot margins) but legend_inside was set to True. Setting bbox_to_anchor to (1.0, 1.0)")

				ax.legend(loc=legend_loc, labelspacing=labelspacing, columnspacing=columnspacing, title=title, title_fontsize=title_fontsize, markerfirst=markerfirst, markerscale=markerscale, labelcolor=labelcolor, ncol=ncol, bbox_to_anchor=bbox_to_anchor, borderaxespad=borderaxespad, edgecolor=legend_edgecolor, facecolor=facecolor, framealpha=framealpha, shadow=shadow)

			else:
				if bbox_to_anchor[0] < 1 and bbox_to_anchor[1] < 1:
					print("Warning: bbox_to_anchor was set less than 1.0 (inside the plot margins) but legend_inside was set to False. Setting bbox_to_anchor to (1.05, 1.0)")

				ax.legend(loc=legend_loc, labelspacing=labelspacing, columnspacing=columnspacing, title=title, title_fontsize=title_fontsize, markerfirst=markerfirst, markerscale=markerscale, ncol=ncol, bbox_to_anchor=bbox_to_anchor, borderaxespad=borderaxespad, edgecolor=legend_edgecolor, facecolor=facecolor, framealpha=framealpha, shadow=shadow)

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

