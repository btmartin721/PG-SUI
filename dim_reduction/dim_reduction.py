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
		self.target = None

	def standard_pca(self, pca_arguments):
		"""[Does standard PCA using scikit-allel. By default uses a Patterson scaler]

		Args:
			all_settings ([dict]): [Dictionary with option names as keys and the settings as values]
		"""
		print("\nDoing PCA with {} principal components...".format(pca_arguments["n_components"]))


		gn = np.array(self.data).transpose()

		self.pca_coords, self.pca_model = allel.pca(gn, n_components=pca_arguments["n_components"], copy=pca_arguments["copy"], scaler=pca_arguments["scaler"], ploidy=pca_arguments["ploidy"])

		print("Done!")

	def plot_pca(self, prefix, pc1=1, pc2=2, figwidth=6, figheight=6, alpha=1.0, colors=None, palette="Set1", legend=True, legend_inside=False, legend_loc="upper left", marker='o', markersize=6, markeredgecolor='k', markeredgewidth=0.5, labelspacing=0.5, columnspacing=2.0, title=None, title_fontsize=None, markerfirst=True, markerscale=1.0, ncol=1, bbox_to_anchor=(1.0, 1.0), borderaxespad=0.5, legend_edgecolor="black", facecolor="white", framealpha=0.8, shadow=False):
		"""[Plot PCA results as a scatterplot and save it as a PDF file]

		Args:
			prefix ([str]): [Prefix for output PDF filename]

			pc1 (int, optional): [First principal component axis to plot; starts at 1]. Defaults to 1.

			pc2 (int, optional): [Second principal component axis to plot; starts at 1]. Defaults to 2.

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
		"""
		print("\nPlotting PCA results...")
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

		self._plot_pca_coords(self.pca_coords, self.pca_model, pc1, pc2, ax, pop_df, targets, colors, alpha, marker, markersize, markeredgecolor, markeredgewidth)

		if legend:
			if legend_inside:
				if bbox_to_anchor[0] > 1 or bbox_to_anchor > 1:
					print("Warning: bbox_to_anchor was set grater than 1.0 (outside plot margins) but legend_inside was set to True. Setting bbox_to_anchor to (1.0, 1.0)")
				ax.legend(loc=legend_loc, labelspacing=labelspacing, columnspacing=columnspacing, title=title, title_fontsize=title_fontsize, markerfirst=markerfirst, markerscale=markerscale, labelcolor=labelcolor, ncol=ncol, bbox_to_anchor=bbox_to_anchor, borderaxespad=borderaxespad, edgecolor=legend_edgecolor, facecolor=facecolor, framealpha=framealpha, shadow=shadow)
			else:
				if bbox_to_anchor[0] < 1 and bbox_to_anchor[1] < 1:
					print("Warning: bbox_to_anchor was set less than 1.0 (inside the plot margins) but legend_inside was set to False. Setting bbox_to_anchor to (1.05, 1.0)")
				ax.legend(loc=legend_loc, labelspacing=labelspacing, columnspacing=columnspacing, title=title, title_fontsize=title_fontsize, markerfirst=markerfirst, markerscale=markerscale, ncol=ncol, bbox_to_anchor=bbox_to_anchor, borderaxespad=borderaxespad, edgecolor=legend_edgecolor, facecolor=facecolor, framealpha=framealpha, shadow=shadow)

		plot_fn = "{}_pca.pdf".format(prefix)
		fig.savefig(plot_fn, bbox_inches="tight")

		print("Done!\nSaved PCA scatterplot to {}".format(plot_fn))
		
	def _plot_pca_coords(self, coords, model, pc1, pc2, ax, populations, unique_populations, pop_colors, alpha, marker, markersize, markeredgecolor, markeredgewidth):
		"""[Map colors to populations and make the scatterplot]

		Args:
			coords ([numpy.array]): [pca_coords object returned from scikit-allel PCA]

			model ([scikit-allel object]): [Second object returned from scikit-allel PCA]

			pc1 ([int]): [First principal component axis to plot. Starts at 1]

			pc2 ([int]): [Second principal component axis to plot. Starts at 1]

			ax ([matplotlib object]): [ax object from matplotlib]

			populations ([pandas.DataFrame]): [pandas.DataFrame with population IDs]

			unique_populations ([list]): [Unique populations in the populations argument]

			pop_colors ([dict]): [Dictionary with unique population IDs as keys and hex-code colors as values]

			alpha ([float]): [Set transparency of points; lower = more transparent. Should be between 0 and 1]

			marker ([str]): [Set the marker shape. See matplotlib.markers documentation].

			markersize ([int]): [Set size of the markers]

			markeredgecolor ([str]): [Set the color of the marker edge]. See matplotlib.pyplot.plot documentation.

			markeredgewidth ([float]): [Set the width of the marker edge].
		"""

		sns.despine(ax=ax, offset=5)

		pc1_idx = pc1 - 1
		pc2_idx = pc2 - 1

		x = coords[:, pc1_idx]
		y = coords[:, pc2_idx]

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

		ax.set_xlabel('PC%s (%.1f%%)' % (pc1, model.explained_variance_ratio_[pc1_idx]*100))

		ax.set_ylabel('PC%s (%.1f%%)' % (pc2, model.explained_variance_ratio_[pc2_idx]*100))

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

