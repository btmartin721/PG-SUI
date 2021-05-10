
# Third-party imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Custom imports
from utils.misc import timer

class DimReduction:

	def __init__(self, gt, pops, prefix, colors=None, palette="Set1"):

		self.gt = gt
		self.pops = pops
		self.prefix = prefix
		self.colors = colors
		self.palette = palette

	def _validate_args(self, _dimreduction, _gt, _pops, _prefix):

		if _dimreduction is None:
			if _gt is None:
				raise TypeError("The 'gt' keyword argument must be defined if dimreduction=None")
			if _pops is None:
				raise TypeError("The 'pops' keyword argument must be defined if dimreduction=None")
			if _prefix is None:
				raise TypeError("The 'prefix' argument must be defined if dimreduction=None")

			gt_df = self._validate_type(_gt)
			self.set_gt(gt_df)

		else: # _dimreduction is not None
			if _gt is not None:
				raise TypeError("The 'dimreduction' and 'gt' arguments cannot both be defined")
			if _pops is not None:
				raise TypeError("The 'dimreduction' and 'pops' arguments cannot both be defined")
			if _prefix is not None:
				raise TypeError("The 'dimreduction and 'prefix' arguments cannot both be defined")

			gt_df = self._validate_type(_dimreduction.gt)
			self.set_gt(gt_df)
			self.set_pops(_dimreduction.pops)
			self.set_prefix(_dimreduction.prefix)
			self.set_colors(_dimreduction.colors)
			self.set_palette(_dimreduction.palette)

	def _validate_type(self, X):
		if isinstance(X, np.ndarray):
			df = pd.DataFrame(X)

		elif isinstance(X, list):
			df = pd.DataFrame.from_records(X)
		
		elif isinstance(X, pd.DataFrame):
			df = X.copy()

		else:
			raise TypeError("\nThe genotype data must be a numpy.ndarray, a pandas.DataFrame, or a 2-dimensional list of shape (n_samples, n_sites)! Any of these can be retrieved from the GenotypeData object")

		return df

	def set_gt(self, _gt):
		self.gt = _gt

	def set_pops(self, _pops):
		self.pops = _pops

	def set_prefix(self, _prefix):
		self.prefix = _prefix

	def set_colors(self, _colors):
		self.colors = _colors

	def set_palette(self, _palette):
		self.palette = _palette

	def plot(self, plot_3d=False, axis1=1, axis2=2, axis3=3, figwidth=6, figheight=6, alpha=1.0, legend=True, legend_inside=False, legend_loc="upper left", marker="o", markersize=6, markeredgecolor="k", markeredgewidth=0.5, labelspacing=0.5, columnspacing=2.0, title=None, title_fontsize=None, markerfirst=True, markerscale=1.0, ncol=1, bbox_to_anchor=(1.0, 1.0), borderaxespad=0.5, legend_edgecolor="black", facecolor="white", framealpha=0.8, shadow=False):
		"""[Plot PCA results as a scatterplot and save it as a PDF file]

		Args:
			axis1 (int, optional): [First axis to plot]. Defaults to 1.

			axis2 (int, optional): [Second axis to plot]. Defaults to 2.

			axis3 (int, optional): [third axis to plot]. Defaults to 3.


		Raises:
			ValueError: [Must be a supported dimensionality reduction method]
			TypeError: [If pca=True, pca_model argument must also be set]
		"""
		if self.method == "PCA":
			if self.pca_model is None:
				raise TypeError("pca_model argument must be provided if pca=True")

			print("\nPlotting PCA results...")

		uniq_pops = list(set(self.pops))

		pop_df = pd.DataFrame(self.pops, columns=["population"])
		
		fig = plt.figure(figsize=(figwidth, figheight))

		if plot_3d:
			ax = fig.add_subplot(111, projection="3d")

		else:
			ax = fig.add_subplot(1,1,1)

		colors = self._get_pop_colors(uniq_pops, self.palette, self.colors)

		if self.method == "PCA":
			self._plot_coords(self.coords, axis1, axis2, axis3, plot_3d, ax, pop_df, uniq_pops, colors, alpha, marker, markersize, markeredgecolor, markeredgewidth, self.method, model=self.pca_model)

		else:
			self._plot_coords(self.coords, axis1, axis2, axis3, plot_3d, ax, pop_df, uniq_pops, colors, alpha, marker, markersize, markeredgecolor, markeredgewidth, self.method)

		if legend:
			if legend_inside:
				if bbox_to_anchor[0] > 1 or \
					bbox_to_anchor > 1:
					print("Warning: bbox_to_anchor was set grater than 1.0 (outside plot margins) but legend_inside was set to True. Setting bbox_to_anchor to (1.0, 1.0)")

			else:
				if bbox_to_anchor[0] < 1 and \
					bbox_to_anchor[1] < 1:
					print("Warning: bbox_to_anchor was set less than 1.0 (inside the plot margins) but legend_inside was set to False. Setting bbox_to_anchor to (1.05, 1.0)")

			ax.legend(loc=legend_loc, labelspacing=labelspacing, columnspacing=columnspacing, title=title, title_fontsize=title_fontsize, markerfirst=markerfirst, markerscale=markerscale, ncol=ncol, bbox_to_anchor=bbox_to_anchor, borderaxespad=borderaxespad, edgecolor=legend_edgecolor, facecolor=facecolor, framealpha=framealpha, shadow=shadow)

		plot_fn = "{}_{}.pdf".format(self.prefix, self.method)

		fig.savefig(plot_fn, bbox_inches="tight")

		print("\nSaved {} scatterplot to {}".format(self.method, plot_fn))
		
	def _plot_coords(self, coords, axis1, axis2, axis3, plot_3d, ax, populations, unique_populations, pop_colors, alpha, marker, markersize, markeredgecolor, markeredgewidth, method, model=None):
		"""[Map colors to populations and make the scatterplot]

		Args:
			coords ([numpy.array]): [pca_coords object returned from scikit-allel PCA or cmds_model or isomds_model objects stored in do_mds()]

			axis1 ([int]): [First axis to plot. Starts at 1]

			axis2 ([int]): [Second axis to plot. Starts at 1]

			axis3 ([int]): [Third axis to plot. Starts at 1]

			plot_3d ([bool]): [True if making 3D plot. False if 2D plot]

			ax ([matplotlib object]): [ax object from matplotlib]

			populations ([pandas.DataFrame]): [pandas.DataFrame with population IDs]

			unique_populations ([list]): [Unique populations in the populations argument]

			pop_colors ([dict]): [Dictionary with unique population IDs as keys and hex-code colors as values]

			alpha ([float]): [Set transparency of points; lower = more transparent. Should be between 0 and 1]

			marker ([str]): [Set the marker shape. See matplotlib.markers documentation].

			markersize ([int]): [Set size of the markers]

			markeredgecolor ([str]): [Set the color of the marker edge]. See matplotlib.pyplot.plot documentation.

			markeredgewidth ([float]): [Set the width of the marker edge]

			method ([str]): [Dimensionality reduction method to use]

			model (scikit-allel object, optional): [Second object returned from scikit-allel PCA. Set model=None if doing cmds or isomds]

		Raises:
			ValueError: [Make sure model argument is set if pca is True]
			ValueError: [Make sure one of pca, cmds, or isomds is True]
		"""
		sns.despine(ax=ax, offset=5)
		axis1_idx = axis1 - 1
		axis2_idx = axis2 - 1
		x = coords[:, axis1_idx]
		y = coords[:, axis2_idx]

		if plot_3d:
			if int(coords.shape[1]) < 3:
				raise ValueError("plot_3d was specified"
								"but there are fewer than 3 coordinate axes!"
				)
			axis3_idx = axis3 - 1
			z = coords[:, axis3_idx]

		for pop in unique_populations:
			flt = (populations.population == pop)

			if plot_3d:
				ax.plot3D(
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

