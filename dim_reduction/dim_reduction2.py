
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from kneed import KneeLocator
from sklearn.decomposition import PCA


from utils.misc import timer

class DimReduction:

	def __init__(self, gt, pops, prefix, colors=None, palette="Set1"):

		self.gt = gt
		self.pops = pops
		self.prefix = prefix
		self.colors = colors
		self.palette = palette

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

class runPCA(DimReduction):

	def __init__(self, gt, pops, prefix, colors=None, palette="Set1", keep_pcs=10, scaler="patterson", plot_cumvar=False, elbow=False, pc_var=None, **kwargs):
		
		super().__init__(gt, pops, prefix, colors, palette)
		self.keep_pcs = keep_pcs
		self.scaler = scaler
		self.coords = None
		self.pca_model = None
		self.method = "PCA"

		# PCA scatterplot settings
		text_size = kwargs.pop("text_size", 14)
		style = kwargs.pop("style", "white")
		figwidth = kwargs.pop("figwidth", 6)
		figheight = kwargs.pop("figheight", 6)

		# PCA Cumulative Variance plot settings
		cumvar_figwidth = kwargs.pop("cumvar_figwidth", 6)
		cumvar_figheight = kwargs.pop("cumvar_figheight", 6)
		cumvar_linecolor = kwargs.pop("cumvar_linecolor", "blue")
		cumvar_linewidth = kwargs.pop("cumvar_linewidth", 3)
		cumvar_xintercept_width = kwargs.pop("cumvar_xintercept_width", 3)
		cumvar_xintercept_color = kwargs.pop("cumvar_xintercept_color", "red")
		cumvar_xintercept_style = kwargs.pop("cumvar_xintercept_style", "--")
		cumvar_style = kwargs.pop("cumvar_style", "white")
		cumvar_text_size = kwargs.pop("cumvar_text_size", 14)

		# If still items left in kwargs, then unrecognized keyword argument
		if kwargs:
			raise TypeError("Unexpected keyword arguments provided: {}".format(list(kwargs.keys())))


		self.coords, self.pca_model = self.fit_transform()

		if plot_cumvar:
			self.keep_pcs = self._plot_pca_cumvar(self.coords, self.pca_model, prefix, elbow, pc_var, cumvar_figwidth, cumvar_figheight, cumvar_linecolor, cumvar_linewidth, cumvar_xintercept_width, cumvar_xintercept_color, cumvar_xintercept_style, cumvar_style, cumvar_text_size)

			self.coords, self.pca_model = self.fit_transform()

	def fit_transform(self):
		"""[Does principal component analysis on 012-encoded genotypes. By default uses a Patterson scaler to standardize, but you can use two other scalers: 'standard' and 'center'. 'standard' centers and standardizes the data, whereas 'center' just centers it and doesn't standardize it]

		Args:
			data ([numpy.ndarray]): [012-encoded genotypes of shape (n_samples, n_features)]

			pca_arguments ([dict]): [Dictionary with option names as keys and the settings as values]
		"""
		print("\nDoing PCA...\n")
		print(
				"PCA Settings:\n"
				"\tn_components: """+str(self.keep_pcs)+"\n"
				"\tscaler: """+str(self.scaler)+"\n"
		)

		# Scale and center the data
		if self.scaler == "patterson":
			X = self._scaler_patterson(self.gt)
		
		elif self.scaler == "standard":
			X = self._scaler_standard(self.gt)

		elif self.scaler == "center":
			X = self._scaler_center(self.gt)

		else:
			raise ValueError("Unsupported scaler argument provided: {}".format(self.scaler))

		pca = PCA(n_components=self.keep_pcs)

		model = pca.fit(X)
		coords = model.transform(X)

		print("\nDone!")

		return coords, model

	def _scaler_patterson(self, data):
		"""[Patterson scaler for PCA. Basically the formula for calculating the unit variance per SNP site is: std = np.sqrt((mean / ploidy) * (1 - (mean / ploidy))). Then center the data by subtracting the mean and scale it by dividing by the unit variance per SNP site.]

		Args:
			data ([numpy.ndarray]): [012-encoded genotypes to transform of shape (n_samples, n_features)]

		Returns:
			[numpy.ndarray]: [Transformed data, centered and scaled with Patterson scaler]
		"""
		# Make sure type is np.ndarray
		if not isinstance(data, np.ndarray):
			data = np.asarray(data)
		
		# Find mean of each site
		mean = np.mean(data, axis=0, keepdims=True)

		# Super Deli only supports ploidy of 2 currently
		ploidy = 2

		# Do Patterson scaling
		# Ploidy is 2
		p = mean / ploidy
		std = np.sqrt(p * (1 - p))

		# Make sure dtype is np.float64
		data = data.astype(np.float64)

		# Center the data
		data -= mean

		# Scale the data using the Patterson scaler
		data /= std

		return data

	def _scaler_standard(self, data):
		"""[Center and standardize data]

		Args:
			data ([numpy.ndarray]): [012-encoded genotypes to transform of shape (n_samples, n_features)]

		Returns:
			[numpy.ndarray]: [Transformed data, centered and standardized]
		"""
		# Make sure data is a numpy.ndarray
		if not isinstance(data, np.ndarray):
			data = np.asarray(data)
		
		# Make sure dtype is np.float64
		data = data.astype(np.float64)

		# Center and standardize the data
		mean = np.mean(data, axis=0, keepdims=True)
		std = np.std(data, axis=0, keepdims=True)
		data -= mean
		data /= std

		return data

	def _scaler_center(self, data):
		"""[Just center the data; don't standardize]

		Args:
			data ([numpy.ndarray]): [012-encoded genoytpes to transform of shape (n_samples, n_features)]

		Returns:
			[numpy.ndarray]: [Centered 012-encoded genotypes; not standardized]
		"""
		if not isinstance(data, np.ndarray):
			data = np.asarray(data)

		# Make sure type is np.float64
		data = data.astype(np.float64)

		# Center the data
		mean = np.mean(data, axis=0, keepdims=True)
		data -= mean
		return data

	def _plot_pca_cumvar(self, coords, model, prefix, elbow, pc_var, figwidth, figheight, linecolor, linewidth, xintercept_width, xintercept_color, xintercept_style, style, text_size):
		"""[Plot cumulative variance for principal components with xintercept line at the inflection point]

		Args:
			coords ([numpy.ndarray]): [PCA coordinates as 2D array of shape (n_samples, n_components)]

			model ([sklearn.decomposision.PCA model object]): [PCA model returned from scikit-allel. See scikit-allel documentation]

			prefix ([str]): [Prefix for output PDF filename]

			user_settings ([dict]): [Dictionary with matplotlib arguments as keys and their corresponding values. Only some or all of the settings can be changed]

		Raises:
			AttributeError: [pca_model must be defined prior to running this function]
		"""
		# Raise error if PCA hasn't been run yet
		if coords is None:
			raise AttributeError("\nA PCA coordinates are not defined!")

		if elbow is True and pc_var is not None:
			raise ValueError("elbow and pc_var cannot both be specified!")

		if pc_var is not None:
			assert isinstance(pc_var, float), "pc_var must be of type float"
			assert pc_var <= 1.0, "pc_var must be a value between 0.0 and 1.0"
			assert pc_var > 0.0, "pc_var must be a value greater than 0.0"

		# Get the cumulative variance of the principal components
		cumsum = np.cumsum(model.explained_variance_ratio_)

		# From the kneed package
		# Gets the knee/ elbow of the curve
		kneedle = KneeLocator(range(1, coords.shape[1]+1), cumsum, curve="concave", direction="increasing")

		# Sets plot background style
		# Uses seaborn
		sns.set(style=style)

		# Plot the results
		# Uses matplotlib.pyplot
		fig = plt.figure(figsize=(figwidth, figheight))
		ax = fig.add_subplot(1,1,1)

		if elbow:
			# Plot the explained variance ratio
			ax.plot(kneedle.y, color=linecolor, linewidth=linewidth)

			# Add text to show inflection point
			ax.text(0.95, 0.01, "Knee={}".format(kneedle.knee), verticalalignment="bottom", horizontalalignment="right", transform=ax.transAxes, color="k", fontsize=text_size)

			knee = kneedle.knee

		if pc_var is not None:
			for idx, item in enumerate(cumsum):
				if cumsum[idx] < pc_var:
					knee = idx + 1
				else:
					break

			ax.plot(cumsum, color=linecolor, linewidth=linewidth)

			ax.text(0.95, 0.01, "# PCs: {}".format(knee), verticalalignment="bottom", horizontalalignment="right", transform=ax.transAxes, color="k", fontsize=text_size)

		ax.axvline(linewidth=xintercept_width, color=xintercept_color, linestyle=xintercept_style, x=knee, ymin=0, ymax=1)
		
		# Set axis labels
		ax.set_xlabel("Number of Components")
		ax.set_ylabel("Cumulative Explained Variance")

		# Add prefix to filename
		plot_fn = "{}_pca_cumvar.pdf".format(prefix)

		# Save as PDF file
		fig.savefig(plot_fn, bbox_inches="tight")

		# Returns number of principal components at elbow
		return knee

	@property
	def	pca_coords(self):
		if self.coords is not None:
			return self.coords
		else:
			raise AttributeError("pca_coodinates attribute is not yet defined")

	@property
	def pca_model_object(self):
		if self.pca_model is not None:
			return self.pca_model
		else:
			raise AttributeError("pca_model_object is not yet defined")
					




