# Standard library imports
import sys

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
import seaborn as sns
sns.set_style("white")
sns.set_style("ticks")

from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.manifold import TSNE
from kneed import KneeLocator # conda install -c conda-forge kneed

# Custom module imports
from utils import settings


def do_pca(data, pca_settings):
	"""[Does principal component analysis on 012-encoded genotypes. By default uses a Patterson scaler to standardize, but you can use two other scalers: 'standard' and 'center'. 'standard' centers and standardizes the data, whereas 'center' just centers it and doesn't standardize it]

	Args:
		data ([numpy.ndarray]): [012-encoded genotypes of shape (n_samples, n_features)]

		pca_arguments ([dict]): [Dictionary with option names as keys and the settings as values]
	"""
	print("\nDoing PCA...\n")
	print(
			"""
			PCA Settings:
				n_components: """+str(pca_settings["n_components"])+"""
				scaler: """+str(pca_settings["scaler"])+"""
			"""
	)

	# Scale and center the data
	if pca_settings["scaler"] == "patterson":
		gt = _scaler_patterson(data)
	
	elif pca_settings["scaler"] == "standard":
		gt = _scaler_standard(data)

	elif pca_settings["scaler"] == "center":
		gt = _scaler_center(data)

	else:
		raise ValueError("Unsupported scaler argument provided: {}".format(pca_settings["scaler"]))

	pca = PCA(n_components=pca_settings["n_components"])

	model = pca.fit(gt)
	coords = model.transform(gt)

	print("\nDone!")

	return coords, model


def _scaler_patterson(data):
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

def _scaler_standard(data):
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

def _scaler_center(data):
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


	
def do_mds(X, mds_arguments, metric=True, do_3d=False):
	
	if metric:
		print("\nDoing cMDS dimensionality reduction...\n")
	else:
		print("\nDoing isoMDS dimensionality reduction...\n")

	if do_3d and mds_arguments["n_dims"] < 3:
		raise ValueError("plot_3d was set to True but mds_settings has n_dims set < 3")
		
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
		cmds_model = mds.fit_transform(X)
		print("\nDone!")
		return cmds_model

	else:
		isomds_model = mds.fit_transform(X)
		print("\nDone!")
		return isomds_model


def tsne(df, settings):
	
	print("\nDoing T-SNE embedding...")

	print(
		"""
		T-SNE Settings:
			n_components: """+str(settings["n_components"])+"""
			perplexity: """+str(settings["perplexity"])+"""
			early exaggeration: """+str(settings["early_exaggeration"])+"""
			learning_rate: """+str(settings["learning_rate"])+"""
			n_iter: """+str(settings["n_iter"])+"""
			min_grad_norm: """+str(settings["min_grad_norm"])+"""
			metric: """+str(settings["metric"])+"""
			init: """+str(settings["init"])+"""
			verbose: """+str(settings["verbose"])+"""
			random_state: """+str(settings["random_state"])+"""
			method: """+str(settings["method"])+"""
			angle: """+str(settings["angle"])+"""
			n_jobs: """+str(settings["n_jobs"])+"""
			square_distances: """+str(settings["square_distances"])+"""
		"""
	)

	t = TSNE(
		n_components=settings["n_components"],
		perplexity=settings["perplexity"],
		early_exaggeration=settings["early_exaggeration"],
		learning_rate=settings["learning_rate"],
		n_iter=settings["n_iter"],
		n_iter_without_progress=settings["n_iter_without_progress"],
		min_grad_norm=settings["min_grad_norm"],
		metric=settings["metric"],
		init=settings["init"],
		verbose=settings["verbose"],
		random_state=settings["random_state"],
		method=settings["method"],
		angle=settings["angle"],
		n_jobs=settings["n_jobs"],
		square_distances=settings["square_distances"]
	)

	tsne_results = t.fit_transform(df.values)

	print("Done!")

	return tsne_results
	
def plot_dimreduction(coords, pops, prefix, method, pca=False, pca_model=None, plot_3d=False, user_settings=None, colors=None, palette="Set1"):
	"""[Plot PCA results as a scatterplot and save it as a PDF file]

	Args:
		coords ([numpy.ndarray, optional]): [Coordinates from PCA, MDS, or T-SNE]

		pops ([list]): [List of population IDs generated in GenotypeData object]

		prefix ([str]): [Prefix for output PDF filename]

		method ([str]): [Dimensionality reduction method to use]

		pca (bool, optional): [True if plotting PCA results. If set, has different axis labels on plot that include PCA variance per axis]. Defaults to False.

		pca_model (sklearn.decomposition.PCA object): [PCA model to plot]. Defaults to None.

		settings (dict, optional): [Dictionary with plot setting arguments as keys and their corresponding values. Some or all of the arguments can be set]. Defaults to None.

		colors (dict, optional): [Dictionary with unique population IDs as keys and hex-code colors as values]. Defaults to None.

		palette (str, optional): [matplotlib.colors palette to be used if colors=None]. Defaults to 'Set1'.

	Raises:
		ValueError: [Must be a supported dimensionality reduction method]
		TypeError: [If pca=True, pca_model argument must also be set]
	"""
	if method == "cmds":
		method = "cMDS"

	elif method == "isomds": 
		method = "isoMDS"

	elif method == "tsne":
		method = "T-SNE"

	elif method == "pca":
		method = "PCA"
	
	else:
		raise ValueError("The dimensionality reduction method {} is not supported!".format(method))

	settings_default = settings.dimreduction_plot_settings()

	if user_settings:
		settings_default.update(user_settings)

	if plot_3d:
		axis3_idx = settings_default["axis3"] - 1

	if pca:
		if not pca_model:
			raise TypeError("pca_model argument must be provided if pca=True")

		print("\nPlotting PCA results...")

	targets = list(set(pops))

	pop_df = pd.DataFrame(pops, columns=["population"])
	
	fig = plt.figure(figsize=(settings_default["figwidth"], settings_default["figheight"]))

	if plot_3d:
		ax = fig.add_subplot(111, projection="3d")
	else:
		ax = fig.add_subplot(1,1,1)

	colors = _get_pop_colors(targets, palette, colors)

	if pca:
		_plot_coords(coords, settings_default["axis1"], settings_default["axis2"], settings_default["axis3"], plot_3d, ax, pop_df, targets, colors, settings_default["alpha"], settings_default["marker"], settings_default["markersize"], settings_default["markeredgecolor"], settings_default["markeredgewidth"], method, model=pca_model)

	else:
		_plot_coords(coords, settings_default["axis1"], settings_default["axis2"], settings_default["axis3"], plot_3d, ax, pop_df, targets, colors, settings_default["alpha"], settings_default["marker"], settings_default["markersize"], settings_default["markeredgecolor"], settings_default["markeredgewidth"], method)

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

	plot_fn = "{}_{}.pdf".format(prefix, method)

	fig.savefig(plot_fn, bbox_inches="tight")

	print("\nSaved {} scatterplot to {}".format(method, plot_fn))

def plot_pca_cumvar(coords, model, prefix, user_settings):
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
	if not model:
		raise AttributeError("\nA PCA object has not yet been created! Please do so with DelimModel([arguments]).dim_reduction(algorithms=['pca'], [other_arguments])")

	# Retrieve default plot settings
	settings_default = settings.pca_cumvar_default_settings()

	# Update plot settings with user-specified settings
	if user_settings:
		settings_default.update(user_settings)

	# Get the cumulative variance of the principal components
	cumsum = np.cumsum(model.explained_variance_ratio_)

	# From the kneed package
	# Gets the knee/ elbow of the curve
	kneedle = KneeLocator(range(1, len(coords)), cumsum, curve="concave", direction="increasing")

	# Sets plot background style
	# Uses seaborn
	sns.set(style=settings_default["style"])

	# Plot the results
	# Uses matplotlib.pyplot
	fig = plt.figure(figsize=(settings_default["figwidth"], settings_default["figheight"]))
	ax = fig.add_subplot(1,1,1)

	# Plot the explained variance ratio
	ax.plot(kneedle.y, color=settings_default["linecolor"], linewidth=settings_default["linewidth"])

	# Set axis labels
	ax.set_xlabel("Number of Components")
	ax.set_ylabel("Cumulative Explained Variance")

	# Add text to show inflection point
	ax.text(0.95, 0.01, "Elbow={}".format(kneedle.knee), verticalalignment="bottom", horizontalalignment="right", transform=ax.transAxes, color="k", fontsize=settings_default["text_size"])

	# Add inflection point
	ax.axvline(linewidth=settings_default["xintercept_width"], color=settings_default["xintercept_color"], linestyle=settings_default["xintercept_style"], x=kneedle.knee, ymin=0, ymax=1)

	# Add prefix to filename
	plot_fn = "{}_pca_cumvar.pdf".format(prefix)

	# Save as PDF file
	fig.savefig(plot_fn, bbox_inches="tight")

	# Returns number of principal components at elbow
	return kneedle.knee
	
def _plot_coords(coords, axis1, axis2, axis3, plot_3d, ax, populations, unique_populations, pop_colors, alpha, marker, markersize, markeredgecolor, markeredgewidth, method, model=None):
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

def _get_pop_colors(uniq_pops, palette, colors):
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
