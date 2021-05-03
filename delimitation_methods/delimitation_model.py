# Standard library imports
import sys
import os

# Third-party imports
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.metrics import pairwise_distances
from sklearn.utils.validation import check_symmetric

# Custom module imports
from read_input.read_input import GenotypeData
import dim_reduction.dim_reduction as dr
from utils import settings
from utils.misc import timer

class DelimModel:
	"""[Parent class for delimitation models]
	"""
	
	def __init__(self, gt, pops, prefix):
		"""[Class constructor]

		Args:
			gt ([list(list), numpy.ndarray, or pandas.DataFrame]): [One of the data formats generated from GenotypeData]

			pops ([list]): [List of populations IDs. Can be strings or integers]

			algorithms ([list or str], optional): [List of dimensionality reduction algorithms to run. If algorithms=None, all of them will be performed]. Defaults to None.

			settings ([dict], optional): [Dictionary of settings for each algorithm with setting names as keys and the settings as values. If settings=None, defaults for each algorithm will be used]. Defaults to None.
		"""
		self.gt = gt 
		self.pops = pops
		self.prefix = prefix
		self.pca_settings = None
		self.mds_settings = None
		self.tsne_settings = None
		self.rf_settings = None
		self.algorithms = None
		self.colors = None
		self.palette = None

		# Model results
		self.rf = None
		self.rf_prox_matrix = None
		self.rf_diss_matrix = None
		self.pca = None
		self.tsne = None

	@timer
	def random_forest_unsupervised(self, rf_settings=None, pca_init=True, pca_settings=None, perc=None, elbow=True):
		"""[Do unsupervised random forest embedding. The results can subsequently be subjected to dimensionality reduction using multidimensional scaling (cMDS and isoMDS). Random forest can also be performed on principal components or raw 012-encoded genotypes by setting pca_init=True or pca_init=False, respectively. 
		
			If pca_init=True, there are several options for setting the number of principal components to retain. 

			First, the process can be automated by specifying either a percentage to retain using, for example, perc=.5. 
			
			Second, the inflection point of the cumulative explained variance can be inferred to choose the number of components by setting elbow=True. 
			
			Third, the number of PCs can be set manually by setting elbow=False and perc=None]

			The random forest and PCA settings can be changed by specifying a dictionary with some or all of the sklearn.ensemble.RandomTreesEmbedding or sklearn.decomposition.PCA arguments specified as the dictionary keys along with their corresponding values. If you only specify some of the arguments, that is fine; it just updates the ones you changed. Otherwise it uses the default arguments for those that weren't manually specified]

		Args:
			rf_settings ([dict], optional): [sklearn.ensemble.RandomTreesEmbeddings arguments as keys with their corresponding values. If not specified, uses the default settings. Some or all of the settings can be changed, and it will only update the ones that were set manaully.]. Defaults to None.

			pca_init (bool, optional): [True if principal component analysis (PCA) should be used to boil down the input into N principal components before running random forest embedding. False if using the raw genotypes]. Defaults to True.

			pca_settings ([dict], optional): [sklearn.decomposition.PCA arguments as keys with their corresponding values. If not specified, uses the default settings. Some or all of the settings can be changed, and it will only update the ones that were set manually]. Defaults to None.

			perc ([float], optional): [Percentage of principal components to retain. Ignored if pca_init=False. Cannot be used in conjuction with elbow=True]. Defaults to None.

			elbow (bool, optional): [True if inflection point should be used to infer the number of principal components to retain. Cannot be used in conjuction with perc]. Defaults to True.

		Raises:
			ValueError: [Only one of the perc or elbow arguments can be set at a time]

			TypeError: [The perc argument must be of type(float)]
		"""

		# Can't be both
		if perc and elbow:
			raise ValueError("\nperc and elbow arguments cannot both be set")

		self.rf_settings = rf_settings

		# Fetch supported settings and defaults
		supported_settings = settings.random_forest_unsupervised_supported()
		rf_settings_default = settings.random_forest_unsupervised_defaults()

		# Update manually specified settings
		if rf_settings:
			rf_settings_default.update(rf_settings)
			self._validate_settings(rf_settings, supported_settings)

		# Make sure genotypes are a supported type and return a pandas.DataFrame
		gt_df = self._validate_type(self.gt)

		# Embed the genotypes with PCA first
		if pca_init:

			# Update settings if manually specified
			pca_settings_default = settings.pca_default_settings()
			
			if pca_settings:
				pca_settings_default.update(pca_settings)
				
			# If inferring n_components with inflection point
			if elbow:
				pca_coords, pca_model, inflection = self.dim_reduction(gt_df, ["standard-pca"], pca_settings=pca_settings, plot_pca_cumvar=True, return_pca=True)

				print("\nRe-doing PCA with n_components set to the inflection point")
				
				pca_coords, pca_model = self.dim_reduction(gt_df, ["standard-pca"], pca_settings={"n_components": int(inflection)}, plot_pca_cumvar=False, return_pca=True)

			# Make sure perc is of type(float)
			elif perc:
				try:
					perc = float(perc)
				except:
					raise TypeError("\nThe perc argument could not be coerced to type(float)")

				# Calculate the percentage of principal components to retain
				indcount = gt_df.shape[0]
				n_components_frac = perc * indcount
				n_components_frac = int(n_components_frac)
				pca_settings_default.update({"n_components": n_components_frac})

				pca_coords, pca_model = self.dim_reduction(gt_df, ["standard-pca"], pca_settings=pca_settings, plot_pca_cumvar=False, return_pca=True)

			# If n_components is manually specified
			else:
				pca_coords, pca_model = self.dim_reduction(gt_df, ["standard-pca"], pca_settings=pca_settings, plot_pca_cumvar=False, return_pca=True)

			print("\nDoing unsupervised random forest...")

			self.rf_embedding(pca_coords, rf_settings_default)

			print("Done!")


		# Don't embed genotypes with PCA first
		# Just use raw 012-encoded genotypes
		else:

			print("\nDoing unsupervised random forest...")

			self.rf_embedding(gt_df, rf_settings_default)

			print("Done!")

	def rf_embedding(self, X, rf_settings_default):
		"""[Do unsupervised random forest embedding. Saves an rf model and a proximity score matrix]

		Args:
			X ([numpy.ndarray or pandas.DataFrame]): [Data to model. Can be principal components or 012-encoded genotypes]

			rf_settings_default ([dict]): [RandomTreesEmbedding settings with argument names as keys and their corresponding values]
		"""
		# Print settings to command-line
		print(
		"""
		Random Forest Embedding Settings:
			n_estimators: """+str(rf_settings_default["rf_n_estimators"])+"""
			max_depth: """+str(rf_settings_default["rf_max_depth"])+"""
			min_samples_split: """+str(rf_settings_default["rf_min_samples_split"])+"""
			min_samples_leaf: """+str(rf_settings_default["rf_min_samples_leaf"])+"""
			rf_min_weight_fraction_leaf: """+str(rf_settings_default["rf_min_weight_fraction_leaf"])+"""
			max_leaf_nodes: """+str(rf_settings_default["rf_max_leaf_nodes"])+"""
			min_impurity_decrease: """+str(rf_settings_default["rf_min_impurity_decrease"])+"""
			min_impurity_split: """+str(rf_settings_default["rf_min_impurity_split"])+"""
			sparse_output: """+str(rf_settings_default["rf_sparse_output"])+"""
			n_jobs: """+str(rf_settings_default["rf_n_jobs"])+"""
			rf_random_state: """+str(rf_settings_default["rf_random_state"])+"""
			rf_verbose: """+str(rf_settings_default["rf_verbose"])+"""
			warm_start: """+str(rf_settings_default["rf_warm_start"])+"""
			"""
		)

		# Initialize a random forest
		clf = RandomTreesEmbedding(
			n_estimators=rf_settings_default["rf_n_estimators"],
			max_depth=rf_settings_default["rf_max_depth"],
			min_samples_split=rf_settings_default["rf_min_samples_split"],
			min_samples_leaf=rf_settings_default["rf_min_samples_leaf"],
			min_weight_fraction_leaf=rf_settings_default["rf_min_weight_fraction_leaf"],
			max_leaf_nodes=rf_settings_default["rf_max_leaf_nodes"],
			min_impurity_decrease=rf_settings_default["rf_min_impurity_decrease"],
			min_impurity_split=rf_settings_default["rf_min_impurity_split"],
			sparse_output=rf_settings_default["rf_sparse_output"],
			n_jobs=rf_settings_default["rf_n_jobs"],
			random_state=rf_settings_default["rf_random_state"],
			verbose=rf_settings_default["rf_verbose"],
			warm_start=rf_settings_default["rf_warm_start"]
		)

		# Fit the model
		clf.fit(X)

		# transform and return the model
		rf_model = clf.transform(X)

		# Cast it to a numpy.ndarray
		self.rf = rf_model.toarray()

		self.prox_matrix, self.diss_matrix = self.calculate_rf_proximity_dissimilarity_mat(X, clf, rf_settings_default["rf_n_estimators"])

	def calculate_rf_proximity_dissimilarity_mat(self, X, clf, n_trees):
		"""[Calculate random forest proximity scores and a dissimilarity matrix from a fit sklearn.ensemble.RandomTreesEmbeddings model. Should be applied after model fitting but not on the transformed data. X is the modeled data]

		Args:
			X ([numpy.ndarray or pandas.DataFrame]): [The features to modeled with RandomTreesEmbeddings]

			clf ([RandomTreesEmbeddings object]): [The classifier returned from initializing RandomForestEmbeddings()]

			n_trees ([int]): [Number of estimator trees used in the RandomTreesEmbedding]

		Returns:
			[numpy.ndarray]: [Proximity score matrix and dissimilarity score matrix]
		"""
		# Apply trees in the forest to X, return leaf indices
		# This allows calculation of the proximity scores
		leaves = clf.apply(X)

		# Initialize proximity matrix
		prox = np.zeros((len(X), len(X)))
			
		# adapted implementation found here: 
		# https://stackoverflow.com/questions/18703136/proximity-matrix-in-sklearn-ensemble-randomforestclassifier

		# Generates proximity scores
		for tree_idx in range(n_trees):
			prox += np.equal.outer(leaves[:,tree_idx],
												leaves[:,tree_idx]).astype(float)
			
		# Divide by the number of estimators to normalize
		prox /= n_trees

		# Calculate dissimilarity matrix
		# Subtracts 1 from each element in numpy.ndarray
		diss = 1 - prox

		# Convert them to pandas.DataFrame objects
		diss = pd.DataFrame(diss)
		prox = pd.DataFrame(prox)

		return prox, diss

	def dim_reduction(
		self,
		data,
		algorithms, 
		pca_settings=None, 
		mds_settings=None, 
		tsne_settings=None,
		plot_pca_scatter=False, 
		pca_scatter_settings=None, 
		plot_pca_cumvar=False, 
		pca_cumvar_settings=None, 
		plot_cmds_scatter=False, 
		cmds_scatter_settings=None, 
		plot_isomds_scatter=False, 
		isomds_scatter_settings=None, 
		plot_tsne_scatter=False,
		tsne_scatter_settings=None,
		colors=None, 
		palette="Set1",
		plot_3d=False,
		return_pca=False
	):
		"""[Perform dimensionality reduction using the specified algorithms. Data can be of type numpy.ndarray or pandas.DataFrame. It can also be raw 012-encoded genotypes or the output from unsupervised random forest embedding. 
		
			For the random forest output, it can either be the dissimilarity matrix generated after running random_forest_unsupervised() or the rf_model object also generated with random_forest_unsupervised().
			
			data can be embedded using several currently supported algorithms, including 'standard-pca', 'cmds', and 'isomds'
			
			The settings for each algorithm can be changed by specifying a dictionary with the sklearn.manifold.MDS and sklearn.decomposition.PCA parameters as the keys and their corresponding values. See the documentation for scikit-learn for more information. For each of the settings dictionaries, default arguments are used if left unspecified and you can set one, some, or all. Arguments not manually included in the settings dictionary will use default settings for those arguments.

			Plots of each algorithm can also be made here by setting plot_pca_scatter=True, plot_cmds_scatter=True, and/ or plot_isomds_scatter=True. The plots can be customized by specifying a dictionary object for pca_scatter_settings, cmds_scatter_settings, and/ or isomds_scatter_settings. If the scatter settings are left unspecified, default arguments are used. Also, only the arguments you want to change need to be in the settings. A list of the supported settings will be included in our documentation.

			The colors argument allows you to map colors to specific population IDs. It is a dictionary object with unique population IDs as keys and hex-code colors as the values

			If colors=None, a default palette will be used, which can be changed by setting the palette argument. Supported options are include: (coming soon)]

		Args:
			data ([numpy.ndarray or pandas.DataFrame]): [Data to embed. Can be raw genotypes or fit models]

			algorithms ([list]): [Dimensionality reduction algorithms to perform]

			pca_settings ([dict], optional): [Dictionary with PCA settings as keys and the corresponding values. If pca_settings=None it will use all default arguments (scikit-allel and matplotlib documentation)]. Defaults to None.

			mds_settings ([dict], optional): [Dictionary with MDS settings as keys and the correpsonding values. If mds_settings=None it will use all default arguments (see sklearn.manifold and matplotlib documentation)]

			plot_pca_scatter (bool, optional): [If True, PCA results will be plotted as a scatterplot]. Defaults to False.

			pca_scatter_settings (dict, optional): [Change some or all settings for the PCA scatterplot]

			plot_pca_cumvar (bool, optional): [If True, creates a cumulative variance plot of the pca results and assesses the inflection point]

			plot_cmds_scatter (bool, optional): [If True, cMDS results will be plotted as a scatterplot]. Defaults to False.

			cmds_scatter_settings (dict, optional): [Change some or all settings for the cMDS plot]

			plot_isomds_scatter (bool, optional): [If True, isoMDS results will be plotted as a scatterplot]. Defaults to False.

			isomds_scatter_settings (dict, optional): [Change some or all settings for the isoMDS scatterplot]

			plot_tsne_scatter (bool, optional): [If True, T-SNE results will be plotted as a scatterplot]. Defaults to False.

			tsne_scatter_settings (dict, optional): [Chnage some or all settings for the T-SNE scatterplot. Setting names should be the dictionary keys with their corresponding values]

			colors (dict, optional): [Dictionary with population IDs as keys and hex-code colors as the values. If colors=None, dim_reduction will use a default color palette that can be changed with the palette argument]. Defaults to None.

			palette (str, optional): [Color palette to use with the scatterplots. See matplotlib.colors documentation]. Defaults to "Set1".

			plot_3d (bool, optional): [True if making 3D plot with 3 axes. False if 2D plot with 2 axes]. Defaults to False.

			return_pca (bool, optional): [True if you want to just return the PCA results. Ignores the MDS and T-SNE arguments]. Defaults to False.

		Returns:
			pca_coords (numpy.ndarray, optional): [If return_pca=True, the PCA coordinates get returned as a 2D numpy.ndarray with shape (n_samples, n_features)]

			pca_model (sklearn.decomposition.PCA, optional): [If return_pca=True, the PCA model gets returned]

			elbow (int, optional): [If return_pca=True and plot_pca_cumvar=True, elbow gets returned along with pca_coords and pca_model. Contains the number of principal components at the elbow of the curve]

		Raises:
			ValueError: [Must be a supported argument in algorithms]
		"""
		self.pca_settings = pca_settings
		self.mds_settings = mds_settings
		self.tsne_settings = tsne_settings
		self.algorithms = algorithms
		self.palette = palette
		self.colors = colors

		# Get default and supported settings for each method
		supported_algs = settings.dim_reduction_supported_algorithms()
		supported_settings = settings.dim_reduction_supported_arguments()
		pca_settings_default = settings.pca_default_settings()
		mds_settings_default = settings.mds_default_settings()
		tsne_settings_default = settings.tsne_default_settings()

		# Make sure the data are of the correct type
		gt_df = self._validate_type(self.gt)
		data_df = self._validate_type(data)

		# Validate that the settings keys are supported and update the default
		# settings with user-defined settings
		if self.pca_settings:
			self._validate_settings(self.pca_settings, supported_settings)
			pca_settings_default.update(self.pca_settings)

		# Make sure all manually specified settings are supported
		if self.mds_settings:
			self._validate_settings(self.mds_settings, supported_settings)
			mds_settings_default.update(self.mds_settings)
		
		if self.tsne_settings:
			self._validate_settings(self.tsne_settings, supported_settings)
			tsne_settings_default.update(self.tsne_settings)

		# Convert to list if user supplied a string
		if isinstance(self.algorithms, str):
			self.algorithms = [self.algorithms]

		for arg in self.algorithms:
			if arg not in supported_algs:
				raise ValueError("\nThe dimensionality reduction algorithm {} is not supported. Supported options include: {})".format(arg, supported_algs))

			if arg == "standard-pca":
				pca_coords, pca_model = dr.do_pca(data_df, pca_settings_default)
				
				# Plot PCA scatterplot
				if plot_pca_scatter:
					dr.plot_dimreduction(
						pca_coords,
						self.pops,
						self.prefix,
						arg,
						pca=True,
						pca_model=pca_model,
						plot_3d=plot_3d,
						user_settings=pca_scatter_settings,
						colors=self.colors,
						palette=self.palette
					)

				if return_pca and not plot_pca_cumvar:
					return pca_coords, pca_model

				# Plot cumulative PCA variance
				if plot_pca_cumvar:
					elbow = dr.plot_pca_cumvar(pca_coords, pca_model, self.prefix, pca_cumvar_settings)

					if return_pca:
						return pca_coords, pca_model, elbow

			elif arg == "cmds":
				cmds_model = dr.do_mds(data_df, mds_settings_default, metric=True, do_3d=plot_3d)

				# cMDS scatterplot
				if plot_cmds_scatter:
					dr.plot_dimreduction(
						cmds_model,
						self.pops,
						self.prefix,
						arg,
						pca=False,
						plot_3d=plot_3d,
						user_settings=cmds_scatter_settings,
						colors=self.colors,
						palette=self.palette
					)

			# isoMDS scatterplot
			elif arg == "isomds":
				isomds_model = dr.do_mds(data_df, mds_settings_default, metric=False, do_3d=plot_3d)

				if plot_isomds_scatter:
					dr.plot_dimreduction(
						isomds_model,
						self.pops,
						self.prefix,
						arg,
						pca=False,
						plot_3d=plot_3d,
						user_settings=isomds_scatter_settings,
						colors=self.colors,
						palette=self.palette
					)

			elif arg == "tsne":
				tsne_model = dr.tsne(data_df, tsne_settings_default)

				if plot_tsne_scatter:
					dr.plot_dimreduction(
						tsne_model,
						self.pops,
						self.prefix,
						arg,
						pca=False,
						plot_3d=plot_3d,
						user_settings=tsne_scatter_settings,
						colors=self.colors,
						palette=self.palette
					)

	def _validate_type(self, geno):
		"""[Validate that the genotypes are of the correct type. Also converts numpy.ndarrays and list(list) into pandas.DataFrame for use with DelimModel]

		Args:
			([numpy.ndarray, pandas.DataFrame, or list(list)]) : [GenotypeData object containing the genotype data]

		Returns:
			([pandas.DataFrame]): [Genotypes converted to pandas DataFrame]

		Raises:
			TypeError: [Must be of type numpy.ndarray, list, or pandas.DataFrame]
		"""
		if isinstance(geno, np.ndarray):
			gt_df = pd.DataFrame(geno)

		elif isinstance(geno, list):
			gt_df = pd.DataFrame.from_records(geno)

		elif isinstance(geno, pd.DataFrame):
			gt_df = geno.copy()

		else:
			raise TypeError("\nThe genotype data must be a numpy.ndarray, a pandas.DataFrame, or a 2-dimensional list! Any of these can be retrieved from the GenotypeData object")

		return gt_df

	def _validate_settings(self, settings, supported_settings):
		"""[Validate that settings are supported arguments]

		Args:
			settings ([dict]): [Settings for dimensionality reduction method]
			supported_settings ([list]): [Settings supported by any of the dimensionality reduction methods]

		Raises:
			ValueError: [One or more settings arguments was not found in supported_settings]
		"""
		# Make sure settings are supported by dimensionality reduction algorithm
		for arg in settings.keys():
			if arg not in supported_settings:
				raise ValueError("The settings argument {} is not supported".format(arg))

	#method will match labels across K -- changes self.results
	def clusterAcrossK(self, method, inplace=True):
		if inplace:
			for i in range(0, len(self.results)):
				self.results[i].clusterAcrossK(method, inplace=True)
		else:
			ret=list()
			for res in self.results:
				ret.append(res.clusterAcrossK(method, inplace=False))

	@property
	def rf_model(self):
		"""[Getter for the embedded random forest matrix]

		Returns:
			[numpy.ndarray]: [2-dimensional embedded random forest matrix]
		"""
		return self.rf

	@property
	def rf_dissimilarity(self):
		"""[Getter for the embedded random forest dissimilarity matrix]

		Returns:
			[pd.DataFrame]: [Square distance matrix with random forest dissimilarity scores]
		"""
		return self.diss_matrix

	@property
	def rf_proximity_scores(self):
		"""[Getter for the random forest proximity scores matrix]

		Returns:
			[numpy.ndarray]: [Square matrix with proximity scores from the random forest embedding]
		"""
		return self.prox_matrix
