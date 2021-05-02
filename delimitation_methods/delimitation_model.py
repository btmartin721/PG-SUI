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
from dim_reduction.dim_reduction import DimReduction
from utils import settings

class DelimModel:
	"""[Parent class for delimitation models]
	"""
	
	def __init__(self, gt, pops, prefix):
		"""[Class constructor]

		Args:
			gt ([list(list), numpy.ndarray, or pandas.DataFrame]): [One of the data formats generated from GenotypeData]

			pops ([list]): [List of populations IDs. Can be strings or integers]

			dim_red_algorithms ([list or str], optional): [List of dimensionality reduction algorithms to run. If dim_red_algorithms=None, all of them will be performed]. Defaults to None.

			settings ([dict], optional): [Dictionary of settings for each algorithm with setting names as keys and the settings as values. If settings=None, defaults for each algorithm will be used]. Defaults to None.
		"""
		self.gt = gt 
		self.pops = pops
		self.prefix = prefix
		self.gt_df = None
		self.pca_settings = None
		self.mds_settings = None
		self.rf_settings = None
		self.dim_red_algorithms = None
		self.colors = None
		self.palette = None
		self.dr = None

		# Model results
		self.rf = None
		self.prox_matrix = None
		self.diss_matrix = None
		self.pca = None
		self.rf_cmds = None
		self.rf_isomds = None

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

		if perc and elbow:
			raise ValueError("\nperc and elbow arguments cannot both be set")

		self.rf_settings = rf_settings

		supported_settings = settings.random_forest_unsupervised_supported()
		rf_settings_default = settings.random_forest_unsupervised_defaults()

		if rf_settings:
			rf_settings_default.update(rf_settings)
			self._validate_settings(rf_settings, supported_settings)

		# Make sure genotypes are a supported type and return a pandas.DataFrame
		self.gt_df = self._validate_gt_type(self.gt)

		# Embed the genotypes with PCA first
		if pca_init:
			pca_settings_default = settings.pca_default_settings()
			
			if pca_settings:
				pca_settings_default.update(pca_settings)
				

			if elbow:
				self.dim_reduction(self.gt_df, ["standard-pca"], pca_settings=pca_settings, plot_pca_cumvar=True)

				inflection = self.dr.pca_components_elbow
				print("\nRe-doing PCA with n_components set to the inflection point")
				
				self.dim_reduction(self.gt_df, ["standard-pca"], pca_settings={"n_components": int(inflection)}, plot_pca_cumvar=False)

			elif perc:
				try:
					perc = float(perc)
				except:
					raise TypeError("\nThe perc argument could not be coerced to type(float)")

				indcount = self.gt_df.shape[0]
				n_components_frac = perc * indcount
				n_components_frac = int(n_components_frac)
				pca_settings_default.update({"n_components": n_components_frac})

				pca_model = self.dim_reduction(self.gt_df, ["standard-pca"], pca_settings=pca_settings, plot_pca_cumvar=False)

			else:
				pca_model = self.dim_reduction(self.gt_df, ["standard-pca"], pca_settings=pca_settings, plot_pca_cumvar=False)

			self.pca = self.dr.get_pca_coords

			print("\nDoing unsupervised random forest...")

			self._rf_prox_matrix(self.pca, rf_settings_default)

			print("Done!")


		# Don't embed genotypes with PCA first
		# Might take a long time
		else:

			print("\nDoing unsupervised random forest...")

			self._rf_prox_matrix(self.gt_df, rf_settings_default)

			print("Done!")

	def _rf_prox_matrix(self, X, rf_settings_default):
		"""[Do unsupervised random forest embedding and calculate proximity scores. Saves an rf model and a proximity score matrix]

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

		# Apply trees in the forest to X, return leaf indices
		# This allows calculation of the proximity scores
		leaves = clf.apply(X)

		# transform and return the model
		rf_model = clf.transform(X)

		# Cast it to a numpy.ndarray
		self.rf = rf_model.toarray()

		# Initialize proximity matrix
		self.prox_matrix = np.zeros((len(X), len(X)))
		
		# adapted implementation found here: 
		# https://stackoverflow.com/questions/18703136/proximity-matrix-in-sklearn-ensemble-randomforestclassifier

		# Generates proximity scores
		for tree_idx in range(rf_settings_default["rf_n_estimators"]):
			self.prox_matrix += np.equal.outer(leaves[:,tree_idx],
												leaves[:,tree_idx]).astype(float)
		
		# Divide by the number of estimators to normalize
		self.prox_matrix /= rf_settings_default["rf_n_estimators"]

		# Calculate dissimilarity matrix
		self.diss_matrix = 1 - self.prox_matrix
		self.diss_matrix = pd.DataFrame(self.diss_matrix)

	def dim_reduction(
		self,
		data,
		dim_red_algorithms, 
		pca_settings=None, 
		mds_settings=None, 
		plot_pca_scatter=False, 
		pca_scatter_settings=None, 
		plot_pca_cumvar=False, 
		pca_cumvar_settings=None, 
		plot_cmds_scatter=False, 
		cmds_scatter_settings=None, 
		plot_isomds_scatter=False, 
		isomds_scatter_settings=None, 
		colors=None, 
		palette="Set1"):
		"""[Perform dimensionality reduction using the algorithms in the dim_red_algorithms list]

		Args:
			data ([numpy.ndarray]): [Data to embed. Can be raw genotypes or fit models]

			dim_red_algorithms ([list]): [Dimensionality reduction algorithms to perform]

			pca_settings ([dict], optional): [Dictionary with PCA settings as keys and the corresponding values. If pca_settings=None it will use all default arguments (scikit-allel and matplotlib documentation)]. Defaults to None.

			mds_settings ([dict], optional): [Dictionary with MDS settings as keys and the correpsonding values. If mds_settings=None it will use all default arguments (see sklearn.manifold and matplotlib documentation)]

			plot_pca_scatter (bool, optional): [If True, PCA results will be plotted as a scatterplot]. Defaults to False.

			pca_scatter_settings (dict, optional): [Change some or all settings for the PCA scatterplot]

			plot_pca_cumvar (bool, optional): [If True, creates a cumulative variance plot of the pca results and assesses the inflection point]

			plot_cmds_scatter (bool, optional): [If True, cMDS results will be plotted as a scatterplot]. Defaults to False.

			cmds_scatter_settings (dict, optional): [Change some or all settings for the cMDS plot]

			plot_isomds_scatter (bool, optional): [If True, isoMDS results will be plotted as a scatterplot]. Defaults to False.

			isomds_scatter_settings (dict, optional): [Change some or all settings for the isoMDS scatterplot]

			colors (dict, optional): [Dictionary with population IDs as keys and hex-code colors as the values. If colors=None, dim_reduction will use a default color palette that can be changed with the palette argument]. Defaults to None.

			palette (str, optional): [Color palette to use with the scatterplots. See matplotlib.colors documentation]. Defaults to "Set1".

		Raises:
			ValueError: [Must be a supported argument in dim_red_algoithms]
		"""
		self.pca_settings = pca_settings
		self.mds_settings = mds_settings
		self.dim_red_algorithms = dim_red_algorithms
		self.palette = palette
		self.colors = colors

		# Get default and supported settings for each method
		supported_algs = settings.dim_reduction_supported_algorithms()
		supported_settings = settings.dim_reduction_supported_arguments()
		pca_settings_default = settings.pca_default_settings()
		mds_settings_default = settings.mds_default_settings()

		# Make sure the data are of the correct type
		self.gt_df = self._validate_gt_type(self.gt)
		data_df = self._validate_gt_type(data)

		# Validate that the settings keys are supported and update the default
		# settings with user-defined settings
		if self.pca_settings:
			self._validate_settings(self.pca_settings, supported_settings)

			pca_settings_default.update(self.pca_settings)

		if self.mds_settings:
			self._validate_settings(self.mds_settings, supported_settings)

			mds_settings_default.update(self.mds_settings)

		# Convert to list if user supplied a string
		if isinstance(self.dim_red_algorithms, str):
			self.dim_red_algorithms = [self.dim_red_algorithms]

		# Do dimensionality reduction
		self.dr = DimReduction(data_df, self.pops)

		for arg in self.dim_red_algorithms:
			if arg not in supported_algs:
				raise ValueError("\nThe dimensionality reduction algorithm {} is not supported. Supported options include: {})".format(arg, supported_algs))

			if arg == "standard-pca":
				self.dr.standard_pca(pca_settings_default)
				
				# Plot PCA scatterplot
				if plot_pca_scatter:
					self.dr.plot_dimreduction(
						self.prefix,
						pca=True,
						cmds=False,
						isomds=False,
						user_settings=pca_scatter_settings,
						colors=self.colors,
						palette=self.palette
					)

				if plot_pca_cumvar:
					self.dr.plot_pca_cumvar(self.prefix, pca_cumvar_settings)

			elif arg == "cmds":
				self.dr.do_mds(data_df, mds_settings_default, metric=True)

				if plot_cmds_scatter:
					self.dr.plot_dimreduction(
						self.prefix,
						pca=False,
						cmds=True,
						isomds=False,
						user_settings=cmds_scatter_settings,
						colors=self.colors,
						palette=self.palette
					)

			elif arg == "isomds":
				self.dr.do_mds(data_df, mds_settings_default, metric=False)

				if plot_isomds_scatter:
					self.dr.plot_dimreduction(
						self.prefix,
						pca=False,
						cmds=False,
						isomds=True,
						user_settings=isomds_scatter_settings,
						colors=self.colors,
						palette=self.palette
					)

	def _validate_gt_type(self, geno):
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
			
class ModelResult:
	"""[Object to hold results from replicates of a single delim model]
	
	"""
	def __init__(self):
		self.model = None
		self.reps = list()
		self.num_reps = 0
	
	def clusterAcrossK(self, method, inplace=True):
		ret=list()
		for i in range(0, len(self.reps)):
			if method == "exhaustive":
				#ret.append()
				pass
			elif method == "heuristic":
				#ret.append()
				pass
			elif method == "graph":
				#ret.append()
				pass
			else:
				sys.exit("\nError: Method",method,"invalid for ModelResult.clusterAcrossK\n")
		
		if inplace:
			for i in range(0, len(ret)):
				self.reps[i] = reps[i]
		else:
			return(ret)
	