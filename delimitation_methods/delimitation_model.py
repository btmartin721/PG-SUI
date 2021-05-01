# Standard library imports
import sys
import os

# Third-party imports
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomTreesEmbedding

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

		# Model results
		self.rf = None
		self.pca = None
		self.cmds = None
		self.isomds = None

	def random_forest_unsupervised(self, rf_settings=None, pca_init=True):

		self.rf_settings = rf_settings

		supported_settings = settings.random_forest_unsupervised_supported()
		rf_settings_default = settings.random_forest_unsupervised_defaults()

		if rf_settings:
			rf_settings_default.update(rf_settings)
			self._validate_settings(rf_settings, supported_settings)

		# Make sure genotypes are a supported type and return a pandas.DataFrame
		self.gt_df = _validate_gt_type(self.gt)

		rf = RandomTreesEmbedding(
			n_estimators=rf_settings["rf_n_estimators"],
			max_depth=rf_settings["rf_max_depth"],
			min_samples_split=rf_settings["rf_min_samples_split"],
			min_samples_leaf=rf_settings["rf_min_samples_leaf"],
			min_weight_fraction_leaf=rf_settings["rf_min_weight_fraction_leaf"],
			max_leaf_nodes=rf_settings["rf_max_leaf_nodes"],
			min_impurity_decrease=rf_settings["rf_min_impurity_decrease"],
			min_impurity_split=rf_settings["rf_min_impurity_split"],
			sparse_output=rf_settings["rf_sparse_output"],
			n_jobs=rf_settings["rf_n_jobs"],
			random_state=rf_settings["rf_random_state"],
			verbose=rf_settings["rf_verbose"],
			warm_start=rf_settings["rf_warm_start"]
		)

	def dim_reduction(
		self, 
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

		# Make sure the genotypes are of the correct type
		self.gt_df = self._validate_gt_type(self.gt)

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
		dr = DimReduction(self.gt_df, self.pops, algorithms=self.dim_red_algorithms)

		for arg in self.dim_red_algorithms:
			if arg not in supported_algs:
				raise ValueError("\nThe dimensionality reduction algorithm {} is not supported. Supported options include: {})".format(arg, supported_algs))

			if arg == "standard-pca":
				dr.standard_pca(pca_settings_default)
				
				# Plot PCA scatterplot
				if plot_pca_scatter:
					dr.plot_dimreduction(
						self.prefix,
						pca=True,
						cmds=False,
						isomds=False,
						user_settings=pca_scatter_settings,
						colors=self.colors,
						palette=self.palette
					)

				if plot_pca_cumvar:
					dr.plot_pca_cumvar(self.prefix, pca_cumvar_settings)

			elif arg == "cmds":
				dr.do_mds(mds_settings_default, metric=True)

				if plot_cmds_scatter:
					dr.plot_dimreduction(
						self.prefix,
						pca=False,
						cmds=True,
						isomds=False,
						user_settings=cmds_scatter_settings,
						colors=self.colors,
						palette=self.palette
					)

			elif arg == "isomds":
				dr.do_mds(mds_settings_default, metric=False)

				if plot_isomds_scatter:
					dr.plot_dimreduction(
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
	