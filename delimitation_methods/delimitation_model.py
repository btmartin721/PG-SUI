# Standard library imports
import sys
import os

# Third-party imports
import numpy as np
import pandas as pd

# Custom module imports
from read_input.read_input import GenotypeData
from dim_reduction.dim_reduction import DimReduction

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
		self.dim_red_algorithms = None
		self.colors = None
		self.palette = None

		# Model results
		self.pca = None

	def dim_reduction(self, dim_red_algorithms, pca_settings=None, plot_pca_scatter=False, colors=None, palette="Set1"):
		"""[Perform dimensionality reduction using the algorithms in the dim_red_algorithms list]

		Args:
			dim_red_algorithms ([list]): [Dimensionality reduction algorithms to perform]

			pca_settings ([dict], optional): [Dictionary with PCA settings at keys and the corresponding values. If pca_settings=None it will use all default arguments]. Defaults to None.

			plot_pca_scatter (bool, optional): [If True, PCA results will be plotted as a scatterplot]. Defaults to False.

			colors ([dict], optional): [Dictionary with population IDs as keys and hex-code colors as the values. If colors=None, dim_reduction will use a default color palette that can be changed with the palette argument]. Defaults to None.

			palette (str, optional): [Color palette to use with the scatterplots. See matplotlib.colors documentation]. Defaults to "Set1".

		Raises:
			ValueError: [Must be a supported argument in dim_red_algoithms]
		"""
		self.pca_settings = pca_settings
		self.dim_red_algorithms = dim_red_algorithms
		self.palette = palette
		self.colors = colors

		supported_algs = ["standard-pca"]

		supported_settings = [
								"n_components", 
								"copy", 
								"scaler", 
								"ploidy",
								"pc_axis1",
								"pc_axis2",
								"figwidth", 
								"figheight", 
								"alpha", 
								"legend", 
								"legend_inside", 
								"legend_loc", 
								"marker", 
								"markersize", 
								"markeredgecolor", 
								"markeredgewidth", 
								"labelspacing", 
								"columnspacing", 
								"title", 
								"title_fontsize",
								"markerfirst", 
								"markerscale", 
								"labelcolor", 
								"ncol", 
								"bbox_to_anchor", 
								"borderaxespad", 
								"legend_edgecolor", 
								"facecolor", 
								"framealpha", 
								"shadow"
							]

		pca_settings_default = 		{
								"n_components": 10, 
								"copy": True, 
								"scaler": "patterson", 
								"ploidy": 2,
								"pc_axis1": 1,
								"pc_axis2": 2,
								"figwidth": 6, 
								"figheight": 6, 
								"alpha": 1.0, 
								"legend": True, 
								"legend_inside": False, 
								"legend_loc": "upper left", 
								"marker": 'o', 
								"markersize": 6, 
								"markeredgecolor": 'k', 
								"markeredgewidth": 0.5, 
								"labelspacing": 0.5, 
								"columnspacing": 2.0, 
								"title": None, 
								"title_fontsize": None,
								"markerfirst": True, 
								"markerscale": 1.0, 
								"labelcolor": "black", 
								"ncol": 1, 
								"bbox_to_anchor": (1.0, 1.0), 
								"borderaxespad": 0.5, 
								"legend_edgecolor": "black", 
								"facecolor": "white", 
								"framealpha": 0.8, 
								"shadow": False
							}

		self.gt_df = self._validate_gt_type(self.gt)

		# Validate that the settings keys are supported and update the default
		# settings with user-defined settings
		if self.pca_settings:
			self._validate_dimred_settings(self.pca_settings, supported_settings)
			pca_settings_default.update(self.pca_settings)

		# Convert to list if user supplied a string
		if isinstance(self.dim_red_algorithms, str):
			self.dim_red_algorithms = [self.dim_red_algorithms]

		dimred = DimReduction(self.gt_df, self.pops, algorithms=self.dim_red_algorithms)

		for arg in self.dim_red_algorithms:
			if arg not in supported_algs:
				raise ValueError("\nThe dimensionality reduction algorithm {} is not supported. Supported options include: {})".format(arg, supported_algs))

			if arg == "standard-pca":
				dimred.standard_pca(pca_settings_default)
				
				if plot_pca_scatter:
					dimred.plot_pca(
					self.prefix, 
					int(pca_settings_default["pc_axis1"]),
					int(pca_settings_default["pc_axis2"]), 
					pca_settings_default["figwidth"], 
					figheight=pca_settings_default["figheight"],
					alpha=pca_settings_default["alpha"],
					legend=pca_settings_default["legend"], 
					legend_inside=pca_settings_default["legend_inside"], 
					legend_loc=pca_settings_default["legend_loc"], 
					marker=pca_settings_default["marker"], 
					markersize=pca_settings_default["markersize"], 
					markeredgecolor=pca_settings_default["markeredgecolor"], 
					markeredgewidth=pca_settings_default["markeredgewidth"], 
					labelspacing=pca_settings_default["labelspacing"], 
					columnspacing=pca_settings_default["columnspacing"], 
					title=pca_settings_default["title"], 
					title_fontsize=pca_settings_default["title_fontsize"], 
					markerfirst=pca_settings_default["markerfirst"], 
					markerscale=pca_settings_default["markerscale"], 
					ncol=pca_settings_default["ncol"], 
					bbox_to_anchor=pca_settings_default["bbox_to_anchor"], 
					borderaxespad=pca_settings_default["borderaxespad"], 
					legend_edgecolor=pca_settings_default["legend_edgecolor"], 
					facecolor=pca_settings_default["facecolor"], 
					framealpha=pca_settings_default["framealpha"], 
					shadow=pca_settings_default["shadow"],
					colors=self.colors,
					palette=self.palette
					)

		print(dimred.get_pca_coords)


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

	def _validate_dimred_settings(self, settings, supported_settings):
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
	