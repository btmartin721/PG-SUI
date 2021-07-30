#!/usr/bin/env python

# Standard library imports
import argparse
import sys

import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn_genetic.space import Continuous, Categorical, Integer

from utils.misc import get_processor_name

print(get_processor_name().strip().startswith("Intel"))

# Custom module imports
from read_input.read_input import GenotypeData
from super_deli_imputer.impute import *

from dim_reduction.dim_reduction import DimReduction
from dim_reduction.embed import *

from clustering.clustering import *

def main():
	"""[Class instantiations and main package body]
	"""

	#args = get_arguments()

	data = GenotypeData(
		filename="example_data/structure_files/test.nopops.str", 
		filetype="structure2row", 
		popmapfile="example_data/popmaps/test.popmap"
	)

	# # Random Forest gridparams - RandomizedSearchCV
	# grid_params = {
	# 	"n_estimators": n_estimators,
	# 	"max_features": max_features,
	# 	"max_depth": max_depth,
	# 	"min_samples_split": min_samples_split, 
	# 	"min_samples_leaf": min_samples_leaf,
	# 	"max_samples": max_samples
	# }

	# Random Forest gridparams - Genetic Algorithms
	# grid_params = {
	# 	"n_estimators": Integer(100, 500),
	# 	"max_features": max_features,
	# 	"max_depth": max_depth,
	# 	"min_samples_split": min_samples_split, 
	# 	"min_samples_leaf": min_samples_leaf,
	# 	"max_samples": max_samples
	# }

	# grid_params = {
	# 	"max_features": Categorical(["sqrt", "log2"]),
	# 	"min_samples_split": Integer(2, 10), 
	# 	"min_samples_leaf": Integer(1, 10),
	# 	"max_depth": Integer(3, 110)
	# }

	# Bayesian Ridge gridparams - RandomizedSearchCV
	# grid_params = {
	# 	"alpha_1": stats.loguniform(1e-6, 1e-3),
	# 	"alpha_2": stats.loguniform(1e-6, 1e-3),
	# 	"lambda_1": stats.loguniform(1e-6, 1e-3),
	# 	"lambda_2": stats.loguniform(1e-6, 1e-3),
	# }

	# Bayesian Ridge gridparams - Genetic algorithm
	# grid_params = {
	# 	"alpha_1": Continuous(1e-6, 1e-3, distribution="log-uniform"),
	# 	"alpha_2": Continuous(1e-6, 1e-3, distribution="log-uniform"),
	# 	"lambda_1": Continuous(1e-6, 1e-3, distribution="log-uniform"),
	# 	"lambda_2": Continuous(1e-6, 1e-3, distribution="log-uniform")
	# }

	# rf_imp = ImputeRandomForest(
	# 		data, 
	# 		prefix=args.prefix, 
	# 		n_estimators=1000,
	# 		n_nearest_features=5, 
	# 		gridparams=grid_params, 
	# 		cv=3, 
	# 		grid_iter=40, 
	# 		n_jobs=32, 
	# 		max_iter=25, 
	# 		column_subset=100,
	# 		ga=True,
	# 		disable_progressbar=True
	# )

	prefix = "profiling_test_ntrees100_maxdepth15"

	rf_imp = ImputeRandomForest(
			data, 
			prefix=prefix, 
			n_estimators=100,
			n_nearest_features=3, 
			n_jobs=1, 
			max_iter=3, 
			disable_progressbar=True,
			extratrees=False, 
			progress_update_percent=20,
			max_features="log2",
			min_samples_split=6,
			min_samples_leaf=4,
			max_depth=15
	)

	# br_imp = ImputeBayesianRidge(data, prefix=prefix, n_iter=100, gridparams=grid_params, grid_iter=3, cv=3, n_jobs=1, max_iter=2, n_nearest_features=3, column_subset=10, ga=True, disable_progressbar=True)

	# colors = {
	# 	"GU": "#FF00FF",
	# 	"EA": "#FF8C00",
	# 	"TT": "#228B22",
	# 	"TC": "#6495ED",
	# 	"DS": "#00FFFF",
	# 	"ON": "#800080",
	# 	"CH": "#696969",
	# 	"FL": "#FFFF00",
	# 	"MX": "#FF0000"
	# }

	# dr = DimReduction(
	# 	data.imputed_rf_df, 
	# 	data.populations, 
	# 	data.individuals,
	# 	args.prefix, 
	# 	colors=colors, 
	# 	reps=2
	#)

	# pca = runPCA(dimreduction=dr, plot_cumvar=False, keep_pcs=10)

	# pca.plot(plot_3d=True)

	#rf = runRandomForestUML(dimreduction=dr, n_estimators=1000, n_jobs=1, min_samples_leaf=4)

	# rf_cmds = runMDS(
	# 	dimreduction=dr, 
	# 	distances=rf.dissimilarity, 
	# 	keep_dims=3, 
	# 	n_jobs=1, 
	# 	max_iter=1000, 
	# 	n_init=25
	# )

	# rf_isomds = runMDS(dr, distances=rf.dissimilarity_matrix, metric=False, keep_dims=3, n_jobs=1, max_iter=1000, n_init=25)

	# rf_cmds.plot(plot_3d=True)
	# rf_isomds.plot(plot_3d=True)

	# tsne = runTSNE(dimreduction=dr, keep_dims=3, n_iter=20000, perplexity=15.0)
	# tsne.plot(plot_3d=True)

	#maxk = 9

	#pam_rf = PamClustering(rf_cmds, use_embedding=False, dimreduction=dr, distances=rf.dissimilarity, maxk=9, max_iter=2500)

	# pam.msw(plot_msw_clusters=True, plot_msw_line=True, axes=3)

	#pam_rf.gapstat(plot_gap=True, show_plot=False)

	# kmeans = KMeansClustering(rf_cmds, dimreduction=dr, sampleids=data.individuals, maxk=9)
	# kmeans.msw(plot_msw_clusters=True, plot_msw_line=True, axes=3)

	# tsne_pam = PamClustering(tsne, dimreduction=dr, sampleids=data.individuals)
	# tsne_pam.msw(plot_msw_clusters=True, plot_msw_line=True, axes=3)

	# cmds_dbscan = DBSCANClustering(
	# 	rf_cmds, dimreduction=dr, sampleids=data.individuals, plot_eps=True
	# )
	
	#cmds_dbscan.plot(plot_3d=True)

	#cmds_affprop = AffinityPropogationClustering(rf_cmds, dimreduction=dr, sampleids=data.individuals)

	#cmds_affprop.plot(plot_3d=True)

	#cmds_aggclust = AgglomHier(rf_cmds, dimreduction=dr, maxk=maxk, sampleids=data.individuals)
	#cmds.msw(plot_msw_clusters=True, plot_msw_line=True, axes=3)

def get_arguments():
	"""[Parse command-line arguments. Imported with argparse]

	Returns:
		[argparse object]: [contains command-line arguments; accessed as method]
	"""

	parser = argparse.ArgumentParser(description="Convert VCF file to BGC format (with genotype uncertainties). Currently only handles three populations maximum (P1, P2, and Admixed).", add_help=False)

	required_args = parser.add_argument_group("Required arguments")
	filetype_args = parser.add_argument_group("File type arguments (choose only one)")
	structure_args = parser.add_argument_group("Structure file arguments")
	optional_args = parser.add_argument_group("Optional arguments")

	# File Type arguments
	filetype_args.add_argument("-s", "--str",
								type=str,
								required=False,
								help="Input structure file")
	filetype_args.add_argument("-p", "--phylip",
								type=str,
								required=False,
								help="Input phylip file")
	
	filetype_args.add_argument("-t", "--treefile",
								type=str,
								required=False,
								default=None,
								help="Newick-formatted treefile")

	# Structure Arguments
	structure_args.add_argument("--onerow_perind",
								default=False,
								action="store_true",
								help="Toggles on one row per individual option in structure file")
	structure_args.add_argument("--pop_ids",
								default=False,
								required=False,
								action="store_true",
								help="Toggles on population ID column (2nd col) in structure file")
	
	
	## Optional Arguments
	optional_args.add_argument("-m", "--popmap",
								type=str,
								required=False,
								default=None,
								help="Two-column tab-separated population map file: inds\tpops. No header line")
	optional_args.add_argument("--prefix",
								type=str,
								required=False,
								default="output",
								help="Prefix for output files")

	optional_args.add_argument("--resume_imputed",
								type=str,
								required=False,
								help="Read in imputed data from a file instead of doing the imputation")						
	# Add help menu							
	optional_args.add_argument("-h", "--help",
								action="help",
								help="Displays this help menu")

	# If no command-line arguments are called then exit and call help menu.
	if len(sys.argv)==1:
		print("\nExiting because no command-line options were called.\n")
		parser.print_help(sys.stderr)
		sys.exit(1)

	args = parser.parse_args()
	return args

if __name__ == "__main__":
	main()
