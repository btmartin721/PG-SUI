# Standard library imports
import argparse
import sys
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

# Make sure python version is >= 3.6
if sys.version_info < (3, 6):
	raise ImportError("Python < 3.6 is not supported!")

# Custom module imports
from delimitation_methods.delimitation_model import DelimModel
from read_input.read_input import GenotypeData
import read_input.impute as impute

from dim_reduction.dim_reduction2 import DimReduction
from dim_reduction.embed import runPCA
from dim_reduction.embed import runRandomForestUML
from dim_reduction.embed import runMDS
from dim_reduction.embed import runTSNE

def main():
	"""[Class instantiations and main package body]
	"""

	args = get_arguments()

	if args.str and args.phylip:
		sys.exit("Error: Only one file type can be specified")

	br_imputation_settings = {
								"br_n_iter": 1000,
								"n_nearest_features": 25
							}
		
		# If VCF file is specified.
	if args.str:
		if not args.pop_ids and args.popmap is None:
			sys.exit("\nError: Either --pop_ids or --popmap must be specified\n")

		if args.pop_ids:
			print("\n--pop_ids was specified as column 2\n".format(args.pop_ids))
		else:
			print("\n--pop_ids was not specified; using the popmap file to get population IDs\n")
		
		if args.onerow_perind:
			print("\nUsing one row per individual...\n")
		else:
			print("\nUsing two rows per individual...\n")
			
		if args.onerow_perind:
			data = GenotypeData(filename=args.str, filetype="structure1row", popmapfile=args.popmap)
		else:
			data = GenotypeData(filename=args.str, filetype="structure2row", popmapfile=args.popmap)

	if args.phylip:
		if (args.pop_ids or 
			args.onerow_perind):

			print("\nPhylip file was used with structure arguments; ignoring structure file arguments\n")
		
		if args.popmap is None:
			sys.exit("\nError: No popmap file supplied with Phylip-formatted input data\n")
		
		data = GenotypeData(filename=args.phylip, filetype="phylip", popmapfile=args.popmap)
		
	if args.resume_imputed:
		data.read_imputed(args.resume_imputed, impute_methods="rf")
		#data.write_imputed(data.imputed_rf_df, args.prefix)

	else:	
		data.impute_missing(impute_methods="freq_pop", impute_settings=br_imputation_settings)

	colors = {
		"GU": "#FF00FF",
		"EA": "#FF8C00",
		"TT": "#228B22",
		"TC": "#6495ED",
		"DS": "#00FFFF",
		"ON": "#800080",
		"CH": "#696969",
		"FL": "#FFFF00",
		"MX": "#FF0000"
	}

	dr = DimReduction(data.imputed_rf_df, data.populations, args.prefix, colors=colors)

	pca = runPCA(dimreduction=dr, plot_cumvar=False, keep_pcs=10)

	pca.plot(plot_3d=True)

	rf = runRandomForestUML(dr, n_estimators=1000, n_jobs=4, min_samples_leaf=4)

	rf_cmds = runMDS(dr, dissimilarity_matrix = rf.dissimilarity_matrix, keep_dims=3, n_jobs=4, max_iter=1000, n_init=25)

	rf_isomds = runMDS(dr, dissimilarity_matrix = rf.dissimilarity_matrix, metric = False, keep_dims=3, n_jobs=4, max_iter=1000, n_init=25)

	rf_cmds.plot(plot_3d=True)
	rf_isomds.plot(plot_3d=True)

	tsne = runTSNE(dr, keep_dims=3, n_iter=20000, perplexity=15.0)
	tsne.plot(plot_3d=True)


	#data.write_imputed(data.imputed_br_df, args.prefix)
	
	# pcs = data.indcount - 1

	# pca_settings = {
	# 	"scaler": "patterson",
	# 	"n_components": pcs
	# }

	# mds_settings = {
	# 	"n_init": 100,
	# 	"max_iter": 1000,
	# 	"n_jobs": 4,
	# 	"eps": 1e-4,
	# 	"dissimilarity": "precomputed",
	# 	"n_dims": 3
	# }

	# tsne_settings = {
	# 	"n_components": 3,
	# 	"perplexity": 15
	# }

	# cmds_scatter_settings = {
	# 	"bbox_to_anchor": (1.1, 1.0)
	# }

	# isomds_scatter_settings = {
	# 	"bbox_to_anchor": (1.1, 1.0)
	# }

	# tsne_scatter_settings = {
	# 	"bbox_to_anchor": (1.1, 1.0)
	# }

	

	# # See matplotlib axvline settings for options
	# pca_cumvar_settings={"text_size": 14, "style": "white", "figwidth": 6, "figheight": 6, "intercept_width": 3, "intercept_color": "r", "intercept_style": "--"}

	# clusters = DelimModel(data.imputed_rf_df, data.populations, args.prefix)

	# clusters.dim_reduction(data.imputed_rf_df, algorithms=["pca"], pca_settings=pca_settings, mds_settings=mds_settings, plot_pca_scatter=True, plot_pca_cumvar=True, pca_cumvar_settings=pca_cumvar_settings, plot_cmds_scatter=True, plot_isomds_scatter=True, plot_3d=True)

	# rf_embed_settings = {"rf_n_estimators": 1000, "rf_n_jobs": 4}

	#clusters.random_forest_unsupervised(pca_settings={"n_components": 10}, pca_init=True, rf_settings=rf_embed_settings, elbow=False)

	#clusters.dim_reduction(clusters.rf_dissimilarity, algorithms=["cmds", "tsne"], plot_cmds_scatter=True, plot_isomds_scatter=True, mds_settings=mds_settings, tsne_settings=tsne_settings, plot_3d=True, cmds_scatter_settings=cmds_scatter_settings, isomds_scatter_settings=isomds_scatter_settings, plot_tsne_scatter=True, tsne_scatter_settings=tsne_scatter_settings, plot_pca_cumvar=True)

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