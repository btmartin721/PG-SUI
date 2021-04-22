# Standard library imports
import argparse
import sys

import numpy as np

# Make sure python version is >= 3.6
if sys.version_info < (3, 6):
	raise ImportError("Python < 3.6 is not supported!")

# Custom module imports
from dim_reduction.pca import DimReduction
from read_input.read_input import GenotypeData

def main():
	"""[Class instantiations and main package body]
	"""

	args = get_arguments()
	
	if args.str and args.phylip:
		sys.exit("Error: Only one file type can be specified")

	imputation_settings = {"n_neighbors": 5}

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
			data = GenotypeData(filename=args.str, filetype="structure1row", popmapfile=args.popmap, impute_methods="knn", impute_settings=imputation_settings)
		else:
			data = GenotypeData(filename=args.str, filetype="structure2row", popmapfile=args.popmap, impute_methods="knn", impute_settings=imputation_settings)

	if args.phylip:
		if (args.pop_ids or 
			args.onerow_perind):

			print("\nPhylip file was used with structure arguments; ignoring structure file arguments\n")
		
		if args.popmap is None:
			sys.exit("\nError: No popmap file supplied with Phylip-formatted input data\n")
		
		data = GenotypeData(filename=args.phylip, filetype="phylip", popmapfile=args.popmap, impute_methods="knn", impute_settings=imputation_settings)

	#pca_settings = {"n_components": data.indcount, "copy": True, "scaler": "patterson", "ploidy": 2}

	#pca = DimReduction(data=data.genotypes, algorithms="standard-pca", settings=pca_settings)

	#print(data.genotypes_list)
	#data.convert_onehot()
	#data.convert_df()

	#print(data.individuals)
	#print(data.populations)


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