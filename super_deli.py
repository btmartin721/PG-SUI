import argparse
import sys

from read_input.structure_file import StrFile
from read_input.phylip_file import PhylipFile

def main():
	"""[Class instantiations and main package body]
	"""

	args = get_arguments()

	if args.str and args.phylip:
		sys.exit("Error: Only one file type can be specified")

	# If VCF file is specified.
	if args.str:

		print("\nUsing {} to as the missing data character\n".format(str(args.na_character)))

		print("\nUsing column {} as the first genotype column\n".format(args.first_snp_column))

		if not args.pop_ids and not args.popmap:
			sys.exit("\nError: Either --pop_ids or --popmap must be specified\n")

		if args.pop_ids:
			print("\n--pop_ids was specified as column {}\n".format(args.pop_ids))
		else:
			print("\n--pop_ids was not specified; using the popmap file to get population IDs\n")

		if args.onerow_perind:
			print("\nUsing one row per individual...\n")
		else:
			print("\nUsing two rows per individual...\n")

		s = StrFile(str(args.str), str(args.na_character))

		str_dict, number_of_snps, number_of_inds = s.read_structure_file(args.column_labels, args.pop_ids, args.first_snp_column, args.onerow_perind, args.popmap)

		data_dict = s.separate_structure_alleles(str_dict, number_of_inds, number_of_snps, args.onerow_perind, args.popmap)

		data_dict = s.structure2onehot(data_dict)
		# print(data_dict["onehot"])

	if args.phylip:

		if (args.pop_ids or 
			args.na_character or 
			args.first_snp_column or 
			args.column_labels or 
			args.onerow_perind):

			print("\nPhylip file was used with structure arguments; ignoring structure file arguments\n")

		p = PhylipFile(str(args.phylip))
		data_dict = p.read_phylip(args.popmap)
		data_dict = p.phylip2onehot(data_dict)

		p.phylip2df(data_dict["snps"])


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
	structure_args.add_argument("--column_labels",
								default=False,
								action="store_true",
								help="Specifies that structure file has column labels")
	structure_args.add_argument("--pop_ids",
								type=int,
								required=False,
								help="Column number for population ids (if present); integer with 1-based indexing")
	structure_args.add_argument("--na_character",
								type=str,
								required=False,
								default="-9",
								help="Specifies value for missing data in structure file; default = -9")
	structure_args.add_argument("--first_snp_column",
								type=int,
								required=False,
								help="Number of column with first SNP site; integer starting at 1")

	## Optional Arguments
	optional_args.add_argument("-m", "--popmap",
								type=str,
								required=True,
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