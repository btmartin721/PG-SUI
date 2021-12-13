
# conda install -c bioconda pyvcf 
import vcf
import sys

class vcfFile:
	"""
	[Class to read and parse VCF files]
	"""
	def __init__(self, vcf_file):
		"""[Class instantiation]

		Args:
			vcf_file ([string]): [Input VCF filename]
		"""
		self.vcf_file = vcf_file

	def read_vcf(self):
		"""[Read VCF file into PyVCF object]

		Returns:
			[dict of lists]: [dictionary of lists containing sample ids and ref and alt alleles]
		"""
		alleles_dict = dict()
		ref_alleles = list()
		alt_alleles = list()
		print("\nReading VCF file...\n")
		with open(self.vcf_file, "r") as fin:
			# Read VCF using PyVCF module.
			vcf_reader = vcf.Reader(fin)
			samples = vcf_reader.samples # Get list of sample ids.
			alleles_dict["sample_id"] = samples
			for record in vcf_reader:
				if len(record.ALT) > 1: # If more than one ALT allele.
					sys.exit("Error: Chromosome {} at position {} is not bi-allelic. All sites must be bi-allelic.\n".format(record.CHROM, record.POS))
				else:
					ref_alleles.append(record.REF)
					alt_alleles.append(str(record.ALT[0]))
		alleles_dict["ref"] = ref_alleles
		alleles_dict["alt"] = alt_alleles
		return alleles_dict

	def vcf2onehot(self, vcf_data):

		onehot_dict = {
			"A" : "1.0,0.0,0.0,0.0",
			"T" : "0.0,1.0,0.0,0.0",
			"G" : "0.0,0.0,1.0,0.0",
			"C" : "0.0,0.0,0.0,1.0",
			"R" : "0.5,0.0,0.5,0.0",
			"Y" : "0.0,0.5,0.0,0.5",
			"S" : "0.0,0.5,0.5,0.0",
			"W" : "0.5,0.0,0.0,0.5",
			"K" : "0.0,0.0,0.5,0.5",
			"M" : "0.5,0.5,0.0,0.0",
			"N" : "0.0,0.0,0.0,0.0",
			"-" : "0.0,0.0,0.0,0.0"
	}


			


