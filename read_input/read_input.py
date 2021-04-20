import sys

import numpy as np
import pandas as pd

from read_input.popmap_file import ReadPopmap

class GenotypeData:
	"""[Class to read genotype data and convert to onehot encoding]
	"""
	def __init__(self, filename, filetype=None, popmapfile=None):
		self.filename = filename
		self.filetype = filetype
		self.popmapfile = popmapfile
		self.samples = list()
		self.snps = list()
		self.pops = list()
		self.onehot = list()
		self.df = None
		
		if self.filetype is not None:
			self.parse_filetype(filetype)
		
		if self.popmapfile is not None:
			self.read_popmap(popmapfile)
	
	def parse_filetype(self, filetype=None):
		if filetype is None:
			sys.exit("\nError: No filetype specified.\n")
		else:
			if filetype == "phylip":
				self.filetype = filetype
				self.read_phylip()
			else:
				sys.exit("\nError: Filetype",filetype,"not implemented!\n")

	def check_filetype(self, filetype):
		if self.filetype is None:
			self.filetype = filetype
		elif self.filetype == filetype:
			pass
		else:
			sys.exit("\nError: GenotypeData read_XX() call does not match filetype!\n")

	def read_phylip(self):
		"""[Read phylip file from disk]

		Args:
			popmap_filename [str]: [Filename for population map file]

		Populates ReadInput object by parsing Phylip
		"""
		self.check_filetype("phylip")
		
		with open(self.filename, "r") as fin:
			header = fin.readline()
			header_cols = header.split()
			num_inds = int(header_cols[0])
			num_snps = int(header_cols[1])
			
			for line in fin:
				line = line.strip()
				if not line: # If blank line.
					continue
				cols = line.split()
				inds = cols[0]
				seqs = cols[1]
				snps = [snp for snp in seqs] # Split each site.

				# Error handling if incorrect sequence length
				if len(snps) != num_snps:
					print(len(snps))
					sys.exit("\nError: All sequences must be the same length; at least one sequence differes from the header line\n")

				self.snps.append(snps)

				self.samples.append(inds)

		# Error hanlding if incorrect number of individuals in header.
		if len(self.samples) != num_inds:
			sys.exit("\nError: Incorrect number of individuals are listed in the header\n")
	
	def convert_df(self):
		self.df = self.df2allelecounts(pd.DataFrame.from_records(self.snps))
	
	def df2allelecounts(self, df):

		df2 = pd.DataFrame()
		homozygotes = ["A", "T", "G", "C"]
		missing_vals = ["N", "-"]
		heterozygotes = ["W", "R", "M", "K", "Y", "S"]

		for i, col in enumerate(df):
			all_counts = self.get_value_counts(df[col])
			uniq_bases = all_counts["index"].to_list()

			if len(uniq_bases) <= 2:
				if uniq_bases[0] in homozygotes:
					ref = uniq_bases[0]
				elif uniq_bases[0] in heterozygotes:
					df.drop(col, axis=1, inplace=True)
					print("\nWarning: Removing site {} from the PCA input because its only reference allele is heterozygous".format(i+1))
					continue
				elif uniq_bases[0] in missing_vals:
					df.drop(col, axis=1, inplace=True)
					print("\nWarning: Removing site {} from the PCA input because it only contains one allele that isn't missing data".format(i+1))
					continue

				if uniq_bases[1] in heterozygotes:
					df.drop(col, axis=1, inplace=True)
					print("\nWarning: Removing site {} from the PCA input because its only alternate allele is heterozygous".format(i+1))
					continue
				
				alt = uniq_bases[1]

				df[col].replace([ref], "0", inplace=True)
				df[col].replace([alt], "2", inplace=True)

			if len(uniq_bases) > 2:
				
				homoz_matches = [key for key, val in enumerate(uniq_bases) if val in set(homozygotes)]
				
				if len(homoz_matches) > 2:
					df.drop(col, axis=1, inplace=True)
					print("Warning: Site {} was not bi-allelic so it was removed.".format(i+1))
					continue

				elif len(homoz_matches) < 2:
					df.drop(col, axis=1, inplace=True)
					print("Warning: Site {} did not have two non-ambiguous alleles so it was removed.".format(i+1))
					continue

				ref = uniq_bases[homoz_matches[0]]
				alt = uniq_bases[homoz_matches[1]]

				df[col].replace([ref], "0", inplace=True)
				df[col].replace([alt], "2", inplace=True)
				
				heteroz_matches = [uniq_bases.index(x) for x in heterozygotes]
				miss_matches = [uniq_bases.index(x) for x in missing_vals]

				if heteroz_matches:
					df[col].replace([heteroz_matches], "1", inplace=True)

				if miss_matches:
					df[col].replace([miss_matches], np.nan, inplace=True)

	def get_value_counts(self, df):
		return df.value_counts().reset_index()
	
	def read_popmap(self, popmapfile):
		self.popmapfile = popmapfile
		# Join popmap file with main object.
		if len(self.samples) < 1:
			sys.exit("\nError: No samples in GenotypeData\n")
		
		#instantiate popmap object
		my_popmap = ReadPopmap(popmapfile)
		
		popmapOK = my_popmap.validate_popmap(self.samples)
		
		if not popmapOK:
			sys.exit("\nError: Not all samples are present in supplied popmap file:", my_popmap.filename,"\n")
		
		if len(my_popmap) != len(self.samples):
			sys.exit("\nError: The number of individuals in the popmap file differs from the number of sequences\n")

		for sample in self.samples:
			if sample in my_popmap:
				self.pops.append(my_popmap[sample])

	def convert_onehot(self):
		"""[Adds onehot encoded dict(list)]

		Args:
			mydict ([dict(list)]): [Object storing the phylip data]

		Returns:
			[dict(list)]: [Adds onehot encoding dict(list) to mydict object]
		"""
		if self.filetype == "phylip":
			self.onehot = phylip2onehot(self.samples, self.snps)

def phylip2onehot(samples, snps):
	onehot_dict = {
		"A": [1.0, 0.0, 0.0, 0.0], 
		"T": [0.0, 1.0, 0.0, 0.0],
		"G": [0.0, 0.0, 1.0, 0.0], 
		"C": [0.0, 0.0, 0.0, 1.0],
		"N": [0.0, 0.0, 0.0, 0.0],
		"W": [0.5, 0.5, 0.0, 0.0],
		"R": [0.5, 0.0, 0.5, 0.0],
		"M": [0.5, 0.0, 0.0, 0.5],
		"K": [0.0, 0.5, 0.5, 0.0],
		"Y": [0.0, 0.5, 0.0, 0.5],
		"S": [0.0, 0.0, 0.5, 0.5],
		"-": [0.0, 0.0, 0.0, 0.0]
	}

	onehot_outer_list = list()
	for i in range(len(samples)):
		onehot_list = list()

		for j in range(len(snps[0])):
			onehot_list.append(onehot_dict[snps[i][j]])
		onehot_outer_list.append(onehot_list)

	onehot = np.array(onehot_outer_list)
	
	return(onehot)