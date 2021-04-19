import sys

import numpy as np
import pandas as pd

from popmap_file import ReadPopmap

class PhylipFile:
	"""[Class to read phylip file and convert to onehot encoding]
	"""
	def __init__(self, filename):
		self.filename = filename

	def read_phylip(self, popmap_filename):
		"""[Read phylip file from disk]

		Args:
			popmap_filename [str]: [Filename for population map file]

		Returns:
			[dict(list)]: [dictionary object with 'samples', 'snps', and 'popids' keys]
		"""
		mydict = dict()
		samples_list = list()
		snps_list = list()
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

				snps_list.append(snps)

				samples_list.append(inds)
		mydict["samples"] = samples_list
		mydict["snps"] = snps_list

		# Error hanlding if incorrect number of individuals in header.
		if len(samples_list) != num_inds:
			sys.exit("\nError: Incorrect number of individuals are listed in the header\n")

		# Join popmap file with main object.
		my_popmap = ReadPopmap(popmap_filename)
		pops = my_popmap.read_popmap()
		mydict = my_popmap.join_popmap_with_data(mydict, pops)

		return mydict

	def phylip2onehot(self, mydict):
		"""[Adds onehot encoded dict(list) to mydict object]

		Args:
			mydict ([dict(list)]): [Object storing the phylip data]

		Returns:
			[dict(list)]: [Adds onehot encoding dict(list) to mydict object]
		"""
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
		for i in range(len(mydict["samples"])):
			onehot_list = list()

			for j in range(len(mydict["snps"][0])):
				onehot_list.append(onehot_dict[mydict["snps"][i][j]])
			onehot_outer_list.append(onehot_list)

		mydict["onehot"] = onehot_outer_list
		
		# Convert to 3D numpy array
		mydict["onehot"] = np.array(mydict["onehot"])

		return mydict

	def phylip2df(self, snp_data):

		df = pd.DataFrame.from_records(snp_data)
		#df = df.T
		#print(df)
		df = self.df2allelecounts(df)

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
					
		print(df)

	def get_value_counts(self, df):
		return df.value_counts().reset_index()
