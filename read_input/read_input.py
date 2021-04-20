import sys

import numpy as np
import pandas as pd

from read_input.popmap_file import ReadPopmap
from utils import sequence_tools

class GenotypeData:
	"""[Class to read genotype data and convert to onehot encoding]
	
	Possible filetype values:
		- phylip
		- structure1row
		- structure2row 
		- vcf (TBD)
		- Any others?
	
	Note: If popmapfile is supplied, read_structure assumes NO popID column
	"""
	def __init__(self, filename=None, filetype=None, popmapfile=None):
		self.filename = filename
		self.filetype = filetype
		self.popmapfile = popmapfile
		self.samples = list()
		self.snps = list()
		self.pops = list()
		self.onehot = list()
		self.df = None
		
		if self.filetype is not None:
			self.parse_filetype(filetype, popmapfile)
		
		if self.popmapfile is not None:
			self.read_popmap(popmapfile)
	
	def parse_filetype(self, filetype=None, popmapfile=None):
		if filetype is None:
			sys.exit("\nError: No filetype specified.\n")
		else:
			if filetype == "phylip":
				self.filetype = filetype
				self.read_phylip()
			elif filetype == "structure1row":
				if popmapfile is not None:
					self.filetype = "structure1row"
					self.read_structure(onerow=True, popids=False)
				else:
					self.filetype = "structure1rowPopID"
					self.read_structure(onerow=True, popids=True)
			elif filetype == "structure2row":
				if popmapfile is not None:
					self.filetype = "structure2row"
					self.read_structure(onerow=False, popids=False)
				else:
					self.filetype = "structure2rowPopID"
					self.read_structure(onerow=False, popids=True)
			else:
				sys.exit("\nError: Filetype",filetype,"not implemented!\n")

	def check_filetype(self, filetype):
		if self.filetype is None:
			self.filetype = filetype
		elif self.filetype == filetype:
			pass
		else:
			sys.exit("\nError: GenotypeData read_XX() call does not match filetype!\n")

	def read_structure(self, onerow=False, popids=True):
		"""[Read a structure file with two row per individual]

		"""
		print("\nReading structure file {}...".format(self.filename))

		with open(self.filename, "r") as fin:
			snp_data=list()
			if not onerow:
				firstline=None
				for line in fin:
					line.strip()
					if not line:
						continue
					if not firstline:
						firstline=line.split()
						continue
					else:
						secondline=line.split()
						if firstline[0] != secondline[0]:
							sys.exit("\nError: Sample names do not match:",str(firstline[0]),str(secondline[0]),"\n")
						ind=firstline[0]
						pop=None
						if popids:
							if firstline[1] != secondline[1]:
								sys.exit("\nError: Population IDs do not match:",str(firstline[1]),str(secondline[1]),"\n")
							pop=firstline[1]
							self.pops.append(pop)
							firstline=firstline[2:]
							secondline=secondline[2:]
						else:
							firstline=firstline[1:]
							secondline=secondline[1:]
						self.samples.append(ind)
						genotypes = merge_alleles(firstline, secondline)
						snp_data.append(genotypes)
						firstline=None
			else:
				for line in fin:
					line.strip()
					if not line:
						continue
					if not firstline:
						firstline=line.split()
						continue
					else:
						ind=firstline[0]
						pop=None
						if popids:
							pop=firstline[1]
							self.pops.append(pop)
							firstline=firstline[2:]
						else:
							firstline=firstline[1:]
						self.samples.append(ind)
						genotypes = merge_alleles(firstline, second=None)
						snp_data.append(genotypes)
						firstline=None
			#convert snp_data to 012 format
			self.convert_012(snp_data, vcf=True)
		fin.close()
		
		num_snps = len(self.snps[0])
		print("\nFound {} SNPs and {} individuals...\n".format(num_snps, len(self.samples)))

		# Make sure all sequences are the same length.
		for item in self.snps:
			try:
				assert len(item) == num_snps
			except AssertionError:
				sys.exit("\nError: There are sequences of different lengths in the structure file\n")


	def read_phylip(self):
		"""[Read phylip file from disk]

		Args:
			popmap_filename [str]: [Filename for population map file]

		Populates ReadInput object by parsing Phylip
		"""
		self.check_filetype("phylip")
		snp_data=list()
		with open(self.filename, "r") as fin:
			num_inds = 0
			num_snps = 0
			first=True
			for line in fin:
				line = line.strip()
				if not line: # If blank line.
					continue
				if first:
					first=False
					header=line.split()
					num_inds=int(header[0])
					num_snps=int(header[1])
					continue
				cols = line.split()
				inds = cols[0]
				seqs = cols[1]
				snps = [snp for snp in seqs] # Split each site.

				# Error handling if incorrect sequence length
				if len(snps) != num_snps:
					#print(len(snps))
					#print(num_snps)
					sys.exit("\nError: All sequences must be the same length; at least one sequence differes from the header line\n")

				snp_data.append(snps)
				
				self.samples.append(inds)
			
			#convert snp_data to 012 format
			self.convert_012(snp_data)
			
		fin.close()

		# Error hanlding if incorrect number of individuals in header.
		if len(self.samples) != num_inds:
			sys.exit("\nError: Incorrect number of individuals are listed in the header\n")
	
	def convert_df(self):
		self.df = self.df2allelecounts(pd.DataFrame.from_records(self.snps))
	
	def convert_012(self, snps, vcf=False):
		skip=0
		new_snps=list()
		for i in range(0, len(self.samples)):
			new_snps.append([])
		for j in range(0, len(snps[0])):
			#print(i)
			loc=list()
			for i in range(0, len(self.samples)):
				if vcf:
					loc.append(snps[i][j])
				else:
					loc.append(snps[i][j].upper())
			#**NOTE**: Here we could switch to !=2 to also remove monomorphic sites?
			if sequence_tools.count_alleles(loc, vcf=vcf) > 2:
				skip+=1
				continue
			else:
				ref, alt = sequence_tools.get_major_allele(loc, 2, vcf=vcf)
				ref=str(ref)
				alt=str(alt)
				if vcf:
					for i in range(0, len(self.samples)):
						gen=snps[i][j].split("/")
						if gen[0] in ["-", "-9", "N"] or gen[1] in ["-", "-9", "N"]:
							new_snps[i].append(-9)
						elif gen[0] == gen[1] and gen[0] == ref:
							new_snps[i].append(0)
						elif gen[0] == gen[1] and gen[0] == alt:
							new_snps[i].append(2)
						else:
							new_snps[i].append(1)
				else:
					for s in snps[i]:
						if loc[i] == ref:
							new_snps[i].append(0)
						elif loc[i] == alt:
							new_snps[i].append(2)
						elif loc[i] in ["-", "-9", "N"]:
							new_snps[i].append(-9)
						else:
							new_snps[i].append(1)
		if skip > 0:
			print("\nWarning: Skipping",str(skip),"non-biallelic sites\n")
		for s in new_snps:
			self.snps.append(s)
	
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

#outputs VCF-style genotypes (i.e. split with "/")
def merge_alleles(first, second=None):
	ret=list()
	if second is not None:
		if len(first) != len(second):
			sys.exit("\nError: First and second lines have different number of alleles\n")
		else:
			for i in range(0, len(first)):
				ret.append(str(first[i])+"/"+str(second[i]))
	else:
		if len(first) % 2 != 0:
			sys.exit("\nError: Line has non-even number of alleles!\n")
		else:
			for i, j in zip(first[::2], first[1::2]):
				ret.append(str(i)+"/"+str(j))
	return(ret)

def count2onehot(samples, snps):
	onehot_dict = {
		"0": [1.0, 0.0], 
		"1": [0.5, 0.5],
		"2": [0.0, 1.0],
		"-": [0.0, 0.0]
	}
	onehot_outer_list = list()
	for i in range(len(samples)):
		onehot_list = list()
		for j in range(len(snps[0])):
			onehot_list.append(onehot_dict[snps[i][j]])
		onehot_outer_list.append(onehot_list)
	onehot = np.array(onehot_outer_list)
	return(onehot)

def seq2onehot(samples, snps):
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