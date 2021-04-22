import sys

import numpy as np
import pandas as pd

from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler

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
	def __init__(self, filename=None, filetype=None, popmapfile=None, impute_methods=None, impute_settings=None):
		self.filename = filename
		self.filetype = filetype
		self.impute_methods = impute_methods
		self.impute_settings = impute_settings
		self.popmapfile = popmapfile
		self.samples = list()
		self.snps = list()
		self.pops = list()
		self.onehot = list()
		self.imputed = list()
		self.df = None
		self.num_snps = 0
		self.num_inds = 0
		self.knn_settings = {"n_neighbors": 5, 
							"weights": "distance", 
							"metric": "nan_euclidean"}
		
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
				sys.exit("\nError: Filetype {} is not supported!\n".format(filetype))

	def check_filetype(self, filetype):
		if self.filetype is None:
			self.filetype = filetype
		elif self.filetype == filetype:
			pass
		else:
			sys.exit("\nError: GenotypeData read_XX() call does not match filetype!\n")

	def read_structure(self, onerow=False, popids=True):
		"""[Read a structure file with two rows per individual]

		"""
		print("\nReading structure file {}...".format(self.filename))

		snp_data=list()
		with open(self.filename, "r") as fin:
			if not onerow:
				firstline=None
				for line in fin:
					line=line.strip()
					if not line:
						continue
					if not firstline:
						firstline=line.split()
						continue
					else:
						secondline=line.split()
						if firstline[0] != secondline[0]:
							sys.exit("\nError: Two rows per individual was specified but sample names do not match: {} and {}\n".format(str(firstline[0]), str(secondline[0])))
						ind=firstline[0]
						pop=None
						if popids:
							if firstline[1] != secondline[1]:
								sys.exit("\nError: Two rows per individual was specified but population IDs do not match: {} {}\n".format(str(firstline[1]), str(secondline[1])))
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
			else: # If onerow:
				for line in fin:
					line=line.strip()
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
		print("Done!")

		print("\nConverting genotypes to one-hot encoding...")
		# Convert snp_data to onehot encoding format
		self.convert_onehot(snp_data)

		print("Done!")

		print("\nConverting genotypes to 012 format...")
		# Convert snp_data to 012 format
		self.convert_012(snp_data, vcf=True)

		print("Done!")
		
		# Get number of samples and snps
		self.num_snps = len(self.snps[0])
		self.num_inds = len(self.samples)

		print("\nFound {} SNPs and {} individuals...\n".format(self.num_snps, self.num_inds))

		# Make sure all sequences are the same length.
		for item in self.snps:
			try:
				assert len(item) == self.num_snps
			except AssertionError:
				sys.exit("\nError: There are sequences of different lengths in the structure file\n")

		if self.impute_methods:
			self.impute_missing()

	def read_phylip(self):
		"""[Populates ReadInput object by parsing Phylip]

		Args:
			popmap_filename [str]: [Filename for population map file]
		"""
		print("\nReading phylip file {}...".format(self.filename))

		self.check_filetype("phylip")
		snp_data=list()
		with open(self.filename, "r") as fin:
			num_inds = 0
			num_snps = 0
			first=True
			for line in fin:
				line=line.strip()
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
					sys.exit("\nError: All sequences must be the same length; at least one sequence differs from the header line\n")

				snp_data.append(snps)
				
				self.samples.append(inds)

		print("Done!")

		print("\nConverting genotypes to one-hot encoding...")
		# Convert snp_data to onehot format
		self.convert_onehot(snp_data)
		print("Done!")

		print("\nConverting genotypes to 012 encoding...")
		# Convert snp_data to 012 format
		self.convert_012(snp_data)
		print("Done!")
		
		self.num_snps = num_snps
		self.num_inds = num_inds
			
		# Error hanlding if incorrect number of individuals in header.
		if len(self.samples) != num_inds:
			sys.exit("\nError: Incorrect number of individuals are listed in the header\n")

		if self.impute_methods:
			self.impute_missing()
	
	def convert_012(self, snps, vcf=False):
		skip=0
		new_snps=list()
		for i in range(0, len(self.samples)):
			new_snps.append([])
		for j in range(0, len(snps[0])):
			loc=list()
			for i in range(0, len(self.samples)):
				if vcf:
					loc.append(snps[i][j])
				else:
					loc.append(snps[i][j].upper())
			#**NOTE**: Here we could switch to !=2 to also remove monomorphic sites? 
			# 
			# **NOTE**I agree. Monomorphic sites might violate assumptions.
			if sequence_tools.count_alleles(loc, vcf=vcf) != 2:
				skip+=1
				continue
			else:
				ref, alt = sequence_tools.get_major_allele(loc, vcf=vcf)
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
					for i in range(0, len(self.samples)):
						if loc[i] in ["-", "-9", "N"]:
							new_snps[i].append(-9)
						elif loc[i] == ref:
							new_snps[i].append(0)
						elif loc[i] == alt:
							new_snps[i].append(2)
						else:
							new_snps[i].append(1)
		if skip > 0:
			print("\nWarning: Skipping",str(skip),"non-biallelic sites\n")
		for s in new_snps:
			self.snps.append(s)

	def convert_onehot(self, snp_data):

		if self.filetype == "phylip":
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

		elif self.filetype == "structure1row" or self.filetype == "structure2row":
			onehot_dict = {
				"1/1": [1.0, 0.0, 0.0, 0.0], 
				"2/2": [0.0, 1.0, 0.0, 0.0],
				"3/3": [0.0, 0.0, 1.0, 0.0], 
				"4/4": [0.0, 0.0, 0.0, 1.0],
				"-9/-9": [0.0, 0.0, 0.0, 0.0],
				"1/2": [0.5, 0.5, 0.0, 0.0],
				"2/1": [0.5, 0.5, 0.0, 0.0],
				"1/3": [0.5, 0.0, 0.5, 0.0],
				"3/1": [0.5, 0.0, 0.5, 0.0],
				"1/4": [0.5, 0.0, 0.0, 0.5],
				"4/1": [0.5, 0.0, 0.0, 0.5],
				"2/3": [0.0, 0.5, 0.5, 0.0],
				"3/2": [0.0, 0.5, 0.5, 0.0],
				"2/4": [0.0, 0.5, 0.0, 0.5],
				"4/2": [0.0, 0.5, 0.0, 0.5],
				"3/4": [0.0, 0.0, 0.5, 0.5],
				"4/3": [0.0, 0.0, 0.5, 0.5]
			}

		onehot_outer_list = list()
		for i in range(len(self.samples)):
			onehot_list = list()
			for j in range(len(snp_data[0])):
				onehot_list.append(onehot_dict[snp_data[i][j]])
			onehot_outer_list.append(onehot_list)
		self.onehot = np.array(onehot_outer_list)
		
	def read_popmap(self, popmapfile):
		self.popmapfile = popmapfile
		# Join popmap file with main object.
		if len(self.samples) < 1:
			sys.exit("\nError: No samples in GenotypeData\n")
		
		# Instantiate popmap object
		my_popmap = ReadPopmap(popmapfile)
		
		popmapOK = my_popmap.validate_popmap(self.samples)
		
		if not popmapOK:
			sys.exit("\nError: Not all samples are present in supplied popmap file:", my_popmap.filename,"\n")
		
		if len(my_popmap) != len(self.samples):
			sys.exit("\nError: The number of individuals in the popmap file differs from the number of sequences\n")

		for sample in self.samples:
			if sample in my_popmap:
				self.pops.append(my_popmap[sample])

	def impute_missing(self):

		if isinstance(self.impute_methods, str):
			if len(self.impute_methods.split(",")) > 1:
				raise TypeError("\nThe method argument must be a list if more than one arguments are specified!")
			if self.impute_methods not in ["knn"]:
				raise ValueError("\nThe value supplied to the method argument is not supported!")
			if self.impute_methods == "knn":
				self.df = self.impute_knn()

		elif isinstance(self.impute_methods, list):
			if self.impute_methods not in ["knn"]:
				raise ValueError("\nThe value supplied to the method argument is not supported!")
				for arg in self.impute_methods:
					if arg == "knn":
						self.df = self.impute_knn()
		else:
			raise ValueError("The methods argument must be either a string or a list!")

		print(self.df)

	def impute_knn(self):

		print("\nDoing K-NN imputation...")

		if self.impute_settings:
			self.knn_settings.update(self.impute_settings)

		# Make sure knn_settings are supported by KNNImputer
		for arg in self.knn_settings.keys():
			if arg not in ["n_neighbors", "weights", "metric"]:
				raise ValueError("The impute_settings argument {} is not supported".format(arg))
		
		df = pd.DataFrame.from_records(self.snps)
		df.replace(-9, np.nan, inplace=True)

		scaler = MinMaxScaler()
		df = pd.DataFrame(scaler.fit_transform(df), columns = df.columns)

		imputer = KNNImputer(n_neighbors=self.knn_settings["n_neighbors"], weights=self.knn_settings["weights"], metric=self.knn_settings["metric"])

		df = pd.DataFrame(imputer.fit_transform(df), columns = df.columns)

		# If there are any remaining missing data raise error.
		if any(df.isna().any().to_list()) == True:
			raise AssertionError("\nThere was a problem with the K-NN imputation. Please inspect your data and try again.")
		
		print("Done!\n")

		return df
		

	@property
	def snpcount(self):
		"""[Getter for number of snps in the dataset]

		Returns:
			[int]: [Number of SNPs per individual]
		"""
		return self.num_snps
	
	@property
	def indcount(self):
		"""[Getter for number of individuals in dataset]

		Returns:
			[int]: [Number of individuals in input sequence data]
		"""
		return self.num_inds

	@property
	def populations(self):
		"""[Getter for population IDs]

		Returns:
			[list]: [Poulation IDs as a list]
		"""
		return self.pops

	@property
	def individuals(self):
		"""[Getter for sample IDs in input order]

		Returns:
			[list]: [sample IDs as a list in input order]
		"""
		return self.samples

	@property
	def genotypes_list(self):
		"""[Getter for the 012 genotypes]

		Returns:
			[list(list)]: [012 genotypes as a 2d list]
		"""
		return self.snps

	@property
	def genotypes_nparray(self):
		"""[Returns 012 genotypes as a numpy array]

		Returns:
			[2D numpy.array]: [012 genotypes as shape (n_samples, n_variants)]
		"""
		return np.array(self.snps)

	@property
	def gentotypes_nparray(self):
		"""[Returns 012 genotypes as a numpy array]

		Returns:
			[2D numpy.array]: [012 genotypes as shape (n_samples, n_variants)]
		"""
		return np.array(self.snps)

	@property
	def genotypes_onehot(self):
		"""[Returns one-hot encoded snps]

		Returns:
			[2D numpy.array]: [One-hot encoded numpy array (n_samples, n_variants)]
		"""
		return self.onehot

def merge_alleles(first, second=None):
	"""[Merges first and second alleles in structure file]

	Args:
		first ([list]): [Alleles on one line]
		second ([list], optional): [Second row of alleles]. Defaults to None.

	Returns:
		[list(str)]: [VCF-style genotypes (i.e. split by "/")]
	"""
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
	return ret

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
