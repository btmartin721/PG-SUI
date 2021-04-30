import sys

import numpy as np
import pandas as pd

from read_input.popmap_file import ReadPopmap
from read_input import impute
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
		self.freq_imputed_global = list()
		self.freq_imputed_pop = list()
		self.knn_imputed = list()
		self.impute_methods = None
		self.df = None
		self.knn_imputed_df = None
		self.rf_imputed_arr = None
		self.gb_imputed_arr = None
		self.br_imputed_arr = None
		self.knn_iter_imputed_arr = None
		self.num_snps = 0
		self.num_inds = 0
		self.supported_methods = [
									"knn", 
									"freq_global", 
									"freq_pop", 
									"rf", 
									"gb", 
									"br",
									"knn_iter"
								]

		
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

	def impute_missing(self, impute_methods=None, impute_settings=None, pops=None, maxk=None, np=1):

		self.impute_methods = impute_methods

		supported_settings = ["n_neighbors", 
								"weights", 
								"metric", 
								"rf_n_estimators",
								"rf_min_samples_leaf",
								"rf_max_features",
								"rf_n_jobs",
								"rf_criterion",
								"rf_random_state", 
								"max_iter", 
								"tol", 
								"n_nearest_features", 
								"initial_strategy", 
								"imputation_order", 
								"skip_complete", 
								"random_state",
								"verbose",
								"gb_n_estimators",
								"gb_min_samples_leaf",
								"gb_max_features",
								"gb_criterion",
								"gb_learning_rate",
								"gb_subsample",
								"gb_loss",
								"gb_min_samples_split",
								"gb_max_depth",
								"gb_random_state",
								"gb_verbose",
								"gb_validation_fraction",
								"gb_n_iter_no_change",
								"gb_tol",
								"br_n_iter",
								"br_tol",
								"br_alpha_1",
								"br_alpha_2",
								"br_lambda_1",
								"br_lambda_2",
								"br_verbose",
								"br_alpha_init",
								"br_lambda_init",
								"br_sample_posterior",
								"knn_it_n_neighbors",
								"knn_it_weights",
								"knn_it_algorithm",
								"knn_it_leaf_size",
								"knn_it_power",
								"knn_it_metric",
								"knn_it_metric_params",
								"knn_it_n_jobs"
							]
							
		supported_settings_opt = ["weights", "metric", "reps"]


		if maxk:
			knn_settings = {"weights": "uniform", 
							"metric": "nan_euclidean", 
							"reps": 1}
		else:
			knn_settings = {"n_neighbors": 5,
							"weights": "uniform", 
							"metric": "nan_euclidean"}

		knn_iterative_settings = {
							"knn_it_n_neighbors": 5,
							"knn_it_weights": "uniform"
							"knn_it_algorithm": "auto",
							"knn_it_leaf_size": 30,
							"knn_it_power": 2
							"knn_it_metric": "minkowski",
							"knn_it_metric_params": None,
							"knn_it_n_jobs": 1
		}

		rf_settings = {
							"rf_n_estimators": 100,
							"rf_min_samples_leaf": 1,
							"rf_max_features": "auto",
							"rf_n_jobs": 1,
							"rf_criterion": "gini",
							"rf_random_state": None,
							"max_iter": 10,
							"tol": 1e-3,
							"n_nearest_features": None,
							"initial_strategy": "most_frequent",
							"imputation_order": "ascending",
							"skip_complete": False,
							"random_state": None,
							"verbose": 0
					}

		gb_settings = {
							"gb_n_estimators": 100,
							"gb_min_samples_leaf": 1,
							"gb_max_features": "auto",
							"gb_criterion": "friedman_mse",
							"gb_learning_rate": 0.1,
							"gb_subsample": 1.0,
							"gb_loss": "deviance",
							"gb_min_samples_split": 2,
							"gb_max_depth": 3,
							"gb_random_state": None,
							"gb_verbose": 0,
							"gb_validation_fraction": 0.1,
							"gb_n_iter_no_change": None,
							"gb_tol": 1e-4,
							"max_iter": 10,
							"tol": 1e-3,
							"n_nearest_features": None,
							"initial_strategy": "most_frequent",
							"imputation_order": "ascending",
							"skip_complete": False,
							"verbose": 0,
							"random_state": None
					}

		br_settings = {
							"br_n_iter": 300,
							"br_tol": 1e-3,
							"br_alpha_1": 1e-6,
							"br_alpha_2": 1e-6,
							"br_lambda_1": 1e-6,
							"br_lambda_2": 1e-6,
							"br_verbose": False,
							"br_alpha_init": None,
							"br_lambda_init": None,
							"br_sample_posterior": True
					}

		# bayesian ridge has a different verbose setting.
		# Make sure user didn't specify the wrong type
		if "br_verbose" in impute_settings.keys()
			if impute_settings["br_verbose"] == 0:
				impute_settings["br_verbose"] = False

			elif impute_settings["br_verbose"] == 1 or impute_settings["br_verbose"] == 2:
				impute_settings["br_verbose"] = True

		# Update settings if non-default ones were specified
		if impute_settings:
			knn_settings.update(impute_settings)
			rf_settings.update(impute_settings)
			gb_settings.update(impute_settings)
			br_settings.update(impute_settings)
			knn_iterative_settings(impute_settings)
		
		# Validate impute settings
		for method in self.impute_methods:
			if maxk and method == "knn":
				self._check_impute_settings(method, knn_settings, supported_settings_opt, opt=True)

			elif not maxk and method == "knn":
				self._check_impute_settings(method, knn_settings, supported_settings)

			elif method == "rf":
				self._check_impute_settings(method, rf_settings, supported_settings)

			elif method == "gb":
				self._check_impute_settings(method, gb_settings, supported_settings)

			elif method == "br":
				self._check_impute_settings(method, br_settings, supported_settings)

			elif method == "knn_iter":
				self._check_impute_settings(method, knn_iterative_settings, supported_settings)

		# If one string value is supplied to impute_methods
		if isinstance(self.impute_methods, str):

			# There can be only one
			if len(self.impute_methods.split(",")) > 1: 
				raise TypeError("\nThe method argument must be a list if more than one arguments are specified!")

			# Must be a supported method
			if self.impute_methods not in self.supported_methods:
				raise ValueError("\nThe value supplied to the method argument is not supported!")
			
			# K-NN imputation with K optimization
			if self.impute_methods == "knn" and maxk:
				optimalk_list = list()
				acc_list = list()
				print("\nDoing K-NN imputation with K optimization")
				for i in range(knn_settings["reps"]):

					# For printing progress updates to screen
					rep = i+1
					if rep % 5 == 0:
						print(rep, end="", flush=True)
					else:
						print(".", end="", flush=True)

					# Run imputations
					optk, acc = impute.impute_knn_optk(self.snps, self.pops, knn_settings, maxk, np=np)

					# Save the output into lists
					optimalk_list.append(optk)
					acc_perc = 100 * acc
					acc_list.append(acc_perc)
				
				# Get most frequent optimal K from replicates
				optimalk, idx, kcount, nreps = impute.most_common(optimalk_list)
				perc_k = (kcount / nreps) * 100

				print("\nDone!\nThe best accuracy had n_neighbors = {} with a prediction accuracy of {:.2f}%!\nOptimal n_neighbors was found {:.2f}% of the time\n".format(optimalk, acc_list[idx], perc_k))

				# Now run final knn with optimal K
				knn_settings["n_neighbors"] = int(optimalk)
				self.knn_imputed_df = impute.impute_knn(self.snps, knn_settings)

			# Run with no k-optmimization
			elif self.impute_methods == "knn" and not maxk:
				self.knn_imputed_df = impute.impute_knn(self.snps, knn_settings)

			# Run imputation my global mode
			if self.impute_methods == "freq_global":
				self.freq_imputed_global = impute.impute_freq(self.snps)

			# Run imputation for by-population mode
			if self.impute_methods == "freq_pop":
				self.freq_imputed_pop = impute.impute_freq(self.snps, pops=self.pops)

			if self.impute_methods == "rf":
				self.rf_imputed_arr = impute.rf_imputer(self.snps, rf_settings)

			if self.impute_methods == "gb":
				self.gb_imputed_arr = impute.gb_imputer(self.snps, gb_settings)

			if self.impute_methods == "br":
				self.br_imputed_arr = impute.bayesianridge_imputer(self.snps, br_settings)

			if self.impute_methods == "knn_iter":
				self.knn_iter_imputed_arr = impute.knn_iterative_imputer(self.snps, knn_iterative_settings)
			

		# If value supplied to impute_methods is a list
		elif isinstance(self.impute_methods, list):
			
			for arg in self.impute_methods:
				# Make sure impute_methods are supported
				if arg not in self.supported_methods:
					raise ValueError("\nThe value supplied to the impute_methods argument is not supported!")

			# For each imputation method
			for arg in self.impute_methods:
				if arg == "knn" and maxk:

					optimalk_list = list()
					acc_list = list()
					print("\nDoing K-NN imputation with K optimization")

					# Run replicates of KNN optimization
					for i in range(knn_settings["reps"]):

						# For printing progress updates
						rep = i+1
						if rep % 5 == 0:
							print(rep, end="", flush=True)
						else:
							print(".", end="", flush=True)

						# Run KNN imputation with K-optimization
						optk, acc = impute.impute_knn_optk(self.snps, self.pops, knn_settings, maxk, np=np)

						# Save output to list
						optimalk_list.append(optk)
						acc_perc = 100 * acc
						acc_list.append(acc_perc)

					# Get most commonly found K among all replicates
					optimalk, idx, kcount, nreps = impute.most_common(optimalk_list)

					perc_k = (kcount / nreps) * 100

					print("\nDone!\nThe best accuracy had n_neighbors = {} with a prediction accuracy of {:.2f}%!\nOptimal n_neighbors was found {:.2f}% of the time\n".format(optimalk, acc_list[idx], perc_k))

					# Run final knn imputation with optimal k
					knn_settings["n_neighbors"] = int(optimalk)
					self.knn_imputed_df = impute.impute_knn(self.snps, knn_settings)

				# If not doing k optimization
				elif arg == "knn" and not maxk:
					impupted_df = impute.impute_knn(self.snps, knn_settings)
					self.knn_imputed_df = impute.impute_knn(self.snps, knn_settings)

				# Run imputation my global mode
				elif arg == "freq_global":
					self.freq_imputed_global = impute.impute_freq(self.snps)

				# Run imputation for by-population mode
				elif arg == "freq_pop":
					self.freq_imputed_pop = impute.impute_freq(self.snps, pops=self.pops)

				elif arg == "rf":
					self.rf_imputed_arr = impute.rf_imputer(self.snps, rf_settings)

				elif arg == "gb":
					self.gb_imputed_arr = impute.gb_imputer(self.snps, gb_settings)

				elif arg == "br":
					self.br_imputed_arr = impute.br_imputer(self.snps, br_settings)

				elif arg == "knn_iter":
					self.knn_iter_imputed_arr = impute.knn_iterative_imputer(self.snps, knn_iterative_settings)

		# impute_methods must be either string or list			
		else:
			raise ValueError("The impute_methods argument must be either a string or a list!")
		
	def _check_impute_settings(self, method, settings, supported_settings, opt=False):
		"""[Validate that impute_settings are supported arguments]

		Args:
			method ([str]): [Imputation method]
			settings ([dict]): [Settings for imputation method]
			supported_settings ([dict]): [Settings supported by any of the imputation methods]
			opt (bool, optional): [True if doing K-NN with K optimization]. Defaults to False.

		Raises:
			ValueError: [One or more impute_settings arguments was not found in supported_settings]
			ValueError: [Only one of maxk or n_neigbors settings can be used at a time]
		"""

		# Make sure settings are supported by imputer
		for arg in settings.keys():
			if arg not in supported_settings:
				raise ValueError("The impute_settings argument {} is not supported".format(arg))

			if opt and arg == "n_neighbors":
				raise ValueError("maxk and n_neighbors cannot both be specified!")
	
	def write_imputed(self, data, prefix):
		"""[Save imputed data to a CSV file]

		Args:
			data ([pandas.DataFrame, numpy.array, or list(list)]): [Object returned from impute_missing()]
			prefix ([str]): [Prefix for output CSV file]

		Raises:
			TypeError: [Must be or type pandas.DataFrame, numpy.array, or list]
		"""
		outfile = "{}_imputed_012.csv".format(prefix)
		if isinstance(data, pd.DataFrame):
			data.to_csv(outfile, header = False, index = False)

		elif isinstance(data, np.ndarray):
			np.savetxt(outfile, data, delimiter=",")

		elif isinstance(data, list):
			with open(outfile, "w") as fout:
				fout.writelines(",".join(str(j) for j in i) + "\n" for i in data)
		else:
			raise TypeError("write_imputed takes either a pandas.DataFrame, numpy.ndarray, or 2-dimensional list")

	def read_imputed(self, filename, impute_methods):
		"""[Read in imputed CSV file as formatted by write_imputed]

		Args:
			filename ([str]): [Name of imputed CSV file to be read]

		Returns:
			[pandas.DataFrame]: [Imputed data as DataFrame of 8-bit integers]

		Raises:
			ValueError: [Must be supported impute_method option]
		"""
		self.impute_methods = impute_methods
		if isinstance(self.impute_methods, list):
			for method in self.impute_methods:
				if method == "knn":
					self.knn_imputed_df = pd.read_csv(filename, dtype="Int8", header=None)
					self.knn_imputed = self.knn_imputed_df.values.tolist()

				elif method == "freq_global":
					self.freq_imputed_global_df = pd.read_csv(filename, dtype="Int8", header=None)
					self.freq_imputed_global = self.freq_imputed_global_df.values.tolist()

				elif method == "freq_pop":
					self.freq_imputed_pop_df = pd.read_csv(filename, dtype="Int8", header=None)
					self.freq_imputed_pop = self.freq_imputed_global_df.values.tolist()

				elif method == "rf":
					rf_df = pd.read_csv(filename, dtype="Int8", header=None)
					self.rf_imputed_arr = rf_df.to_numpy(dtype=np.int)

				elif method == "gb":
					gb_df = pd.read_csv(filename, dtype="Int8", header=None)
					self.gb_imputed_arr = gb_df.to_numpy(dtype=np.int)

				else:
					raise ValueError("\n{} is not a supported option in impute_methods!".format(method))

		elif isinstance(self.impute_methods, str):
			if self.impute_methods == "knn":
				self.knn_imputed_df = pd.read_csv(filename, dtype="Int8", header=None)
				self.knn_imputed = self.knn_imputed_df.values.tolist()

			elif self.impute_methods == "freq_global":
				self.freq_imputed_global_df = pd.read_csv(filename, dtype="Int8", header=None)
				self.freq_imputed_global = self.freq_imputed_global_df.values.tolist()

			elif self.impute_methods == "freq_pop":
				self.freq_imputed_pop_df = pd.read_csv(filename, dtype="Int8", header=None)
				self.freq_imputed_pop = self.freq_imputed_global_df.values.tolist()

			elif self.impute_methods == "rf":
				rf_df = pd.read_csv(filename, dtype="Int8", header=None)
				self.rf_imputed_arr = rf_df.to_numpy(dtype=np.int)

			elif self.impute_methods == "gb":
				gb_df = pd.read_csv(filename, dtype="Int8", header=None)
				self.gb_imputed_arr = gb_df.to_numpy(dtype=np.int)
				
			else:
				raise ValueError("\n{} is not a supported option in impute_methods!".format(self.impute_methods))

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
		"""[Getter for 012 genotypes as a numpy array]

		Returns:
			[2D numpy.array]: [012 genotypes as shape (n_samples, n_variants)]
		"""
		return np.array(self.snps)

	@property
	def genotypes_df(self):
		"""[Getter for 012 genotypes as a pandas DataFrame object]

		Returns:
			[pandas.DataFrame]: [012-encoded genotypes as pandas DataFrame]
		"""
		return pd.DataFrame.from_records(self.snps)

	@property
	def genotypes_onehot(self):
		"""[Getter for one-hot encoded snps format]

		Returns:
			[2D numpy.array]: [One-hot encoded numpy array (n_samples, n_variants)]
		"""
		return self.onehot
	
	@property
	def imputed_knn_df(self):
		"""[Getter for 012 genotypes with K-NN missing data imputation]

		Returns:
			[pandas.DataFrame()]: [pandas DataFrame with the imputed 012 genotypes]
		"""
		return self.knn_imputed_df

	@property
	def imputed_knn(self):
		"""[Getter for 012 genotypes with K-NN missing data imputation]

		Returns:
			[pandas.DataFrame()]: [pandas DataFrame with the imputed 012 genotypes]
		"""
		return self.knn_imputed_df.values.tolist()

	@property
	def imputed_freq_global(self):
		"""[Getter for 012 genotypes imputed by global allele frequency]

		Returns:
			[list(list)]: [Imputed genotype data]
		"""
		return self.freq_imputed_global
	
	@property
	def imputed_freq_pop(self):
		"""[Getter for 012 genotypes imputed by population allele frequency]

		Returns:
			[list(list)]: [Imputed genotype data]
		"""
		return self.freq_imputed_pop

	@property
	def imputed_freq_global_df(self):
		"""[Getter for 012 genotypes imputed by global allele frequency]

		Returns:
			[pandas.DataFrame]: [Imputed genotype data]
		"""
		return pd.DataFrame.from_records(self.freq_imputed_global)
	
	@property
	def imputed_freq_pop_df(self):
		"""[Getter for 012 genotypes imputed by population allele frequency]

		Returns:
			[pandas.DataFrame]: [Imputed genotype data]
		"""
		return pd.DataFrame.from_records(self.freq_imputed_pop)

	@property
	def imputed_rf_np(self):
		"""[Getter for 012 genotypes imputed by random forest iterative imputation]

		Returns:
			[numpy array]: [Imputed 012-encoded genotype data]
		"""
		return self.rf_imputed_arr

	@property
	def imputed_rf_df(self):
		"""[Getter for 012 genotypes imputed by random forest iterative imputation]

		Returns:
			[pandas.DataFrame]: [Imputed 012-encoded genotype data]
		"""
		return pd.DataFrame(self.rf_imputed_arr)

	@property
	def imputed_gb_np(self):
		"""[Getter for 012 genotypes imputed by gradient boosting iterative imputation]

		Returns:
			[numpy array]: [Imputed 012-encoded genotype data]
		"""
		return self.gb_imputed_arr

	@property
	def imputed_gb_df(self):
		"""[Getter for 012 genotypes imputed by gradient boosting iterative imputation]

		Returns:
			[pandas.DataFrame]: [Imputed 012-encoded genotype data]
		"""
		return pd.DataFrame(self.gb_imputed_arr)

	@property
	def imputed_br_df(self):
		return pd.DataFrame(self.br_imputed_arr)

	@property
	def imputed_br_np(self):
		return self.br_imputed_arr

	@property
	def imputed_knn_iter_df(self):
		return pd.DataFrame(self.knn_iter_imputed_arr)

	@property
	def imputed_knn_iter_np(self):
		return self.knn_iter_imputed_arr
		
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
