import sys

import numpy as np

from popmap_file import ReadPopmap

class StrFile:

	def __init__(self, filename, na_value):
		self.filename = filename
		self.na_value = na_value

	def read_structure_file(self, col_labels, pop_ids, first_snp_col, onerow, popmap):
		"""[Read a structure file with two rown per individual]

		Args:
			col_labels ([boolean]): [Header row with column labels? True/ False]
			pop_ids ([integer]): [Column number for population ids, starts at 1]
			first_snp_col ([integer]): [Column number of first SNP site; starts at 1]
			onerow ([boolean]): [True if one row per individual]
			popmap ([str]): [Population map filename]

		Returns:
			[dict(list), integer, integer]: [Object containing samples, popids, and snps]
		"""
		print("\nReading structure file {}...".format(self.filename))

		mydict = dict()
		samples_list = list()
		popids_list = list()
		snps_list = list()
		with open(self.filename, "r") as fin:
			if col_labels:
				first_line = fin.readline()
			for line in fin:
				line = line.strip()
				if not line:
					continue
				cols = line.split()
				samples = cols[0]
				samples_list.append(samples)
				if pop_ids: # If popid column is present.
					pop_ids = int(pop_ids)
					popids = cols[pop_ids-1]
					popids_list.append(popids)

				# Put snps into 2d list.
				first = first_snp_col - 1
				snps = cols[first:]
				snps_list.append(snps)

		# Put data into mydict object.
		mydict["samples"] = samples_list
		mydict["snps"] = snps_list

		if onerow:
			num_inds = len(mydict["snps"])
			num_snps = int(len(mydict["snps"][0]) / 2)
		else:
			num_inds = int(len(mydict["snps"]) / 2)
			num_snps = int(len(mydict["snps"][0]))

		if pop_ids:
			mydict["popids"] = popids_list

		print("\nFound {} SNPs and {} individuals...\n".format(num_snps, num_inds))

		# Make sure all sequences are the same length.
		for item in mydict["snps"]:
			try:
				assert len(item) == num_snps
			except AssertionError:
				sys.exit("\nError: There are sequences of different lengths in the structure file\n")

		return mydict, num_snps, num_inds

	def separate_structure_alleles(self, mydict, num_inds, num_snps, onerow, popmap):
		"""[Separate structure file data into allele 1 and allele 2]

		Args:
			mydict ([dict(list)]): [Object from read_structure_file()]
			num_inds ([integer]): [Number of individuals in dataset]
			num_snps ([type]): [Number of snps in dataset]
			onerow ([boolean]): [True if one row per individual]
			popmap ([str]): [Population map filename]

		Returns:
			[dict(list)]: [Object with sampleIDs, popIDs, allele1, allele2, onehot]
		"""
		ind_list = list()
		allele1 = list()
		allele2 = list()
		allele_dict = dict()
		if not onerow:
			for cnt, snp in enumerate(mydict["snps"]):
				ind = mydict["samples"][cnt]
				if ind_list:
					if ind in ind_list:
						allele2.append(snp)
					else:
						allele1.append(snp)
					
					ind_list.append(ind)
				else:
					ind_list.append(ind)
					allele1.append(snp)
			
			# Make two rows into one.
			allele_dict["samples"] = mydict["samples"][::2]

			if "popids" in mydict:
				allele_dict["popids"] = mydict["popids"][::2]
			else:
				# Join popmap file with main object.
				my_popmap = ReadPopmap(popmap)
				pops = my_popmap.read_popmap()
				mydict = my_popmap.join_popmap_with_data(allele_dict, pops)
				allele_dict["popids"] = mydict["popids"]

		else:
			for cnt, snp in enumerate(mydict["snps"]):
				ind = mydict["samples"][cnt]
				for alleles in snp[::2]:
					allele1.append(alleles)
				for alleles in snp[1::2]:
					allele2.append(alleles)

			allele_dict["samples"] = mydict["samples"]

			if "popids" in mydict:
				allele_dict["popids"] = mydict["popids"]
			else:
				my_popmap = ReadPopmap(popmap)
				pops = my_popmap.read_popmap(mydict, popmap)
				mydict = my_popmap.join_popmap_with_data(mydict, pops)
				allele_dict["popids"] = mydict["popids"]
				
		
		allele_dict["allele1"] = allele1
		allele_dict["allele2"] = allele2

		return allele_dict

	def structure2onehot(self, allele_dict):

		missing = "{}{}".format(self.na_value, self.na_value)

		onehot_dict = {
			"11": [1.0, 0.0, 0.0, 0.0], 
			"22": [0.0, 1.0, 0.0, 0.0],
			"33": [0.0, 0.0, 1.0, 0.0], 
			"44": [0.0, 0.0, 0.0, 1.0],
			missing: [0.0, 0.0, 0.0, 0.0],
			"12": [0.5, 0.5, 0.0, 0.0],
			"13": [0.5, 0.0, 0.5, 0.0],
			"14": [0.5, 0.0, 0.0, 0.5],
			"23": [0.0, 0.5, 0.5, 0.0],
			"24": [0.0, 0.5, 0.0, 0.5],
			"34": [0.0, 0.0, 0.5, 0.5]
		}
		onehot_outer_list = list()
		a_key_list = list()
		for i in range(len(allele_dict["samples"])):
			onehot_list = list()

			for j in range(len(allele_dict["allele1"][0])):
				a1 = allele_dict["allele1"][i][j]
				a2 = allele_dict["allele2"][i][j]
				a_key = "{}{}".format(a1, a2)
				onehot_list.append(onehot_dict[a_key])
			onehot_outer_list.append(onehot_list)
		allele_dict["onehot"] = onehot_outer_list

		# Convert to 3D numpy array
		allele_dict["onehot"] = np.array(allele_dict["onehot"])

		return allele_dict

