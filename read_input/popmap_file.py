import sys

class ReadPopmap:

	def __init__(self, filename):
		"""[Class constructor]

		Args:
			filename ([str]): [Filename for population map]
		"""
		self.filename = filename

	def read_popmap(self):
		"""[Read a population map file from disk into a dictionary]

		Args:
			popmap_filename ([string]): [Name of popmap file]

		Returns:
			[dict]: [Dictionary with indID: popID]
		"""
		popdict = dict()
		with open(self.filename, "r") as fin:
			for line in fin:
				line = line.strip()
				if not line:
					continue
				cols = line.split()
				ind = cols[0]
				pop = cols[1]
				popdict[ind] = pop

		return popdict

	def join_popmap_with_data(self, mydict, pops):
		"""[Joins popmap dictionary with data dictionary]

		Args:
			mydict ([dict(list)]): [Data dictionary with 'samples', 'snps' keys]
			pops ([dict]): [Popmap dictionary with {samples: popids}]

		Returns:
			[dict(list)]: [Object with data joined with popmap info]
		"""
		pop_list = list()
		if len(pops) != len(mydict["samples"]):
			sys.exit("\nError: The number of individuals in the popmap file differs from the number of sequences\n")

		for sample in mydict["samples"]:
			for k, v in pops.items():
				if k == sample:
					pop_list.append(v)
		mydict["popids"] = pop_list

		return mydict
