import sys

class ReadPopmap:
	"""[Class to read and parse a population map file]
	"""

	def __init__(self, filename):
		"""[Class constructor]

		Args:
			filename ([str]): [Filename for population map]
		"""
		self.filename = filename
		self.popdict = dict()
		if filename is not None:
			self.read_popmap()

	def read_popmap(self):
		"""[Read a population map file from disk into a dictionary]

		Args:
			popmap_filename ([str]): [Name of popmap file]

		Returns:
			[dict]: [Dictionary with indID: popID]
		"""
		with open(self.filename, "r") as fin:
			for line in fin:
				line = line.strip()
				if not line:
					continue
				cols = line.split()
				ind = cols[0]
				pop = cols[1]
				self.popdict[ind] = pop

	def validate_popmap(self, samples):
		#print(self.popdict)
		for samp in samples:
			if samp not in self.popdict:
				return(False)
		return(True)
	
	def __len__(self):
		return(len(list(self.popdict.keys())))
	
	def __getitem__(self, idx):
		if idx in self.popdict:
			return(self.popdict[idx])
		else:
			sys.exit("\nSample",idx,"not in popmap:",self.filename,"\n")
	
	def __contains__(self, idx):
		if idx in self.popdict:
			return(True)
		else:
			return(False)

	# DEPRECATED
	# def join_popmap_with_data(self, mydict, pops):
	# 	"""[Joins popmap dictionary with data dictionary]
	# 
	# 	Args:
	# 		mydict ([dict(list)]): [Data dictionary with 'samples', 'snps' keys]
	# 		pops ([dict]): [Popmap dictionary with {samples: popids}]
	# 
	# 	Returns:
	# 		[dict(list)]: [Object with data joined with popmap info]
	# 	"""
	# 	pop_list = list()
	# 	if len(pops) != len(mydict["samples"]):
	# 		sys.exit("\nError: The number of individuals in the popmap file differs from the number of sequences\n")
	# 
	# 	for sample in mydict["samples"]:
	# 		for k, v in pops.items():
	# 			if k == sample:
	# 				pop_list.append(v)
	# 	mydict["popids"] = pop_list
	# 
	# 	return mydict
