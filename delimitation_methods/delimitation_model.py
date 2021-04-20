import sys
import os 

from read_input.read_input import GenotypeData

class DelimitationModel:
	"""[Parent class for delimitation models]
	
	"""
	
	def __init__(self):
		self.results = list() #list of ModelResult objects
	
	#method will match labels across K -- changes self.results
	def clusterAcrossK(self, method, inplace=True):
		if inplace:
			for i in range(0, len(self.results)):
				self.results[i].clusterAcrossK(method, inplace=True)
		else:
			ret=list()
			for res in self.results:
				ret.append(res.clusterAcrossK(method, inplace=False))
			
	
class ModelResult:
	"""[Object to hold results from replicates of a single delim model]
	
	"""
	def __init__(self):
		self.model = None
		self.reps = list()
		self.num_reps = 0
	
	def clusterAcrossK(self, method, inplace=True):
		ret=list()
		for i in range(0, len(self.reps)):
			if method == "exhaustive":
				#ret.append()
				pass
			elif method == "heuristic":
				#ret.append()
				pass
			elif method == "graph":
				#ret.append()
				pass
			else:
				sys.exit("\nError: Method",method,"invalid for ModelResult.clusterAcrossK\n")
		
		if inplace:
			for i in range(0, len(ret)):
				self.reps[i] = reps[i]
		else:
			return(ret)
	